#!/usr/bin/env python3
"""Train the physics-informed POD-DeepONet volume-field operator (design doc
docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md).

--stage 1: branch-only coefficient regression (L_POD);
--stage 2: branch + trunk field training (L_POD + lambda_f * L_field);
--stage 3: physics-informed fine-tuning from a stage-2 checkpoint
           (+ lambda_phys(progress) * (L_c + L_m + L_wall + L_ff)).

Snapshots follow the npz contract of deep_sdf.cfd.volume (fields (G, 4)
[u, v, w, Cp] on the shared reference grid, bc (4,) = [U, dir], shape) and
are nondimensionalized at load (u /= U). POD is fit per variable (cPOD) on
the TRAINING shapes only; the common rank is max of the per-variable
energy-truncated ranks unless --pod_rank fixes it. With --synthetic the
physics-consistent potential-flow fields (deep_sdf.cfd.flow_synth) are
written to <experiment>/PipodONet/snapshots/ and read back through the same
npz path as real data.
"""

import argparse
import glob
import json
import logging
import os
import random
import time

import numpy as np
import torch

import deep_sdf
import deep_sdf.cfd
import deep_sdf.data
import deep_sdf.utils
import deep_sdf.workspace as ws
from deep_sdf.cfd.deeponet import BranchNet
from deep_sdf.cfd.flow_synth import parse_ellipsoid_axes, potential_flow_field
from deep_sdf.cfd.volume import (
    load_snapshot,
    make_reference_grid,
    pod_fit,
    save_snapshot,
    snapshot_filename,
)
from train_pressure_surrogate import (
    load_or_fit_latent,
    make_bc,
    sample_flow_direction,
)

FIELD_NAMES = ["u", "v", "w", "p"]


def build_shapes(args, decoder, latent_size, saved_model_epoch, npz_filenames,
                 grid_points, grid_shape, num_points, snapshots_dir, rng,
                 device):
    """Per shape: latent (loaded or fitted, train_volume_rom convention) +
    per-case snapshots (nondimensionalized at load: velocity /= U)."""
    snap_index = {}
    if args.snapshots:
        for path in sorted(glob.glob(os.path.join(snapshots_dir, "*.npz"))):
            data = np.load(path, allow_pickle=False)
            snap_index.setdefault(str(data["shape"]), []).append(path)

    shapes = []
    for npz in npz_filenames:
        if "npz" not in npz:
            continue
        logging.info("processing {}".format(npz))
        latent = load_or_fit_latent(
            args, decoder, latent_size, saved_model_epoch, npz
        )
        cases = []
        if args.synthetic:
            axes = parse_ellipsoid_axes(npz)
            case_paths = []
            for case_idx in range(args.cases_per_shape):
                direction = sample_flow_direction(rng, args.dir_cone_deg)
                velocity = rng.uniform(args.u_range[0], args.u_range[1])
                bc = make_bc(direction, velocity)
                path = os.path.join(
                    snapshots_dir, snapshot_filename(npz, case_idx))
                regenerate = True
                if os.path.isfile(path):
                    try:
                        old = load_snapshot(path, num_points)
                        regenerate = not (
                            old["shape"] == npz
                            and np.allclose(old["bc"].numpy(), bc, atol=1e-5)
                        )
                    except (ValueError, KeyError):
                        regenerate = True
                if regenerate:
                    fields = potential_flow_field(
                        axes, (0.0, 0.0, 0.0), bc, grid_points,
                        wake_amp=args.wake_amp, wake_sigma=args.wake_sigma,
                    )
                    save_snapshot(path, fields, bc, npz)
                case_paths.append(path)
            for path in case_paths:
                snap = load_snapshot(path, num_points)
                if snap["shape"] != npz:
                    raise RuntimeError(
                        "snapshot {} belongs to shape {}, expected {}".format(
                            path, snap["shape"], npz))
                cases.append({"bc": snap["bc"].to(device),
                              "fields": snap["fields"].to(device),
                              "case_id": os.path.basename(path)})
        else:
            paths = snap_index.get(npz)
            if not paths:
                raise RuntimeError(
                    "no snapshots found for shape {} in {}".format(
                        npz, snapshots_dir))
            for path in paths:
                snap = load_snapshot(path, num_points)
                cases.append({"bc": snap["bc"].to(device),
                              "fields": snap["fields"].to(device),
                              "case_id": os.path.basename(path)})
        for c in cases:
            c["fields"] = c["fields"].clone()
            c["fields"][:, :3] /= c["bc"][0]  # nondimensionalize velocity
        shapes.append({"name": npz, "latent": latent.detach(),
                       "cases": cases})
    if not shapes:
        raise RuntimeError("no shapes produced snapshots; nothing to train on")
    return shapes


def fit_pod_bases(train_cases, energy, rank, device):
    """Per-variable cPOD on the training snapshots. Returns (bases, r) with
    r the common rank (max of energy-truncated per-variable ranks unless
    ``rank`` fixes it)."""
    S = torch.stack([c["fields"] for _, c in train_cases])  # (N, G, 4)
    bases = [pod_fit(S[:, :, v], energy=energy, rank=None, device=device)
             for v in range(4)]
    r = max(b.rank for b in bases) if rank is None else rank
    bases = [pod_fit(S[:, :, v], energy=energy, rank=r, device=device)
             for v in range(4)]
    return bases, r


def set_targets(bases, model_cases):
    """True projection coefficients per case: coef (4, r)."""
    for s, c in model_cases:
        c["coef"] = torch.stack([
            bases[v].project(c["fields"][:, v].unsqueeze(0))[0]
            for v in range(4)
        ]).detach()


def set_normalizations(branch, train_shapes, train_cases, val_cases):
    train_z = torch.cat([s["latent"] for s in train_shapes], 0)
    branch.set_z_normalization(train_z.mean(0), train_z.std(0))
    train_bcs = torch.stack([c["bc"] for _, c in train_cases])
    branch.set_bc_normalization(train_bcs.mean(0), train_bcs.std(0))
    train_coefs = torch.stack([c["coef"] for _, c in train_cases])  # (N,4,r)
    branch.set_coef_normalization(train_coefs.mean(0), train_coefs.std(0))
    # 标准化目标对 train 与 val 案例都要设置（val 评估需要 target）
    for _, c in train_cases + val_cases:
        c["target"] = ((c["coef"] - branch.coef_mean)
                       / branch.coef_std).detach()


def evaluate_stage1(branch, bases, flat_cases):
    """Mean over cases: standardized coef MSE, per-variable reconstructed
    relative L2 (mean), projection lower bound, mean-field baseline."""
    loss_fn = torch.nn.MSELoss()
    coef_mses, rel_l2s, projs, baselines = [], [], [], []
    with torch.no_grad():
        for s, c in flat_cases:
            pred_n = branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0))
            coef_mses.append(loss_fn(pred_n, c["target"].unsqueeze(0)).item())
            a_pred = branch(s["latent"], c["bc"].unsqueeze(0))  # (1,4,r)
            errs, ps, bs = [], [], []
            for v in range(4):
                Y = c["fields"][:, v].unsqueeze(0)
                errs.append(bases[v].relative_error(Y, a_pred[:, v])[0].item())
                ps.append(bases[v].projection_error(Y)[0].item())
                bs.append(bases[v].relative_error(
                    Y, torch.zeros_like(a_pred[:, v]))[0].item())
            rel_l2s.append(float(np.mean(errs)))
            projs.append(float(np.mean(ps)))
            baselines.append(float(np.mean(bs)))
    return {"coef_mse": float(np.mean(coef_mses)),
            "rel_l2": float(np.mean(rel_l2s)),
            "proj": float(np.mean(projs)),
            "baseline": float(np.mean(baselines))}


def train_stage1(args, branch, branch_kwargs, bases, train_shapes,
                 train_cases, val_cases, out_dir, rng):
    optimizer = torch.optim.Adam(branch.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    start = time.time()
    loss_num = 0.0
    best = None
    ckpt_path = os.path.join(out_dir, "stage1.pth")

    def save(val_metrics=None):
        checkpoint = {
            "model_type": "pipod_deeponet_stage1",
            "model_state_dict": branch.state_dict(),
            "model_kwargs": branch_kwargs,
            "rank": bases[0].rank,
            "pod_basis_files": ["pod_basis_{}.pth".format(v)
                                for v in FIELD_NAMES],
            "bc_fields": list(deep_sdf.cfd.BC_FIELDS),
            "grid_resolution": args.grid_resolution,
            "u_range": list(args.u_range),
            "dir_cone_deg": args.dir_cone_deg,
            "wake_amp": args.wake_amp,
            "train_coef_mse": loss_num,
            "seed": args.seed,
        }
        if val_metrics is not None:
            checkpoint.update({
                "val_coef_mse": val_metrics["coef_mse"],
                "val_rel_l2": val_metrics["rel_l2"],
                "val_projection_error": val_metrics["proj"],
                "val_mean_field_rel_l2": val_metrics["baseline"],
            })
        torch.save(checkpoint, ckpt_path)

    for e in range(int(args.iterations)):
        optimizer.zero_grad()
        s, c = rng.choice(train_cases)
        pred = branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0))
        loss = loss_fn(pred, c["target"].unsqueeze(0))
        loss.backward()
        optimizer.step()
        loss_num = loss.item()
        if e % 50 == 0:
            if val_cases:
                metrics = evaluate_stage1(branch, bases, val_cases)
                logging.info(
                    "iter {} mse: {:.6e} | val coef mse: {:.6e} rel L2: "
                    "{:.6e} proj bound: {:.6e} mean-field: {:.6e}".format(
                        e, loss_num, metrics["coef_mse"], metrics["rel_l2"],
                        metrics["proj"], metrics["baseline"]))
                if best is None or metrics["rel_l2"] < best:
                    best = metrics["rel_l2"]
                    save(metrics)
            else:
                logging.info("iter {} mse: {:.6e}".format(e, loss_num))
    if val_cases and best is None:
        metrics = evaluate_stage1(branch, bases, val_cases)
        save(metrics)
    elif not val_cases:
        save()
    logging.info("stage-1 training time: {:.2f}s; saved {}".format(
        time.time() - start, ckpt_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the physics-informed POD-DeepONet volume operator "
        "(staged: 1 coefficients, 2 field, 3 physics fine-tuning)."
    )
    parser.add_argument("--experiment", "-e", dest="experiment_directory",
                        required=True)
    parser.add_argument("--checkpoint", "-c", dest="checkpoint",
                        default="latest")
    parser.add_argument("--data", "-d", dest="data_source", required=True)
    parser.add_argument("--split", "-s", dest="split_filename", required=True)
    parser.add_argument("--snapshots", dest="snapshots", default=None)
    parser.add_argument("--synthetic", dest="synthetic", action="store_true")
    parser.add_argument("--grid_resolution", type=int, default=64)
    parser.add_argument("--pod_energy", type=float, default=0.999)
    parser.add_argument("--pod_rank", type=int, default=None)
    parser.add_argument("--cases_per_shape", type=int, default=4)
    parser.add_argument("--u_range", type=float, nargs=2, default=[10.0, 20.0])
    parser.add_argument("--dir_cone_deg", type=float, default=180.0)
    parser.add_argument("--wake_amp", type=float, default=0.15)
    parser.add_argument("--wake_sigma", type=float, default=0.5)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--init_from", default=None,
                        help="checkpoint to initialize branch (stage 2) or "
                        "the full operator (stage 3) from")
    parser.add_argument("--iters", dest="iterations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--bc_hidden", type=int, default=64)
    parser.add_argument("--trunk_hidden", type=int, nargs="+",
                        default=[256, 512])
    parser.add_argument("--lambda_pod", type=float, default=1.0)
    parser.add_argument("--lambda_field", type=float, default=1.0)
    parser.add_argument("--lambda_phys", type=float, default=None,
                        help="fixed physics weight (default: the 0/0.01/0.05/"
                        "0.1 progress schedule of the design doc)")
    parser.add_argument("--re", type=float, default=1e4)
    parser.add_argument("--n_field", type=int, default=16384)
    parser.add_argument("--n_collocation", type=int, default=4096)
    parser.add_argument("--phys_chunk", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--wall_bc", choices=["slip", "noslip"],
                        default="slip")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--reconstruct_iters", dest="reconstruct_iterations",
                        type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    deep_sdf.add_common_args(parser)
    args = parser.parse_args()
    deep_sdf.configure_logging(args)

    if bool(args.snapshots) == bool(args.synthetic):
        raise RuntimeError("pass exactly one of --snapshots <dir> or --synthetic")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs = json.load(open(os.path.join(
        args.experiment_directory, "specs.json")))
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(os.path.join(
        args.experiment_directory, ws.model_params_subdir,
        args.checkpoint + ".pth"))
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda().eval()
    for param in decoder.parameters():
        param.requires_grad = False

    with open(args.split_filename) as f:
        split = json.load(f)
    npz_filenames = deep_sdf.data.get_instance_filenames(
        args.data_source, split)

    device = torch.device("cuda")
    out_dir = os.path.join(args.experiment_directory, "PipodONet")
    os.makedirs(out_dir, exist_ok=True)
    snapshots_dir = (args.snapshots if args.snapshots
                     else os.path.join(out_dir, "snapshots"))
    if args.synthetic:
        os.makedirs(snapshots_dir, exist_ok=True)

    grid_points, grid_shape = make_reference_grid(args.grid_resolution)
    grid_points = grid_points.to(device)
    num_points = grid_points.shape[0]

    rng = random.Random(args.seed)
    shapes = build_shapes(args, decoder, latent_size, saved_model_epoch,
                          npz_filenames, grid_points, grid_shape, num_points,
                          snapshots_dir, rng, device)

    indices = list(range(len(shapes)))
    rng.shuffle(indices)
    num_val = 0
    if args.val_fraction > 0.0 and len(shapes) >= 2:
        num_val = max(1, min(len(shapes) - 1,
                             int(round(args.val_fraction * len(shapes)))))
    val_idx = set(indices[:num_val])
    train_shapes = [s for i, s in enumerate(shapes) if i not in val_idx]
    val_shapes = [s for i, s in enumerate(shapes) if i in val_idx]
    if not val_shapes:
        train_shapes = shapes
    train_cases = [(s, c) for s in train_shapes for c in s["cases"]]
    val_cases = [(s, c) for s in val_shapes for c in s["cases"]]
    logging.info("snapshot ensemble: %d train / %d val cases",
                 len(train_cases), len(val_cases))

    bases, rank = fit_pod_bases(train_cases, args.pod_energy, args.pod_rank,
                                device)
    for v, basis in zip(FIELD_NAMES, bases):
        basis.save(os.path.join(out_dir, "pod_basis_{}.pth".format(v)))
        logging.info(
            "train projection error ({}): {:.6e}".format(
                v, basis.projection_error(torch.stack(
                    [c["fields"][:, FIELD_NAMES.index(v)]
                     for _, c in train_cases])).mean().item()))
    logging.info("common POD rank r = %d", rank)
    set_targets(bases, train_cases + val_cases)

    branch_kwargs = {
        "latent_size": latent_size, "bc_dim": len(deep_sdf.cfd.BC_FIELDS),
        "rank": rank, "n_outputs": 4, "hidden": args.hidden,
        "num_layers": args.num_layers, "bc_hidden": args.bc_hidden,
    }
    branch = BranchNet(**branch_kwargs).to(device)
    set_normalizations(branch, train_shapes, train_cases, val_cases)

    if args.stage == 1:
        train_stage1(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, out_dir, rng)
    else:
        raise SystemExit(
            "stage {} is implemented in a later task of the plan".format(
                args.stage))

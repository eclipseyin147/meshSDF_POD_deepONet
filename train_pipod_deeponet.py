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

With --latent_manifest <npz> (generate_openfoam_snapshots.save_manifest
format: names + latents) the shape list and latents come from the manifest
instead of the split: z is taken directly (reshape (1, L)), skipping
load_or_fit_latent's reconstruction. Mutually exclusive with --synthetic;
snapshots are still indexed by shape name from --snapshots.

Every stage appends evaluation records to
<experiment>/PipodONet/metrics_stage<1|2|3>.jsonl (overwritten per run).
"""

import argparse
import glob
import json
import logging
import math
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
from deep_sdf.cfd.deeponet import BranchNet, PODDeepONet, TrunkNet
from deep_sdf.cfd.flow_synth import parse_ellipsoid_axes, potential_flow_field
from deep_sdf.cfd.physics import (
    CollocationSampler,
    FluidMaskEmpty,
    IncompressibleNS,
    PDEInformer,
    farfield_loss,
    noslip_loss,
    physics_weight_schedule,
    wall_slip_loss,
)
from deep_sdf.cfd.volume import (
    load_snapshot,
    make_reference_grid,
    pod_fit,
    save_snapshot,
    snapshot_filename,
)
from deep_sdf.differentiable_mesh import compute_sdf_gradients
from generate_openfoam_snapshots import load_manifest
from train_pressure_surrogate import (
    load_or_fit_latent,
    make_bc,
    sample_flow_direction,
)

FIELD_NAMES = ["u", "v", "w", "p"]


def make_scheduler(optimizer, args):
    """LR schedule: 'constant' (default, old behavior) or 'cosine' with a
    2% warmup and a floor at lr * lr_final_ratio."""
    if args.lr_schedule == "constant":
        return None
    warmup = max(1, int(0.02 * args.iterations))

    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / warmup
        t = min(1.0, (e - warmup) / max(1, args.iterations - warmup))
        floor = args.lr_final_ratio
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class MetricsLogger:
    """Per-stage JSONL metrics log at <out_dir>/metrics_stage<N>.jsonl.

    The file is created (overwritten) at the start of every run and flushed
    per record so training curves can be followed live. NaN/Inf floats are
    written as null to keep every line strict JSON."""

    def __init__(self, out_dir, stage):
        self.path = os.path.join(out_dir, "metrics_stage{}.jsonl".format(stage))
        self.handle = open(self.path, "w")

    @staticmethod
    def _clean(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, (list, tuple)):
            return [MetricsLogger._clean(v) for v in value]
        return value

    def log(self, record):
        self.handle.write(json.dumps({k: self._clean(v)
                                      for k, v in record.items()}) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def _val_record(metrics):
    return {"val_" + k: v for k, v in metrics.items()}


def build_shapes(args, decoder, latent_size, saved_model_epoch, npz_filenames,
                 grid_points, grid_shape, num_points, snapshots_dir, rng,
                 device):
    """Per shape: latent (loaded or fitted, train_volume_rom convention) +
    per-case snapshots (nondimensionalized at load: velocity /= U).

    Shape enumeration has two branches: the default enumerates the split
    npz names and obtains each latent via load_or_fit_latent; with
    --latent_manifest the shapes and latents come straight from the manifest
    (z reshaped to (1, L), no reconstruction)."""
    snap_index = {}
    if args.snapshots:
        for path in sorted(glob.glob(os.path.join(snapshots_dir, "*.npz"))):
            data = np.load(path, allow_pickle=False)
            snap_index.setdefault(str(data["shape"]), []).append(path)

    if args.latent_manifest:
        names, latents = load_manifest(args.latent_manifest)
        if latents.ndim != 2 or latents.shape[1] != latent_size:
            raise RuntimeError(
                "manifest {} latents have shape {}, expected (N, {})".format(
                    args.latent_manifest, latents.shape, latent_size))
        logging.info("latent manifest {}: {} shapes, latents taken directly "
                     "(reconstruction skipped)".format(args.latent_manifest,
                                                       len(names)))
        shape_entries = [
            (name, torch.from_numpy(latents[i]).reshape(
                1, latent_size).float().to(device))
            for i, name in enumerate(names)]
    else:
        shape_entries = [(npz, None) for npz in npz_filenames if "npz" in npz]

    shapes = []
    for npz, manifest_latent in shape_entries:
        logging.info("processing {}".format(npz))
        if manifest_latent is not None:
            latent = manifest_latent
        else:
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
                              "fields": snap["fields"],  # CPU: moved per-use
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
                              "fields": snap["fields"],  # CPU: moved per-use
                              "case_id": os.path.basename(path)})
        for c in cases:
            c["fields"] = c["fields"].clone()
            c["fields"][:, :3] /= c["bc"][0].cpu()  # nondim. velocity
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
            bases[v].project(c["fields"][:, v].unsqueeze(0)
                             .to(bases[v].mean.device))[0]
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
                Y = c["fields"][:, v].unsqueeze(0).to(
                    bases[v].mean.device)
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


def build_shape_geometry(decoder, shapes, grid_points, grid_shape,
                         max_batch=2 ** 18, near_band=None):
    """Per-shape detached SDF + SDF-gradient features on the reference grid
    (data-loss branch), cached on the shape dict. With ``near_band`` (an
    absolute |sdf| threshold), also caches ``s["near_idx"]`` (CPU int64 of
    grid points within the band) for importance-sampled field losses."""
    spacing = float(grid_points[:, 0].max() - grid_points[:, 0].min())
    spacing /= max(grid_shape[0] - 1, 1)
    for s in shapes:
        if "sdf" not in s:
            sds = []
            with torch.no_grad():
                head = 0
                while head < grid_points.shape[0]:
                    chunk = grid_points[head:head + max_batch]
                    sds.append(deep_sdf.utils.decode_sdf(
                        decoder, s["latent"], chunk).squeeze(1).float())
                    head += max_batch
            s["sdf"] = torch.cat(sds, 0)
            s["sdf_grad"] = compute_sdf_gradients(
                decoder, s["latent"], grid_points, max_batch)
            s["h"] = spacing
        if near_band is not None and "near_idx" not in s:
            near = torch.nonzero(
                s["sdf"].abs().cpu() < near_band).squeeze(1)
            if near.numel() == 0:
                logging.warning("near band %.4g empty for a shape; falling "
                                "back to uniform sampling for it", near_band)
            else:
                s["near_idx"] = near


def make_features(grid_points, shape, idx):
    """(n, 8) trunk features [x, y, z, sdf, dsdf/dx, dsdf/dy, dsdf/dz, h]."""
    n = idx.numel()
    return torch.cat([
        grid_points[idx],
        shape["sdf"][idx].unsqueeze(1),
        shape["sdf_grad"][idx],
        torch.full((n, 1), shape["h"], device=grid_points.device),
    ], dim=1)


def sample_field_idx(shape, num_points, n, gen, near_frac=0.0):
    """Field-loss point indices: ``near_frac`` of the batch from the shape's
    near-wall band (s["near_idx"], requires build_shape_geometry(near_band)),
    the rest uniform. near_frac=0 reproduces the old uniform behavior."""
    n_near = int(n * near_frac)
    if n_near <= 0 or "near_idx" not in shape:
        return torch.randint(num_points, (n,), generator=gen)
    parts = []
    if n_near < n:
        parts.append(torch.randint(num_points, (n - n_near,), generator=gen))
    sel = torch.randint(shape["near_idx"].numel(), (n_near,), generator=gen)
    parts.append(shape["near_idx"][sel])
    return torch.cat(parts)

def predict_field(model, shape, bc, grid_points, chunk=2 ** 16):
    """Full-grid field prediction (no_grad, chunked) -> (G, 4)."""
    outs = []
    with torch.no_grad():
        for i in range(0, grid_points.shape[0], chunk):
            idx = torch.arange(i, min(i + chunk, grid_points.shape[0]),
                               device=grid_points.device)
            outs.append(model(shape["latent"], bc.unsqueeze(0),
                              make_features(grid_points, shape, idx)))
    return torch.cat(outs, 0)


def evaluate_field(model, bases, flat_cases, grid_points):
    """Per-variable and mean relative L2 of the full-grid prediction, plus
    the projection lower bound and the mean-field baseline (both from the
    POD bases) and the standardized coefficient MSE."""
    loss_fn = torch.nn.MSELoss()
    rel_l2_v, projs, baselines, coef_mses = [], [], [], []
    for s, c in flat_cases:
        pred = predict_field(model, s, c["bc"], grid_points)
        truth = c["fields"].to(grid_points.device)
        per_var = ((pred - truth).pow(2).sum(0)
                   / truth.pow(2).sum(0).clamp_min(1e-30)).sqrt()
        rel_l2_v.append(per_var.cpu())
        coef_mses.append(loss_fn(
            model.branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0)),
            c["target"].unsqueeze(0)).item())
        ps, bs = [], []
        for v in range(4):
            Y = truth[:, v].unsqueeze(0)
            ps.append(bases[v].projection_error(Y)[0].item())
            bs.append(bases[v].relative_error(
                Y, torch.zeros(1, bases[v].rank, device=Y.device))[0].item())
        projs.append(float(np.mean(ps)))
        baselines.append(float(np.mean(bs)))
    per_var = torch.stack(rel_l2_v).mean(0)
    return {"coef_mse": float(np.mean(coef_mses)),
            "rel_l2": float(per_var.mean()),
            "rel_l2_per_var": [float(x) for x in per_var],
            "proj": float(np.mean(projs)),
            "baseline": float(np.mean(baselines))}


def load_branch_from_checkpoint(branch, path):
    """Initialize branch weights from a stage-1 checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_type") != "pipod_deeponet_stage1":
        raise ValueError("stage-2 --init_from expects a stage-1 checkpoint, "
                         "got {}".format(state.get("model_type")))
    branch.load_state_dict(state["model_state_dict"])


def train_stage2(args, branch, branch_kwargs, bases, train_shapes,
                 train_cases, val_cases, shapes, decoder, grid_points,
                 grid_shape, out_dir, rng, device):
    """Branch + trunk field training: L = lambda_pod * L_POD + lambda_field *
    L_field (per-variable pointwise MSE summed over variables, the
    ChannelwiseMSE convention of mPOD-DeepONet)."""
    if args.init_from:
        load_branch_from_checkpoint(branch, args.init_from)
        logging.info("initialized branch from {}".format(args.init_from))
    trunk = TrunkNet(in_dim=8, rank=branch_kwargs["rank"], n_outputs=4,
                     hidden_sizes=tuple(args.trunk_hidden))
    model = PODDeepONet(branch, trunk).to(device)
    build_shape_geometry(decoder, shapes, grid_points, grid_shape,
                         near_band=(args.field_near_band
                                    if args.field_near_frac > 0 else None))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(args.seed)
    num_points = grid_points.shape[0]
    start = time.time()
    loss_num = 0.0
    best = None
    ckpt_path = os.path.join(out_dir, "stage2.pth")
    metrics_log = MetricsLogger(out_dir, 2)

    def save(val_metrics=None):
        checkpoint = {
            "model_type": "pipod_deeponet_stage2",
            "model_state_dict": model.state_dict(),
            "branch_kwargs": branch_kwargs,
            "trunk_kwargs": {"in_dim": 8, "rank": branch_kwargs["rank"],
                             "n_outputs": 4,
                             "hidden_sizes": list(args.trunk_hidden)},
            "rank": branch_kwargs["rank"],
            "pod_basis_files": ["pod_basis_{}.pth".format(v)
                                for v in FIELD_NAMES],
            "bc_fields": list(deep_sdf.cfd.BC_FIELDS),
            "grid_resolution": args.grid_resolution,
            "train_loss": loss_num,
            "seed": args.seed,
        }
        if val_metrics is not None:
            checkpoint.update({
                "val_coef_mse": val_metrics["coef_mse"],
                "val_rel_l2": val_metrics["rel_l2"],
                "val_rel_l2_per_var": val_metrics["rel_l2_per_var"],
                "val_projection_error": val_metrics["proj"],
                "val_mean_field_rel_l2": val_metrics["baseline"],
            })
        torch.save(checkpoint, ckpt_path)

    for e in range(int(args.iterations)):
        optimizer.zero_grad()
        s, c = rng.choice(train_cases)
        idx = sample_field_idx(s, num_points, args.n_field, gen,
                               args.field_near_frac)
        q = model(s["latent"], c["bc"].unsqueeze(0),
                  make_features(grid_points, s, idx.to(device)))
        loss_field = ((q - c["fields"][idx].to(device)) ** 2
                      ).mean(dim=0).sum()
        loss_pod = loss_fn(
            model.branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0)),
            c["target"].unsqueeze(0))
        loss = args.lambda_pod * loss_pod + args.lambda_field * loss_field
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_num = loss.item()
        if e % 50 == 0:
            if val_cases:
                metrics = evaluate_field(model, bases, val_cases, grid_points)
                metrics_log.log({"iter": e, "loss": loss_num,
                                 "lr": optimizer.param_groups[0]["lr"],
                                 **_val_record(metrics)})
                logging.info(
                    "iter {} loss: {:.6e} (pod {:.6e} field {:.6e}) | val rel "
                    "L2: {:.6e} per-var {} proj: {:.6e} mean-field: "
                    "{:.6e}".format(e, loss_num, loss_pod.item(),
                                    loss_field.item(), metrics["rel_l2"],
                                    ["%.3f" % v for v in
                                     metrics["rel_l2_per_var"]],
                                    metrics["proj"], metrics["baseline"]))
                if best is None or metrics["rel_l2"] < best:
                    best = metrics["rel_l2"]
                    save(metrics)
            else:
                logging.info("iter {} loss: {:.6e}".format(e, loss_num))
    if val_cases and best is None:
        metrics = evaluate_field(model, bases, val_cases, grid_points)
        best = metrics["rel_l2"]
        save(metrics)
    elif not val_cases:
        save()
    metrics_log.log({"final": True, "time_s": time.time() - start,
                     "best_val_rel_l2": best})
    metrics_log.close()
    logging.info("stage-2 training time: {:.2f}s; saved {}".format(
        time.time() - start, ckpt_path))


def physics_losses(model, informer, decoder, latent, bc, points, h,
                   max_batch):
    """Continuity + momentum residuals at collocation points (second-order
    autodiff). d is evaluated through the frozen decoder WITH the graph
    attached; grad d is detached (the ReLU decoder's second derivative
    vanishes a.e.). Losses accumulate over chunks, normalized by count."""
    device = points.device
    lc = torch.zeros((), device=device)
    lm = torch.zeros((), device=device)
    n_tot = 0
    for chunk in points.split(max_batch):
        p = chunk.detach().requires_grad_(True)
        with torch.enable_grad():
            d = deep_sdf.utils.decode_sdf(decoder, latent, p)
            g = torch.autograd.grad(d.sum(), p, create_graph=True)[0]
            feats = torch.cat(
                [p, d, g.detach(), torch.full_like(d, h)], dim=1)
            q = model(latent, bc, feats)
            res = informer({"coordinates": p, "u": q[:, 0:1], "v": q[:, 1:2],
                            "w": q[:, 2:3], "cp": q[:, 3:4]})
            lc = lc + (res["continuity"] ** 2).sum()
            lm = lm + sum((res["momentum_" + k] ** 2).sum() for k in "uvw")
        n_tot += p.shape[0]
    return lc / max(n_tot, 1), lm / max(n_tot, 1)


def boundary_losses(model, shape, bc, grid_points, idx_wall, idx_far,
                    wall_bc):
    """Wall loss on the near-wall band (slip: (u.n)^2 with n from grad SDF;
    noslip: |u|^2) + far-field loss (u -> unit flow direction)."""
    l_wall = torch.zeros((), device=grid_points.device)
    l_far = torch.zeros((), device=grid_points.device)
    if idx_wall.numel():
        q_w = model(shape["latent"], bc.unsqueeze(0),
                    make_features(grid_points, shape, idx_wall))
        if wall_bc == "slip":
            n_w = shape["sdf_grad"][idx_wall]
            n_w = n_w / n_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
            l_wall = wall_slip_loss(q_w, n_w)
        else:
            l_wall = noslip_loss(q_w)
    if idx_far.numel():
        q_f = model(shape["latent"], bc.unsqueeze(0),
                    make_features(grid_points, shape, idx_far))
        d = bc[1:4]
        l_far = farfield_loss(q_f, d / d.norm().clamp_min(1e-12))
    return l_wall, l_far


def load_operator_from_checkpoint(model, path):
    """Initialize the full PODDeepONet from a stage-2 checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_type") != "pipod_deeponet_stage2":
        raise ValueError("stage-3 --init_from expects a stage-2 checkpoint, "
                         "got {}".format(state.get("model_type")))
    model.load_state_dict(state["model_state_dict"])


def evaluate_physics(model, informer, decoder, bases, val_cases, grid_points,
                     grid_shape, sampler, args, device):
    """evaluate_field + physics residuals on a fixed-seed val collocation
    set + wall violation."""
    metrics = evaluate_field(model, bases, val_cases, grid_points)
    conts, moms, walls = [], [], []
    gen = torch.Generator().manual_seed(args.seed + 12345)  # 固定评估集
    for s, c in val_cases:
        try:
            picks = sampler.sample(grid_points, grid_shape, s["sdf"],
                                   c["fields"].to(device), c["bc"],
                                   args.n_collocation, gen)
        except FluidMaskEmpty:
            continue
        with torch.enable_grad():
            lc, lm = physics_losses(model, informer, decoder, s["latent"],
                                    c["bc"].unsqueeze(0),
                                    grid_points[picks["collocation"]], s["h"],
                                    args.phys_chunk)
        conts.append(lc.sqrt().item())
        moms.append(lm.sqrt().item())
        if picks["wall"].numel():
            with torch.no_grad():
                q_w = model(s["latent"], c["bc"].unsqueeze(0),
                            make_features(grid_points, s, picks["wall"]))
                n_w = s["sdf_grad"][picks["wall"]]
                n_w = n_w / n_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
                walls.append(
                    ((q_w[:, :3] * n_w).sum(1).abs()
                     / q_w[:, :3].norm(dim=1).clamp_min(1e-12)
                     ).mean().item())
    metrics["continuity"] = float(np.mean(conts)) if conts else float("nan")
    metrics["momentum"] = float(np.mean(moms)) if moms else float("nan")
    metrics["wall"] = float(np.mean(walls)) if walls else float("nan")
    return metrics


def train_stage3(args, branch, branch_kwargs, bases, train_shapes,
                 train_cases, val_cases, shapes, decoder, grid_points,
                 grid_shape, out_dir, rng, device):
    """Physics-informed fine-tuning from a stage-2 checkpoint:
    L = lambda_pod L_POD + lambda_field L_field
        + lambda_phys(progress) * (L_c + L_m + L_wall + L_ff)."""
    trunk = TrunkNet(in_dim=8, rank=branch_kwargs["rank"], n_outputs=4,
                     hidden_sizes=tuple(args.trunk_hidden))
    model = PODDeepONet(branch, trunk).to(device)
    if not args.init_from:
        raise SystemExit("stage 3 requires --init_from <stage2.pth>")
    load_operator_from_checkpoint(model, args.init_from)
    logging.info("initialized operator from {}".format(args.init_from))
    build_shape_geometry(decoder, shapes, grid_points, grid_shape,
                         near_band=(args.field_near_band
                                    if args.field_near_frac > 0 else None))
    informer = PDEInformer(IncompressibleNS(re=args.re).equations)
    sampler = CollocationSampler(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(args.seed)
    num_points = grid_points.shape[0]
    start = time.time()
    loss_num = 0.0
    best = None
    ckpt_path = os.path.join(out_dir, "stage3.pth")
    metrics_log = MetricsLogger(out_dir, 3)

    def save(val_metrics=None):
        checkpoint = {
            "model_type": "pipod_deeponet_stage3",
            "model_state_dict": model.state_dict(),
            "branch_kwargs": branch_kwargs,
            "trunk_kwargs": {"in_dim": 8, "rank": branch_kwargs["rank"],
                             "n_outputs": 4,
                             "hidden_sizes": list(args.trunk_hidden)},
            "rank": branch_kwargs["rank"],
            "re": args.re,
            "pod_basis_files": ["pod_basis_{}.pth".format(v)
                                for v in FIELD_NAMES],
            "bc_fields": list(deep_sdf.cfd.BC_FIELDS),
            "grid_resolution": args.grid_resolution,
            "train_loss": loss_num,
            "seed": args.seed,
        }
        if val_metrics is not None:
            checkpoint.update({
                "val_rel_l2": val_metrics["rel_l2"],
                "val_rel_l2_per_var": val_metrics["rel_l2_per_var"],
                "val_projection_error": val_metrics["proj"],
                "val_mean_field_rel_l2": val_metrics["baseline"],
                "val_continuity": val_metrics["continuity"],
                "val_momentum": val_metrics["momentum"],
                "val_wall": val_metrics["wall"],
            })
        torch.save(checkpoint, ckpt_path)

    for e in range(int(args.iterations)):
        progress = e / max(int(args.iterations), 1)
        lam = (args.lambda_phys if args.lambda_phys is not None
               else physics_weight_schedule(progress))
        optimizer.zero_grad()
        s, c = rng.choice(train_cases)
        idx = sample_field_idx(s, num_points, args.n_field, gen,
                               args.field_near_frac)
        q = model(s["latent"], c["bc"].unsqueeze(0),
                  make_features(grid_points, s, idx.to(device)))
        loss_field = ((q - c["fields"][idx].to(device)) ** 2
                      ).mean(dim=0).sum()
        loss_pod = loss_fn(
            model.branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0)),
            c["target"].unsqueeze(0))
        loss = args.lambda_pod * loss_pod + args.lambda_field * loss_field
        phys_terms = {}
        if lam > 0.0:
            try:
                picks = sampler.sample(grid_points, grid_shape, s["sdf"],
                                       c["fields"].to(device), c["bc"],
                                       args.n_collocation, gen)
            except FluidMaskEmpty:
                logging.warning("case {} has no fluid points; skipping "
                                "physics".format(c["case_id"]))
                picks = None
            if picks is not None:
                lc, lm = physics_losses(
                    model, informer, decoder, s["latent"],
                    c["bc"].unsqueeze(0),
                    grid_points[picks["collocation"]], s["h"],
                    args.phys_chunk)
                lw, lf = boundary_losses(model, s, c["bc"], grid_points,
                                         picks["wall"], picks["far"],
                                         args.wall_bc)
                loss = loss + lam * (lc + lm + lw + lf)
                phys_terms = {"cont": lc.item(), "mom": lm.item(),
                              "wall": lw.item(), "far": lf.item()}
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_num = loss.item()
        if e % 50 == 0:
            if val_cases:
                metrics = evaluate_physics(model, informer, decoder, bases,
                                           val_cases, grid_points,
                                           grid_shape, sampler, args, device)
                metrics_log.log({"iter": e, "loss": loss_num,
                                 "lr": optimizer.param_groups[0]["lr"],
                                 "lambda_phys": lam,
                                 **_val_record(metrics)})
                logging.info(
                    "iter {} loss: {:.6e} lam: {:.3g} {} | val rel L2: "
                    "{:.6e} val cont: {:.6e} mom: {:.6e} wall: "
                    "{:.6e}".format(e, loss_num, lam,
                                    " ".join("{}={:.2e}".format(k, v)
                                             for k, v in phys_terms.items()),
                                    metrics["rel_l2"], metrics["continuity"],
                                    metrics["momentum"], metrics["wall"]))
                if best is None or metrics["rel_l2"] < best:
                    best = metrics["rel_l2"]
                    save(metrics)
            else:
                logging.info("iter {} loss: {:.6e} lam: {:.3g}".format(
                    e, loss_num, lam))
    if val_cases and best is None:
        metrics = evaluate_physics(model, informer, decoder, bases, val_cases,
                                   grid_points, grid_shape, sampler, args,
                                   device)
        best = metrics["rel_l2"]
        save(metrics)
    elif not val_cases:
        save()
    metrics_log.log({"final": True, "time_s": time.time() - start,
                     "best_val_rel_l2": best})
    metrics_log.close()
    logging.info("stage-3 training time: {:.2f}s; saved {}".format(
        time.time() - start, ckpt_path))


def train_stage1(args, branch, branch_kwargs, bases, train_shapes,
                 train_cases, val_cases, out_dir, rng):
    optimizer = torch.optim.Adam(branch.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args)
    loss_fn = torch.nn.MSELoss()
    start = time.time()
    loss_num = 0.0
    best = None
    ckpt_path = os.path.join(out_dir, "stage1.pth")
    metrics_log = MetricsLogger(out_dir, 1)

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
        if scheduler is not None:
            scheduler.step()
        loss_num = loss.item()
        if e % 50 == 0:
            if val_cases:
                metrics = evaluate_stage1(branch, bases, val_cases)
                metrics_log.log({"iter": e, "loss": loss_num,
                                 "lr": optimizer.param_groups[0]["lr"],
                                 **_val_record(metrics)})
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
        best = metrics["rel_l2"]
        save(metrics)
    elif not val_cases:
        save()
    metrics_log.log({"final": True, "time_s": time.time() - start,
                     "best_val_rel_l2": best})
    metrics_log.close()
    logging.info("stage-1 training time: {:.2f}s; saved {}".format(
        time.time() - start, ckpt_path))


if __name__ == "__main__":
    # Pass 1: --config / --stage only, so the JSON can seed parser defaults
    # (CLI flags still override; see pass 2 below).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                     help="JSON config: top-level keys are defaults for any "
                     "CLI flag (dest or dashed name); an optional \"stages\" "
                     "list holds per-stage overrides selected by --stage")
    pre.add_argument("--stage", type=int, default=None)
    pre_args, _ = pre.parse_known_args()

    cfg = {}
    if pre_args.config:
        raw = json.load(open(pre_args.config))
        stage_no = pre_args.stage or raw.get("stage", 1)
        stages = {int(s["stage"]): s for s in raw.pop("stages", [])}
        raw.pop("stage", None)
        cfg.update(raw)
        if stage_no in stages:
            cfg.update({k: v for k, v in stages[stage_no].items()
                        if k != "stage"})
        cfg["stage"] = stage_no

    parser = argparse.ArgumentParser(
        description="Train the physics-informed POD-DeepONet volume operator "
        "(staged: 1 coefficients, 2 field, 3 physics fine-tuning)."
    )
    parser.add_argument("--config", default=None,
                        help="JSON config file (see pass-1 help); CLI flags "
                        "override config values")
    parser.add_argument("--experiment", "-e", dest="experiment_directory",
                        required=True)
    parser.add_argument("--checkpoint", "-c", dest="checkpoint",
                        default="latest")
    parser.add_argument("--data", "-d", dest="data_source", required=True)
    parser.add_argument("--split", "-s", dest="split_filename", required=True)
    parser.add_argument("--snapshots", dest="snapshots", default=None)
    parser.add_argument("--latent_manifest", dest="latent_manifest",
                        default=None,
                        help="npz manifest (names + latents, the "
                        "generate_openfoam_snapshots.save_manifest format): "
                        "shapes and latents are taken from the manifest "
                        "instead of the split, skipping latent "
                        "reconstruction; requires --snapshots, mutually "
                        "exclusive with --synthetic")
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
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"],
                        default="constant",
                        help="'constant' (old behavior) or 'cosine' with 2%% "
                        "warmup decaying to lr*lr_final_ratio at the last iter")
    parser.add_argument("--lr_final_ratio", type=float, default=0.01)
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
    parser.add_argument("--field_near_frac", type=float, default=0.0,
                        help="fraction of each field-loss batch sampled from "
                        "the shape's near-wall band (0 = old uniform "
                        "sampling)")
    parser.add_argument("--field_near_band", type=float, default=0.15,
                        help="absolute |sdf| threshold of the near-wall band "
                        "used when field_near_frac > 0")
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

    if cfg:
        aliases = {"experiment": "experiment_directory",
                   "data": "data_source",
                   "split": "split_filename",
                   "iters": "iterations"}
        actions = {a.dest: a for a in parser._actions}
        resolved, unknown = {}, []
        for k, v in cfg.items():
            dest = aliases.get(k, k)
            (resolved.__setitem__(dest, v) if dest in actions
             else unknown.append(k))
        if unknown:
            raise RuntimeError("config key(s) not matching any CLI flag: "
                               + ", ".join(sorted(unknown)))
        parser.set_defaults(**resolved)
        for dest in ("experiment_directory", "data_source", "split_filename"):
            if dest in resolved:
                actions[dest].required = False
    args = parser.parse_args()
    deep_sdf.configure_logging(args)

    # Stage chaining: default --init_from to the previous stage checkpoint.
    if args.config and args.stage > 1 and args.init_from is None:
        chained = os.path.join(args.experiment_directory, "PipodONet",
                               f"stage{args.stage - 1}.pth")
        if os.path.isfile(chained):
            args.init_from = chained
        else:
            logging.warning("stage %d: chained checkpoint %s not found; "
                            "pass --init_from explicitly", args.stage, chained)

    if bool(args.snapshots) == bool(args.synthetic):
        raise RuntimeError("pass exactly one of --snapshots <dir> or --synthetic")
    if args.latent_manifest and args.synthetic:
        raise RuntimeError("--synthetic and --latent_manifest are mutually "
                           "exclusive (synthetic shapes come from the split)")

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

    npz_filenames = None
    if not args.latent_manifest:
        with open(args.split_filename) as f:
            split = json.load(f)
        npz_filenames = deep_sdf.data.get_instance_filenames(
            args.data_source, split)

    device = torch.device("cuda")
    out_dir = os.path.join(args.experiment_directory, "PipodONet")
    os.makedirs(out_dir, exist_ok=True)
    if args.config:
        with open(os.path.join(out_dir, f"config_stage{args.stage}.json"),
                  "w") as f:
            json.dump({k: v for k, v in vars(args).items()
                       if isinstance(v, (int, float, str, bool, list))
                       or v is None}, f, indent=1, sort_keys=True)
            f.write("\n")
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
                     for _, c in train_cases]).to(
                         basis.mean.device)).mean().item()))
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
    elif args.stage == 2:
        train_stage2(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, shapes, decoder, grid_points,
                     grid_shape, out_dir, rng, device)
    else:
        train_stage3(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, shapes, decoder, grid_points,
                     grid_shape, out_dir, rng, device)

#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

"""Train the volume-field POD + coefficient-regression ROM (design doc
docs/superpowers/specs/2026-09-10-volume-pod-rom-design.md).

Pipeline: per-shape latents are loaded/fitted exactly as in
``train_pressure_surrogate.py`` (``load_or_fit_latent``); volume snapshots
(fields (G, 4) [u, v, w, p] on the shared reference grid + bc [U, dir] per
case, one npz per (shape, case)) come either from a snapshot directory
(``--snapshots``, real or pre-generated data) or are generated on the fly
from the decoder SDF with the synthetic stand-in field (``--synthetic``,
pipeline validation only - written to ``<experiment>/VolumeROM/snapshots/``
and then loaded through the same npz path as real data). A POD basis is fit
on the *training* shapes' snapshots only (GPU randomized SVD,
``pod_basis.pth``), and a ``VolumeCoefficientRegressor`` learns (z, BC) ->
POD coefficients with a standardized-coefficient MSE. Validation is by
held-out shapes (--val_fraction): coefficient MSE, reconstructed full-field
relative L2, the true-projection lower bound and the predict-the-mean-field
baseline are reported side by side; the checkpoint (``latest.pth``) is saved
on the best validation field error.
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
import deep_sdf.workspace as ws
from train_pressure_surrogate import (
    load_or_fit_latent,
    make_bc,
    sample_flow_direction,
)
from deep_sdf.cfd.volume import load_snapshot, save_snapshot, snapshot_filename


def evaluate_cases(model, basis, flat_cases):
    """Mean metrics over (shape, case) pairs: standardized coefficient MSE,
    reconstructed full-field relative L2, the true-projection lower bound
    and the predict-the-mean-field baseline."""
    loss_fn = torch.nn.MSELoss()
    coef_mses, rel_l2s, proj_errs, baselines = [], [], [], []
    with torch.no_grad():
        for s, c in flat_cases:
            a_pred = model(s["latent"], c["bc"])
            coef_mses.append(
                loss_fn(
                    model.forward_normalized(s["latent"], c["bc"]),
                    c["target"].unsqueeze(0),
                ).item()
            )
            Y = c["fields"].unsqueeze(0)
            rel_l2s.append(basis.relative_error(Y, a_pred)[0].item())
            proj_errs.append(basis.projection_error(Y)[0].item())
            baselines.append(
                basis.relative_error(Y, torch.zeros_like(a_pred))[0].item()
            )
    return {
        "coef_mse": float(np.mean(coef_mses)),
        "rel_l2": float(np.mean(rel_l2s)),
        "proj": float(np.mean(proj_errs)),
        "baseline": float(np.mean(baselines)),
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Train the volume-field POD + coefficient-regression ROM "
        + "over the shapes of a split: fit a POD basis of the training "
        + "snapshots on the GPU and regress the POD coefficients from "
        + "(shape latent, BC). Snapshots come from --snapshots (npz "
        + "directory) or are generated synthetically from the decoder SDF "
        + "(--synthetic, pipeline validation only)."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved "
        + "model files to use.",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The decoder checkpoint weights to use.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split of shapes to build the ROM for.",
    )
    arg_parser.add_argument(
        "--snapshots",
        dest="snapshots",
        default=None,
        help="Directory of per-case snapshot npz files (fields (G, 4), bc "
        + "(4,), shape). Mutually exclusive with --synthetic.",
    )
    arg_parser.add_argument(
        "--synthetic",
        dest="synthetic",
        action="store_true",
        help="Generate snapshots on the fly from the decoder SDF with the "
        + "synthetic stand-in field (pipeline validation only, NOT a "
        + "physical simulation). Files are written to "
        + "<experiment>/VolumeROM/snapshots/ and loaded through the same "
        + "path as real snapshots.",
    )
    arg_parser.add_argument(
        "--grid_resolution",
        dest="grid_resolution",
        default=64,
        type=int,
        help="Reference-grid resolution N (G = N^3 points, domain "
        + "[-1.5, 1.5]^3).",
    )
    arg_parser.add_argument(
        "--pod_energy",
        dest="pod_energy",
        default=0.999,
        type=float,
        help="Captured-energy threshold for POD rank truncation.",
    )
    arg_parser.add_argument(
        "--pod_rank",
        dest="pod_rank",
        default=None,
        type=int,
        help="Fixed POD rank (overrides --pod_energy).",
    )
    arg_parser.add_argument(
        "--cases_per_shape",
        dest="cases_per_shape",
        default=4,
        type=int,
        help="Number of boundary-condition cases sampled per shape "
        + "(--synthetic only).",
    )
    arg_parser.add_argument(
        "--u_range",
        dest="u_range",
        default=[10.0, 20.0],
        type=float,
        nargs=2,
        help="Free-stream speed sampling range (--synthetic only).",
    )
    arg_parser.add_argument(
        "--dir_cone_deg",
        dest="dir_cone_deg",
        default=180.0,
        type=float,
        help="Flow directions are sampled uniformly on the sphere (180) or "
        + "inside a cone of this half-angle (degrees) around the +x axis "
        + "(--synthetic only).",
    )
    arg_parser.add_argument(
        "--delta",
        dest="delta",
        default=0.05,
        type=float,
        help="Near-wall sigmoid decay length of the synthetic field.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=20000,
        type=int,
        help="Number of regressor training iterations (Adam).",
    )
    arg_parser.add_argument(
        "--lr",
        dest="lr",
        default=1e-3,
        type=float,
        help="Regressor learning rate.",
    )
    arg_parser.add_argument(
        "--hidden",
        dest="hidden",
        default=256,
        type=int,
        help="Hidden width of the coefficient regressor.",
    )
    arg_parser.add_argument(
        "--num_layers",
        dest="num_layers",
        default=4,
        type=int,
        help="Number of hidden layers of the coefficient regressor.",
    )
    arg_parser.add_argument(
        "--val_fraction",
        dest="val_fraction",
        default=0.2,
        type=float,
        help="Fraction of shapes held out for validation (best-on-val "
        + "checkpoint). 0 disables validation and saves the final model.",
    )
    arg_parser.add_argument(
        "--reconstruct_iters",
        dest="reconstruct_iterations",
        default=800,
        type=int,
        help="Iterations of implicit-domain latent fitting for shapes without "
        + "a stored latent code.",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        default=0,
        type=int,
        help="Random seed for case sampling, the train/val split, POD and "
        + "the training loop (fully reproduces a run).",
    )
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    if args.snapshots and args.synthetic:
        raise RuntimeError("pass exactly one of --snapshots <dir> or --synthetic")
    if not args.snapshots and not args.synthetic:
        raise RuntimeError("pass exactly one of --snapshots <dir> or --synthetic")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    with open(args.split_filename, "r") as f:
        split = json.load(f)
    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)

    rom_dir = os.path.join(args.experiment_directory, "VolumeROM")
    if not os.path.isdir(rom_dir):
        os.makedirs(rom_dir)
    snapshots_dir = (
        args.snapshots if args.snapshots else os.path.join(rom_dir, "snapshots")
    )
    if args.synthetic and not os.path.isdir(snapshots_dir):
        os.makedirs(snapshots_dir)

    grid_points, grid_shape = deep_sdf.cfd.make_reference_grid(args.grid_resolution)
    grid_points = grid_points.cuda()
    num_points = grid_points.shape[0]
    logging.info(
        "reference grid: {}^3 = {} points on [-1.5, 1.5]^3".format(
            args.grid_resolution, num_points
        )
    )

    rng = random.Random(args.seed)

    # --snapshots: index the directory by shape up front (files are validated
    # on load below)
    snap_index = {}
    if args.snapshots:
        for path in sorted(glob.glob(os.path.join(snapshots_dir, "*.npz"))):
            data = np.load(path, allow_pickle=False)
            snap_index.setdefault(str(data["shape"]), []).append(path)

    # per-shape data: one latent per shape (shared by all of its cases),
    # per-case volume snapshots
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
            case_paths = []
            for case_idx in range(args.cases_per_shape):
                direction = sample_flow_direction(rng, args.dir_cone_deg)
                velocity = rng.uniform(args.u_range[0], args.u_range[1])
                bc = make_bc(direction, velocity)
                path = os.path.join(
                    snapshots_dir, snapshot_filename(npz, case_idx)
                )
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
                    fields = deep_sdf.cfd.synthetic_volume_field(
                        decoder,
                        latent,
                        grid_points,
                        grid_shape,
                        torch.tensor(bc, dtype=torch.float32).cuda(),
                        delta=args.delta,
                    )
                    save_snapshot(path, fields, bc, npz)
                case_paths.append(path)
            # unified loading path: synthetic snapshots are read back from
            # disk exactly like real ones
            for path in case_paths:
                snap = load_snapshot(path, num_points)
                if snap["shape"] != npz:
                    raise RuntimeError(
                        "snapshot {} belongs to shape {}, expected {}".format(
                            path, snap["shape"], npz
                        )
                    )
                cases.append(
                    {
                        "bc": snap["bc"].cuda(),
                        "fields": snap["fields"].reshape(-1),
                        "case_id": os.path.basename(path),
                    }
                )
        else:
            paths = snap_index.get(npz)
            if not paths:
                raise RuntimeError(
                    "no snapshots found for shape {} in {}".format(
                        npz, snapshots_dir
                    )
                )
            for path in paths:
                snap = load_snapshot(path, num_points)
                cases.append(
                    {
                        "bc": snap["bc"].cuda(),
                        "fields": snap["fields"].reshape(-1),
                        "case_id": os.path.basename(path),
                    }
                )

        shapes.append({"name": npz, "latent": latent.detach(), "cases": cases})

    if not shapes:
        raise RuntimeError("no shapes produced snapshots; nothing to train on")

    # shape-level train/val split (at least one shape kept for training)
    indices = list(range(len(shapes)))
    rng.shuffle(indices)
    num_val = 0
    if args.val_fraction > 0.0 and len(shapes) >= 2:
        num_val = int(round(args.val_fraction * len(shapes)))
        num_val = max(1, min(len(shapes) - 1, num_val))
    val_idx = set(indices[:num_val])
    train_shapes = [s for i, s in enumerate(shapes) if i not in val_idx]
    val_shapes = [s for i, s in enumerate(shapes) if i in val_idx]
    if val_shapes:
        logging.info(
            "train/val split: {} train / {} val shapes (val: {})".format(
                len(train_shapes),
                len(val_shapes),
                [s["name"] for s in val_shapes],
            )
        )
    else:
        train_shapes = shapes

    train_cases = [(s, c) for s in train_shapes for c in s["cases"]]
    val_cases = [(s, c) for s in val_shapes for c in s["cases"]]
    logging.info(
        "snapshot ensemble: {} train cases / {} val cases".format(
            len(train_cases), len(val_cases)
        )
    )

    # POD on the training snapshots only - validation fields never enter the
    # basis
    S = torch.stack([c["fields"] for _, c in train_cases])
    basis = deep_sdf.cfd.pod_fit(
        S, energy=args.pod_energy, rank=args.pod_rank, device="cuda"
    )
    basis_path = os.path.join(rom_dir, "pod_basis.pth")
    basis.save(basis_path)
    logging.info("saved POD basis to {}".format(basis_path))
    logging.info(
        "train projection error (irreducible bound at r={}): {:.6e}".format(
            basis.rank, basis.projection_error(S).mean().item()
        )
    )

    # true projection coefficients for every case (regression targets)
    for s in shapes:
        for c in s["cases"]:
            c["coef"] = basis.project(c["fields"].unsqueeze(0))[0].detach()

    model_kwargs = {
        "latent_size": latent_size,
        "rank": basis.rank,
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "bc_dim": len(deep_sdf.cfd.BC_FIELDS),
    }
    model = deep_sdf.cfd.VolumeCoefficientRegressor(**model_kwargs).cuda()
    # standardize inputs/targets with the training-set statistics
    train_z = torch.cat([s["latent"] for s in train_shapes], 0)
    model.set_z_normalization(train_z.mean(0), train_z.std(0))
    train_bcs = torch.stack([c["bc"] for _, c in train_cases])
    model.set_bc_normalization(train_bcs.mean(0), train_bcs.std(0))
    train_coefs = torch.stack([c["coef"] for _, c in train_cases])
    model.set_coef_normalization(train_coefs.mean(0), train_coefs.std(0))
    for _, c in train_cases + val_cases:
        c["target"] = ((c["coef"] - model.coef_mean) / model.coef_std).detach()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    logging.info(
        "training volume ROM (rank {}) on {} cases from {} shapes".format(
            basis.rank, len(train_cases), len(train_shapes)
        )
    )

    def save_checkpoint(val_metrics=None):
        checkpoint = {
            "model_type": "volume_pod_rom",
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "latent_size": latent_size,
            "bc_fields": deep_sdf.cfd.BC_FIELDS,
            "u_range": list(args.u_range),
            "dir_cone_deg": args.dir_cone_deg,
            "grid_resolution": args.grid_resolution,
            "pod_energy": args.pod_energy,
            "pod_rank": args.pod_rank,
            "pod_basis_file": os.path.basename(basis_path),
            "snapshot_source": "synthetic" if args.synthetic else args.snapshots,
            "snapshot_dir": snapshots_dir,
            "cases_per_shape": args.cases_per_shape,
            "decoder_checkpoint": args.checkpoint,
            "decoder_epoch": saved_model_epoch,
            "shapes": [s["name"] for s in shapes],
            "val_shapes": [s["name"] for s in val_shapes],
            # train-set coefficient MSE at the time this checkpoint was saved
            "train_coef_mse": loss_num,
            "seed": args.seed,
        }
        if val_metrics is not None:
            checkpoint.update(
                {
                    "val_coef_mse": val_metrics["coef_mse"],
                    "val_rel_l2": val_metrics["rel_l2"],
                    "val_projection_error": val_metrics["proj"],
                    "val_mean_field_rel_l2": val_metrics["baseline"],
                }
            )
        torch.save(checkpoint, os.path.join(rom_dir, "latest.pth"))

    start = time.time()
    loss_num = 0.0
    best_val_rel_l2 = None
    for e in range(int(args.iterations)):
        optimizer.zero_grad()
        # one random (shape, case) per iteration
        s, c = rng.choice(train_cases)
        pred = model.forward_normalized(s["latent"], c["bc"])
        loss = loss_fn(pred, c["target"].unsqueeze(0))
        loss.backward()
        optimizer.step()
        loss_num = loss.item()
        if e % 200 == 0:
            if val_cases:
                metrics = evaluate_cases(model, basis, val_cases)
                logging.info(
                    "iter {} mse: {:.6e} | val coef mse: {:.6e} rel L2: {:.6e} "
                    "proj bound: {:.6e} mean-field: {:.6e}".format(
                        e,
                        loss_num,
                        metrics["coef_mse"],
                        metrics["rel_l2"],
                        metrics["proj"],
                        metrics["baseline"],
                    )
                )
                if best_val_rel_l2 is None or metrics["rel_l2"] < best_val_rel_l2:
                    best_val_rel_l2 = metrics["rel_l2"]
                    save_checkpoint(metrics)
            else:
                logging.info("iter {} mse: {:.6e}".format(e, loss_num))
    logging.info("ROM training time: {:.2f}s".format(time.time() - start))

    if val_cases:
        if best_val_rel_l2 is None:
            # fewer iterations than the eval interval: evaluate once now
            metrics = evaluate_cases(model, basis, val_cases)
            best_val_rel_l2 = metrics["rel_l2"]
            save_checkpoint(metrics)
        logging.info(
            "best val rel L2: {:.6e} (checkpoint saved on best val)".format(
                best_val_rel_l2
            )
        )
    else:
        save_checkpoint()
    logging.info(
        "saved volume ROM to {}".format(os.path.join(rom_dir, "latest.pth"))
    )

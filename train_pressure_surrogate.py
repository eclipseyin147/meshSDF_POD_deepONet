#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
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
import deep_sdf.workspace as ws
from reconstruct import reconstruct


def load_or_fit_latent(args, decoder, latent_size, saved_model_epoch, npz):
    latent_filename = os.path.join(
        args.experiment_directory,
        ws.reconstructions_subdir,
        str(saved_model_epoch),
        ws.reconstruction_codes_subdir,
        npz[:-4] + ".pth",
    )
    if os.path.isfile(latent_filename):
        logging.info("loading stored latent code from {}".format(latent_filename))
        return torch.load(latent_filename).reshape(1, latent_size).float().cuda()

    full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
    data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
    data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
    data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]
    _, latent = reconstruct(
        decoder,
        int(args.reconstruct_iterations),
        latent_size,
        data_sdf,
        0.01,
        0.1,
        num_samples=8000,
        lr=5e-3,
        l2reg=True,
    )
    return latent.reshape(1, latent_size).detach()


def sample_flow_direction(rng, cone_deg):
    """Uniform direction on the sphere (cone_deg >= 180) or inside the cone
    of half-angle cone_deg around the +x axis."""
    if cone_deg >= 180.0:
        cos_theta = rng.uniform(-1.0, 1.0)
    else:
        cos_theta = rng.uniform(math.cos(math.radians(cone_deg)), 1.0)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta ** 2))
    return (cos_theta, sin_theta * math.cos(phi), sin_theta * math.sin(phi))


def make_bc(direction, velocity):
    """Boundary-condition vector following deep_sdf.cfd.BC_FIELDS."""
    return [float(velocity)] + [float(d) for d in direction]


# Free-stream conditions of the preprocessed DrivAerNet++ pressure labels
# (arXiv:2406.09624, App. A.3): u_inf = 30 m/s, flow along +x. Real Cp
# labels are single-condition, so shapes using them get exactly one case.
REAL_CP_U_INF = 30.0
REAL_CP_DIR = (1.0, 0.0, 0.0)


def find_cp_filename(cp_dir, npz):
    """Location of the preprocessed pressure npz for a split instance.

    The pressure files mirror the SdfSamples layout below `cp_dir`
    (e.g. <cp_dir>/DrivAerNet/Cars/F_D_WM_WW_0001.npz, as written by
    drivaernet/scripts/preprocess_pressure.py)."""
    path = os.path.join(cp_dir, npz)
    return path if os.path.isfile(path) else None


def load_real_cp(path):
    """Load a preprocessed pressure npz as torch cuda tensors.

    Returns {"verts": (V,3) f32, "faces": (F,3) i64, "cp": (F,) f32}: the
    original STL triangulation with the per-face pressure coefficient,
    so labels correspond 1:1 to the returned mesh.
    """
    d = np.load(path)
    return {
        "verts": torch.from_numpy(np.asarray(d["verts"], dtype=np.float32)).cuda(),
        "faces": torch.from_numpy(np.asarray(d["faces"], dtype=np.int64)).cuda(),
        "cp": torch.from_numpy(np.asarray(d["face_cp"], dtype=np.float32)).cuda(),
    }


def evaluate_split(model, shapes, loss_fn):
    """Mean Cp MSE and mean Cd relative error over all (shape, case) pairs."""
    mses, cd_errs = [], []
    with torch.no_grad():
        for s in shapes:
            for case in s["cases"]:
                pred = model(
                    s["latent"],
                    s["verts"],
                    s["faces"],
                    mean_curvature=s["kappa"],
                    bc=case["bc"],
                    sdf_values=s["sdf"],
                    sdf_grad_norm=s["sdf_grad_norm"],
                )
                mses.append(loss_fn(pred, case["cp"]).item())
                cd_pred, _ = deep_sdf.cfd.drag_coefficient(
                    s["verts"], s["faces"], pred, case["direction"]
                )
                cd_true, _ = deep_sdf.cfd.drag_coefficient(
                    s["verts"], s["faces"], case["cp"], case["direction"]
                )
                cd_errs.append(
                    (cd_pred - cd_true).abs().item() / max(abs(cd_true.item()), 1e-12)
                )
    return float(np.mean(mses)), float(np.mean(cd_errs))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Train a pressure surrogate (DeepMesh section 4.3) over the "
        + "shapes of a split: fit/reuse a latent code per shape, extract the "
        + "mesh, generate per-case pressure-coefficient labels (OpenFOAM if "
        + "requested and available, geometric proxy otherwise) and regress "
        + "them. --model local trains a purely local per-face MLP baseline; "
        + "--model operator trains the shape/BC-conditioned "
        + "PressureNeuralOperator. With --cases_per_shape > 1 the operator is "
        + "conditioned on the boundary-condition vector [U, dir_x, dir_y, "
        + "dir_z] and validated on held-out shapes (--val_fraction)."
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
        help="The split of shapes to build pressure labels for.",
    )
    arg_parser.add_argument(
        "--model",
        "-m",
        dest="model",
        default="operator",
        choices=["local", "operator"],
        help="Surrogate model: local per-face MLP baseline, or the "
        + "shape-conditioned pressure neural operator (default).",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        default=128,
        type=int,
        help="Iso-surface extraction resolution.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=2000,
        type=int,
        help="Number of surrogate training iterations (Adam).",
    )
    arg_parser.add_argument(
        "--lr",
        dest="lr",
        default=1e-3,
        type=float,
        help="Surrogate learning rate.",
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
        "--openfoam",
        dest="openfoam",
        action="store_true",
        help="Generate labels with OpenFOAM (simpleFoam) instead of the "
        + "geometric proxy. Requires OpenFOAM in PATH.",
    )
    arg_parser.add_argument(
        "--cp_data",
        dest="cp_data",
        default=None,
        help="Directory with preprocessed per-shape pressure npz files "
        + "(face_cp on the original STL triangulation, as produced by "
        + "drivaernet/scripts/preprocess_pressure.py). Files are looked up "
        + "relative to this directory using the split instance names, e.g. "
        + "<cp_data>/DrivAerNet/Cars/F_D_WM_WW_0001.npz. Defaults to "
        + "<data_source>/Pressure when that directory exists. Shapes with a "
        + "pressure file are trained on the real Cp labels (single case at "
        + "u = 30 m/s, dir = +x); shapes without one fall back to the proxy "
        + "(or OpenFOAM with --openfoam).",
    )
    arg_parser.add_argument(
        "--flow_dir",
        dest="flow_dir",
        default=[1.0, 0.0, 0.0],
        type=float,
        nargs=3,
        help="Free-stream flow direction (single-case mode).",
    )
    arg_parser.add_argument(
        "--base_suction",
        dest="base_suction",
        default=0.3,
        type=float,
        help="Leeward suction coefficient of the geometric proxy labels.",
    )
    arg_parser.add_argument(
        "--blockage_beta",
        dest="blockage_beta",
        default=1.0,
        type=float,
        help="Strength of the non-local blockage amplification in the "
        + "geometric proxy labels.",
    )
    arg_parser.add_argument(
        "--cases_per_shape",
        dest="cases_per_shape",
        default=4,
        type=int,
        help="Number of boundary-condition cases sampled per shape. Values "
        + "> 1 enable BC conditioning of the operator ([U, dir] input); 1 "
        + "recovers the single-case (old) behavior with --flow_dir.",
    )
    arg_parser.add_argument(
        "--u_range",
        dest="u_range",
        default=[10.0, 20.0],
        type=float,
        nargs=2,
        help="Free-stream speed sampling range for multi-case training.",
    )
    arg_parser.add_argument(
        "--dir_cone_deg",
        dest="dir_cone_deg",
        default=180.0,
        type=float,
        help="Flow directions are sampled uniformly on the sphere (180) or "
        + "inside a cone of this half-angle (degrees) around the +x axis.",
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
        "--gnn_layers",
        dest="gnn_layers",
        default=0,
        type=int,
        help="Face-level message-passing layers before the FiLM trunk.",
    )
    arg_parser.add_argument(
        "--sdf_features",
        dest="sdf_features",
        action="store_true",
        help="Append per-face [sdf, |grad sdf|] decoder field features to "
        + "the trunk input.",
    )
    arg_parser.add_argument(
        "--no_curvature",
        dest="no_curvature",
        action="store_true",
        help="Do not use mean curvature as a trunk input feature.",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        default=0,
        type=int,
        help="Random seed for case sampling, the train/val split and the "
        + "training loop (fully reproduces a run).",
    )
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.openfoam and not deep_sdf.cfd.openfoam_available():
        raise RuntimeError(
            "--openfoam was passed but OpenFOAM ('simpleFoam') is not "
            "available on this system; rerun without --openfoam to use the "
            "geometric proxy labels."
        )
    if args.cp_data is None:
        default_cp = os.path.join(args.data_source, "Pressure")
        if os.path.isdir(default_cp):
            args.cp_data = default_cp
            logging.info(
                "found preprocessed pressure data at {}; shapes with a "
                "pressure npz will use the real Cp labels".format(default_cp)
            )
    if args.cp_data is not None and args.cases_per_shape > 1:
        logging.warning(
            "real Cp labels are single-condition (u = 30 m/s, dir = +x); "
            "shapes using them get one case per shape and the bc input "
            "conveys no label information. Cases > 1 only affect shapes "
            "falling back to the proxy/OpenFOAM labels."
        )
    if args.openfoam and args.cases_per_shape > 1:
        logging.warning(
            "the OpenFOAM case template fixes the inflow direction to +x, and "
            "for incompressible flow Cp is essentially speed-invariant - all "
            "cases of a shape will therefore carry (nearly) identical labels "
            "and the bc input conveys no label information. Use multiple "
            "cases with the geometric proxy, or vary the geometry/angle "
            "of attack in the OpenFOAM setup, for meaningful BC conditioning."
        )
    if args.model == "local" and (
        args.cases_per_shape > 1 or args.gnn_layers > 0 or args.sdf_features
    ):
        logging.warning(
            "--model local ignores --cases_per_shape/--gnn_layers/"
            "--sdf_features (single case, purely local features)"
        )

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

    surrogate_dir = os.path.join(args.experiment_directory, "CfdSurrogate")
    if not os.path.isdir(surrogate_dir):
        os.makedirs(surrogate_dir)

    use_bc = args.model == "operator" and args.cases_per_shape > 1
    use_curvature = args.model == "operator" and not args.no_curvature
    rng = random.Random(args.seed)

    # per-shape data: one mesh + geometric features per shape (shared by all
    # of its cases), per-case Cp labels
    shapes = []
    label_sources = {"real_cp": 0, "proxy": 0, "openfoam": 0}
    for npz in npz_filenames:
        if "npz" not in npz:
            continue
        logging.info("processing {}".format(npz))

        real_cp = None
        cp_filename = (
            find_cp_filename(args.cp_data, npz) if args.cp_data else None
        )
        if cp_filename is not None:
            logging.info("using real Cp labels from {}".format(cp_filename))
            real_cp = load_real_cp(cp_filename)

        latent = load_or_fit_latent(
            args, decoder, latent_size, saved_model_epoch, npz
        )

        if real_cp is not None:
            # train on the original STL triangulation the Cp labels are
            # defined on (1:1 face correspondence, no transfer error).
            # NB: extract_differentiable_mesh samples [-1, 1]^3, which does
            # not cover metric-scale shapes like the DrivAerNet cars.
            verts, faces = real_cp["verts"], real_cp["faces"]
            label_sources["real_cp"] += 1
        else:
            with torch.no_grad():
                verts, faces, _ = deep_sdf.differentiable_mesh.extract_differentiable_mesh(
                    decoder, latent, resolution=args.resolution
                )
            if verts.shape[0] == 0:
                logging.warning("empty mesh for {}; skipping".format(npz))
                continue
            label_sources["proxy" if not args.openfoam else "openfoam"] += 1

        kappa = None
        if use_curvature:
            kappa = deep_sdf.cfd.compute_mean_curvature(decoder, latent, verts)

        sdf_values, sdf_grad_norm = None, None
        if args.sdf_features:
            with torch.no_grad():
                centroids = verts[faces].mean(dim=1)
                sdf_values = (
                    deep_sdf.utils.decode_sdf(decoder, latent, centroids)
                    .squeeze(1)
                    .detach()
                )
                sdf_grad_norm = (
                    deep_sdf.differentiable_mesh.compute_sdf_gradients(
                        decoder, latent, centroids
                    )
                    .norm(dim=1)
                    .detach()
                )

        cases = []
        if real_cp is not None:
            # real DrivAerNet labels: single case at the dataset conditions
            direction = REAL_CP_DIR
            velocity = REAL_CP_U_INF
            cases.append(
                {
                    "direction": direction,
                    "velocity": velocity,
                    "bc": torch.tensor(
                        make_bc(direction, velocity), dtype=torch.float32
                    ).cuda(),
                    "cp": real_cp["cp"],
                }
            )
        else:
            num_cases = args.cases_per_shape if args.model == "operator" else 1
            for case_idx in range(num_cases):
                if use_bc:
                    if args.openfoam:
                        direction = (1.0, 0.0, 0.0)  # fixed by the case template
                    else:
                        direction = sample_flow_direction(rng, args.dir_cone_deg)
                    velocity = rng.uniform(args.u_range[0], args.u_range[1])
                else:
                    direction = tuple(args.flow_dir)
                    velocity = 0.5 * (args.u_range[0] + args.u_range[1])

                if args.openfoam:
                    case_dir = os.path.join(
                        surrogate_dir,
                        "openfoam_cases",
                        npz[:-4].replace("/", "_") + "_case{:03d}".format(case_idx),
                    )
                    stl_path = os.path.join(case_dir, "input.stl")
                    deep_sdf.cfd.export_stl(verts, faces, stl_path)
                    pressure = torch.from_numpy(
                        deep_sdf.cfd.run_openfoam(stl_path, case_dir, velocity=velocity)
                    ).float()
                    if pressure.shape[0] != faces.shape[0]:
                        raise RuntimeError(
                            "OpenFOAM returned {} patch pressures for {} input "
                            "faces; the snappyHexMesh patch faces do not "
                            "correspond 1:1 to the input mesh - map pressures "
                            "back by face centroid before training.".format(
                                pressure.shape[0], faces.shape[0]
                            )
                        )
                    cp = deep_sdf.cfd.pressure_to_cp(pressure, velocity).cuda()
                else:
                    cp = deep_sdf.cfd.proxy_pressure_coefficient(
                        verts,
                        faces,
                        flow_dir=direction,
                        base_suction=args.base_suction,
                        beta=args.blockage_beta,
                    )

                cases.append(
                    {
                        "direction": direction,
                        "velocity": velocity,
                        "bc": torch.tensor(
                            make_bc(direction, velocity), dtype=torch.float32
                        ).cuda(),
                        "cp": cp.detach(),
                    }
                )

        shapes.append(
            {
                "name": npz,
                "latent": latent.detach(),
                "verts": verts.detach(),
                "faces": faces.detach(),
                "kappa": kappa,
                "sdf": sdf_values,
                "sdf_grad_norm": sdf_grad_norm,
                "cases": cases,
            }
        )

    logging.info(
        "label sources: {} real Cp / {} proxy / {} openfoam".format(
            label_sources["real_cp"],
            label_sources["proxy"],
            label_sources["openfoam"],
        )
    )

    if not shapes:
        raise RuntimeError("no shapes produced usable meshes; nothing to train on")

    # shape-level train/val split (at least one shape kept for training); the
    # local baseline has no per-shape conditioning and always trains on all
    # shapes
    indices = list(range(len(shapes)))
    rng.shuffle(indices)
    num_val = 0
    if (
        args.val_fraction > 0.0
        and len(shapes) >= 2
        and args.model == "operator"
    ):
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

    model_kwargs = None
    if args.model == "operator":
        model_kwargs = {
            "latent_size": latent_size,
            "hidden": 128,
            "num_layers": 3,
            "use_curvature": use_curvature,
            "bc_dim": len(deep_sdf.cfd.BC_FIELDS) if use_bc else 0,
            "use_sdf_features": args.sdf_features,
            "gnn_layers": args.gnn_layers,
        }
        model = deep_sdf.cfd.PressureNeuralOperator(**model_kwargs).cuda()
        # standardize trunk inputs with the pooled training-face statistics
        train_feats = torch.cat(
            [
                deep_sdf.cfd.face_features(
                    s["verts"], s["faces"], s["kappa"], s["sdf"], s["sdf_grad_norm"]
                )
                for s in train_shapes
            ],
            0,
        ).detach()
        model.set_feature_normalization(train_feats.mean(0), train_feats.std(0))
        if use_bc:
            train_bcs = torch.stack(
                [c["bc"] for s in train_shapes for c in s["cases"]]
            )
            model.set_bc_normalization(train_bcs.mean(0), train_bcs.std(0))
    else:
        model = deep_sdf.cfd.LocalPressureMLP().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    logging.info(
        "training {} surrogate on {} faces from {} shapes ({} cases/shape)".format(
            args.model,
            sum(s["faces"].shape[0] for s in train_shapes),
            len(train_shapes),
            len(train_shapes[0]["cases"]),
        )
    )
    if args.model == "local":
        pooled_features = torch.cat(
            [
                deep_sdf.cfd.face_features(s["verts"], s["faces"])
                for s in train_shapes
            ],
            0,
        ).detach()
        pooled_pressures = torch.cat([s["cases"][0]["cp"] for s in train_shapes], 0)

    def save_checkpoint(val_mse=None, val_cd_rel_err=None):
        checkpoint = {
            "model_type": args.model,
            "model_state_dict": model.state_dict(),
            "latent_size": latent_size,
            "flow_dir": list(args.flow_dir),
            "base_suction": args.base_suction,
            "blockage_beta": args.blockage_beta,
            "labels": "+".join(k for k, v in label_sources.items() if v > 0),
            "decoder_checkpoint": args.checkpoint,
            "decoder_epoch": saved_model_epoch,
            "resolution": args.resolution,
            "shapes": [s["name"] for s in shapes],
            "val_shapes": [s["name"] for s in val_shapes],
            # train-set MSE at the time this checkpoint was saved
            "train_mse": loss_num,
            "final_mse": loss_num,  # backward-compatible key
            "seed": args.seed,
        }
        if model_kwargs is not None:
            checkpoint.update(
                {
                    "model_kwargs": model_kwargs,
                    "bc_fields": deep_sdf.cfd.BC_FIELDS,
                    "u_range": list(args.u_range),
                    "dir_cone_deg": args.dir_cone_deg,
                    "val_mse": val_mse,
                    "val_cd_rel_err": val_cd_rel_err,
                }
            )
        torch.save(checkpoint, os.path.join(surrogate_dir, "latest.pth"))

    start = time.time()
    loss_num = 0.0
    best_val_mse = None
    for e in range(int(args.iterations)):
        optimizer.zero_grad()
        if args.model == "operator":
            # one random (shape, case) per iteration
            s = rng.choice(train_shapes)
            case = rng.choice(s["cases"])
            pred = model(
                s["latent"],
                s["verts"],
                s["faces"],
                mean_curvature=s["kappa"],
                bc=case["bc"],
                sdf_values=s["sdf"],
                sdf_grad_norm=s["sdf_grad_norm"],
            )
            loss = loss_fn(pred, case["cp"])
        else:
            loss = loss_fn(model(pooled_features), pooled_pressures)
        loss.backward()
        optimizer.step()
        loss_num = loss.item()
        if e % 200 == 0:
            if val_shapes and args.model == "operator":
                val_mse, val_cd_err = evaluate_split(model, val_shapes, loss_fn)
                logging.info(
                    "iter {} mse: {:.6e} | val mse: {:.6e} val Cd rel err: "
                    "{:.3%}".format(e, loss_num, val_mse, val_cd_err)
                )
                if best_val_mse is None or val_mse < best_val_mse:
                    best_val_mse = val_mse
                    save_checkpoint(val_mse, val_cd_err)
            else:
                logging.info("iter {} mse: {:.6e}".format(e, loss_num))
    logging.info("surrogate training time: {:.2f}s".format(time.time() - start))

    if val_shapes and args.model == "operator":
        if best_val_mse is None:
            # fewer iterations than the eval interval: evaluate once now
            val_mse, val_cd_err = evaluate_split(model, val_shapes, loss_fn)
            best_val_mse = val_mse
            save_checkpoint(val_mse, val_cd_err)
        logging.info(
            "best val mse: {:.6e} (checkpoint saved on best val)".format(
                best_val_mse
            )
        )
    else:
        save_checkpoint()
    logging.info(
        "saved surrogate to {}".format(os.path.join(surrogate_dir, "latest.pth"))
    )

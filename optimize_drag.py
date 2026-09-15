#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import math
import os
import time

import torch

import deep_sdf
import deep_sdf.cfd
import deep_sdf.workspace as ws
from reconstruct import reconstruct


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Optimize a latent code for low pressure drag (DeepMesh "
        + "section 4.3): L = Cd + reg_lambda * ||z - z_0||^2, where the drag "
        + "coefficient Cd is predicted by a trained pressure surrogate on "
        + "the differentiably extracted mesh (re-extracted every iteration, "
        + "topology may change). Surrogates trained before the "
        + "case-generalization upgrade (no model_kwargs in the checkpoint) "
        + "use the un-normalized drag loss instead."
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
        "--surrogate",
        dest="surrogate",
        default="latest",
        help="The pressure surrogate checkpoint in <experiment>/CfdSurrogate.",
    )
    arg_parser.add_argument(
        "--latent",
        "-l",
        dest="latent",
        required=True,
        help="Starting shape: either a .pth latent code file or an .npz SDF "
        + "samples file (fitted with reconstruct() first).",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=150,
        type=int,
        help="Number of drag optimization iterations.",
    )
    arg_parser.add_argument(
        "--lr",
        dest="lr",
        default=1e-3,
        type=float,
        help="Learning rate for latent optimization.",
    )
    arg_parser.add_argument(
        "--reg_lambda",
        dest="reg_lambda",
        default=0.1,
        type=float,
        help="Weight of the ||z - z_0||^2 regularizer keeping the latent on "
        + "the learned shape manifold.",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        default=128,
        type=int,
        help="Iso-surface extraction resolution (per iteration).",
    )
    arg_parser.add_argument(
        "--alpha",
        dest="alpha",
        default=None,
        type=float,
        help="Flow angle of attack in degrees: the free-stream direction is "
        + "(cos(alpha), sin(alpha), 0). Only used by BC-conditioned "
        + "surrogates; defaults to 0.",
    )
    arg_parser.add_argument(
        "--velocity",
        dest="velocity",
        default=None,
        type=float,
        help="Free-stream speed for BC-conditioned surrogates; defaults to "
        + "the middle of the training --u_range (or 15).",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default=None,
        help="Data source directory (only needed when --latent is an npz "
        + "referenced relative to the data source).",
    )
    arg_parser.add_argument(
        "--bounds",
        dest="bounds",
        default=None,
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Extraction domain (smallest containing cube; default "
        + "[-1, 1]^3). Set to the data bounding box for metric-scale "
        + "decoders, e.g. --bounds -1.2 -1.2 -0.2 4.2 1.2 1.9.",
    )
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

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
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    surrogate_filename = os.path.join(
        args.experiment_directory, "CfdSurrogate", args.surrogate + ".pth"
    )
    if not os.path.isfile(surrogate_filename):
        raise Exception(
            "surrogate checkpoint '{}' not found; train it with "
            "train_pressure_surrogate.py first".format(surrogate_filename)
        )
    surrogate_data = torch.load(surrogate_filename)
    model_type = surrogate_data.get("model_type", "local")
    model_kwargs = surrogate_data.get("model_kwargs")
    if model_type == "operator":
        if model_kwargs is not None:
            surrogate = deep_sdf.cfd.PressureNeuralOperator(**model_kwargs).cuda()
        else:
            surrogate = deep_sdf.cfd.PressureNeuralOperator(latent_size).cuda()
    else:
        surrogate = deep_sdf.cfd.LocalPressureMLP().cuda()
    incompatible = surrogate.load_state_dict(
        surrogate_data["model_state_dict"], strict=False
    )
    if incompatible.missing_keys:
        logging.info(
            "surrogate checkpoint does not contain {}; initialized to "
            "identity defaults".format(list(incompatible.missing_keys))
        )
    surrogate.eval()
    for param in surrogate.parameters():
        param.requires_grad = False

    # Cd-normalized loss for new-format operator checkpoints; raw pressure
    # drag for old checkpoints and the local baseline
    use_cd = model_type == "operator" and model_kwargs is not None
    use_bc = model_type == "operator" and getattr(surrogate, "bc_dim", 0) > 0

    if use_bc:
        alpha = 0.0 if args.alpha is None else args.alpha
        u_range = surrogate_data.get("u_range")
        if args.velocity is not None:
            velocity = args.velocity
        elif u_range is not None:
            velocity = 0.5 * (u_range[0] + u_range[1])
        else:
            velocity = 15.0
        flow_dir = (
            math.cos(math.radians(alpha)),
            math.sin(math.radians(alpha)),
            0.0,
        )
        bc_values = [velocity] + list(flow_dir)
        bc = torch.tensor(bc_values, dtype=torch.float32).cuda()
        logging.info(
            "BC-conditioned surrogate: alpha={} deg, U={}, direction={}, "
            "bc={} ({})".format(
                alpha,
                velocity,
                tuple(round(d, 4) for d in flow_dir),
                bc_values,
                surrogate_data.get("bc_fields", deep_sdf.cfd.BC_FIELDS),
            )
        )
    else:
        if args.alpha is not None or args.velocity is not None:
            logging.warning(
                "--alpha/--velocity given but the surrogate has no BC "
                "conditioning; they are ignored"
            )
        bc = None
        flow_dir = surrogate_data.get("flow_dir", [1.0, 0.0, 0.0])

    if args.latent.endswith(".pth"):
        z0 = torch.load(args.latent).reshape(1, latent_size).float().cuda()
        stem = os.path.splitext(os.path.basename(args.latent))[0]
    else:
        npz_path = args.latent
        if not os.path.isfile(npz_path) and args.data_source is not None:
            npz_path = os.path.join(args.data_source, ws.sdf_samples_subdir, args.latent)
        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(npz_path)
        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]
        _, latent = reconstruct(
            decoder,
            800,
            latent_size,
            data_sdf,
            0.01,
            0.1,
            num_samples=8000,
            lr=5e-3,
            l2reg=True,
        )
        z0 = latent.reshape(1, latent_size).detach()
        stem = os.path.splitext(os.path.basename(args.latent))[0]

    def predict_pressure(latent, verts, faces):
        if model_type == "operator":
            # gradients reach the latent both through the operator branch
            # (direct z input) and through the geometric features of the
            # differentiably extracted mesh (DeepMesh IFT path)
            kappa = None
            if getattr(surrogate, "use_curvature", True):
                kappa = deep_sdf.cfd.compute_mean_curvature(decoder, latent, verts)
            sdf_values, sdf_grad_norm = None, None
            if getattr(surrogate, "use_sdf_features", False):
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
            return surrogate(
                latent,
                verts,
                faces,
                mean_curvature=kappa,
                bc=bc,
                sdf_values=sdf_values,
                sdf_grad_norm=sdf_grad_norm,
            )
        return surrogate(deep_sdf.cfd.face_features(verts, faces))

    def drag_objective(verts, faces, pressure):
        if use_cd:
            cd, _ = deep_sdf.cfd.drag_coefficient(verts, faces, pressure, flow_dir)
            return cd
        return deep_sdf.cfd.drag_from_pressure(verts, faces, pressure, flow_dir)

    z = z0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=args.lr)

    out_dir = os.path.join(args.experiment_directory, "DragOptimization")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    quantity = "Cd" if use_cd else "drag"
    initial_drag = None
    drag_num = None
    empty_streak = 0
    for e in range(int(args.iterations)):
        optimizer.zero_grad()

        verts, faces, _ = deep_sdf.differentiable_mesh.extract_differentiable_mesh(
            decoder, z, resolution=args.resolution, bounds=args.bounds
        )
        if verts.shape[0] == 0:
            empty_streak += 1
            logging.warning("empty mesh at iteration {}".format(e))
            if empty_streak > 10:
                raise RuntimeError(
                    "iso-surface vanished; retry with a smaller --lr or a "
                    "larger --reg_lambda"
                )
            continue
        empty_streak = 0

        pressure = predict_pressure(z, verts, faces)
        drag = drag_objective(verts, faces, pressure)
        reg = args.reg_lambda * (z - z0).pow(2).sum()
        loss = drag + reg
        loss.backward()
        optimizer.step()

        drag_num = drag.item()
        if initial_drag is None:
            initial_drag = drag_num
        if e % 10 == 0:
            logging.info(
                "iter {} {}: {:.6f} reg: {:.6f}".format(e, quantity, drag_num, reg.item())
            )

    with torch.no_grad():
        verts_init, faces_init, _ = (
            deep_sdf.differentiable_mesh.extract_differentiable_mesh(
                decoder, z0, resolution=args.resolution, bounds=args.bounds
            )
        )
        verts_opt, faces_opt, _ = (
            deep_sdf.differentiable_mesh.extract_differentiable_mesh(
                decoder, z.detach(), resolution=args.resolution, bounds=args.bounds
            )
        )

    deep_sdf.differentiable_mesh.save_mesh(
        verts_init, faces_init, os.path.join(out_dir, stem + "_init.ply")
    )
    deep_sdf.differentiable_mesh.save_mesh(
        verts_opt, faces_opt, os.path.join(out_dir, stem + "_optimized.ply")
    )
    torch.save(z.detach().cpu(), os.path.join(out_dir, stem + "_optimized_latent.pth"))

    # surrogate forward timing: the "fast CFD prediction" - a single forward
    # pass replaces a minutes-long OpenFOAM solve
    num_faces = faces_opt.shape[0]
    kappa_opt = None
    sdf_opt, grad_norm_opt = None, None
    if model_type == "operator":
        if getattr(surrogate, "use_curvature", True):
            kappa_opt = deep_sdf.cfd.compute_mean_curvature(
                decoder, z.detach(), verts_opt
            )
        if getattr(surrogate, "use_sdf_features", False):
            centroids_opt = verts_opt[faces_opt].mean(dim=1)
            sdf_opt = (
                deep_sdf.utils.decode_sdf(decoder, z.detach(), centroids_opt)
                .squeeze(1)
                .detach()
            )
            grad_norm_opt = (
                deep_sdf.differentiable_mesh.compute_sdf_gradients(
                    decoder, z.detach(), centroids_opt
                )
                .norm(dim=1)
                .detach()
            )
    torch.cuda.synchronize()
    start = time.time()
    reps = 100
    for _ in range(reps):
        if model_type == "operator":
            surrogate(
                z.detach(),
                verts_opt,
                faces_opt,
                mean_curvature=kappa_opt,
                bc=bc,
                sdf_values=sdf_opt,
                sdf_grad_norm=grad_norm_opt,
            )
        else:
            surrogate(deep_sdf.cfd.face_features(verts_opt, faces_opt))
    torch.cuda.synchronize()
    forward_ms = (time.time() - start) / reps * 1000.0

    if initial_drag is None or drag_num is None:
        # only possible if every iteration produced an empty mesh, which
        # raises earlier; guard anyway
        logging.warning("no valid optimization iteration completed")
        improvement = float("nan")
    else:
        improvement = (initial_drag - drag_num) / max(abs(initial_drag), 1e-12) * 100.0
        logging.info("initial {}: {:.6f}".format(quantity, initial_drag))
        logging.info("final {}:   {:.6f}".format(quantity, drag_num))
        logging.info("relative {} improvement: {:.2f}%".format(quantity, improvement))
    logging.info(
        "surrogate forward pass: {:.3f} ms ({} faces) - versus minutes for an "
        "OpenFOAM solve".format(forward_ms, num_faces)
    )
    logging.info("optimized mesh saved to {}".format(out_dir))

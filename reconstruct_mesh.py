#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch

import deep_sdf
import deep_sdf.workspace as ws
from reconstruct import reconstruct


def chamfer_distance(p, q, chunk=2048):
    """Two-sided Chamfer distance, computed in chunks over q."""

    def min_dist(a, b):
        dmin = torch.full((a.shape[0],), float("inf"), device=a.device)
        for i in range(0, b.shape[0], chunk):
            d = torch.cdist(a, b[i : i + chunk])
            dmin = torch.minimum(dmin, d.min(dim=1).values)
        return dmin

    return min_dist(p, q).mean() + min_dist(q, p).mean()


def sample_mesh_surface(verts, faces, num_points):
    """Uniformly sample points on a mesh surface, weighted by triangle area."""
    v = verts[faces]  # (F, 3, 3)
    areas = 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1
    )
    probs = areas / areas.sum()
    fi = np.random.choice(faces.shape[0], size=num_points, p=probs)
    u = np.sqrt(np.random.rand(num_points, 1))
    w = np.random.rand(num_points, 1)
    points = (1 - u) * v[fi, 0] + u * (1 - w) * v[fi, 1] + u * w * v[fi, 2]
    return points.astype(np.float32)


def adjust_learning_rate(
    initial_lr, optimizer, iteration, decreased_by=10, adjust_lr_every=None
):
    lr = initial_lr * ((1 / decreased_by) ** (iteration // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def optimize_latent_mesh(
    decoder,
    latent_init,
    target_points,
    num_iterations,
    resolution,
    lr,
    num_surface_samples=10000,
    log_every=50,
    field_type="sdf",
    mc_backend="skimage",
    bounds=None,
):
    """Optimize a latent code against a surface point cloud with a Chamfer
    loss defined on the differentiably extracted mesh. The mesh is
    re-extracted every iteration, so topology may change.
    """
    latent = latent_init.detach().clone().cuda().requires_grad_(True)
    target = target_points.detach().cuda()

    optimizer = torch.optim.Adam([latent], lr=lr)
    adjust_lr_every = max(int(num_iterations / 2), 1)

    loss_num = 0.0
    empty_streak = 0
    for e in range(num_iterations):
        adjust_learning_rate(lr, optimizer, e, adjust_lr_every=adjust_lr_every)

        optimizer.zero_grad()

        verts, faces, _ = deep_sdf.differentiable_mesh.extract_differentiable_mesh(
            decoder,
            latent,
            resolution=resolution,
            field_type=field_type,
            mc_backend=mc_backend,
            bounds=bounds,
        )

        if verts.shape[0] == 0:
            empty_streak += 1
            logging.warning("empty mesh at iteration {}".format(e))
            if empty_streak > 10:
                raise RuntimeError(
                    "iso-surface vanished; retry with --init sdf or a smaller --lr"
                )
            continue
        empty_streak = 0

        if verts.shape[0] > num_surface_samples:
            sel = torch.randperm(verts.shape[0])[:num_surface_samples]
            verts_sampled = verts[sel]
        else:
            verts_sampled = verts

        loss = chamfer_distance(verts_sampled, target)
        loss.backward()
        optimizer.step()

        if e % log_every == 0:
            logging.info("iter {} chamfer loss: {}".format(e, loss.item()))
        loss_num = loss.item()

    return loss_num, latent


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape by "
        + "optimizing the latent code against a target surface through "
        + "differentiable iso-surface extraction (DeepMesh)."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved "
        + "model files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use.",
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
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=300,
        type=int,
        help="The number of iterations of mesh-level latent optimization.",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        default=128,
        type=int,
        help="Iso-surface extraction resolution.",
    )
    arg_parser.add_argument(
        "--lr",
        dest="lr",
        default=5e-4,
        type=float,
        help="Learning rate for mesh-level latent optimization.",
    )
    arg_parser.add_argument(
        "--init",
        dest="init",
        default="sdf",
        choices=["sdf", "rand"],
        help="Latent initialization: implicit-domain SDF fitting or random.",
    )
    arg_parser.add_argument(
        "--field_type",
        dest="field_type",
        default="sdf",
        choices=["sdf", "occupancy"],
        help="The type of implicit field the decoder was trained to output.",
    )
    arg_parser.add_argument(
        "--mc_backend",
        dest="mc_backend",
        default="skimage",
        choices=["skimage", "torch"],
        help="Marching cubes backend: CPU skimage or GPU torch.",
    )
    arg_parser.add_argument(
        "--bounds",
        dest="bounds",
        default=None,
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Extraction domain: the decoder field is sampled on the smallest "
        + "cube containing these bounds (default [-1, 1]^3). Required when "
        + "the decoder was trained on metric-scale data outside the unit "
        + "cube, e.g. the DrivAerNet cars: "
        + "--bounds -1.2 -1.2 -0.2 4.2 1.2 1.9",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
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
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()

    # mesh-level optimization updates the latent code only
    for param in decoder.parameters():
        param.requires_grad = False

    with open(args.split_filename, "r") as f:
        split = json.load(f)
    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)
    random.shuffle(npz_filenames)

    reconstruction_dir = os.path.join(
        args.experiment_directory,
        ws.reconstructions_subdir + "_mesh",
        str(saved_model_epoch),
    )
    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    for d in (reconstruction_meshes_dir, reconstruction_codes_dir):
        if not os.path.isdir(d):
            os.makedirs(d)

    for npz in npz_filenames:
        if "npz" not in npz:
            continue

        mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
        latent_filename = os.path.join(reconstruction_codes_dir, npz[:-4] + ".pth")
        if (
            args.skip
            and os.path.isfile(mesh_filename + ".ply")
            and os.path.isfile(latent_filename)
        ):
            continue

        logging.info("reconstructing {}".format(npz))

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        # target point cloud: implicit-domain latent fit -> mesh -> surface samples
        _, target_latent = reconstruct(
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
        with torch.no_grad():
            target_verts, target_faces, _ = (
                deep_sdf.differentiable_mesh.extract_differentiable_mesh(
                    decoder,
                    target_latent,
                    resolution=256,
                    field_type=args.field_type,
                    mc_backend=args.mc_backend,
                    bounds=args.bounds,
                )
            )
        target_points = torch.from_numpy(
            sample_mesh_surface(
                target_verts.cpu().numpy(), target_faces.cpu().numpy(), 10000
            )
        )

        if args.init == "sdf":
            _, latent_init = reconstruct(
                decoder,
                200,
                latent_size,
                data_sdf,
                0.01,
                0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
            )
        else:
            latent_init = torch.ones(1, latent_size).normal_(mean=0, std=0.01)

        start = time.time()
        err, latent = optimize_latent_mesh(
            decoder,
            latent_init,
            target_points,
            int(args.iterations),
            args.resolution,
            args.lr,
            field_type=args.field_type,
            mc_backend=args.mc_backend,
            bounds=args.bounds,
        )
        logging.info("mesh optimization time: {}".format(time.time() - start))
        logging.info("final chamfer loss: {}".format(err))

        with torch.no_grad():
            verts, faces, _ = deep_sdf.differentiable_mesh.extract_differentiable_mesh(
                decoder,
                latent,
                resolution=256,
                field_type=args.field_type,
                mc_backend=args.mc_backend,
                bounds=args.bounds,
            )
        deep_sdf.differentiable_mesh.save_mesh(verts, faces, mesh_filename + ".ply")
        torch.save(latent.unsqueeze(0).detach().cpu(), latent_filename)

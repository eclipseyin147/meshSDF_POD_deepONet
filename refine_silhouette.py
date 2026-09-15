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
import deep_sdf.soft_renderer as soft_renderer
import deep_sdf.workspace as ws
from reconstruct import reconstruct


# fixed camera positions (looking at the origin, orthographic projection);
# none of the viewing directions is parallel to the up vector (0, 1, 0)
VIEWS = [
    (0.0, 0.0, 2.0),
    (0.0, 0.0, -2.0),
    (2.0, 0.0, 0.0),
    (-2.0, 0.0, 0.0),
    (1.4, 1.4, 1.4),
    (-1.4, 1.4, 1.4),
    (1.4, -1.4, 1.4),
    (-1.4, -1.4, 1.4),
]


def adjust_learning_rate(
    initial_lr, optimizer, iteration, decreased_by=10, adjust_lr_every=None
):
    lr = initial_lr * ((1 / decreased_by) ** (iteration // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def optimize_latent_silhouette(
    decoder,
    latent_init,
    target_silhouette,
    eye,
    num_iterations,
    resolution,
    lr,
    image_size=64,
    sigma=1e-4,
    log_every=50,
    save_every=50,
    mesh_filename=None,
    field_type="sdf",
):
    """Optimize a latent code so that the silhouette of the differentiably
    extracted mesh matches a target silhouette (DeepMesh Eq. 14:
    L = |DR(M(z)) - T|_1). The mesh is re-extracted every iteration, so
    topology may change during optimization.
    """
    latent = latent_init.detach().clone().cuda().requires_grad_(True)

    optimizer = torch.optim.Adam([latent], lr=lr)
    adjust_lr_every = max(int(num_iterations / 2), 1)

    loss_num = 0.0
    empty_streak = 0
    for e in range(num_iterations):
        adjust_learning_rate(lr, optimizer, e, adjust_lr_every=adjust_lr_every)

        optimizer.zero_grad()

        verts, faces, _ = deep_sdf.differentiable_mesh.extract_differentiable_mesh(
            decoder, latent, resolution=resolution, field_type=field_type
        )

        if verts.shape[0] == 0:
            empty_streak += 1
            logging.warning("empty mesh at iteration {}".format(e))
            if empty_streak > 10:
                raise RuntimeError("iso-surface vanished; retry with a smaller --lr")
            continue
        empty_streak = 0

        silhouette = soft_renderer.render_silhouette(
            verts, faces, eye, image_size=image_size, sigma=sigma
        )[0]

        loss = (silhouette - target_silhouette).abs().mean()
        loss.backward()
        optimizer.step()

        if e % log_every == 0:
            logging.info("iter {} silhouette loss: {}".format(e, loss.item()))
        loss_num = loss.item()

        if mesh_filename is not None and e % save_every == 0:
            deep_sdf.differentiable_mesh.save_mesh(
                verts, faces, mesh_filename + "-{:04d}.ply".format(e)
            )

    return loss_num, latent


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to refine a latent code "
        + "against a target silhouette through differentiable iso-surface "
        + "extraction and differentiable soft rasterization (DeepMesh)."
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
        help="The number of iterations of silhouette-level latent optimization.",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        default=128,
        type=int,
        help="Iso-surface extraction resolution during optimization.",
    )
    arg_parser.add_argument(
        "--lr",
        dest="lr",
        default=5e-4,
        type=float,
        help="Learning rate for silhouette-level latent optimization.",
    )
    arg_parser.add_argument(
        "--view_index",
        dest="view_index",
        default=0,
        type=int,
        help="Index into the fixed set of camera positions (0-7).",
    )
    arg_parser.add_argument(
        "--image_size",
        dest="image_size",
        default=64,
        type=int,
        help="Target/rendered silhouette resolution (square).",
    )
    arg_parser.add_argument(
        "--sigma",
        dest="sigma",
        default=1e-4,
        type=float,
        help="Softness of the soft rasterizer (normalized coordinates).",
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
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    if args.view_index < 0 or args.view_index >= len(VIEWS):
        raise Exception(
            "--view_index must be in [0, {}]".format(len(VIEWS) - 1)
        )
    eye = [VIEWS[args.view_index]]
    logging.info("using camera eye {}".format(eye[0]))

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

    # silhouette-level optimization updates the latent code only
    for param in decoder.parameters():
        param.requires_grad = False

    with open(args.split_filename, "r") as f:
        split = json.load(f)
    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)
    random.shuffle(npz_filenames)

    reconstruction_dir = os.path.join(
        args.experiment_directory,
        "ReconstructionsSilhouette",
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

        for d in (os.path.dirname(mesh_filename), os.path.dirname(latent_filename)):
            if not os.path.isdir(d):
                os.makedirs(d)

        logging.info("reconstructing {}".format(npz))

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        # target silhouette: implicit-domain latent fit -> mesh -> soft render
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
                    resolution=128,
                    field_type=args.field_type,
                )
            )
            target_silhouette = soft_renderer.render_silhouette(
                target_verts,
                target_faces,
                eye,
                image_size=args.image_size,
                sigma=args.sigma,
            )[0]
        deep_sdf.differentiable_mesh.save_mesh(
            target_verts, target_faces, mesh_filename + "_target.ply"
        )
        soft_renderer.save_silhouette(target_silhouette, mesh_filename + "_target.npy")

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
        err, latent = optimize_latent_silhouette(
            decoder,
            latent_init,
            target_silhouette,
            eye,
            int(args.iterations),
            args.resolution,
            args.lr,
            image_size=args.image_size,
            sigma=args.sigma,
            mesh_filename=mesh_filename,
            field_type=args.field_type,
        )
        logging.info("silhouette optimization time: {}".format(time.time() - start))
        logging.info("final silhouette loss: {}".format(err))

        with torch.no_grad():
            verts, faces, _ = deep_sdf.differentiable_mesh.extract_differentiable_mesh(
                decoder, latent, resolution=128, field_type=args.field_type
            )
        deep_sdf.differentiable_mesh.save_mesh(verts, faces, mesh_filename + ".ply")
        torch.save(latent.unsqueeze(0).detach().cpu(), latent_filename)

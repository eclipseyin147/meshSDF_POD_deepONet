#!/usr/bin/env python3
"""DeepSDF + PhysicsNeMo DeepONet MVP library (roadmap section 63; spec
docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md).

Model: [z(16), bc(4)] -> branch; [x,y,z,sdf,normal] + Fourier -> trunk;
xDeepONet Hadamard + mlp decoder -> [u, v, w, Cp] (normalized space).
Data/geometry/split/eval helpers are appended by later tasks.
"""

import json
import os

import numpy as np
import torch

from deep_sdf.cfd import physicsnemo_compat as _physicsnemo_compat  # noqa: F401
from deep_sdf.utils import decode_sdf

DEFAULT_CFG = {
    # model
    "latent_size": 16, "bc_dim": 4, "width": 256,
    "branch_hidden": 512, "branch_layers": 4,
    "trunk_hidden": 512, "trunk_layers": 6,
    "decoder_hidden": 128, "decoder_layers": 2,
    "fourier_bands": 6, "domain_half": 1.5,
    # geometry cache / sampling
    "near_band": 0.15, "near_frac": 0.3,
    "batch_cases": 4, "n_points": 8192,
    # training
    "iters": 20000, "lr": 1e-3, "lr_min": 1e-5, "weight_decay": 1e-5,
    "amp": True, "eval_every": 500, "metrics_every": 50,
    "eval_points": 16384, "final_eval_points": 65536,
    # split / stats
    "n_clusters": 12, "seed": 0, "stats_sample": 4096,
}

OUT_VARS = ("u", "v", "w", "p")


def fourier_encode(x, n_bands):
    """x (..., d) -> (..., d * 2 * n_bands): [sin(2^k pi x), cos(2^k pi x)]
    concatenated per band pair (roadmap section 21)."""
    outs = []
    for k in range(n_bands):
        w = (2.0 ** k) * np.pi
        outs.append(torch.sin(w * x))
        outs.append(torch.cos(w * x))
    return torch.cat(outs, dim=-1)


def trunk_features(xyz, sdf, normal, n_bands, domain_half):
    """(N,3),(N,1),(N,3) -> (N, 3+1+3+3*2*n_bands): normalized coords,
    raw (tanh-space) SDF, unit normal, Fourier features of the coords."""
    xn = xyz / domain_half
    return torch.cat([xn, sdf, normal, fourier_encode(xn, n_bands)], dim=-1)


def build_model(cfg):
    """Assemble the xDeepONet (CPU; caller moves to device). Branch:
    [z, bc] -> width; trunk: 43 -> width; mlp decoder -> 4 channels."""
    from physicsnemo.models.mlp import FullyConnected
    from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet

    trunk_in = 3 + 1 + 3 + 3 * 2 * cfg["fourier_bands"]
    branch = FullyConnected(
        in_features=cfg["latent_size"] + cfg["bc_dim"],
        layer_size=cfg["branch_hidden"], num_layers=cfg["branch_layers"],
        out_features=cfg["width"], activation_fn="silu")
    trunk = FullyConnected(
        in_features=trunk_in, layer_size=cfg["trunk_hidden"],
        num_layers=cfg["trunk_layers"], out_features=cfg["width"],
        activation_fn="silu")
    return DeepONet(
        branch, trunk=trunk, dimension=3, width=cfg["width"],
        out_channels=len(OUT_VARS), decoder_type="mlp",
        decoder_width=cfg["decoder_hidden"],
        decoder_layers=cfg["decoder_layers"], decoder_activation_fn="silu")


def predict_normalized(model, latent, bc, xyz, sdf, normal, stats, cfg,
                       amp=False):
    """Single-case forward (xDeepONet core mode pairs one branch with one
    query set). latent (16,) / bc (4,) CPU-or-GPU; xyz/sdf/normal (N,..)
    on the model device. Returns (N, 4) normalized-space prediction
    (float32)."""
    z = (latent - stats["z_mean"]) / stats["z_std"]
    b = (bc - stats["bc_mean"]) / stats["bc_std"]
    xb = torch.cat([z, b]).unsqueeze(0)                      # (1, 20)
    xt = trunk_features(xyz, sdf, normal, cfg["fourier_bands"],
                        cfg["domain_half"])                   # (N, 43)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        y = model(xb, xt)[0]                                  # (N, 4)
    return y.float()


def load_frozen_decoder(specs_path, experiment_dir, checkpoint="latest"):
    """Rebuild the DeepSDF autodecoder from specs.json + ModelParameters
    checkpoint (trained with DataParallel -> strip the 'module.' prefix).
    Returns (decoder cuda/eval/frozen, latent_size)."""
    specs = json.load(open(specs_path))
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(specs["CodeLength"], **specs["NetworkSpecs"])
    path = os.path.join(experiment_dir, "ModelParameters", checkpoint + ".pth")
    saved = torch.load(path, map_location="cpu")
    state = saved["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    decoder.load_state_dict(state)
    decoder = decoder.cuda().eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder, specs["CodeLength"]


def cache_key(name):
    """Manifest shape name -> cache file stem ('lhs/shape_001.npz' ->
    'lhs_shape_001'; mirrors volume.snapshot_filename's '/'->'_')."""
    return name[:-4].replace("/", "_") if name.endswith(".npz") \
        else name.replace("/", "_")


def build_geometry_cache(decoder, name_latents, grid_points, cache_dir,
                         near_band=0.15, chunk=2 ** 18):
    """Per shape: chunked frozen-decoder SDF + autograd gradient on the
    shared grid; unit normal; fluid mask (sdf>0) and near-wall mask
    (|sdf|<near_band, tanh space). Writes sdf_cache/<cache_key>.npz with
    keys sdf (G,)f32 / normal (G,3)f32 / fluid_idx / near_idx (int64).
    Existing files are skipped."""
    os.makedirs(cache_dir, exist_ok=True)
    gp = grid_points.cuda()
    n_total = grid_points.shape[0]
    for i, (name, latent) in enumerate(name_latents):
        out = os.path.join(cache_dir, cache_key(name) + ".npz")
        if os.path.isfile(out):
            continue
        lat = latent.reshape(1, -1).float().cuda()
        sdf_chunks, grad_chunks = [], []
        for head in range(0, n_total, chunk):
            q = gp[head:head + chunk].clone().requires_grad_(True)
            d = decode_sdf(decoder, lat, q)
            g = torch.autograd.grad(d.sum(), q)[0]
            sdf_chunks.append(d.detach().squeeze(1).float().cpu())
            grad_chunks.append(g.detach().float().cpu())
        sdf = torch.cat(sdf_chunks)
        grad = torch.cat(grad_chunks)
        normal = grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-8)
        fluid = torch.nonzero(sdf > 0).squeeze(1)
        near = torch.nonzero(sdf.abs() < near_band).squeeze(1)
        if fluid.numel() == 0:
            raise RuntimeError("no fluid points for shape {}".format(name))
        if near.numel() == 0:
            near = fluid
        np.savez(out, sdf=sdf.numpy().astype(np.float32),
                 normal=normal.numpy().astype(np.float32),
                 fluid_idx=fluid.numpy().astype(np.int64),
                 near_idx=near.numpy().astype(np.int64))
        print("[cache] %d/%d %s (fluid %d, near %d)" % (
            i + 1, len(name_latents), name, fluid.numel(), near.numel()))


def load_geometry_cache(cache_dir, name):
    path = os.path.join(cache_dir, cache_key(name) + ".npz")
    data = np.load(path)
    return {"sdf": torch.from_numpy(data["sdf"]),
            "normal": torch.from_numpy(data["normal"]),
            "fluid_idx": torch.from_numpy(data["fluid_idx"]),
            "near_idx": torch.from_numpy(data["near_idx"])}

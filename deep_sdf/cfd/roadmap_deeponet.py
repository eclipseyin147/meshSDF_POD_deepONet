#!/usr/bin/env python3
"""DeepSDF + PhysicsNeMo DeepONet MVP library (roadmap section 63; spec
docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md).

Model: [z(16), bc(4)] -> branch; [x,y,z,sdf,normal] + Fourier -> trunk;
xDeepONet Hadamard + mlp decoder -> [u, v, w, Cp] (normalized space).
Data/geometry/split/eval helpers are appended by later tasks.
"""

import numpy as np
import torch

from deep_sdf.cfd import physicsnemo_compat as _physicsnemo_compat  # noqa: F401

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

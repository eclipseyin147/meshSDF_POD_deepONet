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
    # analysis / adaptive sampling
    "analyze_points": 65536, "adaptive_points": 4096, "adaptive_eps": 0.1,
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


def snapshot_index(snapshots_dir):
    """Glob snapshot npz files and map internal 'shape' field -> path."""
    import glob
    idx = {}
    for path in sorted(glob.glob(os.path.join(snapshots_dir, "*.npz"))):
        if "_case" not in os.path.basename(path):
            continue
        data = np.load(path, allow_pickle=False)
        if "shape" not in data.files:
            continue
        idx[str(data["shape"])] = path
    return idx


def load_shapes(names, latents, snap_idx, cache_dir, expected_points):
    """Join manifest latents with snapshots (nondim u /= U at load) and the
    geometry cache. All tensors CPU-resident."""
    from deep_sdf.cfd.volume import load_snapshot
    shapes = []
    for i, name in enumerate(names):
        if name not in snap_idx:
            raise RuntimeError("no snapshot for manifest shape {}".format(name))
        snap = load_snapshot(snap_idx[name], expected_points)
        fields = snap["fields"].clone()
        fields[:, :3] /= snap["bc"][0]
        if not torch.isfinite(fields).all():
            raise RuntimeError("non-finite fields in {}".format(snap_idx[name]))
        geom = load_geometry_cache(cache_dir, name)
        shapes.append({"name": name,
                       "latent": torch.from_numpy(latents[i]).float(),
                       "bc": snap["bc"].float(), "fields": fields, **geom})
    return shapes


def cluster_split(names, latents, n_clusters=12, seed=0):
    """Roadmap section 30: k-means on z-scored latents, whole clusters
    assigned 8/2/2 to train/val/test (cluster order shuffled by seed)."""
    from scipy.cluster.vq import kmeans2
    X = latents.astype(np.float64)
    X = (X - X.mean(0)) / X.std(0).clip(1e-8)
    _, labels = kmeans2(X, n_clusters, seed=seed, minit="++")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_clusters)
    groups = {"train": perm[:8], "val": perm[8:10], "test": perm[10:]}
    out = {}
    for key, clusters in groups.items():
        member = set(int(c) for c in clusters)
        out[key] = [n for n, lab in zip(names, labels) if int(lab) in member]
    return out, labels


def compute_stats(shapes, n_sample=4096, seed=0):
    """z/bc/output-channel z-score statistics from the TRAIN shapes; the
    output stats aggregate n_sample random fluid points per shape."""
    z = torch.stack([s["latent"] for s in shapes])
    bc = torch.stack([s["bc"] for s in shapes])
    gen = torch.Generator().manual_seed(seed)
    ys = []
    for s in shapes:
        pool = s["fluid_idx"]
        pick = pool[torch.randint(0, pool.numel(),
                                  (min(n_sample, pool.numel()),),
                                  generator=gen)]
        ys.append(s["fields"][pick])
    Y = torch.cat(ys)
    return {"z_mean": z.mean(0), "z_std": z.std(0).clamp_min(1e-8),
            "bc_mean": bc.mean(0), "bc_std": bc.std(0).clamp_min(1e-8),
            "y_mean": Y.mean(0), "y_std": Y.std(0).clamp_min(1e-8)}


def sample_case_batch(shape, grid_points, n_points, near_frac, gen, device):
    """Draw n_points from one shape: near_frac from the near-wall pool, the
    rest uniform fluid; returns GPU tensors ready for predict_normalized."""
    n_near = int(round(n_points * near_frac))
    pool_near, pool_fluid = shape["near_idx"], shape["fluid_idx"]
    take_near = min(n_near, pool_near.numel())
    take_fluid = n_points - take_near
    idx = torch.cat([
        pool_near[torch.randint(0, pool_near.numel(), (take_near,),
                                generator=gen)],
        pool_fluid[torch.randint(0, pool_fluid.numel(), (take_fluid,),
                                 generator=gen)]])
    return {"latent": shape["latent"].to(device),
            "bc": shape["bc"].to(device),
            "xyz": grid_points[idx].to(device),
            "sdf": shape["sdf"][idx].unsqueeze(1).to(device),
            "normal": shape["normal"][idx].to(device),
            "y": shape["fields"][idx].to(device)}


@torch.no_grad()
def evaluate(model, shapes, grid_points, stats, cfg, n_points, seed,
             chunk=2 ** 18):
    """Physical-space per-variable rel L2 on a fixed-seed fluid subsample
    per shape; baseline predicts the train channel mean (y_mean)."""
    import zlib
    device = next(model.parameters()).device
    stats_g = {k: v.to(device) for k, v in stats.items()}
    per_var, per_case, baselines = [], {}, []
    for s in shapes:
        gen = torch.Generator().manual_seed(
            seed + zlib.crc32(s["name"].encode()))
        pool = s["fluid_idx"]
        idx = pool[torch.randperm(pool.numel(), generator=gen)[:n_points]]
        xyz = grid_points[idx].to(device)
        sdf = s["sdf"][idx].unsqueeze(1).to(device)
        normal = s["normal"][idx].to(device)
        y = s["fields"][idx].to(device)
        preds = []
        for head in range(0, idx.numel(), chunk):
            sl = slice(head, min(head + chunk, idx.numel()))
            pn = predict_normalized(
                model, s["latent"].to(device), s["bc"].to(device),
                xyz[sl], sdf[sl], normal[sl], stats_g, cfg, amp=False)
            preds.append(pn * stats_g["y_std"] + stats_g["y_mean"])
        pred = torch.cat(preds)
        rel_v = [((pred[:, v] - y[:, v]).norm() /
                  y[:, v].norm().clamp_min(1e-12)).item() for v in range(4)]
        per_var.append(rel_v)
        per_case[s["name"]] = float(np.mean(rel_v))
        baselines.append([((stats_g["y_mean"][v] - y[:, v]).norm() /
                           y[:, v].norm().clamp_min(1e-12)).item()
                          for v in range(4)])
    return {"rel_l2": float(np.mean([np.mean(r) for r in per_var])),
            "per_var": [float(np.mean([r[v] for r in per_var]))
                        for v in range(4)],
            "baseline": float(np.mean([np.mean(b) for b in baselines])),
            "per_case": per_case}


@torch.no_grad()
def predict_field(model, shape, grid_points, stats, cfg, chunk=2 ** 18):
    """Full-grid physical-space prediction (G,4), fp32, chunked."""
    device = next(model.parameters()).device
    stats_g = {k: v.to(device) for k, v in stats.items()}
    lat = shape["latent"].to(device)
    bc = shape["bc"].to(device)
    out = []
    for head in range(0, grid_points.shape[0], chunk):
        sl = slice(head, min(head + chunk, grid_points.shape[0]))
        xyz = grid_points[sl].to(device)
        sdf = shape["sdf"][sl].unsqueeze(1).to(device)
        normal = shape["normal"][sl].to(device)
        pn = predict_normalized(model, lat, bc, xyz, sdf, normal,
                                stats_g, cfg, amp=False)
        out.append((pn * stats_g["y_std"] + stats_g["y_mean"]).cpu())
    return torch.cat(out)


def prepare_experiment(experiment_dir, specs_path, checkpoint, data_root,
                       cfg):
    """Shared data pipeline for analysis / surface / physics stages: frozen
    decoder geometry cache (built if missing), snapshot index, cluster
    split, train shapes + stats, and a lazy shape loader. Returns the dict
    documented in the plan (grid_points/grid_shape/axis/names/latents/
    split/labels/snap_idx/cache_dir/train_shapes/stats/load)."""
    from deep_sdf.cfd.volume import make_stretched_grid
    from generate_openfoam_snapshots import load_manifest
    decoder, _ = load_frozen_decoder(specs_path, experiment_dir, checkpoint)
    grid_points, grid_shape, axis = make_stretched_grid()
    names, latents = load_manifest(
        os.path.join(data_root, "lhs_latents.npz"))
    cache_dir = os.path.join(data_root, "sdf_cache")
    build_geometry_cache(
        decoder, [(n, torch.from_numpy(latents[i]))
                  for i, n in enumerate(names)],
        grid_points, cache_dir, near_band=cfg["near_band"])
    del decoder
    torch.cuda.empty_cache()
    snap_idx = snapshot_index(os.path.join(data_root, "snapshots"))
    split, labels = cluster_split(names, latents,
                                  n_clusters=cfg["n_clusters"],
                                  seed=cfg["seed"])
    n_grid = grid_points.shape[0]

    def load(name_list):
        sel = [names.index(n) for n in name_list]
        return load_shapes(name_list, latents[sel], snap_idx, cache_dir,
                           n_grid)

    train_shapes = load(split["train"])
    stats = compute_stats(train_shapes, n_sample=cfg["stats_sample"],
                          seed=cfg["seed"])
    return {"grid_points": grid_points, "grid_shape": grid_shape,
            "axis": axis, "names": names, "latents": latents,
            "split": split, "labels": labels, "snap_idx": snap_idx,
            "cache_dir": cache_dir, "train_shapes": train_shapes,
            "stats": stats, "load": load}


@torch.no_grad()
def evaluate_detailed(model, shapes, grid_points, stats, cfg, n_points, seed,
                      near_band=0.15, chunk=2 ** 18):
    """evaluate() plus per-case near-wall (|sdf|<near_band) vs far masked
    rel L2. Returns {"cases": {name: {"rel_l2", "per_var", "near_rel",
    "far_rel"}}}; a mask subset that is empty yields None for that entry."""
    import zlib
    device = next(model.parameters()).device
    stats_g = {k: v.to(device) for k, v in stats.items()}
    cases = {}
    for s in shapes:
        gen = torch.Generator().manual_seed(
            seed + zlib.crc32(s["name"].encode()))
        pool = s["fluid_idx"]
        idx = pool[torch.randperm(pool.numel(), generator=gen)[:n_points]]
        xyz = grid_points[idx].to(device)
        sdf = s["sdf"][idx].unsqueeze(1).to(device)
        normal = s["normal"][idx].to(device)
        y = s["fields"][idx].to(device)
        preds = []
        for head in range(0, idx.numel(), chunk):
            sl = slice(head, min(head + chunk, idx.numel()))
            pn = predict_normalized(
                model, s["latent"].to(device), s["bc"].to(device),
                xyz[sl], sdf[sl], normal[sl], stats_g, cfg, amp=False)
            preds.append(pn * stats_g["y_std"] + stats_g["y_mean"])
        pred = torch.cat(preds)
        rel_v = [((pred[:, v] - y[:, v]).norm() /
                  y[:, v].norm().clamp_min(1e-12)).item() for v in range(4)]

        def masked_rel(mask):
            if int(mask.sum()) == 0:
                return None
            return float(np.mean([
                ((pred[mask, v] - y[mask, v]).norm() /
                 y[mask, v].norm().clamp_min(1e-12)).item()
                for v in range(4)]))

        near = sdf.squeeze(1).abs() < near_band
        cases[s["name"]] = {
            "rel_l2": float(np.mean(rel_v)),
            "per_var": [float(r) for r in rel_v],
            "near_rel": masked_rel(near),
            "far_rel": masked_rel(~near),
        }
    return {"cases": cases}

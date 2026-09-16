#!/usr/bin/env python3
"""Surface stage (DeepONet v2 spec section 5): surface Cp/Cf DeepONet head
sharing the field branch, dual-path Cd/Cl (surface integration + direct
force head) with consistency loss, and the surface stage runner."""

import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from deep_sdf.cfd import physicsnemo_compat as _compat  # noqa: F401
from deep_sdf.cfd.roadmap_deeponet import (cache_key, fourier_encode,
                                           prepare_experiment)
from deep_sdf.cfd.openfoam_runner import integrate_cd_cl

SURF_VARS = ("cp", "cfx", "cfy", "cfz")


def load_surface_exclusions(surface_dir):
    """Read surface/excluded_shapes.json (degenerate latent-extrapolated
    shapes, plan Global Constraints); returns an empty set when absent."""
    path = os.path.join(surface_dir, "excluded_shapes.json")
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return set(json.load(f)["excluded"])


def load_surface(surface_dir, name):
    path = os.path.join(surface_dir, cache_key(name) + ".npz")
    from deep_sdf.cfd.openfoam_runner import validate_surface_npz
    validate_surface_npz(path)
    data = np.load(path)
    out = {k: torch.from_numpy(np.asarray(data[k], dtype=np.float32))
           for k in ("centers", "normals", "areas", "cp", "cf")}
    out["cd_gt"] = float(np.asarray(data["cd_gt"]))
    out["cl_gt"] = float(np.asarray(data["cl_gt"]))
    return out


def attach_surface(shapes, surface_dir):
    """Attach the 'surf' dict to each shape. Shapes on the exclusion list
    are skipped (no 'surf' key) - callers must filter them out afterwards:
    ``shapes = [s for s in shapes if "surf" in s]``."""
    excluded = load_surface_exclusions(surface_dir)
    for s in list(shapes):
        if s["name"] in excluded:
            logging.info("surface: excluding degenerate shape %s",
                         s["name"])
            continue
        if "surf" not in s:
            s["surf"] = load_surface(surface_dir, s["name"])
    return shapes


def compute_surface_stats(train_shapes):
    """z-score over all train surface points; force std floored at
    0.1*std_Cd per component (spec section 5.1)."""
    cp = torch.cat([s["surf"]["cp"] for s in train_shapes])
    cf = torch.cat([s["surf"]["cf"] for s in train_shapes])
    y = torch.cat([cp.unsqueeze(1), cf], dim=1)
    cd = torch.tensor([s["surf"]["cd_gt"] for s in train_shapes])
    cl = torch.tensor([s["surf"]["cl_gt"] for s in train_shapes])
    fstd = torch.stack([cd.std(), cl.std()])
    fstd = fstd.clamp_min(0.1 * float(cd.std()))
    return {"surf_mean": y.mean(0), "surf_std": y.std(0).clamp_min(1e-8),
            "force_mean": torch.stack([cd.mean(), cl.mean()]),
            "force_std": fstd}


def surface_features(xyz, normal, n_bands, domain_half):
    """(N,3),(N,3) -> (N, 6 + 3*2*n_bands): normalized coords + face
    normal + Fourier(coords)."""
    xn = xyz / domain_half
    return torch.cat([xn, normal, fourier_encode(xn, n_bands)], dim=-1)


def build_surface_model(field_model, cfg):
    """Surface DeepONet sharing field_model.branch1; plus the direct force
    head (z,bc)->(Cd,Cl). Returns (surface_model, force_head)."""
    from physicsnemo.models.mlp import FullyConnected
    from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
    trunk_in = 3 + 3 + 3 * 2 * cfg["fourier_bands"]
    surface_trunk = FullyConnected(
        in_features=trunk_in, layer_size=cfg["surface_trunk_hidden"],
        num_layers=cfg["surface_trunk_layers"], out_features=cfg["width"],
        activation_fn="silu")
    surface_model = DeepONet(
        field_model.branch1, trunk=surface_trunk, dimension=3,
        width=cfg["width"], out_channels=len(SURF_VARS),
        decoder_type="mlp", decoder_width=cfg["decoder_hidden"],
        decoder_layers=cfg["decoder_layers"], decoder_activation_fn="silu")
    force_head = FullyConnected(
        in_features=cfg["latent_size"] + cfg["bc_dim"],
        layer_size=cfg["force_hidden"], num_layers=cfg["force_layers"],
        out_features=2, activation_fn="silu")
    return surface_model, force_head


def _branch_input(latent, bc, stats):
    z = (latent - stats["z_mean"]) / stats["z_std"]
    b = (bc - stats["bc_mean"]) / stats["bc_std"]
    return torch.cat([z, b]).unsqueeze(0)                    # (1, 20)


def predict_surface_normalized(surface_model, latent, bc, xyz, normal,
                               stats, cfg, amp=False):
    xt = surface_features(xyz, normal, cfg["fourier_bands"],
                          cfg["domain_half"])
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        y = surface_model(_branch_input(latent, bc, stats), xt)[0]
    return y.float()                                         # (N, 4)


def predict_force_normalized(force_head, latent, bc, stats):
    return force_head(_branch_input(latent, bc, stats))[0]   # (2,)


def integrate_cd_cl_mc(cp, cf, normals, area_total, direction):
    """Area-weighted Monte-Carlo estimate of the force integral from
    sampled faces: each sample represents area_total / n of surface."""
    n_pts = cp.shape[0]
    a = torch.full((n_pts,), float(area_total) / n_pts,
                   device=cp.device, dtype=cp.dtype)
    return integrate_cd_cl(cp, cf, normals, a, direction)


def _rel_err(pred, gt, floor=0.05):
    return float(abs(float(pred) - float(gt)) / max(abs(float(gt)), floor))


@torch.no_grad()
def evaluate_surface(surface_model, force_head, shapes, stats, cfg,
                     chunk=2 ** 18):
    """Full-surface eval per shape: Cp rel L2, per-component Cf rel L2,
    Cd/Cl relative error for the head and integral paths. Returns the dict
    documented in the plan."""
    device = next(surface_model.parameters()).device
    stats_g = {k: v.to(device) for k, v in stats.items()}
    per_case = {}
    for s in shapes:
        surf = s["surf"]
        n_faces = surf["centers"].shape[0]
        lat = s["latent"].to(device)
        bc = s["bc"].to(device)
        preds = []
        for head in range(0, n_faces, chunk):
            sl = slice(head, min(head + chunk, n_faces))
            pn = predict_surface_normalized(
                surface_model, lat, bc, surf["centers"][sl].to(device),
                surf["normals"][sl].to(device), stats_g, cfg, amp=False)
            preds.append(pn * stats_g["surf_std"] + stats_g["surf_mean"])
        pred = torch.cat(preds)                                # (F, 4)
        cp_gt = surf["cp"].to(device)
        cf_gt = surf["cf"].to(device)
        cp_rel = ((pred[:, 0] - cp_gt).norm()
                  / cp_gt.norm().clamp_min(1e-12)).item()
        cf_rel = [((pred[:, 1 + v] - cf_gt[:, v]).norm() /
                   cf_gt[:, v].norm().clamp_min(1e-12)).item()
                  for v in range(3)]
        direction = bc[1:4]
        cd_int, cl_int = integrate_cd_cl(pred[:, 0], pred[:, 1:4],
                                         surf["normals"].to(device),
                                         surf["areas"].to(device),
                                         direction)
        fh = predict_force_normalized(force_head, lat, bc, stats_g)
        fh = fh * stats_g["force_std"] + stats_g["force_mean"]
        per_case[s["name"]] = {
            "cp_rel": float(cp_rel),
            "cf_rel": [float(r) for r in cf_rel],
            "cd_head": _rel_err(fh[0], surf["cd_gt"]),
            "cd_int": _rel_err(cd_int, surf["cd_gt"]),
            "cl_head": _rel_err(fh[1], surf["cl_gt"]),
            "cl_int": _rel_err(cl_int, surf["cl_gt"]),
            "cd_gt": surf["cd_gt"], "cl_gt": surf["cl_gt"],
            "cd_int_val": float(cd_int), "cd_head_val": float(fh[0]),
        }

    def median(key):
        vals = [r[key] for r in per_case.values()]
        return float(np.median(vals)) if vals else None

    worst = sorted(per_case.items(), key=lambda kv: -kv[1]["cd_int"])[:5]
    return {"cp_rel": median("cp_rel"),
            "cf_rel": [float(np.median([r["cf_rel"][v]
                                        for r in per_case.values()]))
                       for v in range(3)],
            "cd_head": median("cd_head"), "cd_int": median("cd_int"),
            "cl_head": median("cl_head"), "cl_int": median("cl_int"),
            "per_case": per_case,
            "worst5": [{"name": n, "cd_int": r["cd_int"],
                        "cd_gt": r["cd_gt"], "cd_int_val": r["cd_int_val"]}
                       for n, r in worst]}


def run_surface_stage(args, cfg):
    """Surface stage runner: init branch from the field best checkpoint,
    train surface trunk/decoder + force head (branch at 0.1x lr), save
    best_surface.mdlus + best_surface_force.pth + metrics_surface.jsonl,
    then write eval_surface.json."""
    from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
    out_dir = args._out_dir
    device = torch.device("cuda")
    init_from = args.init_from or os.path.join(
        args.experiment_directory, "RoadmapONet", "best.mdlus")
    field_model = DeepONet.from_checkpoint(init_from)
    surface_model, force_head = build_surface_model(field_model, cfg)
    surface_model.to(device)
    force_head.to(device)
    del field_model

    prep = prepare_experiment(args.experiment_directory, args.specs,
                              args.checkpoint, args.data_root, cfg)
    surface_dir = os.path.join(args.data_root, "surface")
    grid_points = prep["grid_points"]
    split = prep["split"]
    train_shapes = [s for s in attach_surface(prep["train_shapes"],
                                              surface_dir)
                    if "surf" in s]
    if not train_shapes:
        raise RuntimeError("no train shapes with valid surface data")
    stats = prep["stats"]
    stats.update(compute_surface_stats(train_shapes))
    stats_g = {k: v.to(device) for k, v in stats.items()}
    torch.save(stats, os.path.join(out_dir, "stats_surface.pth"))
    with open(os.path.join(out_dir, "config_surface.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    iters = 500 if args.smoke else cfg["surface_iters"]
    eval_every = 100 if args.smoke else cfg["eval_every"]
    param_groups = [
        {"params": surface_model.branch1.parameters(),
         "lr": cfg["lr"] * cfg["branch_lr_scale"]},
        {"params": [p for n, p in surface_model.named_parameters()
                    if not n.startswith("branch1.")], "lr": cfg["lr"]},
        {"params": force_head.parameters(), "lr": cfg["lr"]},
    ]
    opt = torch.optim.AdamW(param_groups,
                            weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    def lr_at(it):
        t = min(it / max(iters, 1), 1.0)
        return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (
            1 + np.cos(np.pi * t))

    val_shapes = [s for s in attach_surface(prep["load"](split["val"]),
                                            surface_dir) if "surf" in s]
    if args.smoke:
        val_shapes = val_shapes[:8]
    metrics_path = os.path.join(out_dir, "metrics_surface.jsonl")
    state_path = os.path.join(out_dir, "train_state_surface.pth")
    gen = torch.Generator().manual_seed(cfg["seed"])
    start_iter, best = 0, float("inf")
    if args.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=device)
        surface_model.load_state_dict(st["model_state_dict"])
        force_head.load_state_dict(st["force_head_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        start_iter, best = st["iter"], st["best"]
        logging.info("surface resumed from iter %d (best %.4f)",
                     start_iter, best)

    def val_metric():
        surface_model.eval(); force_head.eval()
        res = evaluate_surface(surface_model, force_head, val_shapes,
                               stats_g, cfg)
        surface_model.train(); force_head.train()
        return res

    t0 = time.time()
    for it in range(start_iter, iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it) * (
                cfg["branch_lr_scale"]
                if g is param_groups[0] else 1.0)
        cases = [train_shapes[int(torch.randint(0, len(train_shapes),
                                                (1,), generator=gen))]
                 for _ in range(cfg["batch_cases"])]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for c in cases:
            surf = c["surf"]
            probs = surf["areas"]
            idx = torch.multinomial(probs, cfg["surface_points"],
                                    replacement=True, generator=gen)
            xyz = surf["centers"][idx].to(device)
            nrm = surf["normals"][idx].to(device)
            y = torch.cat([surf["cp"][idx].unsqueeze(1),
                           surf["cf"][idx]], dim=1).to(device)
            lat = c["latent"].to(device)
            bc = c["bc"].to(device)
            pred = predict_surface_normalized(
                surface_model, lat, bc, xyz, nrm, stats_g, cfg,
                amp=cfg["amp"])
            yn = (y - stats_g["surf_mean"]) / stats_g["surf_std"]
            l_surf = sum(F.mse_loss(pred[:, v], yn[:, v]) for v in range(4))
            pred_phys = pred * stats_g["surf_std"] + stats_g["surf_mean"]
            cd_int, cl_int = integrate_cd_cl_mc(
                pred_phys[:, 0], pred_phys[:, 1:4], nrm,
                surf["areas"].sum(), bc[1:4])
            fh = predict_force_normalized(force_head, lat, bc, stats_g)
            gt = torch.stack([
                (torch.as_tensor(c["surf"]["cd_gt"], device=device)
                 - stats_g["force_mean"][0]) / stats_g["force_std"][0],
                (torch.as_tensor(c["surf"]["cl_gt"], device=device)
                 - stats_g["force_mean"][1]) / stats_g["force_std"][1]])
            ints = torch.stack([
                (cd_int - stats_g["force_mean"][0])
                / stats_g["force_std"][0],
                (cl_int - stats_g["force_mean"][1])
                / stats_g["force_std"][1]])
            l_force = (ints - gt).abs().sum() + (fh - gt).abs().sum()
            l_cons = ((fh - ints) ** 2).sum()
            loss = loss + l_surf \
                + cfg["lambda_force"] * l_force \
                + cfg["lambda_consistency"] * l_cons
        loss = loss / len(cases)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (it + 1) % cfg["metrics_every"] == 0 or it == start_iter:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iter": it + 1, "train_loss": float(loss.item()),
                    "lr": lr_at(it),
                    "sec_per_iter": (time.time() - t0)
                    / (it + 1 - start_iter)}) + "\n")

        if (it + 1) % eval_every == 0 or it + 1 == iters:
            res = val_metric()
            crit = res["cp_rel"] + res["cd_int"]
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iter": it + 1, "val_cp_rel": res["cp_rel"],
                    "val_cf_rel": res["cf_rel"],
                    "val_cd_int": res["cd_int"],
                    "val_cd_head": res["cd_head"]}) + "\n")
            logging.info("surface iter %d loss %.4f cp_rel %.4f "
                         "cd_int %.4f cd_head %.4f",
                         it + 1, float(loss.item()), res["cp_rel"],
                         res["cd_int"], res["cd_head"])
            if crit < best:
                best = crit
                surface_model.save(os.path.join(out_dir,
                                                "best_surface.mdlus"))
                torch.save(force_head.state_dict(),
                           os.path.join(out_dir, "best_surface_force.pth"))
            torch.save({"model_state_dict": surface_model.state_dict(),
                        "force_head_state_dict": force_head.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "iter": it + 1, "best": best}, state_path)

    if not args.smoke:
        surface_model.eval(); force_head.eval()
        test_shapes = [s for s in attach_surface(prep["load"](split["test"]),
                                                 surface_dir)
                       if "surf" in s]
        res = {"val": evaluate_surface(surface_model, force_head,
                                       val_shapes, stats_g, cfg),
               "test": evaluate_surface(surface_model, force_head,
                                        test_shapes, stats_g, cfg)}
        with open(os.path.join(out_dir, "eval_surface.json"), "w") as f:
            json.dump(res, f, indent=1)
        logging.info("surface test: cp_rel %.4f cd_int %.4f cd_head %.4f",
                     res["test"]["cp_rel"], res["test"]["cd_int"],
                     res["test"]["cd_head"])
    logging.info("surface stage done. best crit %.4f", best)

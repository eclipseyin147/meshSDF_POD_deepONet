#!/usr/bin/env python3
"""Physics stage (DeepONet v2 spec section 6): PDE fine-tuning of the
field model with steady incompressible NS residuals (PDEInformer), wall /
far-field boundary losses, and the physics stage runner. Follows the
PiPOD train_pipod_deeponet.physics_losses pattern: SDF through the frozen
decoder WITH graph, SDF gradient detached, per-chunk backward, AMP off
for the residual path."""

import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from deep_sdf.cfd.roadmap_deeponet import (trunk_features,
                                           predict_normalized,
                                           sample_case_batch, evaluate,
                                           prepare_experiment)
from deep_sdf.utils import decode_sdf


def predict_physical(model, latent, bc, xyz, sdf, normal, stats, cfg,
                     amp=False):
    pn = predict_normalized(model, latent, bc, xyz, sdf, normal, stats,
                            cfg, amp=amp)
    return pn * stats["y_std"] + stats["y_mean"]


def physics_losses(model, decoder, latent, bc, points, stats, cfg,
                   informer, max_batch=256, backward_scale=None):
    """Continuity + momentum residuals at collocation points (second-order
    autodiff). ``latent`` is the RAW latent (decoder input); the branch
    z-score is applied inside. With ``backward_scale`` each chunk's loss
    is backwarded immediately (scaled) and detached scalars returned."""
    device = points.device
    n_tot = points.shape[0]
    lc = torch.zeros((), device=device)
    lm = torch.zeros((), device=device)
    z = (latent.reshape(-1) - stats["z_mean"]) / stats["z_std"]
    b = (bc.reshape(-1) - stats["bc_mean"]) / stats["bc_std"]
    xb = torch.cat([z, b]).unsqueeze(0)                        # (1, 20)
    for lo in range(0, n_tot, max_batch):
        chunk = points[lo:lo + max_batch]
        p = chunk.detach().requires_grad_(True)
        with torch.enable_grad():
            d = decode_sdf(decoder, latent.reshape(1, -1), p)
            g = torch.autograd.grad(d.sum(), p, create_graph=True)[0]
            n = (g / g.norm(dim=1, keepdim=True).clamp_min(1e-8)).detach()
            xt = trunk_features(p, d, n, cfg["fourier_bands"],
                                cfg["domain_half"])
            with torch.amp.autocast("cuda", enabled=False):
                qn = model(xb, xt)[0].float()
                q = qn * stats["y_std"] + stats["y_mean"]
                res = informer({"coordinates": p, "u": q[:, 0:1],
                                "v": q[:, 1:2], "w": q[:, 2:3],
                                "cp": q[:, 3:4]})
                l_c = (res["continuity"] ** 2).sum()
                l_m = sum((res["momentum_" + k] ** 2).sum()
                          for k in "uvw")
            if backward_scale is not None:
                (backward_scale * (l_c + l_m) / max(n_tot, 1)).backward()
        lc = lc + l_c.detach()
        lm = lm + l_m.detach()
    return lc / max(n_tot, 1), lm / max(n_tot, 1)


def boundary_losses(model, shape, bc, grid_points, idx_wall, idx_far,
                    stats, cfg):
    """noslip |u|^2 on the wall band + far-field u->dir, cp->0 (PiPOD
    boundary_losses pattern; features from the geometry cache)."""
    device = grid_points.device
    l_wall = torch.zeros((), device=device)
    l_far = torch.zeros((), device=device)
    direction = bc[1:4] / bc[1:4].norm()
    if idx_wall.numel():
        qw = predict_physical(
            model, shape["latent"].to(device), bc, grid_points[idx_wall],
            shape["sdf"][idx_wall.cpu()].unsqueeze(1).to(device),
            shape["normal"][idx_wall.cpu()].to(device), stats, cfg,
            amp=False)
        l_wall = (qw[:, :3] ** 2).sum(1).mean()
    if idx_far.numel():
        qf = predict_physical(
            model, shape["latent"].to(device), bc, grid_points[idx_far],
            shape["sdf"][idx_far.cpu()].unsqueeze(1).to(device),
            shape["normal"][idx_far.cpu()].to(device), stats, cfg,
            amp=False)
        tgt = torch.zeros_like(qf)
        tgt[:, :3] = direction.to(device)
        l_far = ((qf - tgt) ** 2).sum(1).mean()
    return l_wall, l_far


def _shape_pools(cache, sampler, grid_g, grid_shape, shape, device):
    """Per-shape collocation pools (deterministic in sdf/fields/bc) cached
    across iterations - recomputing the full-grid FD-gradient pools costs
    ~0.4s per case, dominated by launch-bound tensor ops."""
    key = (shape["name"], tuple(float(x) for x in shape["bc"]))
    if key not in cache:
        cache[key] = sampler._pools(grid_g, grid_shape,
                                    shape["sdf"].to(device),
                                    shape["fields"].to(device),
                                    shape["bc"])
    return cache[key]


@torch.no_grad()
def eval_residuals(model, decoder, shapes, grid_points, grid_shape, stats,
                   cfg, informer, sampler, n_points, seed, device):
    """Mean continuity/momentum/wall/far metrics over shapes on a fixed
    collocation sample per shape (no backward)."""
    from deep_sdf.cfd.physics import CollocationSampler  # noqa: F401
    grid_g = grid_points.to(device)
    gen = torch.Generator().manual_seed(seed)
    cache = {}
    out = {k: [] for k in ("continuity", "momentum", "wall", "far")}
    for s in shapes:
        picks = sampler.sample(
            grid_g, grid_shape, s["sdf"].to(device),
            s["fields"].to(device), s["bc"], n_points, gen,
            pools=_shape_pools(cache, sampler, grid_g, grid_shape, s,
                               device))
        lc, lm = physics_losses(
            model, decoder, s["latent"].to(device),
            s["bc"].to(device), grid_g[picks["collocation"]], stats, cfg,
            informer, max_batch=cfg["phys_chunk"], backward_scale=None)
        lw, lf = boundary_losses(model, s, s["bc"].to(device), grid_g,
                                 picks["wall"], picks["far"], stats, cfg)
        out["continuity"].append(float(lc))
        out["momentum"].append(float(lm))
        out["wall"].append(float(lw))
        out["far"].append(float(lf))
    return {k: float(np.mean(v)) for k, v in out.items()}


def run_physics_stage(args, cfg):
    """Physics stage runner: init from the field best, fine-tune with
    L_field + lambda_phys(t) * (L_cont + L_mom + L_wall + L_far)."""
    from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
    from deep_sdf.cfd.physics import (PDEInformer, IncompressibleNS,
                                      CollocationSampler,
                                      physics_weight_schedule)
    from deep_sdf.cfd.roadmap_deeponet import (build_model,
                                               load_frozen_decoder)
    out_dir = args._out_dir
    device = torch.device("cuda")
    init_from = args.init_from or os.path.join(
        args.experiment_directory, "RoadmapONet", "best.mdlus")
    model = DeepONet.from_checkpoint(init_from).to(device)
    decoder, _ = load_frozen_decoder(args.specs,
                                     args.experiment_directory,
                                     args.checkpoint)

    prep = prepare_experiment(args.experiment_directory, args.specs,
                              args.checkpoint, args.data_root, cfg)
    grid_points, grid_shape = prep["grid_points"], prep["grid_shape"]
    train_shapes = prep["train_shapes"]
    stats = prep["stats"]
    stats_g = {k: v.to(device) for k, v in stats.items()}
    with open(os.path.join(out_dir, "config_physics.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    iters = 500 if args.smoke else cfg["physics_iters"]
    eval_every = 100 if args.smoke else cfg["eval_every"]
    informer = PDEInformer(IncompressibleNS(re=cfg["re"]).equations)
    sampler = CollocationSampler()
    grid_g = grid_points.to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["lr"] * cfg["phys_lr_scale"],
                            weight_decay=cfg["weight_decay"])
    lambda_fixed = args.lambda_phys

    def lr_at(it):
        base = cfg["lr"] * cfg["phys_lr_scale"]
        lo = cfg["lr_min"] * cfg["phys_lr_scale"]
        t = min(it / max(iters, 1), 1.0)
        return lo + 0.5 * (base - lo) * (1 + np.cos(np.pi * t))

    val_shapes = prep["load"](prep["split"]["val"])
    if args.smoke:
        val_shapes = val_shapes[:8]
    metrics_path = os.path.join(out_dir, "metrics_physics.jsonl")
    state_path = os.path.join(out_dir, "train_state_physics.pth")
    gen = torch.Generator().manual_seed(cfg["seed"])
    start_iter, best = 0, float("inf")
    if args.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=device)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        start_iter, best = st["iter"], st["best"]
        logging.info("physics resumed from iter %d (best %.4f)",
                     start_iter, best)

    def run_eval(n_points):
        model.eval()
        res = evaluate(model, val_shapes, grid_points, stats_g, cfg,
                       n_points=n_points, seed=12345)
        model.train()
        return res

    t0 = time.time()
    model.train()
    pool_cache = {}
    for it in range(start_iter, iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        progress = it / max(iters, 1)
        lam = (lambda_fixed if lambda_fixed is not None
               else physics_weight_schedule(progress))
        cases = [train_shapes[int(torch.randint(0, len(train_shapes),
                                                (1,), generator=gen))]
                 for _ in range(cfg["batch_cases"])]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        phys_terms = {}
        for c in cases:
            b = sample_case_batch(c, grid_points, cfg["n_points"],
                                  cfg["near_frac"], gen, device)
            pred = predict_normalized(model, b["latent"], b["bc"],
                                      b["xyz"], b["sdf"], b["normal"],
                                      stats_g, cfg, amp=cfg["amp"])
            yn = (b["y"] - stats_g["y_mean"]) / stats_g["y_std"]
            loss = loss + sum(F.mse_loss(pred[:, v], yn[:, v])
                              for v in range(4))
            if lam > 0.0:
                picks = sampler.sample(
                    grid_g, grid_shape, c["sdf"].to(device),
                    c["fields"].to(device), c["bc"],
                    cfg["n_collocation"], gen,
                    pools=_shape_pools(pool_cache, sampler, grid_g,
                                       grid_shape, c, device))
                lc, lm = physics_losses(
                    model, decoder, c["latent"].to(device),
                    c["bc"].to(device), grid_g[picks["collocation"]],
                    stats_g, cfg, informer,
                    max_batch=cfg["phys_chunk"], backward_scale=lam)
                lw, lf = boundary_losses(
                    model, c, c["bc"].to(device), grid_g, picks["wall"],
                    picks["far"], stats_g, cfg)
                loss = loss + lam * (lw + lf)
                phys_terms = {"cont": float(lc), "mom": float(lm),
                              "wall": float(lw), "far": float(lf),
                              "lambda_phys": lam}
        loss = loss / len(cases)
        loss.backward()
        opt.step()

        if (it + 1) % cfg["metrics_every"] == 0 or it == start_iter:
            rec = {"iter": it + 1, "train_loss": float(loss.item()),
                   "lr": lr_at(it),
                   "sec_per_iter": (time.time() - t0)
                   / (it + 1 - start_iter)}
            rec.update(phys_terms)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if (it + 1) % eval_every == 0 or it + 1 == iters:
            res = run_eval(cfg["eval_points"])
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iter": it + 1, "val_rel_l2": res["rel_l2"],
                    "val_u": res["per_var"][0], "val_v": res["per_var"][1],
                    "val_w": res["per_var"][2], "val_p": res["per_var"][3],
                    "val_baseline": res["baseline"]}) + "\n")
            logging.info("physics iter %d loss %.4f val %.4f lam %.3g",
                         it + 1, float(loss.item()), res["rel_l2"], lam)
            if res["rel_l2"] < best:
                best = res["rel_l2"]
                model.save(os.path.join(out_dir, "best_physics.mdlus"))
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "iter": it + 1, "best": best}, state_path)

    if not args.smoke:
        model.eval()
        best_model = DeepONet.from_checkpoint(
            os.path.join(out_dir, "best_physics.mdlus")).to(device).eval()
        field_model = DeepONet.from_checkpoint(init_from).to(device).eval()
        res = {
            "val": evaluate(best_model, val_shapes, grid_points, stats_g,
                            cfg, cfg["final_eval_points"], 777),
            "residuals_physics": eval_residuals(
                best_model, decoder, val_shapes, grid_points, grid_shape,
                stats_g, cfg, informer, sampler, cfg["n_collocation"],
                4242, device),
            "residuals_field_only": eval_residuals(
                field_model, decoder, val_shapes, grid_points, grid_shape,
                stats_g, cfg, informer, sampler, cfg["n_collocation"],
                4242, device),
        }
        with open(os.path.join(out_dir, "eval_physics.json"), "w") as f:
            json.dump(res, f, indent=1)
        logging.info("physics eval: val %.4f | cont %.4g -> %.4g | "
                     "mom %.4g -> %.4g",
                     res["val"]["rel_l2"],
                     res["residuals_field_only"]["continuity"],
                     res["residuals_physics"]["continuity"],
                     res["residuals_field_only"]["momentum"],
                     res["residuals_physics"]["momentum"])
    logging.info("physics stage done. best val %.4f", best)

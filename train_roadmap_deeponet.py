#!/usr/bin/env python3
"""DeepSDF + PhysicsNeMo DeepONet MVP trainer (roadmap section 63; spec
docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md).

    .venv/bin/python train_roadmap_deeponet.py --smoke
    .venv/bin/python train_roadmap_deeponet.py --iters 20000
    .venv/bin/python train_roadmap_deeponet.py --eval_only --resume

Outputs under <experiment>/RoadmapONet/ (RoadmapONet_smoke/ with --smoke):
config.json, split.json, stats.pth, metrics.jsonl, best.mdlus,
train_state.pth, eval.json.
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import deep_sdf.cfd.physicsnemo_compat  # noqa: F401 (before physicsnemo)
from deep_sdf.cfd import roadmap_deeponet as rd
from deep_sdf.cfd.volume import make_stretched_grid
from generate_openfoam_snapshots import load_manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_directory", default="examples/ellipsoids")
    p.add_argument("--specs", default="examples/ellipsoids_of4/specs.json")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--data_root", default="data/openfoam/ellipsoids_u10")
    p.add_argument("--config", default=None,
                   help="JSON with DEFAULT_CFG overrides")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--out_name", default=None,
                   help="output dir under <experiment>/ (default "
                        "RoadmapONet, or RoadmapONet_smoke with --smoke)")
    p.add_argument("--analyze", action="store_true",
                   help="write analysis.json (near/far masked per-case "
                        "errors, worst-10) from <out_name>/best.mdlus")
    p.add_argument("--adaptive_sampling", action="store_true",
                   help="shape-level error-adaptive case sampling "
                        "(spec section 4.2)")
    p.add_argument("--stage", choices=["field", "surface", "physics"],
                   default="field")
    p.add_argument("--init_from", default=None,
                   help="checkpoint to init surface/physics stages from "
                        "(default <experiment>/RoadmapONet/best.mdlus)")
    p.add_argument("--lambda_phys", type=float, default=None,
                   help="fixed physics loss weight (default: schedule)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = dict(rd.DEFAULT_CFG)
    if args.config:
        cfg.update(json.load(open(args.config)))
    if args.iters is not None:
        cfg["iters"] = args.iters
    if args.smoke:
        cfg.update(iters=500, eval_every=100, eval_points=4096)

    out_name = args.out_name or (
        "RoadmapONet_smoke" if args.smoke else "RoadmapONet")
    out_dir = os.path.join(args.experiment_directory, out_name)
    os.makedirs(out_dir, exist_ok=True)
    args._out_dir = out_dir
    if args.stage == "surface":
        from deep_sdf.cfd import roadmap_surface
        roadmap_surface.run_surface_stage(args, cfg)
        return
    if args.stage == "physics":
        from deep_sdf.cfd import roadmap_physics
        roadmap_physics.run_physics_stage(args, cfg)
        return
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")

    # --- data -------------------------------------------------------------
    decoder, latent_size = rd.load_frozen_decoder(
        args.specs, args.experiment_directory, args.checkpoint)
    assert latent_size == cfg["latent_size"]
    grid_points, grid_shape, _ = make_stretched_grid()
    n_grid = grid_points.shape[0]
    names, latents = load_manifest(
        os.path.join(args.data_root, "lhs_latents.npz"))
    cache_dir = os.path.join(args.data_root, "sdf_cache")
    rd.build_geometry_cache(
        decoder, [(n, torch.from_numpy(latents[i]))
                  for i, n in enumerate(names)],
        grid_points, cache_dir, near_band=cfg["near_band"])
    del decoder
    torch.cuda.empty_cache()

    snap_idx = rd.snapshot_index(os.path.join(args.data_root, "snapshots"))
    split, labels = rd.cluster_split(names, latents,
                                     n_clusters=cfg["n_clusters"], seed=seed)
    logging.info("split sizes: %s",
                 {k: len(v) for k, v in split.items()})

    def load(name_list):
        sel = [names.index(n) for n in name_list]
        return rd.load_shapes(name_list, latents[sel], snap_idx, cache_dir,
                              n_grid)

    train_shapes = load(split["train"])
    stats = rd.compute_stats(train_shapes, n_sample=cfg["stats_sample"],
                             seed=seed)

    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({**split,
                   "labels": {n: int(l) for n, l in zip(names, labels)}},
                  f, indent=1)
    torch.save(stats, os.path.join(out_dir, "stats.pth"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    # --- model / optimizer -------------------------------------------------
    model = rd.build_model(cfg).to(device)
    stats_g = {k: v.to(device) for k, v in stats.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    start_iter, best = 0, float("inf")
    state_path = os.path.join(out_dir, "train_state.pth")
    if args.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=device)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        start_iter, best = st["iter"], st["best"]
        logging.info("resumed from iter %d (best %.4f)", start_iter, best)

    def lr_at(it):
        t = min(it / max(cfg["iters"], 1), 1.0)
        return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (
            1 + np.cos(np.pi * t))

    val_shapes = load(split["val"])
    if args.smoke:
        val_shapes = val_shapes[:8]

    def run_eval(n_points):
        model.eval()
        res = rd.evaluate(model, val_shapes, grid_points, stats_g, cfg,
                          n_points=n_points, seed=12345)
        model.train()
        return res

    if args.analyze:
        from physicsnemo.experimental.models.xdeeponet.deeponet import (
            DeepONet)
        model = DeepONet.from_checkpoint(
            os.path.join(out_dir, "best.mdlus")).to(device).eval()
        val_shapes = load(split["val"])
        test_shapes = load(split["test"])
        lat_t = torch.from_numpy(latents).float()
        label_map = {n: int(l) for n, l in zip(names, labels)}
        # 簇心只用 train 形状的 latent；val/test 的簇按 cluster_split 不含
        # train 成员，故 dist_to_centroid 取到最近 train 簇心的距离
        train_idx = [names.index(n) for n in split["train"]]
        centroids = {}
        for c in np.unique(labels[train_idx]):
            rows = [i for i in train_idx if labels[i] == c]
            centroids[int(c)] = lat_t[rows].mean(0)

        out = {"cases": []}
        for split_name, shapes in (("val", val_shapes),
                                   ("test", test_shapes)):
            det = rd.evaluate_detailed(
                model, shapes, grid_points, stats_g, cfg,
                n_points=cfg["analyze_points"], seed=777)
            for nm, rec in det["cases"].items():
                zi = lat_t[names.index(nm)]
                cen = min(centroids.values(),
                          key=lambda c: float((zi - c).norm()))
                rec = dict(rec)
                rec.update(split=split_name, name=nm,
                           cluster=label_map[nm],
                           dist_to_centroid=float((zi - cen).norm()))
                out["cases"].append(rec)
        out["cases"].sort(key=lambda r: -r["rel_l2"])
        out["worst10"] = [r["name"] for r in out["cases"][:10]]
        with open(os.path.join(out_dir, "analysis.json"), "w") as f:
            json.dump(out, f, indent=1)
        logging.info("worst-10 by rel_l2:")
        for r in out["cases"][:10]:
            logging.info("  %-58s %s rel %.4f near %s far %s cluster %d "
                         "dist %.3f", r["name"], r["split"], r["rel_l2"],
                         "%.4f" % r["near_rel"] if r["near_rel"] is not None
                         else "n/a",
                         "%.4f" % r["far_rel"] if r["far_rel"] is not None
                         else "n/a",
                         r["cluster"], r["dist_to_centroid"])
        return

    if args.eval_only:
        from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
        model = DeepONet.from_checkpoint(
            os.path.join(out_dir, "best.mdlus")).to(device).eval()
        test_shapes = load(split["test"])
        res = {"val": rd.evaluate(model, val_shapes, grid_points, stats_g,
                                  cfg, cfg["final_eval_points"], 777),
               "test": rd.evaluate(model, test_shapes, grid_points, stats_g,
                                   cfg, cfg["final_eval_points"], 999)}
        shape0 = test_shapes[0]
        t0 = time.time()
        rd.predict_field(model, shape0, grid_points, stats_g, cfg)
        res["infer_sec_per_case"] = time.time() - t0
        res["speedup_vs_cfd_100s"] = 100.0 / res["infer_sec_per_case"]
        with open(os.path.join(out_dir, "eval.json"), "w") as f:
            json.dump(res, f, indent=1)
        logging.info("eval: %s", json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in res.items() if k != "val" and k != "test"}))
        logging.info("val rel_l2 %.4f (baseline %.4f) | test rel_l2 %.4f "
                     "(baseline %.4f) | infer %.2fs/case (~%.0fx)",
                     res["val"]["rel_l2"], res["val"]["baseline"],
                     res["test"]["rel_l2"], res["test"]["baseline"],
                     res["infer_sec_per_case"], res["speedup_vs_cfd_100s"])
        return

    # --- train --------------------------------------------------------------
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    gen = torch.Generator().manual_seed(seed)
    shape_weights = None
    if args.adaptive_sampling:
        shape_weights = torch.ones(len(train_shapes))
    model.train()
    t0 = time.time()
    for it in range(start_iter, cfg["iters"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        if shape_weights is not None:
            picks = torch.multinomial(shape_weights, cfg["batch_cases"],
                                      replacement=True,
                                      generator=gen).tolist()
            cases = [train_shapes[i] for i in picks]
        else:
            cases = [train_shapes[int(torch.randint(0, len(train_shapes),
                                                    (1,), generator=gen))]
                     for _ in range(cfg["batch_cases"])]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for c in cases:
            b = rd.sample_case_batch(c, grid_points, cfg["n_points"],
                                     cfg["near_frac"], gen, device)
            pred = rd.predict_normalized(model, b["latent"], b["bc"],
                                         b["xyz"], b["sdf"], b["normal"],
                                         stats_g, cfg, amp=cfg["amp"])
            yn = (b["y"] - stats_g["y_mean"]) / stats_g["y_std"]
            loss = loss + sum(F.mse_loss(pred[:, v], yn[:, v])
                              for v in range(4))
        loss = loss / len(cases)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (it + 1) % cfg["metrics_every"] == 0 or it == start_iter:
            rec = {"iter": it + 1, "train_loss": float(loss.item()),
                   "lr": lr_at(it),
                   "sec_per_iter": (time.time() - t0) / (it + 1 - start_iter)}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if (it + 1) % cfg["eval_every"] == 0 or it + 1 == cfg["iters"]:
            res = run_eval(cfg["eval_points"])
            rec = {"iter": it + 1, "val_rel_l2": res["rel_l2"],
                   "val_u": res["per_var"][0], "val_v": res["per_var"][1],
                   "val_w": res["per_var"][2], "val_p": res["per_var"][3],
                   "val_baseline": res["baseline"]}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            logging.info("iter %d loss %.4f val %.4f (baseline %.4f)",
                         it + 1, float(loss.item()), res["rel_l2"],
                         res["baseline"])
            if res["rel_l2"] < best:
                best = res["rel_l2"]
                model.save(os.path.join(out_dir, "best.mdlus"))
            if args.adaptive_sampling:
                model.eval()
                quick = rd.evaluate(model, train_shapes, grid_points,
                                    stats_g, cfg,
                                    n_points=cfg["adaptive_points"],
                                    seed=1000 + it)
                model.train()
                errs = torch.tensor([quick["per_case"][s["name"]]
                                     for s in train_shapes])
                shape_weights = errs / errs.mean().clamp_min(1e-12) \
                    + cfg["adaptive_eps"]
                top = torch.argsort(shape_weights, descending=True)[:5]
                logging.info("adaptive top-5: %s",
                             [train_shapes[i]["name"] for i in top])
                rec = {"iter": it + 1, "adaptive_mean_err":
                       float(errs.mean()),
                       "adaptive_worst_err": float(errs.max())}
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "iter": it + 1, "best": best}, state_path)

    logging.info("done. best val rel_l2 %.4f", best)


if __name__ == "__main__":
    main()

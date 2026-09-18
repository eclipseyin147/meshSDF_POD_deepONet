#!/usr/bin/env python3
"""Plot RoadmapONet (DeepSDF + PhysicsNeMo DeepONet) training curves from
the JSONL metrics logs written by train_roadmap_deeponet.py:

    <dir>/metrics.jsonl          field stage   (train_loss / val_rel_l2 / val_baseline)
    <dir>/metrics_surface.jsonl  surface stage (train_loss / val_cp_rel / val_cd_int / val_cd_head / val_cf_rel)
    <dir>/metrics_physics.jsonl  physics stage (train_loss / val_rel_l2 / cont / mom / wall / far / lambda_phys)

Unlike plot_pipod_metrics.py (per-stage files metrics_stageN.jsonl with a
"loss" key), the roadmap logs interleave training and eval records in one
file per stage and use "train_loss". Panels are emitted only for files
that exist.

Examples
--------
.venv/bin/python plot_roadmap_metrics.py -d examples/ellipsoids/RoadmapONet
.venv/bin/python plot_roadmap_metrics.py -d examples/ellipsoids/RoadmapONet \
    --overlay examples/ellipsoids/RoadmapONet_adaptive -o /tmp/roadmap.png
.venv/bin/python plot_roadmap_metrics.py -d examples/ellipsoids/RoadmapONet -w 30
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ("field", "surface", "physics")


def stage_file(directory, stage):
    name = {"field": "metrics.jsonl",
            "surface": "metrics_surface.jsonl",
            "physics": "metrics_physics.jsonl"}[stage]
    return os.path.join(directory, name)


def load_metrics(directory, stage):
    path = stage_file(directory, stage)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return [json.loads(l) for l in f]


def split_recs(recs):
    """Interleaved records -> (train recs, eval recs, adaptive recs)."""
    train = [r for r in recs if "train_loss" in r]
    ev = [r for r in recs
          if "val_rel_l2" in r or "val_cp_rel" in r]
    adp = [r for r in recs if "adaptive_mean_err" in r]
    return train, ev, adp


def draw(ax, directory, stage, label=None):
    recs = load_metrics(directory, stage)
    if not recs:
        ax.set_title("{} (no metrics)".format(stage))
        return
    train, ev, adp = split_recs(recs)
    sfx = "" if label is None else " ({})".format(label)
    if train:
        it = [r["iter"] for r in train]
        ax.plot(it, [r["train_loss"] for r in train], lw=1,
                label="train loss" + sfx)
        if stage == "physics" and any("cont" in r for r in train):
            for key in ("cont", "mom", "wall", "far"):
                pts = [(r["iter"], r[key]) for r in train if key in r]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1,
                        label=key + sfx)
    if ev:
        ite = [r["iter"] for r in ev]
        if "val_rel_l2" in ev[0]:
            ax.plot(ite, [r["val_rel_l2"] for r in ev], lw=1.2,
                    label="val rel L2" + sfx)
            ax.axhline(ev[0]["val_baseline"], ls="--", c="gray", lw=1,
                       alpha=0.6,
                       label="mean baseline {:.3f}".format(
                           ev[0]["val_baseline"]))
        if "val_cp_rel" in ev[0]:
            ax.plot(ite, [r["val_cp_rel"] for r in ev], lw=1.2,
                    label="val Cp rel" + sfx)
            ax.plot(ite, [r["val_cd_int"] for r in ev], lw=1.2,
                    label="val Cd int" + sfx)
            ax.plot(ite, [r["val_cd_head"] for r in ev], lw=1.2,
                    label="val Cd head" + sfx)
            cf = [sum(r["val_cf_rel"]) / len(r["val_cf_rel"]) for r in ev]
            ax.plot(ite, cf, lw=1, alpha=0.7, label="val Cf rel" + sfx)
    if adp:
        ax.plot([r["iter"] for r in adp],
                [r["adaptive_mean_err"] for r in adp], lw=1, ls="-.",
                label="adaptive mean err" + sfx)
    if stage == "physics" and any("lambda_phys" in r for r in train):
        ax2 = ax.twinx()
        pts = [(r["iter"], r["lambda_phys"]) for r in train
               if "lambda_phys" in r]
        ax2.step([p[0] for p in pts], [p[1] for p in pts], c="purple",
                 ls=":", lw=1)
        ax2.set_ylabel("lambda_phys", color="purple")
    ax.set_yscale("log")
    ax.set_xlabel("iter")
    ax.set_title(stage)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def make_figure(directory, overlay=None, out_path=None):
    stages = [s for s in STAGES if os.path.isfile(stage_file(directory, s))]
    if not stages:
        raise SystemExit("no metrics files under {}".format(directory))
    fig, axes = plt.subplots(1, len(stages),
                             figsize=(5.3 * len(stages), 4.5),
                             squeeze=False)
    for ax, stage in zip(axes[0], stages):
        draw(ax, directory, stage)
        if overlay and stage == "field":
            draw(ax, overlay, stage,
                 label=os.path.basename(os.path.normpath(overlay)))
    fig.suptitle(os.path.basename(os.path.normpath(directory)))
    fig.tight_layout()
    if out_path is None:
        out_path = os.path.join(directory, "metrics.png")
    fig.savefig(out_path, dpi=120)
    return out_path


def watch(directory, interval):
    from matplotlib.animation import FuncAnimation

    matplotlib.use("TkAgg", force=True)
    stages = [s for s in STAGES if os.path.isfile(stage_file(directory, s))]
    fig, axes = plt.subplots(1, len(stages),
                             figsize=(5.3 * len(stages), 4.5),
                             squeeze=False)

    def update(_frame):
        for ax, stage in zip(axes[0], stages):
            ax.clear()
            draw(ax, directory, stage)
        fig.tight_layout()

    FuncAnimation(fig, update, interval=interval * 1000,
                  cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot RoadmapONet staged metrics (metrics*.jsonl)")
    parser.add_argument("--dir", "-d", dest="directory", required=True,
                        help="RoadmapONet output directory")
    parser.add_argument("--overlay", default=None,
                        help="second directory to overlay on the field "
                        "panel (e.g. RoadmapONet_adaptive)")
    parser.add_argument("--output", "-o", default=None,
                        help="output image path (default: <dir>/metrics.png)")
    parser.add_argument("--watch", "-w", dest="watch_interval", type=float,
                        default=None, metavar="SECONDS",
                        help="live mode: reload the JSONL logs and redraw "
                        "every SECONDS")
    args = parser.parse_args()

    if args.watch_interval is not None:
        watch(args.directory, args.watch_interval)
    else:
        print("saved", make_figure(args.directory, overlay=args.overlay,
                                   out_path=args.output))

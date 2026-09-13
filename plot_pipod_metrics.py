#!/usr/bin/env python3
"""Plot PIPOD-DeepONet staged training curves from the JSONL metrics logs
(<experiment>/PipodONet/metrics_stage<1|2|3>.jsonl written by
train_pipod_deeponet.py).

Per stage: train loss, val rel L2, POD projection bound (and stage-3 physics
residuals + the lambda_phys schedule on a twin axis). Optionally overlays a
second experiment (e.g. the synthetic line vs the OpenFOAM line).

Examples
--------
.venv/bin/python plot_pipod_metrics.py -e examples/ellipsoids_of
.venv/bin/python plot_pipod_metrics.py -e examples/ellipsoids_of \
    --overlay examples/ellipsoids -o /tmp/pipod.png
.venv/bin/python plot_pipod_metrics.py -e examples/ellipsoids_of -w 30
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELD_NAMES = ["u", "v", "w", "p"]


def load_metrics(experiment_directory, stage):
    path = os.path.join(experiment_directory, "PipodONet",
                        "metrics_stage{}.jsonl".format(stage))
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return [r for r in (json.loads(l) for l in f) if not r.get("final")]


def draw_stage(ax, recs, stage, label=None, physics=True, show_lambda=True):
    it = [r["iter"] for r in recs]
    loss = ax.plot(it, [r["loss"] for r in recs], lw=1,
                   label="train loss" + ("" if label is None else
                                         " ({})".format(label)))
    color = loss[0].get_color()
    ax.plot(it, [r["val_rel_l2"] for r in recs], lw=1.2,
            label="val rel L2" + ("" if label is None else
                                  " ({})".format(label)))
    if "val_proj" in recs[0]:
        ax.axhline(recs[0]["val_proj"], ls="--", c=color, lw=1, alpha=0.6,
                   label="proj bound {:.3f}".format(recs[0]["val_proj"]))
    if physics and "val_continuity" in recs[0]:
        ax.plot(it, [r["val_continuity"] for r in recs], lw=1,
                label="val continuity")
        ax.plot(it, [r["val_momentum"] for r in recs], lw=1,
                label="val momentum")
        ax.plot(it, [r["val_wall"] for r in recs], lw=1, label="val wall")
    if show_lambda and "lambda_phys" in recs[0]:
        ax2 = ax.twinx()
        ax2.step(it, [r["lambda_phys"] for r in recs], c="purple", ls=":",
                 lw=1)
        ax2.set_ylabel("lambda_phys", color="purple")
    ax.set_yscale("log")
    ax.set_xlabel("iter")
    ax.set_title("stage {}".format(stage))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def make_figure(experiment_directory, overlay=None, stages=(1, 2, 3),
                out_path=None):
    fig, axes = plt.subplots(1, len(stages),
                             figsize=(5.3 * len(stages), 4.5),
                             squeeze=False)
    axes = axes[0]
    for ax, stage in zip(axes, stages):
        recs = load_metrics(experiment_directory, stage)
        if recs is None:
            ax.set_title("stage {} (no metrics)".format(stage))
            continue
        draw_stage(ax, recs, stage)
        if overlay:
            recs2 = load_metrics(overlay, stage)
            if recs2:
                draw_stage(ax, recs2, stage,
                           label=os.path.basename(
                               os.path.normpath(overlay)), physics=False)
    name = os.path.basename(os.path.normpath(experiment_directory))
    fig.suptitle(name)
    fig.tight_layout()
    if out_path is None:
        out_path = os.path.join(experiment_directory, "PipodONet",
                                "metrics.png")
    fig.savefig(out_path, dpi=120)
    return out_path


def watch(experiment_directory, interval, stages):
    from matplotlib.animation import FuncAnimation

    matplotlib.use("TkAgg", force=True)
    fig, axes = plt.subplots(1, len(stages),
                             figsize=(5.3 * len(stages), 4.5),
                             squeeze=False)
    axes = axes[0]

    def update(_frame):
        for ax, stage in zip(axes, stages):
            recs = load_metrics(experiment_directory, stage)
            ax.clear()
            if recs:
                draw_stage(ax, recs, stage, show_lambda=False)
        fig.tight_layout()

    FuncAnimation(fig, update, interval=interval * 1000,
                  cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot PIPOD-DeepONet staged metrics (metrics_stageN.jsonl)")
    parser.add_argument("--experiment", "-e", dest="experiment_directory",
                        required=True)
    parser.add_argument("--overlay", default=None,
                        help="second experiment directory to overlay "
                        "(train loss / val rel L2 / proj bound only)")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3])
    parser.add_argument("--output", "-o", default=None,
                        help="output image path (default: "
                        "<experiment>/PipodONet/metrics.png)")
    parser.add_argument("--watch", "-w", dest="watch_interval", type=float,
                        default=None, metavar="SECONDS",
                        help="live mode: reload the JSONL logs and redraw "
                        "every SECONDS")
    args = parser.parse_args()

    if args.watch_interval is not None:
        watch(args.experiment_directory, args.watch_interval, args.stages)
    else:
        out = make_figure(args.experiment_directory, overlay=args.overlay,
                          stages=args.stages, out_path=args.output)
        print("saved", out)

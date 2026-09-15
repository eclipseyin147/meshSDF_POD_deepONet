#!/usr/bin/env python3
"""Generate the 27-ellipsoid DeepSDF dataset + experiment specs (design doc
docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md section 6).

Writes, under --out (default: the repo root):
- data/ellipsoids/SdfSamples/ellipsoids/ellipsoid/ellipsoid_a{A}_b{B}_c{C}.npz
  for axes in {0.5, 0.7, 0.9}^3: DeepSDF-convention pos/neg (N, 4) float32
  [x, y, z, sdf] samples (pos = near-surface with sigma 0.005 / 0.05 noise,
  neg = uniform volume);
- examples/ellipsoids/split.json  ({"ellipsoids": {"ellipsoid": [...]}});
- examples/ellipsoids/specs.json  (small decoder: dims [128]*4, latent 16).

The SDF uses iq's ellipsoid approximation k0*(k0-1)/k1 (accurate near the
surface, which is what the clamped DeepSDF loss weights). Data files are
generated on demand and not committed to git.
"""

import argparse
import json
import logging
import os

import numpy as np

AXES_VALUES = (0.5, 0.7, 0.9)
SURFACE_NOISE = (0.005, 0.05)
POS_FRACTIONS = (0.5, 0.25)  # per noise level; remainder is uniform neg


def ellipsoid_sdf(points, axes):
    """iq's ellipsoid SDF approximation: k0 = |p/r|, k1 = |p/r^2|,
    sd = k0 (k0 - 1) / k1."""
    p = points / np.asarray(axes, dtype=np.float64)
    k0 = np.linalg.norm(p, axis=1)
    k1 = np.linalg.norm(points / np.asarray(axes, dtype=np.float64) ** 2,
                        axis=1).clip(1e-12)
    return (k0 * (k0 - 1.0) / k1).astype(np.float32)


def sample_shape(axes, n_samples, rng):
    """pos/neg (N, 4) samples for one ellipsoid (semi-axes, centered origin)."""
    n1 = int(n_samples * POS_FRACTIONS[0])
    n2 = int(n_samples * POS_FRACTIONS[1])
    n3 = n_samples - n1 - n2
    # uniform directions on the unit sphere -> ellipsoid surface points
    dirs = rng.normal(size=(n1 + n2, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    surface = dirs * np.asarray(axes)
    pts1 = surface[:n1] + rng.normal(scale=SURFACE_NOISE[0], size=(n1, 3))
    pts2 = surface[n1:] + rng.normal(scale=SURFACE_NOISE[1], size=(n2, 3))
    pts3 = rng.uniform(-1.25, 1.25, size=(n3, 3))
    pos = np.concatenate([pts1, pts2], 0).astype(np.float32)
    neg = pts3.astype(np.float32)
    pos = np.concatenate([pos, ellipsoid_sdf(pos, axes)[:, None]], 1)
    neg = np.concatenate([neg, ellipsoid_sdf(neg, axes)[:, None]], 1)
    return pos.astype(np.float32), neg.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.dirname(
        os.path.abspath(__file__)))
    parser.add_argument("--samples_per_shape", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(args.seed)

    data_dir = os.path.join(
        args.out, "data/ellipsoids/SdfSamples/ellipsoids/ellipsoid")
    exp_dir = os.path.join(args.out, "examples/ellipsoids")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    names = []
    for a in AXES_VALUES:
        for b in AXES_VALUES:
            for c in AXES_VALUES:
                name = "ellipsoid_a{}_b{}_c{}".format(a, b, c)
                pos, neg = sample_shape((a, b, c), args.samples_per_shape, rng)
                np.savez(os.path.join(data_dir, name + ".npz"),
                         pos=pos, neg=neg)
                names.append(name)
    logging.info("wrote %d shapes to %s", len(names), data_dir)

    split = {"ellipsoids": {"ellipsoid": names}}
    with open(os.path.join(exp_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    specs = {
        "Description": "DeepSDF autodecoder on 27 analytic ellipsoids "
                       "(PIPOD-DeepONet validation family)",
        "DataSource": os.path.join(os.path.abspath(args.out),
                                   "data/ellipsoids"),
        "TrainSplit": os.path.join(os.path.abspath(args.out),
                                   "examples/ellipsoids/split.json"),
        "NetworkArch": "deep_sdf_decoder",
        "NetworkSpecs": {
            "dims": [128, 128, 128, 128],
            "dropout": [],
            "dropout_prob": 0.0,
            "norm_layers": [],
            "latent_in": [2],
            "xyz_in_all": False,
            "use_tanh": False,
            "latent_dropout": False,
            "weight_norm": False,
        },
        "CodeLength": 16,
        "NumEpochs": 1001,
        "SnapshotFrequency": 200,
        "AdditionalSnapshots": [100, 500],
        "LearningRateSchedule": [
            {"Type": "Step", "Initial": 0.0005, "Interval": 500,
             "Factor": 0.5},
            {"Type": "Step", "Initial": 0.001, "Interval": 500,
             "Factor": 0.5},
        ],
        "SamplesPerScene": 16384,
        "ScenesPerBatch": 27,
        "DataLoaderThreads": 4,
        "ClampingDistance": 0.1,
        "CodeRegularization": True,
        "CodeRegularizationLambda": 0.0001,
        "CodeBound": 1.0,
    }
    with open(os.path.join(exp_dir, "specs.json"), "w") as f:
        json.dump(specs, f, indent=2)
    logging.info("wrote split.json + specs.json to %s", exp_dir)


if __name__ == "__main__":
    main()

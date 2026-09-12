#!/usr/bin/env python3
"""Generate real-CFD volume snapshots from OpenFOAM (Task 10 data line).

One case per (shape, boundary condition): STL export -> blockMesh +
snappyHexMesh wind-tunnel mesh with boundary layers -> laminar simpleFoam
(OpenFOAM 14 ``foamRun -solver incompressibleFluid``) -> probes sampling on
the shared 64^3 reference grid -> npz snapshot contract
(``deep_sdf.cfd.volume.load_snapshot``).

Default invocation (no arguments) runs the two Task-10 validation cases of
the ellipsoid a=0.9, b=0.7, c=0.5:

    bc=[15, 1, 0, 0]           and
    bc=[12, 0.48, 0.64, 0.6]

under ``data/openfoam/ellipsoids/`` (cases/ for the OpenFOAM case
directories, snapshots/ for the npz output; data is generated on demand and
not committed to git). Custom cases can be supplied with --cases_json, a JSON
list of {"shape": <SdfSamples npz name>, "bc": [U, dx, dy, dz], "name":
<case name>} entries (name optional; axes default to the ones parsed from the
shape name).

Requires the OpenFOAM 14 environment: ``source /opt/openfoam14/etc/bashrc``.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_sdf.cfd import openfoam_runner
from deep_sdf.cfd.flow_synth import parse_ellipsoid_axes
from deep_sdf.cfd.volume import load_snapshot, make_reference_grid

VALIDATION_CASES = [
    {
        "name": "ellipsoid_a0.9_b0.7_c0.5_case000",
        "shape": "ellipsoids/ellipsoid/ellipsoid_a0.9_b0.7_c0.5.npz",
        "bc": [15.0, 1.0, 0.0, 0.0],
    },
    {
        "name": "ellipsoid_a0.9_b0.7_c0.5_case001",
        "shape": "ellipsoids/ellipsoid/ellipsoid_a0.9_b0.7_c0.5.npz",
        "bc": [12.0, 0.48, 0.64, 0.6],
    },
]


def run_one_case(spec, cases_root, snapshots_root, grid_points, n_procs,
                 skip_existing=False):
    """STL -> case -> run -> sample -> validate one snapshot npz."""
    name = spec.get("name")
    shape_name = spec["shape"]
    bc = [float(x) for x in spec["bc"]]
    if name is None:
        base = shape_name.rsplit("/", 1)[-1][:-4]
        name = "{}_bc{}".format(base, "_".join("{:g}".format(x) for x in bc))

    out_path = os.path.join(snapshots_root, name + ".npz")
    if skip_existing and os.path.isfile(out_path):
        logging.info("snapshot %s already exists; skipping", out_path)
        return out_path

    case_dir = os.path.join(cases_root, name)
    stl_path = os.path.join(case_dir, "body.stl")

    axes = tuple(spec["axes"]) if "axes" in spec else parse_ellipsoid_axes(shape_name)
    openfoam_runner.write_stl_from_geometry(stl_path, axes=axes)
    openfoam_runner.make_case(case_dir, stl_path, bc)
    openfoam_runner.run_case(case_dir, n_procs=n_procs)
    openfoam_runner.sample_to_snapshot(case_dir, grid_points, shape_name, bc,
                                       out_path)

    snap = load_snapshot(out_path, expected_points=grid_points.shape[0])
    logging.info(
        "snapshot %s: fields %s, bc %s, shape %s",
        out_path, tuple(snap["fields"].shape), snap["bc"].tolist(),
        snap["shape"],
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases_json",
        default=None,
        help="JSON file with a list of case specs ({shape, bc, name?}); "
        "defaults to the two Task-10 validation cases.",
    )
    parser.add_argument(
        "--root",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data/openfoam/ellipsoids",
        ),
        help="output root (cases/ and snapshots/ are created inside)",
    )
    parser.add_argument("--grid_resolution", type=int, default=64)
    parser.add_argument("--grid_domain", type=float, nargs=2,
                        default=(-1.5, 1.5))
    parser.add_argument("--n_procs", type=int, default=1,
                        help="MPI ranks per case (>1 needs mpiexec on PATH)")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cases_json is not None:
        with open(args.cases_json) as f:
            cases = json.load(f)
    else:
        cases = VALIDATION_CASES

    cases_root = os.path.join(args.root, "cases")
    snapshots_root = os.path.join(args.root, "snapshots")
    os.makedirs(cases_root, exist_ok=True)
    os.makedirs(snapshots_root, exist_ok=True)

    grid_points, grid_shape = make_reference_grid(
        args.grid_resolution, tuple(args.grid_domain)
    )
    logging.info(
        "reference grid %s (%d points)", grid_shape, grid_points.shape[0]
    )

    for spec in cases:
        logging.info("=== case %s (bc=%s) ===",
                     spec.get("name", spec["shape"]), spec["bc"])
        run_one_case(spec, cases_root, snapshots_root, grid_points,
                     args.n_procs, args.skip_existing)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate real-CFD volume snapshots from OpenFOAM (Task 10/11 data line).

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

Batch mode (--lhs N, Task 11): Latin-hypercube sample N latent codes inside
the per-dimension [min, max] range of the 27 training latents expanded by 10%
(``--seed`` reproducible), keep the samples whose decoder mesh passes the
validity check (non-empty, more than 500 faces, vertices within
[-1.2, 1.2]^3, no NaN), and write the latent manifest
``<root>/lhs_latents.npz`` (arrays ``names`` + ``latents``: the 27 training
ellipsoids keyed by their split npz names plus the accepted
``lhs/shape_XXX.npz`` samples), so the training side can resolve every
snapshot's ``shape`` field to a latent code. Every shape - the 27 training
ellipsoids (analytic STL + analytic SDF mask) and the accepted LHS samples
(decoder STL + decoder SDF mask) - gets ``--cases_per_shape`` boundary
conditions (U ~ U(--u_range), direction uniform on the sphere, per-shape
deterministic rng following train_pressure_surrogate's
sample_flow_direction/make_bc). Cases are cloned from a template case (the
Task-10-validated configuration written by openfoam_runner.make_case) with
per-case dictionary edits via foamlib (0/U internalField + freestreamValue =
U * dir, 0/p freestream value, snappyHexMeshDict wake box along the flow
direction, controlDict endTime) and run with ``--jobs`` concurrent workers;
an existing snapshot whose stored bc matches is skipped. Failures
(meshing/solver/sampling) are skipped, collected, and summarized in
``<root>/batch_summary.json``.

Requires the OpenFOAM 14 environment: ``source /opt/openfoam14/etc/bashrc``.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import sys
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deep_sdf.data
from deep_sdf.cfd import openfoam_runner
from deep_sdf.cfd.flow_synth import parse_ellipsoid_axes
from deep_sdf.cfd.labels import export_stl
from deep_sdf.cfd.volume import (
    load_snapshot,
    make_reference_grid,
    make_stretched_grid,
    snapshot_filename,
)
from train_pressure_surrogate import make_bc, sample_flow_direction

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

MESH_MIN_FACES = 500
MESH_ABS_BOUND = 1.2


def run_one_case(spec, cases_root, snapshots_root, grid_points, n_procs,
                 skip_existing=False, keep_case=False):
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
    if not keep_case:
        shutil.rmtree(case_dir)
    return out_path


# --------------------------------------------------------------------------
# Task 11: latent-space LHS sampling + batch generation
# --------------------------------------------------------------------------


def load_decoder(experiment_dir, checkpoint="latest"):
    """CPU eval-mode DeepSDF decoder from an experiment directory."""
    with open(os.path.join(experiment_dir, "specs.json")) as f:
        specs = json.load(f)
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(specs["CodeLength"], **specs["NetworkSpecs"])
    state = torch.load(
        os.path.join(experiment_dir, "ModelParameters", checkpoint + ".pth"),
        map_location="cpu",
        weights_only=True,
    )
    state_dict = state["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {
            k.replace("module.", "", 1): v for k, v in state_dict.items()
        }
    decoder.load_state_dict(state_dict)
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    return decoder, int(specs["CodeLength"])


def load_training_latents(latents_path, split_path, data_source):
    """Training latent codes in split order.

    Returns (names, latents): names are the split's SdfSamples npz names
    (``deep_sdf.data.get_instance_filenames`` order, which is also the
    embedding row order used at training time) and latents a (N, L) float32
    numpy array. The checkpoint's ``latent_codes`` entry may be an
    nn.Embedding state dict (``weight``), an nn.Embedding module, or a raw
    tensor.
    """
    ckpt = torch.load(latents_path, map_location="cpu", weights_only=True)
    codes = ckpt["latent_codes"] if isinstance(ckpt, dict) and "latent_codes" in ckpt else ckpt
    if isinstance(codes, torch.nn.Embedding):
        weight = codes.weight.detach()
    elif isinstance(codes, dict):
        weight = codes["weight"].detach()
    else:
        weight = torch.as_tensor(codes).detach()
    with open(split_path) as f:
        split = json.load(f)
    names = deep_sdf.data.get_instance_filenames(data_source, split)
    if weight.dim() != 2 or weight.shape[0] != len(names):
        raise RuntimeError(
            "latent checkpoint {} holds {} codes but the split lists {} "
            "shapes".format(latents_path, weight.shape[0], len(names))
        )
    return names, weight.float().numpy().copy()


def lhs_bounds(latents, margin=0.1):
    """Per-dimension LHS bounds: [min, max] of the training latents expanded
    by ``margin`` (fraction of the per-dimension span) on both sides."""
    latents = np.asarray(latents, dtype=np.float64)
    lo = latents.min(axis=0)
    hi = latents.max(axis=0)
    span = hi - lo
    return np.stack([lo - margin * span, hi + margin * span], axis=1)


def lhs_sample(n_samples, bounds, seed):
    """Latin-hypercube sample of ``n_samples`` points inside ``bounds``
    ((D, 2) array): per dimension, one sample per stratum with a random
    permutation of the strata. Deterministic for a given ``seed`` (numpy
    RandomState). Returns (n_samples, D) float32."""
    rng = np.random.RandomState(int(seed))
    n = int(n_samples)
    bounds = np.asarray(bounds, dtype=np.float64)
    d = bounds.shape[0]
    unit = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        unit[:, j] = (perm + rng.random_sample(n)) / n
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (lo + unit * (hi - lo)).astype(np.float32)


def validate_latent_mesh(verts, faces, min_faces=MESH_MIN_FACES,
                         bound=MESH_ABS_BOUND):
    """Validity check for a decoder-extracted mesh; returns (ok, reason)."""
    v = verts.detach().cpu().numpy() if torch.is_tensor(verts) else np.asarray(verts)
    f = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    if v.shape[0] == 0 or f.shape[0] == 0:
        return False, "empty mesh"
    if not np.isfinite(v).all():
        return False, "NaN/inf vertices"
    if f.shape[0] <= min_faces:
        return False, "too few faces ({})".format(f.shape[0])
    if float(np.abs(v).max()) > bound:
        return False, "vertex outside [-{}, {}]^3".format(bound, bound)
    return True, ""


def save_manifest(path, names, latents):
    """Write the latent manifest npz: ``names`` (str array) + ``latents``
    ((N, L) float32) - one row per shape, keyed by the snapshot ``shape``
    field (split npz names for training ellipsoids, ``lhs/shape_XXX.npz``
    for LHS samples)."""
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.savez(
        path,
        names=np.asarray([str(n) for n in names]),
        latents=np.asarray(latents, dtype=np.float32).reshape(len(names), -1),
    )
    return path


def load_manifest(path):
    """Read a manifest written by ``save_manifest``; returns
    (names, latents (N, L) float32)."""
    data = np.load(path, allow_pickle=False)
    names = [str(s) for s in data["names"]]
    latents = np.asarray(data["latents"], dtype=np.float32)
    if latents.shape[0] != len(names):
        raise ValueError(
            "manifest {}: {} names but {} latent rows".format(
                path, len(names), latents.shape[0]
            )
        )
    return names, latents


def sample_case_bcs(seed, shape_name, n_cases, u_range=(10.0, 20.0)):
    """Deterministic per-shape boundary conditions: U ~ U(u_range) and a
    uniform direction on the sphere (train_pressure_surrogate's
    sample_flow_direction with cone_deg=180 + make_bc). The rng is seeded
    from (seed, shape_name) so the BC set of a shape is independent of the
    rest of the batch."""
    rng = np.random.RandomState(
        (int(seed) * 1000003 + zlib.crc32(shape_name.encode("utf-8")))
        & 0xFFFFFFFF
    )
    bcs = []
    for _ in range(int(n_cases)):
        direction = sample_flow_direction(rng, 180.0)
        velocity = rng.uniform(float(u_range[0]), float(u_range[1]))
        bcs.append(make_bc(direction, velocity))
    return bcs


def ensure_template_case(template_dir):
    """Create the template case once (Task-10-validated configuration via
    openfoam_runner.make_case with a placeholder bc; every per-case value is
    overwritten by foamlib edits in make_case_from_template)."""
    if os.path.isfile(os.path.join(template_dir, "0", "U")):
        return template_dir
    if os.path.isdir(template_dir):
        shutil.rmtree(template_dir)
    stub_stl = template_dir + "_stub.stl"
    openfoam_runner.write_stl_from_geometry(stub_stl, axes=(0.5, 0.5, 0.5))
    openfoam_runner.make_case(template_dir, stub_stl, [1.0, 1.0, 0.0, 0.0])
    os.remove(stub_stl)
    return template_dir


def make_case_from_template(template_dir, case_dir, stl_path, bc):
    """Clone the template case and apply the per-case values with foamlib:
    the body STL, 0/U internalField + farfield freestreamValue = U * dir,
    0/p farfield freestreamValue = 0, and the snappyHexMeshDict wake
    refinement box along the actual flow direction."""
    from foamlib import FoamFile

    U, direction = openfoam_runner._parse_bc(bc)
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(template_dir, case_dir)
    shutil.copyfile(
        stl_path, os.path.join(case_dir, "constant", "geometry", "body.stl")
    )
    velocity = [float(x) for x in U * direction]

    u_file = FoamFile(os.path.join(case_dir, "0", "U"))
    u_file["internalField"] = velocity
    u_file["boundaryField", "farfield", "freestreamValue"] = velocity

    p_file = FoamFile(os.path.join(case_dir, "0", "p"))
    p_file["boundaryField", "farfield", "freestreamValue"] = 0.0

    lo, hi = openfoam_runner.wake_refinement_box(direction)
    shm = FoamFile(os.path.join(case_dir, "system", "snappyHexMeshDict"))
    shm["geometry", "wakeBox", "min"] = [float(x) for x in lo]
    shm["geometry", "wakeBox", "max"] = [float(x) for x in hi]
    return case_dir


def snapshot_matches(path, bc, expected_points):
    """True when an existing snapshot loads and stores the requested bc."""
    if not os.path.isfile(path):
        return False
    try:
        snap = load_snapshot(path, expected_points=expected_points)
    except Exception:
        return False
    return bool(
        np.allclose(
            snap["bc"].numpy(),
            np.asarray(bc, dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
    )


def run_batch_case(spec, template_dir, grid_points, n_procs, keep_case=False):
    """One batch case: clone template + foamlib edits -> run -> sample ->
    validate. Never raises; returns a result dict with status
    ok/failed (+ stage/error)."""
    result = {
        "case": spec["name"],
        "shape": spec["shape"],
        "bc": [float(x) for x in spec["bc"]],
    }
    t0 = time.time()
    stage = "case setup"
    try:
        make_case_from_template(
            template_dir, spec["case_dir"], spec["stl"], spec["bc"]
        )
        stage = "meshing/solver"
        openfoam_runner.run_case(spec["case_dir"], n_procs=n_procs)
        stage = "sampling"
        openfoam_runner.sample_to_snapshot(
            spec["case_dir"],
            grid_points,
            spec["shape"],
            spec["bc"],
            spec["out_path"],
            sdf_mask=spec["sdf_mask"],
        )
        stage = "validation"
        load_snapshot(spec["out_path"], expected_points=grid_points.shape[0])
    except Exception as e:
        result.update(
            status="failed", stage=stage, error="{}: {}".format(
                type(e).__name__, e
            ),
        )
    else:
        if not keep_case:
            shutil.rmtree(spec["case_dir"], ignore_errors=True)
        result.update(status="ok")
    result["elapsed"] = round(time.time() - t0, 2)
    return result


def _stl_cache_path(stl_dir, shape_name):
    base = shape_name[:-4] if shape_name.endswith(".npz") else shape_name
    return os.path.join(stl_dir, base.replace("/", "_") + ".stl")


def run_batch(args, cases_root, snapshots_root, grid_points):
    """Task-11 batch: LHS latent sampling -> validity filtering -> manifest
    -> per-shape STLs/SDF masks -> concurrent case generation."""
    decoder, latent_size = load_decoder(args.experiment)
    split_path = os.path.join(args.experiment, "split.json")
    latent_names, train_latents = load_training_latents(
        args.latents, split_path, args.data_source
    )
    logging.info(
        "training latents: %d codes of size %d",
        len(latent_names), latent_size,
    )

    bounds = lhs_bounds(train_latents, margin=0.1)
    z_samples = lhs_sample(args.lhs, bounds, args.seed)
    offset = int(args.lhs_offset)
    logging.info(
        "LHS: %d samples in %d dims (bounds expanded by 10%%), seed %d, "
        "offset %d",
        z_samples.shape[0], z_samples.shape[1], args.seed, offset,
    )

    # decoder mesh extraction + validity check (serial, CPU-only)
    lhs_accepted = []  # (name, z float32)
    lhs_rejected = []
    for i, z in enumerate(z_samples):
        name = "lhs/shape_{:03d}.npz".format(offset + i)
        try:
            verts, faces = openfoam_runner.mesh_from_latent(
                decoder, z, resolution=63
            )
            ok, reason = validate_latent_mesh(verts, faces)
            if not ok:
                lhs_rejected.append({"name": name, "reason": reason})
                logging.warning("LHS sample %s rejected: %s", name, reason)
                continue
        except Exception as e:
            lhs_rejected.append(
                {"name": name, "reason": "{}: {}".format(type(e).__name__, e)}
            )
            logging.warning("LHS sample %s failed extraction: %s", name, e)
            continue
        lhs_accepted.append((name, z, verts, faces))
    logging.info(
        "LHS validity: %d accepted / %d rejected",
        len(lhs_accepted), len(lhs_rejected),
    )

    # latent manifest: existing manifest (if any) is merged so --lhs_offset
    # batches accumulate; otherwise start from the 27 training ellipsoids
    if os.path.isfile(args.manifest):
        prev_names, prev_latents = load_manifest(args.manifest)
        manifest_names = list(prev_names)
        manifest_latents = prev_latents
        logging.info(
            "merging into existing manifest: %d shapes", len(manifest_names)
        )
    else:
        manifest_names = list(latent_names)
        manifest_latents = train_latents.astype(np.float32)
    seen = set(manifest_names)
    for n, z, _, _ in lhs_accepted:
        if n in seen:
            logging.warning("manifest already contains %s, skipped", n)
            continue
        manifest_names.append(n)
        manifest_latents = np.vstack(
            [manifest_latents, z.reshape(1, -1).astype(np.float32)]
        )
        seen.add(n)
    save_manifest(args.manifest, manifest_names, manifest_latents)
    logging.info(
        "manifest %s: %d shapes", args.manifest, len(manifest_names)
    )

    # per-shape STL + SDF mask. Shape set = 27 analytic ellipsoids (unless
    # skipped) + newly accepted LHS samples + LHS shapes carried over from
    # a pre-existing manifest (e.g. re-running an earlier batch in a new
    # root without re-sampling).
    stl_dir = os.path.join(args.root, "stls")
    os.makedirs(stl_dir, exist_ok=True)
    shape_infos = []
    if not args.skip_ellipsoids:
        ellipsoid_names = latent_names
        if args.limit_ellipsoids is not None:
            ellipsoid_names = ellipsoid_names[: args.limit_ellipsoids]
        for name in ellipsoid_names:
            stl = _stl_cache_path(stl_dir, name)
            openfoam_runner.write_stl_from_geometry(
                stl, axes=parse_ellipsoid_axes(name)
            )
            shape_infos.append({"name": name, "stl": stl, "sdf_mask": None})
    for name, z, verts, faces in lhs_accepted:
        stl = _stl_cache_path(stl_dir, name)
        export_stl(verts, faces, stl)
        mask = openfoam_runner.decoder_sdf_mask(decoder, z, grid_points)
        shape_infos.append({"name": name, "stl": stl, "sdf_mask": mask})
    # carried-over LHS shapes from a merged manifest (already validated)
    fresh = {n for n, _, _, _ in lhs_accepted}
    for n, z in zip(manifest_names, manifest_latents):
        if n in latent_names or n in fresh:
            continue
        stl = _stl_cache_path(stl_dir, n)
        if not os.path.isfile(stl):
            verts, faces = openfoam_runner.mesh_from_latent(
                decoder, z, resolution=63
            )
            export_stl(verts, faces, stl)
        mask = openfoam_runner.decoder_sdf_mask(decoder, z, grid_points)
        shape_infos.append({"name": n, "stl": stl, "sdf_mask": mask})
    logging.info("batch shapes: %d", len(shape_infos))

    template_dir = os.path.join(args.root, "template_case")
    ensure_template_case(template_dir)

    # case list with skip-on-matching-bc
    specs = []
    skipped = []
    for info in shape_infos:
        if args.fixed_bc is not None:
            bcs = [np.asarray(args.fixed_bc, dtype=np.float32)]
        else:
            bcs = sample_case_bcs(
                args.seed, info["name"], args.cases_per_shape, args.u_range
            )
        for case_idx, bc in enumerate(bcs):
            snap_name = snapshot_filename(info["name"], case_idx)
            out_path = os.path.join(snapshots_root, snap_name)
            if snapshot_matches(out_path, bc, grid_points.shape[0]):
                skipped.append(snap_name)
                continue
            case_name = snap_name[:-4]
            specs.append({
                "name": case_name,
                "shape": info["name"],
                "bc": bc,
                "stl": info["stl"],
                "sdf_mask": info["sdf_mask"],
                "case_dir": os.path.join(cases_root, case_name),
                "out_path": out_path,
            })
    logging.info(
        "batch cases: %d to run, %d skipped (existing snapshot with "
        "matching bc)", len(specs), len(skipped),
    )

    results = [
        {"case": s[:-4], "status": "skipped"} for s in skipped
    ]
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.jobs)
    ) as pool:
        futures = [
            pool.submit(
                run_batch_case, spec, template_dir, grid_points, args.n_procs,
                args.keep_cases
            )
            for spec in specs
        ]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if res["status"] == "ok":
                logging.info(
                    "case %d/%d %s: ok in %.1fs",
                    done, len(specs), res["case"], res["elapsed"],
                )
            else:
                logging.warning(
                    "case %d/%d %s: FAILED at %s in %.1fs: %s",
                    done, len(specs), res["case"], res["stage"],
                    res["elapsed"], res["error"],
                )

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    elapsed = np.array([r["elapsed"] for r in ok], dtype=np.float64)
    summary = {
        "seed": args.seed,
        "lhs": int(args.lhs),
        "cases_per_shape": int(args.cases_per_shape),
        "u_range": [float(x) for x in args.u_range],
        "bounds": bounds.tolist(),
        "shapes": {
            "ellipsoids": int(sum(1 for i in shape_infos
                                  if i["sdf_mask"] is None)),
            "lhs_accepted": len(lhs_accepted),
            "lhs_rejected": lhs_rejected,
        },
        "cases": {
            "total": len(specs) + len(skipped),
            "ok": len(ok),
            "skipped": len(skipped),
            "failed": failed,
        },
        "elapsed_seconds": {
            "total": round(time.time() - t0, 1),
            "per_case_min": float(elapsed.min()) if len(elapsed) else None,
            "per_case_mean": float(elapsed.mean()) if len(elapsed) else None,
            "per_case_max": float(elapsed.max()) if len(elapsed) else None,
        },
    }
    summary_path = os.path.join(args.root, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(
        "batch done: %d ok, %d skipped, %d failed (summary: %s)",
        len(ok), len(skipped), len(failed), summary_path,
    )
    if failed:
        logging.warning("failed cases: %s",
                        [r["case"] for r in failed])
    return summary


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
    parser.add_argument("--grid_stretch", action="store_true",
                        help="sample on the fixed anisotropic stretched grid "
                        "(dense near the body) instead of the uniform grid")
    parser.add_argument("--grid_h_fine", type=float, default=0.022)
    parser.add_argument("--grid_dense_half", type=float, default=1.1)
    parser.add_argument("--grid_domain", type=float, nargs=2,
                        default=(-1.5, 1.5))
    parser.add_argument("--n_procs", type=int, default=1,
                        help="MPI ranks per case (>1 needs mpiexec on PATH)")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--keep_cases", action="store_true",
                        help="keep the OpenFOAM case directories after "
                        "successful sampling (default: delete them to save "
                        "disk space; failures are always kept for debugging)")
    # Task 11 batch mode
    parser.add_argument("--lhs", type=int, default=0,
                        help="batch mode: number of latent-space LHS samples")
    parser.add_argument("--lhs_offset", type=int, default=0,
                        help="batch mode: name new samples shape_{offset+i} "
                        "and merge into an existing manifest (for extending "
                        "a previous LHS batch instead of overwriting it)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4,
                        help="concurrent cases in batch mode")
    parser.add_argument("--cases_per_shape", type=int, default=4)
    parser.add_argument("--u_range", type=float, nargs=2, default=(10.0, 20.0))
    parser.add_argument("--fixed_bc", type=float, nargs=4, default=None,
                        metavar=("U", "DX", "DY", "DZ"),
                        help="one case per shape with this exact bc "
                        "(overrides --cases_per_shape/--u_range sampling)")
    parser.add_argument(
        "--experiment",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "examples/ellipsoids",
        ),
        help="DeepSDF experiment directory (specs.json, ModelParameters/, "
        "split.json)",
    )
    parser.add_argument(
        "--data_source",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data/ellipsoids"
        ),
    )
    parser.add_argument(
        "--latents",
        default=None,
        help="latent checkpoint (default: <experiment>/LatentCodes/latest.pth)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="latent manifest path (default: <root>/lhs_latents.npz)",
    )
    parser.add_argument("--skip_ellipsoids", action="store_true",
                        help="batch mode: only run the LHS shapes")
    parser.add_argument("--limit_ellipsoids", type=int, default=None,
                        help="batch mode: only run the first K split "
                        "ellipsoids")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.latents is None:
        args.latents = os.path.join(
            args.experiment, "LatentCodes", "latest.pth"
        )
    if args.manifest is None:
        args.manifest = os.path.join(args.root, "lhs_latents.npz")

    if args.cases_json is not None:
        with open(args.cases_json) as f:
            cases = json.load(f)
    else:
        cases = VALIDATION_CASES

    cases_root = os.path.join(args.root, "cases")
    snapshots_root = os.path.join(args.root, "snapshots")
    os.makedirs(cases_root, exist_ok=True)
    os.makedirs(snapshots_root, exist_ok=True)

    if args.grid_stretch:
        grid_points, grid_shape, _ = make_stretched_grid(
            hi=args.grid_domain[1], h_fine=args.grid_h_fine,
            dense_half=args.grid_dense_half)
    else:
        grid_points, grid_shape = make_reference_grid(
            args.grid_resolution, tuple(args.grid_domain)
        )
    logging.info(
        "reference grid %s (%d points)", grid_shape, grid_points.shape[0]
    )

    if args.lhs > 0:
        if args.cases_json is not None:
            raise RuntimeError("--lhs batch mode and --cases_json are "
                               "mutually exclusive")
        run_batch(args, cases_root, snapshots_root, grid_points)
        return

    for spec in cases:
        logging.info("=== case %s (bc=%s) ===",
                     spec.get("name", spec["shape"]), spec["bc"])
        run_one_case(spec, cases_root, snapshots_root, grid_points,
                     args.n_procs, args.skip_existing, args.keep_cases)


if __name__ == "__main__":
    main()

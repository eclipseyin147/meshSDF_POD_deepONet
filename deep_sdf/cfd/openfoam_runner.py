#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

"""OpenFOAM 14 laminar external-flow case pipeline (real-CFD data line for the
PIPOD-DeepONet volume ROM, design doc
``docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md`` addendum).

One case per (shape, boundary condition):

- ``write_stl_from_geometry`` exports the body surface STL, either analytic
  (``axes=(a, b, c)``: a trimesh icosphere with ``subdivisions=3`` scaled by
  the semi-axes - analytically exact) or from a decoder latent code via
  ``mesh_from_latent`` (dense-grid skimage marching cubes, CPU only - no GPU
  occupancy);
- ``make_case`` writes a complete OpenFOAM 14 case: blockMesh base grid
  (``n_base``^3 over ``[-domain_half, domain_half]^3``), snappyHexMesh
  surface refinement (``surface_level``), a wake refinement box extending 6
  length units downstream of the *actual* free-stream direction (level 2),
  snap, and ``layers`` boundary layers with ``expansionRatio`` and absolute
  ``firstLayerThickness``; laminar ``constant/momentumTransport`` (the v14
  name for turbulenceProperties), freestream BCs on all outer patches
  (``freestreamVelocity``/``freestreamPressure``, value ``U * dir`` - any
  inflow direction without rotating the geometry) and noSlip walls;
- ``run_case`` executes blockMesh -> checkMesh -> snappyHexMesh -> checkMesh
  -> solver (the ``simpleFoam`` compatibility wrapper of
  ``foamRun -solver incompressibleFluid``), logging each step to
  ``case_dir/log.<app>``;
- ``sample_to_snapshot`` probes the converged field on the shared reference
  grid (``deep_sdf.cfd.volume.make_reference_grid``) with the ``probes``
  function object (``interpolationScheme cellPoint``, point order preserved -
  OpenFOAM writes probes whose location has no containing cell, e.g. inside
  the removed body interior, as ``-1e30`` sentinel columns that keep the
  input ordering), converts kinematic pressure to Cp = p/(0.5 U^2), applies
  the no-slip extension (u = 0 where the SDF is negative; sentinel rows take
  the inside-body convention u = 0, Cp = 1 of
  ``deep_sdf.cfd.flow_synth.potential_flow_field``) and writes the npz
  snapshot contract via ``deep_sdf.cfd.volume.save_snapshot``.

Everything lives in the DeepSDF normalized coordinate system: bodies of
semi-axis <= 0.9 centered at the origin, domain [-9, 9]^3 (>5x the longest
full axis), U of O(10), kinematic viscosity nu = 0.1 (Re ~ 270, laminar).
"""

import logging
import os
import shutil
import subprocess

import numpy as np
import torch
import trimesh

try:
    from skimage.measure import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes_lewiner as marching_cubes

import deep_sdf.utils
from deep_sdf.cfd.labels import export_stl
from deep_sdf.cfd.volume import save_snapshot

DEFAULT_NU = 0.1
DEFAULT_DOMAIN_HALF = 9.0
DEFAULT_N_BASE = 36
DEFAULT_LAYERS = 6
DEFAULT_LAYER_RATIO = 1.25
DEFAULT_FIRST_LAYER = 0.011
DEFAULT_SURFACE_LEVEL = 4
WAKE_LEVEL = 2
WAKE_REACH = 6.0  # box extends from -1 to +6 along the free-stream direction
WAKE_HALF_WIDTH = 1.5

PROBE_UNSET = -1.0e30  # probes sentinel for locations without a cell

# OpenFOAM 14 installation root (sourced per command when the calling shell
# has not sourced it - the bashrc's LD_LIBRARY_PATH conflicts with the venv
# torch CUDA libraries, so the Python driver runs without it)
OF_BASHRC = "/opt/openfoam14/etc/bashrc"

# OpenFOAM 14 is built against the system OpenMPI (FOAM_MPI=openmpi-system);
# a user-local mpiexec earlier on PATH (e.g. a CUDA build) breaks inside the
# sourced OpenFOAM environment, so prefer the system launcher explicitly.
MPIEXEC = "/usr/bin/mpiexec" if os.path.isfile("/usr/bin/mpiexec") else "mpiexec"


def _foam_header(object_name, cls="dictionary", location="system"):
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\n"
        "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox       |\n"
        "|  \\\\    /   O peration     | Version:  14                                  |\n"
        "|   \\\\  /    A nd           | Web:      https://openfoam.org               |\n"
        "|    \\\\/     M anipulation  |                                                 |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{{\n"
        "    format      ascii;\n"
        "    class       {cls};\n"
        "    location    \"{location}\";\n"
        "    object      {object};\n"
        "}}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
    ).format(cls=cls, location=location, object=object_name)


def _foam_footer():
    return "\n\n// ************************************************************************* //\n"


def _write(path, content):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(path, "w") as f:
        f.write(content)


def _vec(v):
    return "({:.8g} {:.8g} {:.8g})".format(float(v[0]), float(v[1]), float(v[2]))


def _parse_bc(bc):
    """Split bc [U, dir_x, dir_y, dir_z] and normalize the direction."""
    bc = np.asarray(bc, dtype=np.float64).reshape(-1)
    if bc.shape[0] != 4:
        raise ValueError("bc must be [U, dir_x, dir_y, dir_z], got {}".format(bc))
    U = float(bc[0])
    direction = bc[1:4]
    norm = np.linalg.norm(direction)
    if norm <= 0.0:
        raise ValueError("bc flow direction has zero norm")
    return U, direction / norm


def wake_refinement_box(direction):
    """Axis-aligned wake refinement box for a unit free-stream direction:
    the AABB of the corridor [-1, +WAKE_REACH] * dir inflated by
    WAKE_HALF_WIDTH in the transverse directions only (no extra refinement
    beyond the upstream/downstream reach). For dir = +x this is exactly
    [-1, -1.5, -1.5] x [6, 1.5, 1.5]."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    a, b = -1.0 * d, WAKE_REACH * d
    transverse = WAKE_HALF_WIDTH * np.sqrt((1.0 - d ** 2).clip(0.0))
    lo = np.minimum(a, b) - transverse
    hi = np.maximum(a, b) + transverse
    return lo, hi


def mesh_from_latent(decoder, latent, resolution=63, max_batch=2 ** 18):
    """CPU extraction of the decoder's zero level set for ``latent``.

    The SDF is evaluated with ``deep_sdf.utils.decode_sdf`` (in ``max_batch``
    chunks) on a dense regular ``resolution``^3 grid over [-1, 1]^3 and the
    iso-surface extracted with the skimage marching-cubes implementation.
    Everything runs on the CPU: ``deep_sdf.differentiable_mesh`` cannot be
    used here because it hardcodes ``.cuda()`` and the GPU is reserved for
    training. Returns (verts (M, 3) float32, faces (F, 3) int64); an empty
    iso-surface gives (0, 3) / (0, 3) tensors.
    """
    import deep_sdf.differentiable_mesh as dm  # for _grid_coords/_domain

    if hasattr(decoder, "eval"):
        decoder.eval()
    latent = torch.as_tensor(latent, dtype=torch.float32).reshape(1, -1).cpu()
    origin, extent = dm._domain(None)
    n = int(resolution)
    coords = dm._grid_coords(n, origin, extent)
    values = torch.zeros(n ** 3)
    with torch.no_grad():
        head = 0
        while head < n ** 3:
            chunk = coords[head : head + max_batch]
            values[head : head + max_batch] = (
                deep_sdf.utils.decode_sdf(decoder, latent, chunk)
                .squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch
    grid = values.reshape(n, n, n).numpy()
    voxel_size = extent / (n - 1)
    try:
        verts_np, faces_np, _, _ = marching_cubes(
            grid, level=0.0, spacing=(voxel_size,) * 3
        )
    except (ValueError, RuntimeError) as e:
        logging.warning("iso-surface extraction failed: {}".format(e))
        return torch.zeros(0, 3), torch.zeros(0, 3).long()
    if verts_np.shape[0] == 0:
        return torch.zeros(0, 3), torch.zeros(0, 3).long()
    verts = torch.from_numpy(verts_np.copy()).float() + torch.as_tensor(
        origin, dtype=torch.float32
    )
    faces = torch.from_numpy(faces_np.copy()).long()
    return verts, faces


def decoder_sdf_mask(decoder, latent, grid_points, max_batch=2 ** 18):
    """Inside-body mask (sdf < 0) of a decoder latent on arbitrary points.

    CPU evaluation in ``max_batch`` chunks; returns a bool numpy array of
    shape (N,). Used as the ``sdf_mask`` argument of ``sample_to_snapshot``
    for non-analytic (latent) shapes."""
    if hasattr(decoder, "eval"):
        decoder.eval()
    latent = torch.as_tensor(latent, dtype=torch.float32).reshape(1, -1).cpu()
    pts = torch.as_tensor(grid_points, dtype=torch.float32).reshape(-1, 3).cpu()
    values = torch.zeros(pts.shape[0])
    with torch.no_grad():
        head = 0
        while head < pts.shape[0]:
            chunk = pts[head : head + max_batch]
            values[head : head + max_batch] = (
                deep_sdf.utils.decode_sdf(decoder, latent, chunk)
                .squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch
    return (values < 0.0).numpy()


def write_stl_from_geometry(path, axes=None, decoder=None, latent=None,
                            resolution=63, subdivisions=4):
    """Write the body surface as an ASCII STL.

    Analytic path (``axes=(a, b, c)``): a trimesh icosphere with
    ``subdivisions`` subdivisions (default 4, 5120 faces) scaled by the
    semi-axes (analytically exact for the
    ellipsoid family). Latent path (``decoder`` + ``latent``): the zero level
    set extracted with ``mesh_from_latent`` (dense ``resolution``^3 SDF grid,
    CPU skimage marching cubes - the GPU is reserved for training). Both
    paths export through ``deep_sdf.cfd.labels.export_stl`` (solid name
    "deepsdf").
    """
    if axes is not None:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
        verts = torch.from_numpy(
            np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(axes)
        ).float()
        faces = torch.from_numpy(np.asarray(mesh.faces))
    elif decoder is not None and latent is not None:
        verts, faces = mesh_from_latent(decoder, latent, resolution=resolution)
        if verts.shape[0] == 0:
            raise RuntimeError(
                "iso-surface extraction produced an empty mesh for the "
                "given latent code"
            )
    else:
        raise ValueError(
            "write_stl_from_geometry needs either axes=(a, b, c) or "
            "decoder + latent"
        )
    export_stl(verts, faces, path)
    return path


def _block_mesh_dict(domain_half, n_base):
    d = float(domain_half)
    n = int(n_base)
    verts = [
        (-d, -d, -d), (d, -d, -d), (d, d, -d), (-d, d, -d),
        (-d, -d, d), (d, -d, d), (d, d, d), (-d, d, d),
    ]
    faces = [(0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
             (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7)]
    out = _foam_header("blockMeshDict")
    out += "vertices\n(\n"
    for v in verts:
        out += "    {}\n".format(_vec(v))
    out += ");\n\nblocks\n(\n"
    out += "    hex (0 1 2 3 4 5 6 7) ({0} {0} {0}) simpleGrading (1 1 1)\n".format(n)
    out += ");\n\nboundary\n(\n"
    out += "    farfield\n    {\n        type patch;\n        faces\n        (\n"
    for f in faces:
        out += "            ({})\n".format(" ".join(str(i) for i in f))
    out += "        );\n    }\n);\n"
    return out + _foam_footer()


def _snappy_hex_mesh_dict(surface_level, layers, layer_ratio, first_layer,
                          direction):
    lo, hi = wake_refinement_box(direction)
    out = _foam_header("snappyHexMeshDict")
    out += "castellatedMesh true;\nsnap            true;\naddLayers       true;\n\n"
    out += "geometry\n{\n"
    out += "    body\n    {\n        type triSurface;\n        file \"body.stl\";\n    }\n"
    out += "    wakeBox\n    {\n        type box;\n"
    out += "        min {};\n        max {};\n".format(_vec(lo), _vec(hi))
    out += "    }\n};\n\n"
    out += """castellatedMeshControls
{
    maxLocalCells       500000;
    maxGlobalCells      8000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features
    (
    );

    refinementSurfaces
    {
        body
        {
            level (%d %d);
        }
    }

    resolveFeatureAngle 30;

    refinementRegions
    {
        wakeBox
        {
            mode    inside;
            level   %d;
        }
    }

    insidePoint (5 5 5);

    allowFreeStandingZoneFaces false;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       4.0;
    nSolveIter      100;
    nRelaxIter      5;

    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes       false;

    layers
    {
        "body.*"
        {
            nSurfaceLayers %d;
        }
    }

    expansionRatio      %.6g;
    firstLayerThickness %.6g;
    minThickness        %.6g;

    nGrow               0;

    featureAngle        130;
    slipFeatureAngle    30;

    nRelaxIter          5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}

meshQualityControls
{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minFlatness         0.5;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.02;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}

mergeTolerance 1e-6;
""" % (
        surface_level,
        surface_level,
        WAKE_LEVEL,
        layers,
        layer_ratio,
        first_layer,
        0.25 * first_layer,
    )
    return out + _foam_footer()


def _control_dict(end_time):
    out = _foam_header("controlDict")
    out += """solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         %d;
deltaT          1;

writeControl    timeStep;
writeInterval   %d;
purgeWrite      0;
writeFormat     binary;
writePrecision  8;
writeCompression off;

timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""" % (end_time, end_time)
    return out + _foam_footer()


def _fv_schemes():
    out = _foam_header("fvSchemes")
    out += """ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(div(phi,U)) Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method meshWave;
}
"""
    return out + _foam_footer()


def _fv_solution(residual_p, residual_u):
    out = _foam_header("fvSolution")
    out += """solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.05;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-08;
        relTol          0.05;
    }

    Phi
    {
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.01;
        smoother        GaussSeidel;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      true;

    residualControl
    {
        p               %.6g;
        U               %.6g;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}

cache
{
    grad(U);
}
""" % (residual_p, residual_u)
    return out + _foam_footer()


def _momentum_transport():
    out = _foam_header("momentumTransport", location="constant")
    out += "simulationType laminar;\n"
    return out + _foam_footer()


def _physical_properties(nu):
    out = _foam_header("physicalProperties", location="constant")
    out += "viscosityModel  constant;\n\nnu      %.6g;\n" % nu
    return out + _foam_footer()


def _u_field(U, direction):
    velocity = U * np.asarray(direction, dtype=np.float64)
    out = _foam_header("U", cls="volVectorField", location="0")
    out += """dimensions      [velocity];

internalField   uniform %s;

boundaryField
{
    farfield
    {
        type            freestreamVelocity;
        freestreamValue uniform %s;
    }
    "body.*"
    {
        type            noSlip;
    }
}
""" % (_vec(velocity), _vec(velocity))
    return out + _foam_footer()


def _p_field():
    out = _foam_header("p", cls="volScalarField", location="0")
    out += """dimensions      [kinematicPressure];

internalField   uniform 0;

boundaryField
{
    farfield
    {
        type            freestreamPressure;
        freestreamValue uniform 0;
    }
    "body.*"
    {
        type            zeroGradient;
    }
}
"""
    return out + _foam_footer()


def _probes_dict(name="probes", points_file="probePoints"):
    """Function-object dictionary for reference-grid sampling. The probe
    locations live in ``system/<points_file>`` (included here) so the
    dictionary itself stays readable; the probes FO preserves the input
    point order and emits -1e30 for locations without a containing cell.
    Output goes to postProcessing/<name>/<time>/."""
    out = _foam_header(name + "Dict")
    out += """%s
{
    type                probes;
    libs                ("libsampling.so");
    writeControl        timeStep;
    writeInterval       1;
    interpolationScheme cellPoint;
    fields              (U p);
    probeLocations
    #include "%s"
    ;
}
""" % (name, points_file)
    return out + _foam_footer()


def make_case(case_dir, stl_path, bc, nu=DEFAULT_NU,
              domain_half=DEFAULT_DOMAIN_HALF, n_base=DEFAULT_N_BASE,
              layers=DEFAULT_LAYERS, layer_ratio=DEFAULT_LAYER_RATIO,
              first_layer=DEFAULT_FIRST_LAYER,
              surface_level=DEFAULT_SURFACE_LEVEL, end_time=2000,
              residual_p=1e-4, residual_u=1e-6):
    """Write a complete OpenFOAM 14 laminar external-flow case.

    The STL is copied to ``constant/geometry/body.stl``; all outer boundary
    faces form a single ``farfield`` patch with freestream BCs set to
    ``U * dir`` (kinematic pressure 0), the body patch (regex ``body.*``) is
    noSlip. ``bc`` = [U, dir_x, dir_y, dir_z]; the direction is normalized.
    Returns ``case_dir``.
    """
    U, direction = _parse_bc(bc)
    geom_dir = os.path.join(case_dir, "constant", "geometry")
    os.makedirs(geom_dir, exist_ok=True)
    shutil.copyfile(stl_path, os.path.join(geom_dir, "body.stl"))
    _write(os.path.join(case_dir, "system", "blockMeshDict"),
           _block_mesh_dict(domain_half, n_base))
    _write(os.path.join(case_dir, "system", "controlDict"),
           _control_dict(end_time))
    _write(os.path.join(case_dir, "system", "fvSchemes"), _fv_schemes())
    _write(os.path.join(case_dir, "system", "fvSolution"),
           _fv_solution(residual_p, residual_u))
    _write(os.path.join(case_dir, "system", "snappyHexMeshDict"),
           _snappy_hex_mesh_dict(surface_level, layers, layer_ratio,
                                 first_layer, direction))
    _write(os.path.join(case_dir, "system", "probesDict"), _probes_dict())
    _write(os.path.join(case_dir, "constant", "momentumTransport"),
           _momentum_transport())
    _write(os.path.join(case_dir, "constant", "physicalProperties"),
           _physical_properties(nu))
    _write(os.path.join(case_dir, "0", "U"), _u_field(U, direction))
    _write(os.path.join(case_dir, "0", "p"), _p_field())
    return case_dir


def _run(cmd, case_dir, log_name):
    """Run an OpenFOAM command, logging to ``case_dir/log_name``. When the
    command is not on PATH (the OpenFOAM bashrc pollutes LD_LIBRARY_PATH for
    torch, so the Python driver normally runs un-sourced), it is executed in
    a subshell that sources ``OF_BASHRC`` first."""
    probe = cmd[0]
    if os.path.basename(probe).startswith("mpiexec"):
        # parallel invocation: the OpenFOAM binary follows "-np <n>"
        probe = cmd[cmd.index("-np") + 2]
    if shutil.which(probe) is None:
        cmd = ["bash", "-c", "source {} >/dev/null 2>&1 && exec \"$@\"".format(OF_BASHRC),
               "of"] + cmd
    log_path = os.path.join(case_dir, log_name)
    logging.info("running: {} (log: {})".format(" ".join(cmd), log_path))
    with open(log_path, "w") as log:
        ret = subprocess.call(
            cmd, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT
        )
    if ret != 0:
        raise RuntimeError(
            "OpenFOAM command '{}' failed; see {}".format(" ".join(cmd), log_path)
        )
    return log_path


def run_case(case_dir, n_procs=1):
    """blockMesh -> checkMesh -> snappyHexMesh -> checkMesh -> potentialFoam
    (initialisation) -> simpleFoam (the OpenFOAM 14 compatibility wrapper of
    ``foamRun -solver incompressibleFluid``; controlDict carries
    ``solver incompressibleFluid`` either way). Meshing is serial; with
    ``n_procs > 1`` the case is decomposed after snappyHexMesh and the two
    solvers run in parallel, reconstructed at the end. Commands fall back to
    a subshell sourcing ``OF_BASHRC`` when they are not on PATH."""
    if shutil.which("blockMesh") is None and not os.path.isfile(OF_BASHRC):
        raise RuntimeError(
            "OpenFOAM not on PATH and no bashrc at {} - install OpenFOAM 14 "
            "or source its bashrc first".format(OF_BASHRC)
        )
    parallel = int(n_procs) > 1
    _run(["blockMesh"], case_dir, "log.blockMesh")
    _run(["checkMesh"], case_dir, "log.checkMesh.blockMesh")
    _run(["snappyHexMesh", "-overwrite"], case_dir, "log.snappyHexMesh")
    _run(["checkMesh"], case_dir, "log.checkMesh")
    if parallel:
        # decompose AFTER snappyHexMesh; plain decomposePar decomposes the
        # 0/ fields and adds the processor-boundary patchField entries
        # (-copyZero would only copy them verbatim and solvers would fail)
        _write(
            os.path.join(case_dir, "system", "decomposeParDict"),
            _foam_header("decomposeParDict")
            + "numberOfSubdomains %d;\n\nmethod          scotch;\n" % int(n_procs)
            + _foam_footer(),
        )
        _run(["decomposePar", "-force"], case_dir, "log.decomposePar")
        _run([MPIEXEC, "-np", str(int(n_procs)), "potentialFoam",
              "-parallel", "-writep"], case_dir, "log.potentialFoam")
        _run([MPIEXEC, "-np", str(int(n_procs)), "simpleFoam", "-parallel"],
             case_dir, "log.simpleFoam")
        _run(["reconstructPar", "-latestTime"], case_dir, "log.reconstructPar")
    else:
        _run(["potentialFoam", "-writep"], case_dir, "log.potentialFoam")
        _run(["simpleFoam"], case_dir, "log.simpleFoam")
    return case_dir


def _write_probe_list(case_dir, rel_path, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    path = os.path.join(case_dir, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("(\n")
        for p in pts:
            f.write("    ({:.8e} {:.8e} {:.8e})\n".format(*p))
        f.write(")\n")
    return path


def _read_probes_file(path, count):
    """Parse one probes output file: returns (count, n_components) float64
    array of the last time row (columns follow the probeLocations order)."""
    with open(path) as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise RuntimeError("probes file {} has no data rows".format(path))
    values = np.fromstring(
        lines[-1].replace("(", " ").replace(")", " "), sep=" "
    )[1:]  # drop the time stamp
    n = values.shape[0]
    if n == count:
        return values.reshape(count, 1)
    if n == 3 * count:
        return values.reshape(count, 3)
    raise RuntimeError(
        "probes file {}: expected {} or {} values, got {}".format(
            path, count, 3 * count, n
        )
    )


def _latest_probe_dir(case_dir, name="probes"):
    root = os.path.join(case_dir, "postProcessing", name)
    if not os.path.isdir(root):
        return None
    best = None
    for entry in os.listdir(root):
        try:
            t = float(entry)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(root, entry)):
            if best is None or t > best[0]:
                best = (t, os.path.join(root, entry))
    return None if best is None else best[1]


def probe_points(case_dir, points, name="probes", run_postprocess=True):
    """Sample (U, p) of the converged case at arbitrary ``points`` (N, 3).

    Writes system/<name>Dict + system/<name>Points, executes the probes
    function object at the latest time (``foamPostProcess -latestTime``) when
    ``run_postprocess`` is set, and parses postProcessing/<name>/<time>/{U,p}.
    Returns (vel (N, 3), pres (N,)) float64 in the exact input point order;
    locations without a containing cell carry the -1e30 sentinel.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = pts.shape[0]
    if run_postprocess:
        _write_probe_list(case_dir, os.path.join("system", name + "Points"), pts)
        _write(
            os.path.join(case_dir, "system", name + "Dict"),
            _probes_dict(name, name + "Points"),
        )
        _run(
            ["foamPostProcess", "-latestTime", "-dict",
             os.path.join("system", name + "Dict")],
            case_dir,
            "log." + name,
        )
    probe_dir = _latest_probe_dir(case_dir, name)
    if probe_dir is None:
        raise RuntimeError(
            "no postProcessing/{} output found in {}".format(name, case_dir)
        )
    vel = _read_probes_file(os.path.join(probe_dir, "U"), count)
    pres = _read_probes_file(os.path.join(probe_dir, "p"), count).reshape(-1)
    return vel, pres


def ellipsoid_sdf_mask(grid_points, shape_name):
    """Inside-body mask (sdf < 0) from the analytic ellipsoid SDF, with the
    semi-axes parsed from the snapshot shape name
    (``deep_sdf.cfd.flow_synth.parse_ellipsoid_axes``)."""
    from generate_ellipsoid_dataset import ellipsoid_sdf
    from deep_sdf.cfd.flow_synth import parse_ellipsoid_axes

    axes = parse_ellipsoid_axes(shape_name)
    sdf = ellipsoid_sdf(np.asarray(grid_points, dtype=np.float64).reshape(-1, 3), axes)
    return sdf < 0.0


def sample_to_snapshot(case_dir, grid_points, shape_name, bc, out_path,
                       sdf_mask=None, run_postprocess=True):
    """Probe the converged (U, p) field on the reference grid and write the
    npz snapshot contract (``deep_sdf.cfd.volume.save_snapshot``).

    The probes function object (system/probesDict + system/probePoints, in
    the exact ``grid_points`` order) is executed at the latest time via
    ``foamPostProcess -latestTime -dict system/probesDict``. Probe columns
    without a containing cell (the removed body interior) come back as the
    -1e30 sentinel; those rows take the inside-body convention u = 0, Cp = 1.
    The SDF mask (sdf < 0: analytic ellipsoid from ``shape_name`` unless
    ``sdf_mask`` is passed) then zeroes u on every inside-body point; Cp is
    kept as sampled everywhere else.
    """
    pts = torch.as_tensor(grid_points, dtype=torch.float32).reshape(-1, 3)
    U, direction = _parse_bc(bc)

    vel, pres = probe_points(
        case_dir, pts.numpy(), name="probes", run_postprocess=run_postprocess
    )

    unset = np.isclose(vel[:, 0], PROBE_UNSET, rtol=1e-3, atol=0.0) | (
        vel[:, 0] < 0.5 * PROBE_UNSET
    )
    cp = pres / (0.5 * U ** 2)
    vel[unset] = 0.0
    cp[unset] = 1.0  # inside-body convention (cf. flow_synth extension)

    if sdf_mask is None:
        sdf_mask = ellipsoid_sdf_mask(pts.numpy(), shape_name)
    sdf_mask = np.asarray(sdf_mask, dtype=bool).reshape(-1)
    if sdf_mask.shape[0] != pts.shape[0]:
        raise ValueError("sdf_mask has {} entries, expected {}".format(
            sdf_mask.shape[0], pts.shape[0]))
    vel[sdf_mask] = 0.0  # no-slip extension into the body

    fields = torch.from_numpy(
        np.concatenate([vel, cp.reshape(-1, 1)], axis=1).astype(np.float32)
    )
    save_snapshot(out_path, fields, np.asarray([U] + list(direction)), shape_name)
    return out_path

#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

"""Per-face pressure labels for CFD supervision.

Two label sources are provided:

- ``run_openfoam``: runs a minimal steady-state simpleFoam case on the
  exported STL and reads back the wall pressure. Only usable when OpenFOAM
  is installed (``openfoam_available()``).
- ``proxy_pressure_coefficient`` (backward-compatible alias
  ``proxy_pressure``): a purely geometric stand-in used to validate the
  end-to-end pipeline when no CFD solver is installed.

Both label styles are normalized to the dimensionless pressure coefficient
Cp before training a surrogate (see ``pressure_to_cp``), so that models
generalize across free-stream velocities. The boundary-condition vector
convention is ``bc = [U, dir_x, dir_y, dir_z]`` (free-stream speed and unit
flow direction), with the field names exported as ``BC_FIELDS``.
"""

import logging
import os
import re
import shutil
import subprocess

import numpy as np
import torch

# boundary-condition vector layout: [free-stream speed, unit flow direction]
BC_FIELDS = ["U", "dir_x", "dir_y", "dir_z"]


def _face_geometry(verts, faces):
    """Outward unit face normals and areas, differentiable w.r.t. ``verts``.

    Face winding coming out of marching cubes is globally consistent but may
    point either way; orientation is fixed via the signed volume of the
    (closed) mesh - positive signed volume means outward normals.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    area = 0.5 * cross.norm(dim=1)
    signed_volume = (v0 * cross).sum() / 6.0
    # piecewise-constant global flip; gradients still flow through `cross`
    orient = 1.0 if signed_volume >= 0 else -1.0
    normals = orient * cross / cross.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return normals, area


def export_stl(verts, faces, path):
    """Write a (verts, faces) mesh as an ASCII STL file."""
    v = verts.detach().cpu().numpy().astype(np.float64)
    f = faces.detach().cpu().numpy()
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(path, "w") as fp:
        fp.write("solid deepsdf\n")
        for tri in f:
            p0, p1, p2 = v[tri[0]], v[tri[1]], v[tri[2]]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            fp.write("  facet normal {:.6e} {:.6e} {:.6e}\n".format(*n))
            fp.write("    outer loop\n")
            for p in (p0, p1, p2):
                fp.write("      vertex {:.6e} {:.6e} {:.6e}\n".format(*p))
            fp.write("    endloop\n")
            fp.write("  endfacet\n")
        fp.write("endsolid deepsdf\n")


def openfoam_available():
    """True when an OpenFOAM installation (simpleFoam) is on PATH."""
    return shutil.which("simpleFoam") is not None


def _write(path, content):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(path, "w") as f:
        f.write(content)


def _run(cmd, case_dir, log_name):
    log_path = os.path.join(case_dir, log_name)
    logging.info("running: {} (log: {})".format(" ".join(cmd), log_path))
    with open(log_path, "w") as log:
        ret = subprocess.call(cmd, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT)
    if ret != 0:
        raise RuntimeError(
            "OpenFOAM command '{}' failed; see {}".format(" ".join(cmd), log_path)
        )


def _read_list_scalar(text, pos):
    """Parse `nonuniform List<scalar>\\n<n>\\n( v0 v1 ... )` starting at pos."""
    m = re.search(r"nonuniform\s+List<scalar>\s*(\d+)\s*\(", text[pos:])
    if m is None:
        return None
    n = int(m.group(1))
    start = pos + m.end()
    end = text.index(")", start)
    values = np.fromstring(text[start:end], sep=" ", count=n)
    if values.shape[0] != n:
        raise RuntimeError("failed to parse {} pressure values".format(n))
    return values


def run_openfoam(stl_path, case_dir, velocity=15.0):
    """Run a minimal steady-state simpleFoam case and return per-face pressure.

    NOTE: this code path requires a working OpenFOAM installation and could
    not be exercised on the development machine (no OpenFOAM available); it
    is provided as a best-effort template. Known caveats:

    - snappyHexMesh re-meshes the surface, so the returned pressures follow
      the OpenFOAM wall-patch face ordering, NOT the input STL face order.
      A production pipeline should map pressures back onto the input mesh
      (e.g. nearest-face-centroid interpolation) before using them as labels.
    - The case setup (laminar, coarse background mesh) is meant for pipeline
      validation, not accurate aerodynamics.

    Returns an (F_patch,) numpy array of static pressure on the wall patch.
    """
    if not openfoam_available():
        raise RuntimeError(
            "OpenFOAM was not found on this system ('simpleFoam' is not in "
            "PATH), so CFD pressure labels cannot be generated. Install "
            "OpenFOAM (e.g. https://develop.openfoam.com/Development/"
            "openfoam/-/wikis) or fall back to deep_sdf.cfd.proxy_pressure, "
            "which provides geometric proxy labels for pipeline validation."
        )

    stl_path = os.path.abspath(stl_path)
    tri_dir = os.path.join(case_dir, "constant", "triSurface")
    if not os.path.isdir(tri_dir):
        os.makedirs(tri_dir)
    case_stl = os.path.join(tri_dir, "mesh.stl")
    shutil.copyfile(stl_path, case_stl)

    # background domain: shapes are normalized to roughly [-1, 1]^3
    _write(
        os.path.join(case_dir, "system", "blockMeshDict"),
        """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
    (-3 -2 -2) (6 -2 -2) (6 2 -2) (-3 2 -2)
    (-3 -2  2) (6 -2  2) (6 2  2) (-3 2  2)
);
blocks (hex (0 1 2 3 4 5 6 7) (45 20 20) simpleGrading (1 1 1));
edges ();
boundary
(
    inlet  { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    walls  { type slip;  faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }
);
""",
    )
    _write(
        os.path.join(case_dir, "system", "controlDict"),
        """FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 500;
deltaT 1;
writeControl timeStep;
writeInterval 500;
purgeWrite 1;
writeFormat ascii;
writePrecision 8;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
""",
    )
    _write(
        os.path.join(case_dir, "system", "fvSchemes"),
        """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss upwind; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
    )
    _write(
        os.path.join(case_dir, "system", "fvSolution"),
        """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p { solver GAMG; tolerance 1e-06; relTol 0.05; smoother GaussSeidel; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-06; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl { p 1e-4; U 1e-4; }
}
relaxationFactors
{
    equations { U 0.7; }
    fields { p 0.3; }
}
""",
    )
    _write(
        os.path.join(case_dir, "system", "snappyHexMeshDict"),
        """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap true;
addLayers false;
geometry { mesh.stl { type triSurfaceMesh; name obstacle; } }
castellatedMeshControls
{
    maxLocalCells 2000000;
    maxGlobalCells 4000000;
    minRefinementCells 10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces { obstacle { level (2 2); } }
    resolveFeatureAngle 30;
    refinementRegions {}
    locationInMesh (4.0 1.5 1.5);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
}
meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality -1;
    minArea -1;
    minTwist 0.05;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}
mergeTolerance 1e-06;
""",
    )
    _write(
        os.path.join(case_dir, "constant", "transportProperties"),
        """FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] 1e-05;
""",
    )
    _write(
        os.path.join(case_dir, "constant", "turbulenceProperties"),
        """FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType laminar;
""",
    )
    _write(
        os.path.join(case_dir, "0", "U"),
        """FoamFile { version 2.0; format ascii; class volVectorField; object U; }
dimensions [0 1 -1 0 0 0 0];
internalField uniform (%(v)g 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (%(v)g 0 0); }
    outlet { type zeroGradient; }
    walls { type slip; }
    obstacle { type noSlip; }
}
"""
        % {"v": velocity},
    )
    _write(
        os.path.join(case_dir, "0", "p"),
        """FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls { type zeroGradient; }
    obstacle { type zeroGradient; }
}
""",
    )

    _run(["blockMesh"], case_dir, "log.blockMesh")
    _run(["snappyHexMesh", "-overwrite"], case_dir, "log.snappyHexMesh")
    _run(["simpleFoam"], case_dir, "log.simpleFoam")

    # latest written time directory
    times = []
    for name in os.listdir(case_dir):
        try:
            times.append((float(name), name))
        except ValueError:
            pass
    if not times:
        raise RuntimeError("simpleFoam produced no time directories in " + case_dir)
    latest = max(times)[1]

    # number of faces on the wall patch
    with open(os.path.join(case_dir, "constant", "polyMesh", "boundary")) as f:
        boundary_text = f.read()
    m = re.search(r"obstacle\s*\{[^}]*?nFaces\s+(\d+)", boundary_text, re.S)
    if m is None:
        raise RuntimeError("could not find wall patch 'obstacle' in polyMesh/boundary")
    num_patch_faces = int(m.group(1))

    with open(os.path.join(case_dir, latest, "p")) as f:
        p_text = f.read()
    patch_pos = p_text.index("obstacle")
    values = _read_list_scalar(p_text, patch_pos)
    if values is None:
        m = re.search(r"value\s+uniform\s+([-\d.eE+]+)", p_text[patch_pos:])
        if m is None:
            raise RuntimeError("could not parse pressure on patch 'obstacle'")
        values = np.full(num_patch_faces, float(m.group(1)))
    if values.shape[0] != num_patch_faces:
        raise RuntimeError(
            "parsed {} pressures but patch has {} faces".format(
                values.shape[0], num_patch_faces
            )
        )
    return values


def pressure_to_cp(p, U):
    """Convert OpenFOAM incompressible pressure to the pressure coefficient.

    OpenFOAM's incompressible solvers (simpleFoam) use kinematic pressure
    p/rho (density folded in, rho == 1), so the dimensionless pressure
    coefficient is

        Cp = p / (0.5 * U^2)

    with U the free-stream speed. Accepts a numpy array or a torch tensor and
    returns the same type.
    """
    return p / (0.5 * float(U) ** 2)


def reference_area(verts, faces, flow_dir):
    """Frontal (windward projected) reference area, differentiable w.r.t.
    ``verts``:

        A_ref = sum_f max(0, -n_f . v_hat) * A_f

    where n_f is the outward unit face normal and v_hat the unit flow
    direction. For a convex body this is the projected cross-section
    perpendicular to the flow; used to normalize force integrals into
    coefficients (Cd = F / A_ref for Cp-integrands).
    """
    normals, areas = _face_geometry(verts, faces)
    v = torch.as_tensor(flow_dir, dtype=verts.dtype, device=verts.device)
    v = v / v.norm()
    return (torch.clamp(-(normals @ v), min=0.0) * areas).sum()


def proxy_pressure_coefficient(
    verts, faces, flow_dir=(1.0, 0.0, 0.0), base_suction=0.3, beta=1.0
):
    """Geometric proxy pressure-coefficient (Cp) labels - NOT a physical
    simulation.

    The values are dimensionless Cp with the stagnation value Cp = 1
    (consistent with the convention Cp = p / (0.5 * rho * U^2); see
    ``pressure_to_cp``). This is a stand-in used to validate the
    surrogate/optimization pipeline end-to-end when no CFD solver (OpenFOAM)
    is available. It assigns, per face, a stagnation-like pressure on
    windward faces and a constant suction on leeward faces:

        Cp = max(0, -n . v_hat)^2 * (1 + beta * blockage)
             - base_suction * max(0, n . v_hat)

    where n is the outward unit face normal and v_hat the unit flow
    direction.

    ``blockage`` is a NON-LOCAL shape factor (the classic slenderness
    parameter of bluff-body aerodynamics): the equivalent frontal diameter
    over the flow-direction body length,

        blockage = D_yz / L_x,
        D_yz = 2 * sqrt(A_frontal / pi),   A_frontal = sum_f max(0, -n.v) * A

    computed from the whole mesh, so windward pressure is amplified for
    bluff bodies and reduced for streamlined ones. A purely local model
    (same normal/area -> same pressure) cannot represent this term; a
    shape-conditioned model (e.g. PressureNeuralOperator) can.

    ``beta`` scales the blockage amplification (default 1.0; set to 0 to
    recover the older purely local proxy).

    Returns an (F,) tensor on the same device as ``verts``.
    """
    with torch.no_grad():
        normals, areas = _face_geometry(verts, faces)
        v = torch.as_tensor(flow_dir, dtype=verts.dtype, device=verts.device)
        v = v / v.norm()
        cos = normals @ v
        frontal_area = (torch.clamp(-cos, min=0.0) * areas).sum()
        flow_axis = int(torch.as_tensor(flow_dir).abs().argmax())
        extent = verts.max(dim=0).values - verts.min(dim=0).values
        length_flow = extent[flow_axis].clamp_min(1e-8)
        d_cross = 2.0 * torch.sqrt(frontal_area / np.pi)
        blockage = d_cross / length_flow
        pressure = torch.clamp(-cos, min=0.0) ** 2 * (1.0 + beta * blockage) - (
            base_suction * torch.clamp(cos, min=0.0)
        )
    return pressure


def proxy_pressure(verts, faces, flow_dir=(1.0, 0.0, 0.0), base_suction=0.3, beta=1.0):
    """Backward-compatible alias of ``proxy_pressure_coefficient``.

    The proxy values have always been dimensionless Cp (stagnation value 1);
    the name predates the explicit Cp convention.
    """
    return proxy_pressure_coefficient(
        verts, faces, flow_dir=flow_dir, base_suction=base_suction, beta=beta
    )

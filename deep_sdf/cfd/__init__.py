#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

"""CFD aerodynamic optimization utilities (DeepMesh paper, section 4.3).

This subpackage implements the pressure-field surrogate pipeline:
per-face pressure-coefficient labels (OpenFOAM when available, geometric
proxy otherwise), per-face pressure surrogates (local MLP baseline and a
shape/BC-conditioned neural operator), and differentiable drag functionals
that backpropagate through the differentiably extracted mesh into the
latent code. A volume-field reduced-order model (POD on a shared reference
grid + (z, BC) coefficient regression, ``deep_sdf.cfd.volume``) complements
the surface operator.
"""

from deep_sdf.cfd.labels import (
    BC_FIELDS,
    export_stl,
    openfoam_available,
    pressure_to_cp,
    proxy_pressure,
    proxy_pressure_coefficient,
    reference_area,
    run_openfoam,
)
from deep_sdf.cfd.surrogate import (
    LocalPressureMLP,
    PressureNeuralOperator,
    PressureSurrogate,
    compute_mean_curvature,
    drag_coefficient,
    drag_from_pressure,
    face_features,
)
from deep_sdf.cfd.volume import (
    PODBasis,
    VolumeCoefficientRegressor,
    load_snapshot,
    make_reference_grid,
    pod_fit,
    save_snapshot,
    snapshot_filename,
    synthetic_volume_field,
)
from deep_sdf.cfd.deeponet import (
    BranchNet,
    PODDeepONet,
    TrunkNet,
)
from deep_sdf.cfd.physics import (
    CollocationSampler,
    FluidMaskEmpty,
    IncompressibleNS,
    PDEInformer,
    farfield_loss,
    fluid_mask,
    noslip_loss,
    physics_weight_schedule,
    wall_slip_loss,
)

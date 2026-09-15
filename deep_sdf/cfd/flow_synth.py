#!/usr/bin/env python3
"""Physics-consistent synthetic volume fields (design doc
docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md section 4).

Base flow: potential flow past a unit sphere, anisotropically mapped to the
ellipsoid with semi-axes (a, b, c): xi = M (x - c) with M = diag(1/a, 1/b,
1/c), u(x) = M^-1 u_hat(xi). The unit-sphere problem is solved with the
mapped free stream V = U M d, so that the far field of u is exactly U d for
any axes (with V = U d it would come out scaled component-wise by the
semi-axes). For constant diagonal M this is exactly
divergence-free (div_x u = div_xi u_hat = 0) and satisfies the slip wall
condition u.n = 0 on the ellipsoid. Pressure from the Bernoulli form
Cp = 1 - |u|^2 / U^2. For the spherical sub-family (a = b = c) the steady
incompressible Navier-Stokes momentum residual is then also exactly zero;
for non-spherical ellipsoids the mapped field is not irrotational and the
momentum residual ground truth is nonzero (bounded) - never assert zero
there.

Optional divergence-free wake: u += curl(A) with the vector potential
A = wake_amp * U * exp(-|x - x_w|^2 / sigma_w^2) * c_hat (constant c_hat
perpendicular to the flow direction), i.e. curl A = wake_amp * U * grad(f)
x c_hat - exactly divergence-free by construction. The wake Gaussian has
small but nonzero amplitude on the rear surface, so with wake_amp > 0 the
slip condition holds only approximately (this is accepted: the wall loss is
evaluated on the model, not on the synthetic truth).

Inside the body (|xi| < 1) u == 0 and Cp == 1 (extension convention of
``synthetic_volume_field``). Velocities are DIMENSIONAL (scale ~ U);
training code nondimensionalizes per case by U. All functions are
differentiable w.r.t. ``grid_points`` and follow its dtype/device.
"""

import torch


def parse_ellipsoid_axes(npz_name):
    """Parse semi-axes (a, b, c) from a sample name such as
    'ellipsoids/ellipsoid/ellipsoid_a0.5_b0.7_c0.9.npz'."""
    base = npz_name.rsplit("/", 1)[-1]
    if base.endswith(".npz"):
        base = base[:-4]
    vals = {}
    for token in base.split("_"):
        if token[:1] in ("a", "b", "c") and len(token) > 1:
            try:
                vals[token[0]] = float(token[1:])
            except ValueError:
                pass
    if sorted(vals) != ["a", "b", "c"]:
        raise ValueError(
            "cannot parse ellipsoid semi-axes from '{}'".format(npz_name)
        )
    return (vals["a"], vals["b"], vals["c"])


def sphere_potential_flow(xi, direction, U):
    """Potential flow past the UNIT sphere centered at the origin.

    xi (N, 3) mapped coordinates; direction (3,) unit free-stream direction;
    U free-stream speed. Returns u_hat (N, 3):

        u_hat = U [ d - (3 (d.n) n - d) / (2 r^3) ],  n = xi / r

    The caller masks points with r < 1 (inside the body).
    """
    r = xi.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = xi / r
    d = direction.reshape(1, 3)
    dn = (n * d).sum(dim=1, keepdim=True)
    return U * (d - (3.0 * dn * n - d) / (2.0 * r ** 3))


def potential_flow_field(axes, center, bc, grid_points, wake_amp=0.15,
                         wake_sigma=0.5, wake_offset=1.5):
    """Physics-consistent synthetic snapshot for one (shape, case).

    axes : (3,) ellipsoid semi-axes. center : (3,) body centroid. bc : (4,)
    [U, dir_x, dir_y, dir_z] (``deep_sdf.cfd.labels.BC_FIELDS``).
    grid_points : (G, 3) query points. Returns (G, 4) [u, v, w, Cp],
    dtype/device following ``grid_points``.
    """
    dtype, device = grid_points.dtype, grid_points.device
    axes_t = torch.as_tensor(axes, dtype=dtype, device=device)
    center_t = torch.as_tensor(center, dtype=dtype, device=device)
    bc = torch.as_tensor(bc, dtype=dtype, device=device).reshape(-1)
    U = float(bc[0])
    d = bc[1:4]
    d = d / d.norm().clamp_min(1e-12)

    rel = grid_points - center_t
    xi = rel / axes_t  # M (x - c), unit-sphere coordinates
    inside = xi.norm(dim=1) < 1.0

    V = U * d / axes_t  # mapped free stream U M d
    Vn = V.norm().clamp_min(1e-12)
    u = sphere_potential_flow(xi, V / Vn, float(Vn)) * axes_t  # M^-1 u_hat

    if wake_amp > 0.0:
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        c_hat = torch.cross(d, z_axis, dim=0)
        if c_hat.norm() < 1e-6:
            y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
            c_hat = torch.cross(d, y_axis, dim=0)
        c_hat = c_hat / c_hat.norm()
        x_w = center_t + wake_offset * axes_t.max() * d
        diff = grid_points - x_w
        f = torch.exp(-(diff * diff).sum(dim=1) / wake_sigma ** 2)
        grad_f = (-2.0 / wake_sigma ** 2) * f.unsqueeze(1) * diff
        u = u + wake_amp * U * torch.cross(
            grad_f, c_hat.reshape(1, 3).expand_as(grad_f), dim=1
        )

    cp = 1.0 - (u.norm(dim=1) / max(U, 1e-8)) ** 2
    u = torch.where(inside.unsqueeze(1), torch.zeros_like(u), u)
    cp = torch.where(inside, torch.ones_like(cp), cp)
    return torch.cat([u, cp.unsqueeze(1)], dim=1)

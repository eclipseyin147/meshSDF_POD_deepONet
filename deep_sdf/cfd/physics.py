#!/usr/bin/env python3
"""Physics-informed residuals, boundary losses and collocation sampling
(design doc docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md
section 5).

The PDE declaration and residual evaluation strictly follow PhysicsNeMo's
PhysicsInformer pattern (third-party/physicsnemo,
physicsnemo/sym/eq/phy_informer.py; the darcy example's ``Diffusion(PDE)``
in examples/cfd/darcy_physics_informed/utils.py): PDEs are declared as SymPy
equation dicts and residuals are evaluated by an autodiff informer mapping
dicts of tensors to dicts of residuals. physicsnemo itself is NOT a
dependency - ``PDEInformer`` is a lean local analogue supporting pure first
and second derivatives, which is all steady incompressible Navier-Stokes
needs.

Nondimensionalization: velocities scaled by U, lengths by L_ref, and the
pressure channel stores Cp = (p - p_inf) / (0.5 rho U^2), so the steady
incompressible momentum residual reads

    (u . grad) u + 0.5 grad(Cp) - (1/Re) laplacian(u) = 0.

All derivative calls use create_graph=True, so the physics losses
backpropagate into the network weights.
"""

import torch
import sympy as sp
from sympy.core.function import AppliedUndef, Derivative


class PDE:
    """Minimal physicsnemo-sym-style PDE base: subclasses fill
    ``self.equations`` (name -> sympy expression)."""

    def __init__(self):
        self.equations = {}


class IncompressibleNS(PDE):
    """Steady incompressible Navier-Stokes, nondimensional (Cp convention).

    equations: continuity, momentum_u, momentum_v, momentum_w.
    """

    def __init__(self, re=1e4):
        super().__init__()
        x, y, z = sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")
        coords = (x, y, z)
        u = sp.Function("u")(x, y, z)
        v = sp.Function("v")(x, y, z)
        w = sp.Function("w")(x, y, z)
        cp = sp.Function("cp")(x, y, z)
        vel = (u, v, w)
        nu = sp.Float(1.0 / float(re))
        self.equations = {"continuity": u.diff(x) + v.diff(y) + w.diff(z)}
        for i, ui in enumerate(vel):
            conv = sum(uj * ui.diff(xj) for uj, xj in zip(vel, coords))
            lap = sum(ui.diff(xj, 2) for xj in coords)
            self.equations["momentum_" + "uvw"[i]] = (
                conv + sp.Rational(1, 2) * cp.diff(coords[i]) - nu * lap
            )


class PDEInformer:
    """Lean autodiff residual evaluator for SymPy-defined PDEs - local
    analogue of PhysicsNeMo's PhysicsInformer (grad_method="autodiff").

    forward(inputs): {"coordinates": (N, 3) with requires_grad=True,
    "u"/"v"/"w"/"cp": (N, 1) tensors derived from it} -> {name: (N, 1)}.
    Only the derivatives actually appearing in the equations are computed.
    """

    FIELDS = ("u", "v", "w", "cp")
    AXES = ("x", "y", "z")

    def __init__(self, equations, grad_method="autodiff"):
        if grad_method != "autodiff":
            raise ValueError("only grad_method='autodiff' is implemented")
        self.equations = dict(equations)
        self._needed = set()
        for expr in self.equations.values():
            for d in expr.atoms(Derivative):
                fname = d.expr.func.__name__
                vc = d.variable_count
                if len(vc) != 1 or vc[0][1] not in (1, 2):
                    raise NotImplementedError(
                        "only pure first/second derivatives are supported")
                self._needed.add((fname, str(vc[0][0]), vc[0][1]))

    def forward(self, inputs):
        coords = inputs["coordinates"]
        if not coords.requires_grad:
            raise ValueError("coordinates must have requires_grad=True")
        derivs = {}
        for fname, axis, order in sorted(self._needed):
            q = inputs[fname]
            j = self.AXES.index(axis)
            g = torch.autograd.grad(
                q.sum(), coords, create_graph=True)[0][:, j:j + 1]
            if order == 2:
                g = torch.autograd.grad(
                    g.sum(), coords, create_graph=True)[0][:, j:j + 1]
            derivs[(fname, axis, order)] = g
        ctx = {"derivs": derivs,
               "fields": {f: inputs[f] for f in self.FIELDS}}
        return {name: self._eval(expr, ctx)
                for name, expr in self.equations.items()}

    def __call__(self, inputs):
        # PhysicsNeMo's PhysicsInformer is a plain class invoked via
        # .forward(); the local callers use informer(inputs) directly.
        return self.forward(inputs)

    def _eval(self, expr, ctx):
        if expr.is_Number:
            return float(expr)
        if isinstance(expr, Derivative):
            fname = expr.expr.func.__name__
            axis, order = expr.variable_count[0]
            return ctx["derivs"][(fname, str(axis), order)]
        if isinstance(expr, AppliedUndef):
            return ctx["fields"][expr.func.__name__]
        if isinstance(expr, sp.Add):
            out = 0.0
            for a in expr.args:
                out = out + self._eval(a, ctx)
            return out
        if isinstance(expr, sp.Mul):
            out = 1.0
            for a in expr.args:
                out = out * self._eval(a, ctx)
            return out
        if isinstance(expr, sp.Pow):
            base, exp = expr.as_base_exp()
            return self._eval(base, ctx) ** float(exp)
        raise NotImplementedError(
            "unsupported sympy node {} in PDE expression".format(type(expr)))


def fluid_mask(sdf, margin):
    """Boolean mask of physics-eligible fluid points (sdf > margin)."""
    return sdf.reshape(-1) > margin


def wall_slip_loss(q, normals):
    """mean (u.n)^2 at wall points; q (N, >=3) velocity in channels 0-2."""
    un = (q[:, :3] * normals).sum(dim=1)
    return (un ** 2).mean()


def noslip_loss(q):
    """mean |u|^2 at wall points (for viscous/no-slip data)."""
    return (q[:, :3] ** 2).sum(dim=1).mean()


def farfield_loss(q, u_inf):
    """mean |u - u_inf|^2 at far-field points (nondim: u_inf = unit dir)."""
    return ((q[:, :3] - u_inf) ** 2).sum(dim=1).mean()


def physics_weight_schedule(progress):
    """lambda_phys ramp (roadmap section 23): 0 -> 0.01 -> 0.05 -> 0.1 at
    20% / 50% / 80% of training."""
    if progress < 0.2:
        return 0.0
    if progress < 0.5:
        return 0.01
    if progress < 0.8:
        return 0.05
    return 0.1


class FluidMaskEmpty(RuntimeError):
    """Raised when a sampling region contains no fluid points."""


class CollocationSampler:
    """Stratified collocation-point sampler over the reference grid (roadmap
    section 20): 30% near-wall (margin*h < sdf <= near_factor*h), 30% wake
    (downstream cone from the body centroid), 20% high-gradient (top
    |grad u| FD quantile of the snapshot), 20% uniform fluid. Physics
    (collocation) points satisfy sdf > margin*h; the wall band
    (0 < sdf <= margin*h) and the far field (sdf > far_sdf) are sampled
    separately for the boundary losses. Sampling is with replacement,
    deterministic under the given torch.Generator.
    """

    def __init__(self, fractions=(0.3, 0.3, 0.2, 0.2), near_factor=10.0,
                 wake_xi_min=0.3, wake_rho_max=1.0, margin=2.0,
                 grad_quantile=0.9, far_sdf=1.0):
        self.fractions = tuple(fractions)
        self.near_factor = near_factor
        self.wake_xi_min = wake_xi_min
        self.wake_rho_max = wake_rho_max
        self.margin = margin
        self.grad_quantile = grad_quantile
        self.far_sdf = far_sdf

    def _pools(self, grid_points, grid_shape, sdf, fields, bc, spacing=None):
        n = grid_shape[0]
        device = grid_points.device
        if spacing is None:
            # legacy uniform grid: scalar spacing
            spacing = float(grid_points[:, 0].max() - grid_points[:, 0].min())
            spacing = torch.full((grid_points.shape[0],),
                                 spacing / max(n - 1, 1), device=device)
        else:
            spacing = spacing.to(device)
        fluid = fluid_mask(sdf, self.margin * spacing)
        if not fluid.any():
            raise FluidMaskEmpty("no fluid points (sdf > margin*h)")
        near_wall = fluid & (sdf <= self.near_factor * spacing)
        d = bc[1:4].to(device)
        d = d / d.norm().clamp_min(1e-12)
        inside = sdf < 0
        centroid = (grid_points[inside].mean(dim=0) if inside.any()
                    else torch.zeros(3, device=device))
        rel = grid_points - centroid
        xi = rel @ d
        rho2 = ((rel - xi.unsqueeze(1) * d) ** 2).sum(dim=1)
        wake = fluid & (xi > self.wake_xi_min) & (rho2 < self.wake_rho_max ** 2)
        # |grad u| by central differences; works for non-uniform tensor grids
        ax = [torch.unique(grid_points[:, i]).sort().values for i in range(3)]
        vel = fields[:, :3].reshape(n, n, n, 3)
        gx = torch.zeros_like(vel)
        gy = torch.zeros_like(vel)
        gz = torch.zeros_like(vel)
        dx = (ax[0][2:] - ax[0][:-2]).reshape(-1, 1, 1, 1)
        dy = (ax[1][2:] - ax[1][:-2]).reshape(1, -1, 1, 1)
        dz = (ax[2][2:] - ax[2][:-2]).reshape(1, 1, -1, 1)
        gx[1:-1] = (vel[2:] - vel[:-2]) / dx
        gy[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / dy
        gz[:, :, 1:-1] = (vel[:, :, 2:] - vel[:, :, :-2]) / dz
        gmag = (gx ** 2 + gy ** 2 + gz ** 2).sum(dim=-1).sqrt().reshape(-1)
        thresh = torch.quantile(gmag[fluid], self.grad_quantile)
        high_grad = fluid & (gmag >= thresh)
        wall = (sdf > 0) & ~fluid
        far = fluid & (sdf > self.far_sdf)
        pools = {
            "near_wall": near_wall, "wake": wake, "high_grad": high_grad,
            "uniform": fluid, "wall": wall, "far": far,
        }
        return {k: v.nonzero(as_tuple=True)[0] for k, v in pools.items()}

    @staticmethod
    def _draw(pool, k, generator, fallback):
        if pool.numel() == 0:
            pool = fallback
        if pool.numel() == 0 or k <= 0:
            return pool[:0]
        # draw on the generator's (CPU) device, then index the (CUDA) pool
        sel = torch.randint(pool.numel(), (k,), generator=generator)
        return pool[sel.to(pool.device)]

    def sample(self, grid_points, grid_shape, sdf, fields, bc, n_points,
               generator, spacing=None):
        """-> {"collocation": (n_points,), "wall": (n_points//8,),
        "far": (n_points//8,)} index tensors. ``spacing``: optional
        per-point local cell size (G,) for stretched grids."""
        pools = self._pools(grid_points, grid_shape, sdf, fields, bc,
                            spacing=spacing)
        counts = [int(f * n_points) for f in self.fractions[:-1]]
        counts.append(n_points - sum(counts))
        keys = ("near_wall", "wake", "high_grad", "uniform")
        collocation = torch.cat([
            self._draw(pools[k], c, generator, pools["uniform"])
            for k, c in zip(keys, counts)
        ])
        return {
            "collocation": collocation,
            "wall": self._draw(pools["wall"], n_points // 8, generator,
                               pools["near_wall"]),
            "far": self._draw(pools["far"], n_points // 8, generator,
                              pools["uniform"]),
        }

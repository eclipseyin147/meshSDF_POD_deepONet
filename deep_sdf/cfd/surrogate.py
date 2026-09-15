#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

"""Pressure surrogate models and differentiable pressure-drag functionals.

Two model families are provided:

- ``PressureSurrogate`` (alias ``LocalPressureMLP``): a purely local per-face
  shared MLP baseline, g_beta([centroid, normal, log area]) -> pressure.
  It cannot represent non-local flow physics (blockage / wake effects that
  depend on the overall shape).

- ``PressureNeuralOperator``: a physical neural operator (DeepONet-style
  branch/trunk with FiLM conditioning) learning the *operator*
  "(shape, boundary conditions) -> Cp field function". The branch maps the
  shape latent code (optionally fused with an encoded boundary-condition
  vector ``bc = [U, dir_x, dir_y, dir_z]``, see
  ``deep_sdf.cfd.labels.BC_FIELDS``) to a global embedding (non-local
  physics), the trunk maps per-face query features [centroid, outward unit
  normal, mean curvature, sdf, |grad sdf|, log area] (curvature and the SDF
  field features optional) to the pressure coefficient at that face, so the
  predicted field is a continuous function queryable on meshes of any
  resolution. Optional face-level message passing (``gnn_layers``) adds
  local neighborhood context before the FiLM trunk. Trunk inputs are
  standardized with feature statistics stored as buffers
  (``set_feature_normalization``).

The drag of a mesh M is approximated by

    L(M) = sum_f  p_f * (n_f . v_hat) * A_f

which is differentiable with respect to the mesh vertices and hence -
through the differentiable iso-surface extraction - with respect to the
latent code of the shape. When the surrogate predicts dimensionless Cp, the
normalized drag coefficient Cd = L / A_ref (``drag_coefficient``) is the
physically meaningful quantity, comparable across free-stream conditions.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import deep_sdf.utils
from deep_sdf.cfd.labels import _face_geometry, reference_area
from deep_sdf.differentiable_mesh import compute_sdf_gradients


def _kappa_autograd(decoder, latent, points, eps):
    """Mean curvature via nested autograd (analytic second derivatives)."""
    chunk = points.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        f = deep_sdf.utils.decode_sdf(decoder, latent, chunk)
        grad = torch.autograd.grad(f.sum(), chunk, create_graph=True)[0]
        laplacian = torch.zeros(chunk.shape[0], device=chunk.device)
        gHg = torch.zeros(chunk.shape[0], device=chunk.device)
        for i in range(3):
            hess_row = torch.autograd.grad(grad[:, i].sum(), chunk, retain_graph=True)[0]
            laplacian = laplacian + hess_row[:, i]
            gHg = gHg + grad[:, i] * (hess_row * grad).sum(-1)
    gnorm = grad.norm(dim=1).clamp_min(eps)
    return (gnorm ** 2 * laplacian - gHg) / (2 * gnorm ** 3)


def _kappa_fd(decoder, latent, points, h, eps):
    """Mean curvature via central differences of the analytic gradient.

    For piecewise-linear fields (ReLU decoders) the analytic Hessian is zero
    almost everywhere; finite differences over a stencil of width h smooth
    over the kinks and recover the curvature of the fitted shape.
    """
    chunk = points.detach()
    grad = compute_sdf_gradients(decoder, latent, chunk)
    laplacian = torch.zeros(chunk.shape[0], device=chunk.device)
    gHg = torch.zeros(chunk.shape[0], device=chunk.device)
    for i in range(3):
        d = torch.zeros(3, device=chunk.device)
        d[i] = h
        g_plus = compute_sdf_gradients(decoder, latent, chunk + d)
        g_minus = compute_sdf_gradients(decoder, latent, chunk - d)
        hess_row = (g_plus - g_minus) / (2 * h)
        laplacian = laplacian + hess_row[:, i]
        gHg = gHg + grad[:, i] * (hess_row * grad).sum(-1)
    gnorm = grad.norm(dim=1).clamp_min(eps)
    return (gnorm ** 2 * laplacian - gHg) / (2 * gnorm ** 3)


def compute_mean_curvature(
    decoder,
    latent_vec,
    points,
    max_batch=2 ** 15,
    method="auto",
    fd_step=0.02,
    eps=1e-8,
):
    """Mean curvature of the zero level set at ``points``:

        kappa = div(grad f / |grad f|) / 2
              = (|grad f|^2 * laplacian f - grad f^T H_f grad f)
                / (2 * |grad f|^3)

    Sign convention: positive for convex regions of the body when f is an SDF
    increasing outward; kappa = (k1 + k2)/2 = 1/r on a sphere of radius r
    (the sum-of-principal-curvatures convention would give 2/r).

    method "autograd" uses nested torch.autograd.grad (exact for smooth
    fields); "fd" uses central differences of the analytic gradient with
    stencil width fd_step; "auto" (default) probes the autograd result and
    falls back to "fd" when it vanishes - piecewise-linear (ReLU) decoders
    have identically zero analytic curvature almost everywhere, since
    tanh(piecewise-linear) level sets coincide with the piecewise-planar
    level sets of the pre-activation.

    Returns a detached (M,) tensor on the same device as ``points``.
    """
    if hasattr(decoder, "eval"):
        decoder.eval()
    latent = latent_vec.detach() if latent_vec is not None else None

    if method == "auto":
        probe = points[: min(256, points.shape[0])]
        k = _kappa_autograd(decoder, latent, probe, eps)
        method = "autograd" if k.abs().max() >= 1e-6 else "fd"

    kappas = []
    num_points = points.shape[0]
    head = 0
    while head < num_points:
        chunk = points[head : min(head + max_batch, num_points)]
        if method == "autograd":
            kappa = _kappa_autograd(decoder, latent, chunk, eps)
        else:
            kappa = _kappa_fd(decoder, latent, chunk, fd_step, eps)
        kappas.append(kappa.detach())
        head += max_batch
    return torch.cat(kappas, 0)


def face_features(
    verts, faces, mean_curvature=None, sdf_values=None, sdf_grad_norm=None
):
    """Per-face query features, differentiable with respect to ``verts``.

    Layout: [centroid(3), outward unit normal(3), log area(1)] -> (F, 7),
    extended by a mean curvature column when ``mean_curvature`` (per-vertex,
    see ``compute_mean_curvature``; vertex values are averaged onto faces)
    is given, and by [sdf, |grad sdf|] columns when ``sdf_values``/
    ``sdf_grad_norm`` (per-face (F,), SDF value and gradient norm evaluated
    with the decoder at the face centroids) are given. The optional columns
    are appended before the log-area column, i.e. the full layout is
    [centroid(3), normal(3), curvature?, sdf?, grad_norm?, log_area(1)].
    Normals are outward-pointing (see
    ``deep_sdf.cfd.labels._face_geometry``), matching the convention used when
    generating pressure labels.
    """
    normals, areas = _face_geometry(verts, faces)
    centroids = verts[faces].mean(dim=1)
    feats = [centroids, normals]
    if mean_curvature is not None:
        kappa_face = mean_curvature[faces].mean(dim=1).unsqueeze(1)
        feats.append(kappa_face)
    if sdf_values is not None:
        feats.append(sdf_values.reshape(-1, 1))
    if sdf_grad_norm is not None:
        feats.append(sdf_grad_norm.reshape(-1, 1))
    feats.append(torch.log(areas.clamp_min(1e-12)).unsqueeze(1))
    return torch.cat(feats, dim=1)


def _build_face_adjacency(faces):
    """Face adjacency of a triangle mesh: two faces are neighbors iff they
    share an edge. Computed in pure torch from the face index tensor: the
    (3F, 2) edge table is canonicalized (sorted), unique edges that occur
    exactly twice contribute one adjacency pair per incident face.

    Returns (face_index (F, K), count (F,)): ``face_index[i]`` lists the
    neighboring face ids of face ``i`` (padded with ``i`` itself up to the
    maximum degree K), ``count[i]`` the true number of neighbors.
    """
    num_faces = faces.shape[0]
    device = faces.device
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )  # (3F, 2)
    edges = torch.sort(edges, dim=1).values
    face_ids = torch.arange(num_faces, device=device).repeat(3)

    # unique() sorts rows lexicographically, so the linearized ids of the
    # shared edges come out sorted as well (no extra sort needed below)
    uniq, edge_counts = torch.unique(edges, dim=0, return_counts=True)
    shared = uniq[edge_counts == 2]  # (E, 2) edges bordering exactly two faces
    if shared.shape[0] == 0:
        return (
            face_ids.unsqueeze(1),
            torch.ones(num_faces, dtype=torch.long, device=device),
        )

    # linearize vertex ids; map every face edge onto its shared-edge slot
    num_verts = int(faces.max().item()) + 1
    shared_lin = shared[:, 0] * num_verts + shared[:, 1]  # (E,) sorted
    edge_lin = edges[:, 0] * num_verts + edges[:, 1]  # (3F,)
    pos = torch.searchsorted(shared_lin, edge_lin).clamp_max(shared.shape[0] - 1)
    is_shared = shared_lin[pos] == edge_lin  # (3F,)
    occ_face = face_ids[is_shared]  # (2E,) face of each shared-edge occurrence
    occ_edge = pos[is_shared]  # (2E,) which shared edge

    # group the two occurrences of each shared edge -> (E, 2) face pairs
    order = torch.argsort(occ_edge, stable=True)
    f_sorted = occ_face[order]
    pairs = torch.stack([f_sorted[0::2], f_sorted[1::2]], dim=1)
    adj = torch.cat([pairs, pairs.flip(1)], dim=0)  # (2E, 2) directed (src, dst)

    count = torch.zeros(num_faces, dtype=torch.long, device=device)
    count.scatter_add_(0, adj[:, 0], torch.ones_like(adj[:, 0]))
    max_degree = int(count.max().item())

    # scatter neighbors into a padded (F, max_degree) table; entries of faces
    # with fewer than max_degree neighbors stay at the self-index default
    src_sorted, order2 = torch.sort(adj[:, 0], stable=True)
    dst_sorted = adj[order2, 1]
    rank = torch.arange(adj.shape[0], device=device) - torch.searchsorted(
        src_sorted, src_sorted
    )
    face_index = (
        torch.arange(num_faces, device=device).unsqueeze(1).repeat(1, max_degree)
    )
    face_index[src_sorted, rank] = dst_sorted
    return face_index, count


class PressureSurrogate(nn.Module):
    """Baseline: per-face shared MLP, purely local features -> pressure."""

    def __init__(self, hidden=128, num_layers=3):
        super(PressureSurrogate, self).__init__()
        layers = [nn.Linear(7, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        # features: (F, 7) -> pressures (F,)
        return self.net(features).squeeze(-1)


# descriptive alias for the baseline
LocalPressureMLP = PressureSurrogate


class PressureNeuralOperator(nn.Module):
    """Physical neural operator: (shape latent, BC) -> Cp field function.

    DeepONet-style architecture with FiLM conditioning:

    - branch: latent code z (concatenated with the encoded boundary
      condition ``bc_encoder(bc)`` when ``bc_dim > 0``; see
      ``deep_sdf.cfd.labels.BC_FIELDS`` for the bc layout) -> global
      embedding h (non-local physics: blockage and wake effects depend on
      the whole body and on the free-stream conditions, not on local
      geometry);
    - trunk: per-face query features [centroid, outward unit normal, mean
      curvature?, sdf?, |grad sdf|?, log area] -> pressure coefficient at
      that face, with every hidden layer FiLM-modulated (scale + shift) by
      h. The trunk is a continuous function of the query point geometry, so
      the same network can be evaluated on meshes of any resolution /
      sampling;
    - optional face-level message passing (``gnn_layers > 0``): before the
      FiLM trunk, the standardized features are refined by N rounds of
      ``h_i <- h_i + ReLU(MLP([h_i, mean_j h_j]))`` over the face adjacency
      graph (faces sharing an edge are neighbors), adding local neighborhood
      context for robustness to noisy meshes and latent extrapolation. The
      residual form keeps the raw features intact at initialization; the
      non-residual variant ``ReLU(MLP(...))`` destroys the (sign-sensitive)
      normal features at init and converges erratically.

    Trunk input features are standardized with per-feature statistics stored
    as buffers (``feat_mean``/``feat_std``, identity by default so unset
    buffers reproduce the old unnormalized behavior); set them with
    ``set_feature_normalization`` from training-set statistics. The raw bc
    vector is likewise standardized with ``bc_mean``/``bc_std`` buffers
    (``set_bc_normalization``), since it mixes O(10) speeds with O(1)
    direction components.

    Gradients flow into the latent code both directly through the branch and
    through the geometric features of the differentiably extracted mesh.
    """

    def __init__(
        self,
        latent_size,
        hidden=128,
        num_layers=3,
        use_curvature=True,
        bc_dim=0,
        use_sdf_features=False,
        gnn_layers=0,
    ):
        super(PressureNeuralOperator, self).__init__()
        self.use_curvature = use_curvature
        self.bc_dim = bc_dim
        self.use_sdf_features = use_sdf_features
        self.gnn_layers = gnn_layers
        self.in_dim = (
            7 + (1 if use_curvature else 0) + (2 if use_sdf_features else 0)
        )
        if bc_dim > 0:
            self.bc_encoder = nn.Sequential(
                nn.Linear(bc_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            # identity by default (old behavior); set from the training-case
            # statistics with set_bc_normalization - the raw bc mixes O(10)
            # speeds with O(1) direction components
            self.register_buffer("bc_mean", torch.zeros(bc_dim))
            self.register_buffer("bc_std", torch.ones(bc_dim))
        self.branch = nn.Sequential(
            nn.Linear(latent_size + (hidden if bc_dim > 0 else 0), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.register_buffer("feat_mean", torch.zeros(self.in_dim))
        self.register_buffer("feat_std", torch.ones(self.in_dim))
        if gnn_layers > 0:
            self.gnn_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.in_dim, hidden),
                        nn.Linear(hidden, self.in_dim),
                    )
                    for _ in range(gnn_layers)
                ]
            )
        self.trunk_lins = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden)]
            + [nn.Linear(hidden, hidden) for _ in range(num_layers - 1)]
        )
        self.films = nn.ModuleList(
            [nn.Linear(hidden, 2 * hidden) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, 1)
        self._warned_missing_bc = False

    def set_feature_normalization(self, mean, std):
        """Set the trunk input standardization statistics (per-feature mean
        and std over the training faces)."""
        mean = torch.as_tensor(mean, dtype=self.feat_mean.dtype)
        std = torch.as_tensor(std, dtype=self.feat_std.dtype).clamp_min(1e-8)
        if mean.numel() != self.in_dim or std.numel() != self.in_dim:
            raise ValueError(
                "expected {}-dimensional feature statistics, got {} and "
                "{}".format(self.in_dim, mean.numel(), std.numel())
            )
        self.feat_mean.copy_(mean.reshape(-1).to(self.feat_mean.device))
        self.feat_std.copy_(std.reshape(-1).to(self.feat_std.device))

    def set_bc_normalization(self, mean, std):
        """Set the standardization statistics applied to the raw bc vector
        before the bc encoder (per-field mean/std over the training cases)."""
        if self.bc_dim == 0:
            raise ValueError("this operator was built with bc_dim=0")
        mean = torch.as_tensor(mean, dtype=self.bc_mean.dtype)
        std = torch.as_tensor(std, dtype=self.bc_std.dtype).clamp_min(1e-8)
        if mean.numel() != self.bc_dim or std.numel() != self.bc_dim:
            raise ValueError(
                "expected {}-dimensional bc statistics, got {} and "
                "{}".format(self.bc_dim, mean.numel(), std.numel())
            )
        self.bc_mean.copy_(mean.reshape(-1).to(self.bc_mean.device))
        self.bc_std.copy_(std.reshape(-1).to(self.bc_std.device))

    def forward(
        self,
        latent,
        verts,
        faces,
        mean_curvature=None,
        bc=None,
        sdf_values=None,
        sdf_grad_norm=None,
    ):
        # latent: (1, L) or (L,); returns per-face Cp (F,)
        if self.use_curvature and mean_curvature is None:
            raise ValueError(
                "this operator was built with use_curvature=True; pass "
                "per-vertex mean curvature from compute_mean_curvature"
            )
        if self.use_sdf_features and (sdf_values is None or sdf_grad_norm is None):
            raise ValueError(
                "this operator was built with use_sdf_features=True; pass "
                "per-face sdf_values and sdf_grad_norm (decoder output and "
                "compute_sdf_gradients norm at the face centroids)"
            )
        if self.bc_dim > 0 and bc is None:
            if not self._warned_missing_bc:
                logging.warning(
                    "this operator was built with bc_dim={} but no bc was "
                    "passed; using a zero boundary-condition vector".format(
                        self.bc_dim
                    )
                )
                self._warned_missing_bc = True
            bc = torch.zeros(self.bc_dim, dtype=verts.dtype, device=verts.device)
        feats = face_features(
            verts, faces, mean_curvature, sdf_values, sdf_grad_norm
        )
        x = (feats - self.feat_mean) / self.feat_std
        if self.gnn_layers > 0:
            face_index, count = _build_face_adjacency(faces)
            mask = (
                torch.arange(face_index.shape[1], device=faces.device)
                < count.unsqueeze(1)
            ).unsqueeze(-1)  # (F, K, 1); padded slots hold the self-index
            for mlp in self.gnn_mlps:
                agg = (x[face_index] * mask).sum(dim=1) / count.unsqueeze(
                    1
                ).clamp_min(1)
                x = x + F.relu(mlp(torch.cat([x, agg], dim=1)))
        h_in = [latent.reshape(1, -1)]
        if self.bc_dim > 0:
            bc_in = (bc.reshape(1, -1) - self.bc_mean) / self.bc_std
            h_in.append(self.bc_encoder(bc_in))
        h = self.branch(torch.cat(h_in, dim=1))  # (1, hidden)
        for lin, film in zip(self.trunk_lins, self.films):
            x = F.relu(lin(x))
            scale, shift = film(h).chunk(2, dim=-1)  # (1, hidden) each
            x = x * (1.0 + scale) + shift
        return self.head(x).squeeze(-1)


def drag_from_pressure(verts, faces, pressure, flow_dir=(1.0, 0.0, 0.0)):
    """Differentiable pressure drag: sum_f p_f * (n_f . v_hat) * A_f.

    Here n_f is the unit normal pointing INTO the body (the negative of the
    outward normal used for label generation), so that p * (n . v_hat) * A is
    the streamwise component of the pressure force exerted on the body: the
    result is the physical pressure drag, positive for bluff bodies and
    reduced by streamlining. Normals and areas are recomputed from ``verts``
    (not taken from the extraction), so the scalar carries gradients
    w.r.t. ``verts`` and, through the differentiable mesh, w.r.t. the latent
    code.
    """
    outward, areas = _face_geometry(verts, faces)
    v = torch.as_tensor(flow_dir, dtype=verts.dtype, device=verts.device)
    v = v / v.norm()
    into_body = -outward
    return (pressure * (into_body @ v) * areas).sum()


def drag_coefficient(verts, faces, cp, flow_dir=(1.0, 0.0, 0.0)):
    """Differentiable drag coefficient from a per-face Cp field.

    Returns ``(Cd, force_integral)`` where

        force_integral = sum_f cp_f * (n_f . v_hat) * A_f
        Cd = force_integral / A_ref

    with n_f the unit normal pointing INTO the body (the convention of
    ``drag_from_pressure``: positive for bluff bodies) and A_ref the frontal
    reference area (``deep_sdf.cfd.labels.reference_area``). Both the
    integral and the normalization carry gradients w.r.t. ``verts``.
    """
    force_integral = drag_from_pressure(verts, faces, cp, flow_dir)
    a_ref = reference_area(verts, faces, flow_dir).clamp_min(1e-12)
    return force_integral / a_ref, force_integral

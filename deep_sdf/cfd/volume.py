#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

"""Volume-field reduced-order model: POD on a reference grid + (z, BC) -> POD
coefficient regression.

Complementing the surface Cp operator (``deep_sdf.cfd.surrogate``), this
module predicts the *volume* flow field (u, v, w, p) on a fixed reference
grid around the body:

- ``make_reference_grid`` builds the shared regular grid on which all
  snapshots live, so the snapshot matrix is aligned across cases;
- ``pod_fit`` computes a POD basis of the snapshot matrix on the GPU with a
  randomized range finder (Omega ~ N(0,1), Y = S Omega, QR, B = Q^T S,
  SVD(B)); the rank is either fixed or truncated by captured energy;
- ``PODBasis`` wraps the mean field, the (D, r) basis and the singular
  values, and provides project/reconstruct plus the full-field relative L2
  error and the irreducible true-projection lower bound;
- ``VolumeCoefficientRegressor`` is an MLP mapping (latent z, encoded BC) to
  the r POD coefficients, with identity-default standardization buffers
  (z/bc/coef) in the style of ``PressureNeuralOperator``;
- ``synthetic_volume_field`` generates stand-in snapshots from the decoder
  SDF for pipeline validation - NOT a physical simulation.

Snapshot format (shared by real CFD exports and the synthetic generator): one
npz per (shape, case) with ``fields`` (G, 4) float32 [u, v, w, p] flattened
in C order over the reference grid, ``bc`` (4,) following
``deep_sdf.cfd.labels.BC_FIELDS`` ([U, dir_x, dir_y, dir_z]) and ``shape``
(the SDF-sample npz name, used to fetch the latent code). Points inside the
body (decoder SDF < 0) carry u == 0 (no-slip extension, as in GINO's
extension operator); the SDF mask is evaluated on the fly, not stored.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn

import deep_sdf.utils
from deep_sdf.cfd.labels import BC_FIELDS


def make_reference_grid(resolution=64, domain=(-1.5, 1.5)):
    """Shared regular reference grid for volume-field snapshots.

    Parameters
    ----------
    resolution : int
        Number of grid points N per axis; the grid has G = N**3 points.
    domain : (float, float)
        Coordinate range of every axis. DeepSDF shapes are normalized to
        roughly [-1, 1]^3, so the default [-1.5, 1.5]^3 leaves room for the
        inflow and the wake on all sides.

    Returns
    -------
    grid_points : torch.Tensor
        (G, 3) float32 point coordinates, C-order flattened (x slowest, z
        fastest), so ``grid_points.reshape(shape + (3,))`` is the (N, N, N)
        grid.
    shape : (int, int, int)
        Grid shape (N, N, N).
    """
    n = int(resolution)
    lin = torch.linspace(domain[0], domain[1], n, dtype=torch.float32)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid_points = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1
    )
    return grid_points, (n, n, n)


def snapshot_filename(npz, case_idx):
    """Snapshot file name for a (shape, case) pair."""
    return npz[:-4].replace("/", "_") + "_case{:03d}.npz".format(case_idx)


def save_snapshot(path, fields, bc, shape_name):
    """Write one (shape, case) snapshot npz: fields (G, 4) float32, bc (4,),
    shape (SDF-sample npz name)."""
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.savez(
        path,
        fields=fields.detach().cpu().numpy().astype(np.float32),
        bc=np.asarray(bc, dtype=np.float32),
        shape=shape_name,
    )


def load_snapshot(path, expected_points=None):
    """Load and validate a snapshot npz; returns dict(fields (G, 4) float32,
    bc (4,) float32, shape str)."""
    data = np.load(path, allow_pickle=False)
    for key in ("fields", "bc", "shape"):
        if key not in data.files:
            raise ValueError("snapshot {} is missing '{}'".format(path, key))
    fields = torch.from_numpy(np.asarray(data["fields"], dtype=np.float32))
    bc = torch.from_numpy(np.asarray(data["bc"], dtype=np.float32).reshape(-1))
    shape_name = str(data["shape"])
    if fields.dim() != 2 or fields.shape[1] != 4:
        raise ValueError(
            "snapshot {}: fields must be (G, 4), got {}".format(
                path, tuple(fields.shape)
            )
        )
    if bc.numel() != len(BC_FIELDS):
        raise ValueError(
            "snapshot {}: bc must have {} entries, got {}".format(
                path, len(BC_FIELDS), bc.numel()
            )
        )
    if expected_points is not None and fields.shape[0] != expected_points:
        raise ValueError(
            "snapshot {}: grid mismatch - {} points, expected {} "
            "(--grid_resolution {})".format(
                path,
                fields.shape[0],
                expected_points,
                int(round(expected_points ** (1.0 / 3.0))),
            )
        )
    return {"fields": fields, "bc": bc, "shape": shape_name}


def _column_chunks(S, mean, device, chunk_elems):
    """Yield (col_start, centered (n, c) chunk of S on ``device``).

    ``S`` is the (n, D) snapshot matrix on any device (typically CPU); chunks
    are uploaded column-wise so peak GPU memory stays bounded by about
    ``chunk_elems`` floats regardless of n * D.
    """
    n, D = S.shape
    cols = max(1, chunk_elems // max(n, 1))
    for c0 in range(0, D, cols):
        c1 = min(c0 + cols, D)
        yield c0, S[:, c0:c1].to(device) - mean[c0:c1]


def _orthonormalize_columns(V, chunk_elems=2 ** 24):
    """Orthonormalize the columns of the tall matrix V (D, r) via the
    Cholesky factor of V^T V - a memory-bounded alternative to a full QR of
    the (D, r) matrix (row chunks only, O(r^2) extra memory)."""
    D, r = V.shape
    rows = max(1, chunk_elems // max(r, 1))
    gram = torch.zeros(r, r, dtype=torch.float64, device=V.device)
    for i in range(0, D, rows):
        c = V[i : i + rows].double()
        gram += c.t() @ c
    gram = 0.5 * (gram + gram.t())
    eye = torch.eye(r, dtype=gram.dtype, device=gram.device)
    jitter = (gram.diag().mean() * 1e-12).clamp_min(1e-30)
    try:
        L = torch.linalg.cholesky(gram + jitter * eye)
    except RuntimeError:
        L = torch.linalg.cholesky(gram + 1e-6 * eye)
    L = L.float()
    out = torch.empty_like(V)
    for i in range(0, D, rows):
        # out_chunk = V_chunk @ L^{-T}: solve L X = V_chunk^T
        out[i : i + rows] = torch.linalg.solve_triangular(
            L, V[i : i + rows].t(), upper=False
        ).t()
    return out


def pod_fit(
    S,
    energy=0.999,
    rank=None,
    randomized=True,
    oversampling=10,
    niter=2,
    device="cuda",
    chunk_elems=2 ** 24,
):
    """Fit a POD basis of the snapshot matrix, on the GPU.

    The snapshot matrix ``S`` (N_cases, D) float32 is column-centered and its
    leading subspace extracted. With ``randomized=True`` a randomized range
    finder is used: Omega ~ N(0, 1) (generated column-chunk-wise, never
    materialized), Y = S_c Omega, ``niter`` stabilized power iterations, Q =
    QR(Y), B = Q^T S_c, SVD(B). Both the power iterations and B B^T = Q^T
    (S_c S_c^T) Q are evaluated through the exact centered Gram matrix (n,
    n) - algebraically identical to the S_c S_c^T products but bounded in
    memory. With ``randomized=False`` the basis comes from the exact Gram
    eigendecomposition (deterministic).

    All big products are chunked over the feature dimension, so peak extra
    GPU memory is ~chunk_elems floats plus the (D, r) basis itself;
    ``S`` itself may live on the CPU (N_cases up to ~2000, D up to ~1M).

    Parameters
    ----------
    S : torch.Tensor
        (N_cases, D) float32 snapshot matrix (rows = flattened fields).
    energy : float
        Captured-energy threshold used for rank truncation when ``rank`` is
        None: r is the smallest mode count whose cumulative exact energy
        reaches ``energy``.
    rank : int or None
        Fixed mode count; overrides ``energy``.
    randomized : bool
        Use the randomized range finder (True) or the exact Gram trick
        (False).
    oversampling : int
        Number of extra sampled modes beyond r in the range finder.
    niter : int
        Number of power iterations (each costs one Gram product).
    device : str or torch.device
        Compute device ("cuda" with a CPU fallback warning when unavailable).
    chunk_elems : int
        Chunk size (in elements) for the feature-dimension passes over S.

    Returns
    -------
    PODBasis
        mean (D,), basis (D, r), singular_values (r,), captured energy. The
        exact spectrum (top-20 cumulative energy) and the chosen rank are
        logged.
    """
    S = torch.as_tensor(S, dtype=torch.float32)
    if S.dim() != 2:
        raise ValueError("expected a 2D snapshot matrix, got {}D".format(S.dim()))
    n, D = S.shape
    if n < 1:
        raise ValueError("need at least one snapshot to fit a POD basis")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("pod_fit: CUDA requested but not available; using CPU")
        device = torch.device("cpu")

    # column mean (chunked pass over the feature dim, float64 accumulation)
    cols = max(1, chunk_elems // n)
    total = torch.zeros(D, dtype=torch.float64, device=device)
    for c0 in range(0, D, cols):
        c1 = min(c0 + cols, D)
        total[c0:c1] += S[:, c0:c1].to(device).sum(dim=0).double()
    mean = (total / n).float()

    # exact centered Gram matrix (n, n) in float64, chunked over features
    gram = torch.zeros(n, n, dtype=torch.float64, device=device)
    for _, chunk in _column_chunks(S, mean, device, chunk_elems):
        gram += (chunk @ chunk.t()).double()
    gram = 0.5 * (gram + gram.t())
    eigvals, eigvecs = torch.linalg.eigh(gram)
    eigvals = eigvals.clamp_min(0.0).flip(0)
    eigvecs = eigvecs.flip(1)
    sigma_exact = eigvals.sqrt()
    total_energy = eigvals.sum().clamp_min(1e-30)
    cum_energy = torch.cumsum(eigvals, 0) / total_energy

    logging.info("POD snapshot matrix: {} cases x {} dims".format(n, D))
    k = min(20, n)
    logging.info("POD exact spectrum (first {} modes):".format(k))
    for i in range(k):
        logging.info(
            "  mode {:3d}: sigma = {:.6e}, cumulative energy = {:.6%}".format(
                i + 1, sigma_exact[i].item(), cum_energy[i].item()
            )
        )

    if rank is None:
        # smallest r whose cumulative energy reaches the target
        r = min(int((cum_energy < energy).sum().item()) + 1, n)
    else:
        r = max(1, min(int(rank), n))

    if randomized:
        l = min(r + int(oversampling), n)
        # random start: Y = S_c @ Omega with Omega ~ N(0, 1) generated
        # chunk-wise (the (D, l) Gaussian matrix is never materialized);
        # consumes the global CUDA RNG, seeded by torch.manual_seed
        Y = torch.zeros(n, l, device=device)
        for c0 in range(0, D, cols):
            c1 = min(c0 + cols, D)
            omega = torch.randn(c1 - c0, l, device=device)
            Y += (S[:, c0:c1].to(device) - mean[c0:c1]) @ omega
        # stabilized power iterations through the exact Gram:
        # Y <- S_c S_c^T Y = G Y (identical to the classic (S S^T)^k S Omega)
        for _ in range(int(niter)):
            q, _ = torch.linalg.qr(Y)
            Y = (gram @ q.double()).float()
        q, _ = torch.linalg.qr(Y)
        # SVD of B = Q^T S_c via its (l, l) Gram B B^T = Q^T G Q - the
        # (D, l) matrix B is never materialized
        qd = q.double()
        bbt = qd.t() @ gram @ qd
        bbt = 0.5 * (bbt + bbt.t())
        evals_b, U_b = torch.linalg.eigh(bbt)
        evals_b = evals_b.clamp_min(0.0).flip(0)
        U_b = U_b.flip(1)
        sigma = evals_b[:r].sqrt()
        floor = (sigma[:1] * 1e-12).clamp_min(1e-30)
        M = (qd @ (U_b[:, :r] / sigma.clamp_min(floor))).float()  # (n, r)
        captured = float((evals_b[:r].sum() / total_energy).item())
    else:
        sigma = sigma_exact[:r]
        floor = (sigma[:1] * 1e-12).clamp_min(1e-30)
        M = (eigvecs[:, :r] / sigma.clamp_min(floor)).float()
        captured = float((eigvals[:r].sum() / total_energy).item())

    # basis vectors V = S_c^T M (chunked over the feature dim), then
    # re-orthonormalized to clean up float32 drift
    V = torch.empty(D, r, device=device)
    for c0, chunk in _column_chunks(S, mean, device, chunk_elems):
        V[c0 : c0 + chunk.shape[1]] = chunk.t() @ M
    V = _orthonormalize_columns(V, chunk_elems)

    logging.info(
        "POD basis: rank r = {} ({} path, captured energy {:.6%})".format(
            r, "randomized" if randomized else "exact", captured
        )
    )
    return PODBasis(mean, V, sigma.float(), captured)


class PODBasis(nn.Module):
    """POD basis of a snapshot ensemble: mean field + orthonormal modes.

    Fields are approximated as ``y ~= mean + basis @ a`` with orthonormal
    ``basis`` (D, r). ``project``/``reconstruct`` map between full fields and
    coefficients; ``relative_error`` gives the per-case full-field relative
    L2 error of a reconstruction from given coefficients, and
    ``projection_error`` the irreducible lower bound obtained by projecting
    the truth onto the basis. Both errors are evaluated with the orthonormal
    identity

        ||y_hat - y||^2 = ||y - mean||^2 - 2 a . a_true + ||a||^2,

    so no (N, D) reconstruction is materialized and inputs may live on the
    CPU (row chunks are uploaded on the fly). Buffers move with ``.to()``;
    persistence uses plain ``torch.save`` of a tensor dict (``save``/``load``).
    """

    def __init__(self, mean, basis, singular_values, energy):
        super(PODBasis, self).__init__()
        self.register_buffer(
            "mean", torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        )
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
        self.register_buffer(
            "singular_values",
            torch.as_tensor(singular_values, dtype=torch.float32).reshape(-1),
        )
        self.register_buffer("energy", torch.tensor(float(energy)))
        if self.basis.dim() != 2 or self.basis.shape[0] != self.mean.numel():
            raise ValueError(
                "basis must be (D, r) with D = mean.numel() = {}; got {}".format(
                    self.mean.numel(), tuple(self.basis.shape)
                )
            )
        if self.basis.shape[1] != self.singular_values.numel():
            raise ValueError(
                "singular_values must have one entry per basis mode ({}), "
                "got {}".format(self.basis.shape[1], self.singular_values.numel())
            )

    @property
    def dim(self):
        """Full-field dimension D."""
        return self.basis.shape[0]

    @property
    def rank(self):
        """Number of retained modes r."""
        return self.basis.shape[1]

    def _check_Y(self, Y):
        Y = torch.as_tensor(Y, dtype=torch.float32)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
        if Y.dim() != 2 or Y.shape[1] != self.dim:
            raise ValueError(
                "expected fields of shape (N, {}) or ({},), got {}".format(
                    self.dim, self.dim, tuple(Y.shape)
                )
            )
        return Y

    def project(self, Y, chunk_elems=2 ** 24):
        """True projection coefficients A = (Y - mean) @ basis -> (N, r).

        Parameters
        ----------
        Y : torch.Tensor
            (N, D) or (D,) full fields (any device; uploaded in row chunks).
        """
        Y = self._check_Y(Y)
        device = self.mean.device
        rows = max(1, chunk_elems // self.dim)
        out = []
        for i in range(0, Y.shape[0], rows):
            chunk = Y[i : i + rows].to(device) - self.mean
            out.append(chunk @ self.basis)
        return torch.cat(out, 0)

    def reconstruct(self, A, chunk_elems=2 ** 24):
        """Field reconstruction mean + A @ basis^T -> (N, D) (or (D,) for a
        1D input).

        Parameters
        ----------
        A : torch.Tensor
            (N, r) or (r,) coefficients.
        """
        A = torch.as_tensor(A, dtype=torch.float32)
        single = A.dim() == 1
        if single:
            A = A.unsqueeze(0)
        if A.dim() != 2 or A.shape[1] != self.rank:
            raise ValueError(
                "expected coefficients of shape (N, {}) or ({},), got {}".format(
                    self.rank, self.rank, tuple(A.shape)
                )
            )
        device = self.mean.device
        rows = max(1, chunk_elems // self.dim)
        out = []
        for i in range(0, A.shape[0], rows):
            chunk = A[i : i + rows].to(device)
            out.append(chunk @ self.basis.t() + self.mean)
        result = torch.cat(out, 0)
        return result[0] if single else result

    def _case_stats(self, Y, chunk_elems):
        """Per-case true coefficients, centered energy and field energy."""
        Y = self._check_Y(Y)
        device = self.mean.device
        rows = max(1, chunk_elems // self.dim)
        a_true, yc2, y2 = [], [], []
        for i in range(0, Y.shape[0], rows):
            raw = Y[i : i + rows].to(device)
            yc = raw - self.mean
            a_true.append(yc @ self.basis)
            yc2.append((yc * yc).sum(dim=1))
            y2.append((raw * raw).sum(dim=1))
        return torch.cat(a_true), torch.cat(yc2), torch.cat(y2)

    def relative_error(self, Y, coef=None, chunk_elems=2 ** 24):
        """Per-case full-field relative L2 error ||y_hat - y|| / ||y|| of the
        reconstruction from ``coef`` (default: the true projection
        coefficients, i.e. the POD-optimal reconstruction). Returns (N,).

        Parameters
        ----------
        Y : torch.Tensor
            (N, D) or (D,) ground-truth fields.
        coef : torch.Tensor or None
            (N, r) or (r,) coefficients to reconstruct from (e.g. regressor
            predictions; pass zeros for the predict-the-mean-field baseline).
        """
        a_true, yc2, y2 = self._case_stats(Y, chunk_elems)
        if coef is None:
            coef = a_true
        else:
            coef = torch.as_tensor(coef, dtype=torch.float32)
            if coef.dim() == 1:
                coef = coef.unsqueeze(0)
            if coef.shape != a_true.shape:
                raise ValueError(
                    "expected coefficients of shape {}, got {}".format(
                        tuple(a_true.shape), tuple(coef.shape)
                    )
                )
            coef = coef.to(a_true.device)
        err2 = yc2 - 2.0 * (coef * a_true).sum(dim=1) + (coef * coef).sum(dim=1)
        return err2.clamp_min(0.0).sqrt() / y2.clamp_min(1e-30).sqrt()

    def projection_error(self, Y, chunk_elems=2 ** 24):
        """Per-case irreducible lower bound: the relative L2 error of the
        truth projected onto the basis, ||y - mean||^2 - ||a_true||^2 over
        ||y||^2. Must be reported alongside any learned reconstruction error.
        Returns (N,).
        """
        a_true, yc2, y2 = self._case_stats(Y, chunk_elems)
        err2 = yc2 - (a_true * a_true).sum(dim=1)
        return err2.clamp_min(0.0).sqrt() / y2.clamp_min(1e-30).sqrt()

    def save(self, path):
        """Serialize to a plain tensor dict via torch.save."""
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        torch.save(
            {
                "format": "PODBasis",
                "mean": self.mean.detach().cpu(),
                "basis": self.basis.detach().cpu(),
                "singular_values": self.singular_values.detach().cpu(),
                "energy": float(self.energy),
            },
            path,
        )

    @staticmethod
    def load(path, device="cpu"):
        """Load a basis saved with ``save`` and move it to ``device``."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        return PODBasis(
            state["mean"], state["basis"], state["singular_values"], state["energy"]
        ).to(device)


class VolumeCoefficientRegressor(nn.Module):
    """MLP regression of POD coefficients from (shape latent, BC).

    Follows the surface-operator convention: ``h_bc = bc_encoder(bc)`` (a
    small MLP), ``cat([z_norm, h_bc])`` -> ``num_layers`` hidden layers ->
    linear head of width ``rank`` (the POD coefficients; reconstruct with
    ``PODBasis.reconstruct``).

    Three identity-default standardization buffer groups, set from
    training-set statistics (``set_z_normalization``/``set_bc_normalization``
    /``set_coef_normalization``, std clamped at 1e-8, in the style of
    ``PressureNeuralOperator``): ``z_mean/z_std`` for the latent code,
    ``bc_mean/bc_std`` for the raw bc vector (which mixes O(10) speeds with
    O(1) direction components) and ``coef_mean/coef_std`` for the POD
    coefficients. The coefficient standardization is essential: POD
    coefficient magnitudes decay with the singular values over orders of
    magnitude, so unstandardized targets under-fit the high-order modes. The
    training loss lives in the standardized space (``forward_normalized``);
    ``forward`` returns physical coefficients.
    """

    def __init__(self, latent_size, rank, hidden=256, num_layers=4, bc_dim=4):
        super(VolumeCoefficientRegressor, self).__init__()
        self.latent_size = latent_size
        self.rank = rank
        self.bc_dim = bc_dim
        in_dim = latent_size
        if bc_dim > 0:
            self.bc_encoder = nn.Sequential(
                nn.Linear(bc_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            in_dim += hidden
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, rank)]
        self.net = nn.Sequential(*layers)
        # identity by default; set from the training-set statistics
        self.register_buffer("z_mean", torch.zeros(latent_size))
        self.register_buffer("z_std", torch.ones(latent_size))
        self.register_buffer("bc_mean", torch.zeros(bc_dim))
        self.register_buffer("bc_std", torch.ones(bc_dim))
        self.register_buffer("coef_mean", torch.zeros(rank))
        self.register_buffer("coef_std", torch.ones(rank))

    def _set(self, mean, std, dim, name):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-8)
        if mean.numel() != dim or std.numel() != dim:
            raise ValueError(
                "expected {}-dimensional {} statistics, got {} and {}".format(
                    dim, name, mean.numel(), std.numel()
                )
            )
        getattr(self, name + "_mean").copy_(
            mean.reshape(-1).to(getattr(self, name + "_mean").device)
        )
        getattr(self, name + "_std").copy_(
            std.reshape(-1).to(getattr(self, name + "_std").device)
        )

    def set_z_normalization(self, mean, std):
        """Set the latent-code standardization statistics (per-dimension
        mean/std over the training shapes)."""
        self._set(mean, std, self.latent_size, "z")

    def set_bc_normalization(self, mean, std):
        """Set the standardization statistics applied to the raw bc vector
        before the bc encoder (per-field mean/std over the training cases)."""
        if self.bc_dim == 0:
            raise ValueError("this regressor was built with bc_dim=0")
        self._set(mean, std, self.bc_dim, "bc")

    def set_coef_normalization(self, mean, std):
        """Set the POD-coefficient standardization statistics (per-mode
        mean/std of the true projection coefficients over the training
        cases)."""
        self._set(mean, std, self.rank, "coef")

    def forward_normalized(self, latent, bc=None):
        """Standardized-space coefficients (the regression target space).

        Parameters
        ----------
        latent : torch.Tensor
            (L,) or (B, L) shape latent code.
        bc : torch.Tensor
            (bc_dim,) or (B, bc_dim) boundary-condition vector
            ([U, dir_x, dir_y, dir_z], see ``deep_sdf.cfd.labels.BC_FIELDS``).
        """
        z = (latent.reshape(-1, self.latent_size) - self.z_mean) / self.z_std
        h_in = [z]
        if self.bc_dim > 0:
            if bc is None:
                raise ValueError(
                    "this regressor was built with bc_dim={} but no bc was "
                    "passed".format(self.bc_dim)
                )
            b = (bc.reshape(-1, self.bc_dim) - self.bc_mean) / self.bc_std
            h_in.append(self.bc_encoder(b))
        return self.net(torch.cat(h_in, dim=1))

    def forward(self, latent, bc=None):
        """Physical POD coefficients a (feed ``PODBasis.reconstruct``)."""
        return self.forward_normalized(latent, bc) * self.coef_std + self.coef_mean


def synthetic_volume_field(
    decoder, latent, grid_points, shape, bc, delta=0.05, max_batch=2 ** 18
):
    """Synthetic stand-in volume field - NOT a physical simulation.

    Pipeline-validation generator in the spirit of the surface geometric
    proxy (``proxy_pressure_coefficient``): a free stream modulated by a
    near-wall decay and a downstream wake deficit, both derived from the
    decoder SDF, with a Bernoulli-form pressure. It produces smooth,
    shape/BC-dependent snapshots in the shared npz format so the POD + ROM
    machinery can be exercised end-to-end without a CFD solver; the values
    have no physical meaning (in particular the field is not
    divergence-free).

    Construction (design doc
    ``docs/superpowers/specs/2026-09-10-volume-pod-rom-design.md`` section
    5): free stream ``u_inf = U * v_hat``; near-wall decay ``u = u_inf *
    sigmoid(d / delta)`` with ``d`` the decoder SDF and ``delta`` the decay
    length; wake deficit for downstream points ``xi = (x - c) . v_hat > 0``
    (``c`` the body centroid) ``u *= 1 - A * exp(-rho^2 / w^2) / (1 + xi)``
    with ``rho`` the transverse distance, amplitude ``A`` proportional to the
    blockage ``D_yz / L_x`` (the non-local slenderness factor of the surface
    proxy: equivalent frontal diameter over flow-direction body length,
    estimated here from the grid SDF instead of mesh faces) and width ``w =
    D_yz / 2``; pressure from the Bernoulli form ``Cp = 1 - ||u||^2 / U^2``.
    Inside the body (sdf < 0) ``u == 0`` (no-slip extension) and the pressure
    takes the formula's defined value Cp = 1.

    Parameters
    ----------
    decoder : nn.Module
        Trained DeepSDF decoder (eval mode).
    latent : torch.Tensor
        (1, L) shape latent code.
    grid_points : torch.Tensor
        (G, 3) reference-grid points (``make_reference_grid``).
    shape : (int, int, int)
        Reference-grid shape (N, N, N).
    bc : torch.Tensor
        (4,) boundary-condition vector [U, dir_x, dir_y, dir_z]
        (``deep_sdf.cfd.labels.BC_FIELDS``).
    delta : float
        Near-wall sigmoid decay length.
    max_batch : int
        Chunk size for the decoder SDF evaluation.

    Returns
    -------
    torch.Tensor
        (G, 4) float32 fields [u, v, w, Cp] on ``grid_points.device``.
    """
    device = grid_points.device
    bc = torch.as_tensor(bc, dtype=torch.float32, device=device).reshape(-1)
    U = float(bc[0])
    v_hat = bc[1:4]
    v_hat = v_hat / v_hat.norm().clamp_min(1e-12)

    if hasattr(decoder, "eval"):
        decoder.eval()
    sds = []
    with torch.no_grad():
        head = 0
        while head < grid_points.shape[0]:
            chunk = grid_points[head : head + max_batch]
            sds.append(
                deep_sdf.utils.decode_sdf(decoder, latent, chunk).squeeze(1).float()
            )
            head += max_batch
    sdf = torch.cat(sds, 0)
    inside = sdf < 0

    # body centroid and flow-axis coordinate
    if inside.any():
        centroid = grid_points[inside].mean(dim=0)
    else:
        centroid = torch.zeros(3, device=device)
    rel = grid_points - centroid
    xi = rel @ v_hat

    # orthonormal transverse basis (e1, e2) perpendicular to v_hat
    axis = torch.tensor([1.0, 0.0, 0.0], device=device)
    if float((v_hat * axis).sum().abs()) > 0.9:
        axis = torch.tensor([0.0, 1.0, 0.0], device=device)
    e1 = torch.cross(v_hat, axis, dim=0)
    e1 = e1 / e1.norm().clamp_min(1e-12)
    e2 = torch.cross(v_hat, e1, dim=0)
    p1 = rel @ e1
    p2 = rel @ e2
    rho2 = p1 ** 2 + p2 ** 2

    # blockage D_yz / L_x from the grid SDF (same slenderness factor as the
    # surface proxy, estimated on the reference grid instead of the mesh)
    spacing = float(
        (grid_points[:, 0].max() - grid_points[:, 0].min())
    ) / max(shape[0] - 1, 1)
    if inside.any():
        xi_in = xi[inside]
        length_flow = (xi_in.max() - xi_in.min()).clamp_min(1e-8)
        q1 = torch.div(p1[inside], spacing, rounding_mode="floor").long()
        q2 = torch.div(p2[inside], spacing, rounding_mode="floor").long()
        q1 = q1 - q1.min()
        q2 = q2 - q2.min()
        keys = q1 * (int(q2.max().item()) + 1) + q2
        frontal_area = float(torch.unique(keys).numel()) * spacing ** 2
        d_yz = 2.0 * float(np.sqrt(frontal_area / np.pi))
        blockage = d_yz / float(length_flow)
    else:
        d_yz = 0.0
        blockage = 0.0

    deficit = float(np.clip(blockage, 0.0, 0.95))
    wake_width = max(0.5 * d_yz, spacing)

    sigmoid_decay = torch.sigmoid(sdf / delta)
    wake = torch.ones_like(xi)
    downstream = xi > 0
    wake[downstream] = 1.0 - (
        deficit
        * torch.exp(-rho2[downstream] / wake_width ** 2)
        / (1.0 + xi[downstream])
    )
    factor = sigmoid_decay * wake

    u = U * v_hat.unsqueeze(0) * factor.unsqueeze(1)
    u[inside] = 0.0  # no-slip extension into the body
    cp = 1.0 - (u.norm(dim=1) / max(U, 1e-8)) ** 2
    return torch.cat([u, cp.unsqueeze(1)], dim=1)

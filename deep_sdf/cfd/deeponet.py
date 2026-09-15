#!/usr/bin/env python3
"""Geometry-conditioned POD-DeepONet (design doc
docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md section 3).

    q_v(x; z, mu) = b_v(x) + sum_k a^v_k(z, mu) * phi^v_k(x, d, grad d, h)

The structure strictly follows third-party/mPOD-DeepONet: the branch maps
the conditioning to per-variable coefficients (B, C, r) like
``BranchNetLinear``; the trunk uses one MLP per output variable
(``MultiTrunkDeepONet``'s ModuleList) and the combination is
``einsum('bcr, cnr -> bcn')``. Deliberate deviations (spec section 9):
SiLU activations (PhysicsNeMo FullyConnected default; ReLU would zero the
second derivatives the momentum residual needs), trunk inputs (x, d, grad d,
h), learned basis + bias trunk instead of a fixed POD basis, and the POD
basis only supervises coefficients (it never enters the forward pass).

Standardization buffers (identity defaults, setters clamp std at 1e-8)
follow ``VolumeCoefficientRegressor`` conventions: z/bc on the branch and
per-variable per-mode coefficient statistics; the training loss lives in
the standardized space (``forward_normalized``).
"""

import os

import torch
import torch.nn as nn


def _mlp(in_dim, out_dim, hidden_sizes, act=nn.SiLU):
    layers = []
    dims = [in_dim] + list(hidden_sizes)
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), act()]
    layers += [nn.Linear(dims[-1], out_dim)]
    return nn.Sequential(*layers)


class BranchNet(nn.Module):
    """(z, bc) -> per-variable POD coefficients (B, n_outputs, rank)."""

    def __init__(self, latent_size, bc_dim=4, rank=64, n_outputs=4,
                 hidden=256, num_layers=4, bc_hidden=64):
        super(BranchNet, self).__init__()
        self.latent_size = latent_size
        self.bc_dim = bc_dim
        self.rank = rank
        self.n_outputs = n_outputs
        self.hidden = hidden
        self.num_layers = num_layers
        self.bc_hidden = bc_hidden
        in_dim = latent_size
        if bc_dim > 0:
            self.bc_encoder = _mlp(bc_dim, bc_hidden, [bc_hidden])
            in_dim += bc_hidden
        self.net = _mlp(in_dim, n_outputs * rank, [hidden] * num_layers)
        self.register_buffer("z_mean", torch.zeros(latent_size))
        self.register_buffer("z_std", torch.ones(latent_size))
        self.register_buffer("bc_mean", torch.zeros(bc_dim))
        self.register_buffer("bc_std", torch.ones(bc_dim))
        self.register_buffer("coef_mean", torch.zeros(n_outputs, rank))
        self.register_buffer("coef_std", torch.ones(n_outputs, rank))

    def _set(self, mean, std, shape, name):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-8)
        if tuple(mean.shape) != tuple(shape) or tuple(std.shape) != tuple(shape):
            raise ValueError(
                "expected {} statistics of shape {}, got {} and {}".format(
                    name, tuple(shape), tuple(mean.shape), tuple(std.shape))
            )
        getattr(self, name + "_mean").copy_(
            mean.to(getattr(self, name + "_mean").device))
        getattr(self, name + "_std").copy_(
            std.to(getattr(self, name + "_std").device))

    def set_z_normalization(self, mean, std):
        self._set(mean, std, (self.latent_size,), "z")

    def set_bc_normalization(self, mean, std):
        if self.bc_dim == 0:
            raise ValueError("this branch was built with bc_dim=0")
        self._set(mean, std, (self.bc_dim,), "bc")

    def set_coef_normalization(self, mean, std):
        self._set(mean, std, (self.n_outputs, self.rank), "coef")

    def forward_normalized(self, latent, bc=None):
        """Standardized coefficients (B, n_outputs, rank) - the L_POD space."""
        z = (latent.reshape(-1, self.latent_size) - self.z_mean) / self.z_std
        h_in = [z]
        if self.bc_dim > 0:
            if bc is None:
                raise ValueError("bc_dim={} but no bc passed".format(self.bc_dim))
            b = (bc.reshape(-1, self.bc_dim) - self.bc_mean) / self.bc_std
            h_in.append(self.bc_encoder(b))
        out = self.net(torch.cat(h_in, dim=1))
        return out.reshape(-1, self.n_outputs, self.rank)

    def forward(self, latent, bc=None):
        """Physical POD coefficients (B, n_outputs, rank)."""
        return self.forward_normalized(latent, bc) * self.coef_std + self.coef_mean


class TrunkNet(nn.Module):
    """Per-variable learned basis over (x, d, grad d, h) plus a shared bias
    trunk producing the per-variable mean field b_v."""

    def __init__(self, in_dim=8, rank=64, n_outputs=4,
                 hidden_sizes=(256, 512)):
        super(TrunkNet, self).__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.n_outputs = n_outputs
        self.hidden_sizes = tuple(hidden_sizes)
        self.trunks = nn.ModuleList(
            [_mlp(in_dim, rank, self.hidden_sizes) for _ in range(n_outputs)]
        )
        self.bias_trunk = _mlp(in_dim, n_outputs, self.hidden_sizes)

    def basis(self, features):
        """features (N, in_dim) -> (n_outputs, N, rank)."""
        return torch.stack([t(features) for t in self.trunks], dim=0)

    def bias(self, features):
        """features (N, in_dim) -> (N, n_outputs)."""
        return self.bias_trunk(features)


class PODDeepONet(nn.Module):
    """q(x) = bias(x) + einsum('bcr, cnr -> bcn'); one (shape, case) per call."""

    def __init__(self, branch, trunk):
        super(PODDeepONet, self).__init__()
        self.branch = branch
        self.trunk = trunk

    def forward(self, latent, bc, features):
        """latent (L,) or (1, L); bc (bc_dim,) or (1, bc_dim);
        features (N, trunk.in_dim). Returns q (N, n_outputs)."""
        a = self.branch(latent.reshape(1, -1), bc.reshape(1, -1))
        if a.shape[0] != 1:
            raise ValueError(
                "PODDeepONet evaluates one (shape, case) per call")
        phi = self.trunk.basis(features)          # (C, N, r)
        q = torch.einsum("bcr,cnr->bcn", a, phi)  # (1, C, N)
        return q[0].t() + self.trunk.bias(features)

    def coefficients(self, latent, bc):
        """Physical coefficients (n_outputs, rank)."""
        return self.branch(latent.reshape(1, -1), bc.reshape(1, -1))[0]

    def save(self, path):
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        torch.save({
            "format": "PODDeepONet",
            "branch_kwargs": {
                "latent_size": self.branch.latent_size,
                "bc_dim": self.branch.bc_dim,
                "rank": self.branch.rank,
                "n_outputs": self.branch.n_outputs,
                "hidden": self.branch.hidden,
                "num_layers": self.branch.num_layers,
                "bc_hidden": self.branch.bc_hidden,
            },
            "trunk_kwargs": {
                "in_dim": self.trunk.in_dim,
                "rank": self.trunk.rank,
                "n_outputs": self.trunk.n_outputs,
                "hidden_sizes": list(self.trunk.hidden_sizes),
            },
            "state_dict": self.state_dict(),
        }, path)

    @staticmethod
    def load(path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        branch = BranchNet(**state["branch_kwargs"])
        trunk = TrunkNet(**state["trunk_kwargs"])
        model = PODDeepONet(branch, trunk)
        model.load_state_dict(state["state_dict"])
        return model.to(device)

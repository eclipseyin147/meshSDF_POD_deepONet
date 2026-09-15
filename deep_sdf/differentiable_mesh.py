#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging

import numpy as np
import plyfile
import torch
import torch.nn.functional as F

try:
    from skimage.measure import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes_lewiner as marching_cubes

import deep_sdf.utils


def _eval_points(decoder, latent_vec, coords, max_batch):
    num_samples = coords.shape[0]
    values = torch.zeros(num_samples)
    head = 0
    while head < num_samples:
        subset = coords[head : min(head + max_batch, num_samples)].cuda()
        values[head : min(head + max_batch, num_samples)] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch
    return values


def _grid_coords(N, origin=(-1.0, -1.0, -1.0), extent=2.0):
    step = extent / (N - 1)
    overall_index = torch.arange(0, N ** 3, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    samples[:, 2] = overall_index % N
    samples[:, 1] = torch.div(overall_index, N, rounding_mode="floor") % N
    samples[:, 0] = torch.div(overall_index, N ** 2, rounding_mode="floor") % N
    origin_t = torch.as_tensor(origin, dtype=torch.float32).reshape(1, 3)
    return samples * step + origin_t


def _domain(bounds):
    """Extraction domain: the smallest cube containing `bounds`
    (xmin, ymin, zmin, xmax, ymax, zmax) as (origin, extent). A cubic
    domain keeps the octree subdivision isotropic; the default is the
    classic [-1, 1]^3 unit cube."""
    if bounds is None:
        return (-1.0, -1.0, -1.0), 2.0
    b = [float(v) for v in bounds]
    origin = (b[0], b[1], b[2])
    extent = max(b[3] - b[0], b[4] - b[1], b[5] - b[2])
    if extent <= 0:
        raise ValueError("extract bounds must have positive extent")
    return origin, extent


def sample_sdf_octree(
    decoder,
    latent_vec,
    resolution,
    max_batch,
    initial_resolution=32,
    threshold_factor=2.0,
    field_type="sdf",
    bounds=None,
):
    """Hierarchically sample the field on an octree-refined grid.

    For SDFs, voxels are subdivided when the smallest absolute corner value is
    below threshold_factor * voxel_size. For occupancy fields, voxels are
    subdivided when the corners are not all on the same side of 0.5.
    Unevaluated fine-grid points are filled by trilinear interpolation of the
    coarse grid, which is sign-consistent and cannot introduce spurious zero
    crossings.

    Returns (sdf, N): sdf is a (N, N, N) FloatTensor over the domain cube
    ([-1, 1]^3 by default, or the smallest cube containing `bounds`) and N
    the reached resolution (N follows the chain initial -> 2N-1, so it may
    exceed `resolution`).
    """
    decoder.eval() if hasattr(decoder, "eval") else None

    origin, extent = _domain(bounds)
    origin_t = torch.as_tensor(origin, dtype=torch.float32).reshape(1, 3)

    N = initial_resolution
    sdf = _eval_points(
        decoder, latent_vec, _grid_coords(N, origin, extent), max_batch
    ).reshape(N, N, N)

    while N < resolution:
        voxel_size = extent / (N - 1)
        threshold = threshold_factor * voxel_size

        a = sdf.abs()
        cell_min = torch.minimum(
            torch.minimum(
                torch.minimum(a[:-1, :-1, :-1], a[1:, :-1, :-1]),
                torch.minimum(a[:-1, 1:, :-1], a[1:, 1:, :-1]),
            ),
            torch.minimum(
                torch.minimum(a[:-1, :-1, 1:], a[1:, :-1, 1:]),
                torch.minimum(a[:-1, 1:, 1:], a[1:, 1:, 1:]),
            ),
        )

        if field_type == "sdf":
            split = cell_min < threshold  # (N-1)^3
        else:
            # occupancy: split cells whose corners straddle the 0.5 level
            b = sdf > 0.5
            cell_all_above = (
                b[:-1, :-1, :-1]
                & b[1:, :-1, :-1]
                & b[:-1, 1:, :-1]
                & b[1:, 1:, :-1]
                & b[:-1, :-1, 1:]
                & b[1:, :-1, 1:]
                & b[:-1, 1:, 1:]
                & b[1:, 1:, 1:]
            )
            nb = ~b
            cell_all_below = (
                nb[:-1, :-1, :-1]
                & nb[1:, :-1, :-1]
                & nb[:-1, 1:, :-1]
                & nb[1:, 1:, :-1]
                & nb[:-1, :-1, 1:]
                & nb[1:, :-1, 1:]
                & nb[:-1, 1:, 1:]
                & nb[1:, 1:, 1:]
            )
            split = ~(cell_all_above | cell_all_below)

        if not split.any():
            logging.debug("octree: no voxel to split at resolution {}".format(N))
            break

        N_fine = 2 * N - 1

        # trilinear interpolation reproduces coarse values at even indices
        fine = (
            F.interpolate(
                sdf.reshape(1, 1, N, N, N),
                size=(N_fine, N_fine, N_fine),
                mode="trilinear",
                align_corners=True,
            )
            .reshape(N_fine, N_fine, N_fine)
        )

        # a fine point needs evaluation if any adjacent coarse cell is split
        dilated = (
            F.max_pool3d(
                split.float().reshape(1, 1, N - 1, N - 1, N - 1),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape(N - 1, N - 1, N - 1)
            > 0
        )
        # padded cell array so that cell index ci -> ci + 1 is always valid
        pad = torch.zeros(N + 1, N + 1, N + 1, dtype=torch.bool)
        pad[1:N, 1:N, 1:N] = dilated

        fine_index = torch.arange(N_fine)
        required = torch.zeros(N_fine, N_fine, N_fine, dtype=torch.bool)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    ii = fine_index // 2 - di + 1
                    jj = fine_index // 2 - dj + 1
                    kk = fine_index // 2 - dk + 1
                    required |= pad[ii][:, jj][:, :, kk]

        odd = (fine_index % 2) == 1
        new_points = required & (
            odd.reshape(-1, 1, 1) | odd.reshape(1, -1, 1) | odd.reshape(1, 1, -1)
        )

        indices = new_points.nonzero()
        if indices.shape[0] > 0:
            coords = (
                indices.float() * (extent / (N_fine - 1)) + origin_t
            )
            values = _eval_points(decoder, latent_vec, coords, max_batch)
            fine[new_points] = values

        sdf = fine
        N = N_fine
        logging.debug(
            "octree: refined to {}, evaluated {} new points".format(
                N, indices.shape[0]
            )
        )

    return sdf, N


def compute_sdf_gradients(decoder, latent_vec, points, max_batch=2 ** 18):
    """Analytic gradient of the network output w.r.t. xyz at each point."""
    if hasattr(decoder, "eval"):
        decoder.eval()
    grads = []
    head = 0
    num_points = points.shape[0]
    while head < num_points:
        chunk = (
            points[head : min(head + max_batch, num_points)]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        with torch.enable_grad():
            sdf = deep_sdf.utils.decode_sdf(decoder, latent_vec, chunk)
            grad = torch.autograd.grad(sdf.sum(), chunk)[0]
        grads.append(grad.detach())
        head += max_batch
    return torch.cat(grads, 0)


def extract_differentiable_mesh(
    decoder,
    latent_vec,
    resolution=128,
    initial_resolution=32,
    max_batch=2 ** 18,
    threshold_factor=2.0,
    eps=1e-8,
    field_type="sdf",
    mc_backend="skimage",
    bounds=None,
):
    """Extract the zero level set as a mesh whose vertices are differentiable
    with respect to the latent code (and decoder weights) via the implicit
    function theorem:

        dx/dc = -(n / ||n||^2) * df/dc,   n = grad_x f(z, x)

    field_type is "sdf" (iso-level 0) or "occupancy" (iso-level 0.5, with
    logit amplification of the sampled grid before marching cubes).
    mc_backend is "skimage" (CPU) or "torch" (GPU, deep_sdf.marching_cubes_torch).

    bounds: optional extraction domain (xmin, ymin, zmin, xmax, ymax, zmax).
    The field is sampled on the smallest cube containing it (default
    [-1, 1]^3); use metric-scale bounds, e.g. the data bounding box, when the
    decoder was trained outside the unit cube.

    Returns (verts, faces, normals): verts (M,3) carries gradients, faces
    (F,3) is an integer index tensor, normals (M,3) are detached.
    """
    iso = 0.0 if field_type == "sdf" else 0.5

    sdf, N = sample_sdf_octree(
        decoder,
        latent_vec,
        resolution,
        max_batch,
        initial_resolution,
        threshold_factor,
        field_type,
        bounds,
    )
    origin, extent = _domain(bounds)
    origin_t = torch.as_tensor(origin, dtype=torch.float32).reshape(1, 3)
    voxel_size = extent / (N - 1)

    if field_type == "occupancy":
        # inverse sigmoid amplification approximates the linear regime of a
        # signed distance function and yields smoother meshes (logit(0.5) = 0)
        grid = torch.logit(sdf.clamp(1e-6, 1.0 - 1e-6))
    else:
        grid = sdf

    if mc_backend == "torch":
        from deep_sdf.marching_cubes_torch import marching_cubes_torch

        verts, faces = marching_cubes_torch(
            grid.cuda(), level=0.0, voxel_size=voxel_size
        )
        if verts.shape[0] == 0:
            logging.warning("iso-surface extraction produced an empty mesh")
            empty = torch.zeros(0, 3)
            return empty.cuda(), torch.zeros(0, 3).long().cuda(), empty.cuda()
        verts = verts + origin_t.cuda()
    else:
        try:
            verts_np, faces_np, _, _ = marching_cubes(
                grid.numpy(), level=0.0, spacing=(voxel_size,) * 3
            )
        except (ValueError, RuntimeError) as e:
            logging.warning("iso-surface extraction failed: {}".format(e))
            empty = torch.zeros(0, 3)
            return empty.cuda(), torch.zeros(0, 3).long().cuda(), empty.cuda()

        if verts_np.shape[0] == 0:
            logging.warning("iso-surface extraction produced an empty mesh")
            empty = torch.zeros(0, 3)
            return empty.cuda(), torch.zeros(0, 3).long().cuda(), empty.cuda()

        verts = torch.from_numpy(verts_np.copy()).float().cuda() + origin_t.cuda()
        faces = torch.from_numpy(faces_np.copy()).long().cuda()

    normals = compute_sdf_gradients(decoder, latent_vec, verts, max_batch)

    # value equals verts; gradient is the implicit function theorem result
    field_at_verts = (
        deep_sdf.utils.decode_sdf(decoder, latent_vec, verts) - iso
    )
    normals_sq = (normals ** 2).sum(-1, keepdim=True).clamp_min(eps)
    verts = (
        verts
        - normals
        / normals_sq
        * (field_at_verts - field_at_verts.detach())
    )

    return verts, faces, normals


def save_mesh(verts, faces, filename):
    """Write a (verts, faces) mesh to a ply file."""
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % filename)
    ply_data.write(filename)

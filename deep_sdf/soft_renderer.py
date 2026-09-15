#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def look_at(eyes, center=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation matrices for cameras placed at `eyes`.

    eyes: (K, 3) array-like of camera positions; center/up: array-like (3,).
    Returns a (K, 3, 3) float tensor R with p_camera = R @ p_world. The camera
    looks down its local -z axis and its local +y matches `up` as closely as
    possible. `eyes` must not be parallel to `up`.
    """
    if not torch.is_tensor(eyes):
        eyes = torch.as_tensor(eyes, dtype=torch.float32)
    eyes = eyes.float()
    if eyes.dim() == 1:
        eyes = eyes.unsqueeze(0)

    center = torch.as_tensor(center, dtype=eyes.dtype, device=eyes.device)
    up = torch.as_tensor(up, dtype=eyes.dtype, device=eyes.device)

    forward = F.normalize(center - eyes, dim=-1)
    right = F.normalize(torch.cross(forward, up.expand_as(forward), dim=-1), dim=-1)
    true_up = torch.cross(right, forward, dim=-1)

    # rows are the camera axes expressed in world coordinates
    return torch.stack([right, true_up, -forward], dim=1)


def _triangle_signed_distance(tri2d, pixels):
    """Signed distance from pixels to projected triangles.

    tri2d: (C, 3, 2) projected triangle vertices; pixels: (P, 2). Returns
    (C, P): the signed distance to the triangle - positive inside, negative
    outside - where the unsigned distance is to the closest edge *segment*
    (edge endpoints included), and inside-ness is decided winding-agnostically
    by the sign of the three edge cross products.
    """

    def edge(va, vb):
        e = vb - va  # (C, 2)
        w = pixels.unsqueeze(0) - va.unsqueeze(1)  # (C, P, 2)
        len2 = (e * e).sum(-1, keepdim=True).clamp_min(1e-20)  # (C, 1)
        t = ((w * e.unsqueeze(1)).sum(-1) / len2).clamp(0.0, 1.0)  # (C, P)
        d_seg = (w - t.unsqueeze(-1) * e.unsqueeze(1)).norm(dim=-1)  # (C, P)
        side = e[:, 0:1] * w[..., 1] - e[:, 1:2] * w[..., 0]  # (C, P)
        return d_seg, side

    v0, v1, v2 = tri2d[:, 0], tri2d[:, 1], tri2d[:, 2]
    d0, s0 = edge(v0, v1)
    d1, s1 = edge(v1, v2)
    d2, s2 = edge(v2, v0)

    dmin = torch.minimum(torch.minimum(d0, d1), d2)
    inside = ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))
    return torch.where(inside, dmin, -dmin)


def render_silhouette(
    verts,
    faces,
    eyes,
    image_size=64,
    sigma=1e-4,
    dist_thresh=0.1,
    face_chunk=1024,
):
    """Soft-rasterize the silhouette (occupancy) of a mesh from K cameras.

    verts: (M, 3) differentiable tensor; faces: (F, 3) long tensor; eyes:
    (K, 3) camera positions looking at the origin. Orthographic projection;
    the image plane spans [-1, 1]^2 in camera coordinates, matching the
    normalized [-1, 1]^3 shape coordinate frame.

    Per-pixel coverage of face f is A_fj = sigmoid(D_fj / sigma) with D_fj the
    signed pixel-to-triangle distance (positive inside). The silhouette
    aggregates faces as S_j = 1 - prod_f(1 - A_fj), computed in log space as
    1 - exp(sum_f logsigmoid(-D_fj / sigma)) for numerical stability. Faces
    are processed in chunks (with gradient checkpointing) so the (F x H x W)
    tensor is never fully materialized; sigma is in normalized [-1, 1]
    coordinates (one pixel is 2 / image_size wide).

    dist_thresh culls faces whose projected bounding box lies further than
    this outside the image square; they cannot contribute to any pixel.

    Returns a (K, image_size, image_size) tensor in [0, 1] (row 0 = top, i.e.
    +y in camera coordinates), differentiable w.r.t. verts.
    """
    device = verts.device
    dtype = verts.dtype
    R = look_at(eyes).to(device=device, dtype=dtype)

    # pixel centers in normalized camera coordinates
    offsets = (
        torch.arange(image_size, device=device, dtype=dtype) + 0.5
    ) * (2.0 / image_size) - 1.0
    rows, cols = torch.meshgrid(-offsets, offsets, indexing="ij")
    pixels = torch.stack([cols.reshape(-1), rows.reshape(-1)], dim=-1)  # (P, 2)
    num_pixels = pixels.shape[0]

    v_faces = verts[faces]  # (F, 3, 3)
    images = []
    for k in range(R.shape[0]):
        tri2d = torch.matmul(v_faces, R[k].transpose(0, 1))[..., :2]  # (F, 3, 2)

        limit = 1.0 + dist_thresh
        bb_min = tri2d.amin(dim=1)  # (F, 2)
        bb_max = tri2d.amax(dim=1)
        keep = ((bb_max >= -limit) & (bb_min <= limit)).all(dim=-1)
        face_idx = keep.nonzero(as_tuple=True)[0]

        # keep a (zero-valued) grad path even if every face is culled
        log_neg = torch.zeros(num_pixels, device=device, dtype=dtype)
        log_neg = log_neg + tri2d.sum() * 0.0
        for start in range(0, face_idx.shape[0], face_chunk):
            chunk = tri2d[face_idx[start : start + face_chunk]]

            def chunk_log(c):
                d = _triangle_signed_distance(c, pixels)
                return F.logsigmoid(-d / sigma).sum(dim=0)

            if torch.is_grad_enabled() and chunk.requires_grad:
                contrib = checkpoint(chunk_log, chunk, use_reentrant=False)
            else:
                contrib = chunk_log(chunk)
            log_neg = log_neg + contrib

        images.append((1.0 - torch.exp(log_neg)).reshape(image_size, image_size))

    return torch.stack(images, dim=0)


def save_silhouette(silhouette, filename):
    """Save a (..., H, W) silhouette tensor as a float32 .npy file."""
    np.save(filename, silhouette.detach().cpu().numpy().astype(np.float32))

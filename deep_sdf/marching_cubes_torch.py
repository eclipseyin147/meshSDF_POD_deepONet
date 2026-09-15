#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import base64

import numpy as np
import torch


def _load_case_table():
    try:
        from skimage.measure._marching_cubes_lewiner_luts import THE_LUTS

        return np.asarray(THE_LUTS.CASESCLASSIC, dtype=np.int64)
    except (ImportError, AttributeError):
        from skimage.measure import _marching_cubes_lewiner_luts as luts

        shape, text = luts.CASESCLASSIC
        table = np.frombuffer(base64.decodebytes(text.encode("utf-8")), np.int8)
        return table.reshape(shape).astype(np.int64)


_CASE_TABLE = _load_case_table()

# corner offsets in standard lorensen numbering (bit c <-> corner c)
_CORNERS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]

# edge -> (axis, di, dj, dk) of the edge's start vertex relative to the cell
_EDGES = [
    (0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0),
    (0, 0, 0, 1), (1, 1, 0, 1), (0, 0, 1, 1), (1, 0, 0, 1),
    (2, 0, 0, 0), (2, 1, 0, 0), (2, 1, 1, 0), (2, 0, 1, 0),
]


def marching_cubes_torch(sdf_grid, level=0.0, voxel_size=1.0):
    """Extract an isosurface mesh from a dense SDF grid.

    sdf_grid: (N, N, N) torch.FloatTensor on cpu or cuda. Returns
    (verts (M, 3) float32, faces (F, 3) int64) on the same device, with
    verts = grid index coordinates * voxel_size, matching
    skimage.measure.marching_cubes(spacing=(voxel_size,) * 3).
    """
    device = sdf_grid.device
    sdf = sdf_grid.detach().to(torch.float32)
    nx, ny, nz = sdf.shape
    inside = sdf > level
    empty = (
        torch.zeros((0, 3), dtype=torch.float32, device=device),
        torch.zeros((0, 3), dtype=torch.int64, device=device),
    )

    # per-axis crossing edges, compacted into global vertex ids via cumsum;
    # each axis keeps its natural (shrunk) grid shape and c-contiguous strides
    id_maps = []
    strides = []
    axis_offset = []
    verts = []
    count = 0
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        cross = inside[tuple(lo)] ^ inside[tuple(hi)]
        ids = torch.cumsum(cross.reshape(-1).to(torch.int32), 0)
        id_maps.append(ids.reshape(cross.shape) - 1)
        shape = cross.shape
        strides.append((shape[1] * shape[2], shape[2], 1))
        axis_offset.append(count)
        count += int(cross.sum())
        if cross.any():
            pos = cross.nonzero().to(torch.float32)
            s0 = sdf[tuple(lo)][cross]
            s1 = sdf[tuple(hi)][cross]
            t = (level - s0) / (s1 - s0)
            pos[:, axis] += t
            verts.append(pos)
    if count == 0:
        return empty

    # case index per cell
    bits = inside.to(torch.int32)
    case = torch.zeros((nx - 1, ny - 1, nz - 1), dtype=torch.int32, device=device)
    for c, (di, dj, dk) in enumerate(_CORNERS):
        case += bits[di : di + nx - 1, dj : dj + ny - 1, dk : dk + nz - 1] << c

    active = (case != 0) & (case != 255)
    sel = active.reshape(-1).nonzero().squeeze(1)
    if sel.numel() == 0:
        return empty

    # global vertex id of each of the 12 edges of every active cell
    ci = sel // ((ny - 1) * (nz - 1))
    rem = sel % ((ny - 1) * (nz - 1))
    cj = rem // (nz - 1)
    ck = rem % (nz - 1)
    cell_edges = torch.empty((sel.numel(), 12), dtype=torch.int32, device=device)
    for e, (axis, di, dj, dk) in enumerate(_EDGES):
        s0, s1, s2 = strides[axis]
        flat = (ci + di) * s0 + (cj + dj) * s1 + (ck + dk) * s2
        cell_edges[:, e] = id_maps[axis].reshape(-1)[flat] + axis_offset[axis]

    table = torch.from_numpy(_CASE_TABLE).to(device)
    tri = table[case.reshape(-1)[sel].to(torch.int64)]
    valid = tri >= 0
    faces = torch.gather(cell_edges, 1, tri.clamp(min=0))[valid]
    faces = faces.reshape(-1, 3).to(torch.int64)

    return torch.cat(verts, dim=0) * voxel_size, faces

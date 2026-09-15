# PIPOD+DeepONet 体积场架构（Stage 1–3）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 DeepSDF 仓库中搭建 physics-informed POD-DeepONet 体积场算子：`q_v(x; z, μ) = b_v(x) + Σ_k a^v_k(z,μ)·φ^v_k(x, d, ∇d, h)`（v ∈ u,v,w,p），含物理一致合成场、逐变量 POD、三阶段训练（系数回归 → 场训练 → 物理正则微调），在 27 椭球族上端到端验证。

**Architecture:** 方案 A（学习式 trunk 基）：DeepSDF（冻结）提供 z 与查询点处 d/∇d；BranchNet（结构对齐 mPOD-DeepONet `BranchNetLinear`）输出逐变量系数 (4, r)；TrunkNet（逐变量 trunk ModuleList + bias trunk，对齐 `MultiTrunkDeepONet`）输出学习基；`einsum('bcr,cnr->bcn')` 融合。POD 只用于定秩/系数监督/评估下界，不进前向。物理约束按 PhysicsNeMo `PhysicsInformer` 模式本地轻量实现（SymPy PDE + autodiff）。

**Tech Stack:** 项目 `.venv`（torch 2.5.1+cu121、sympy 1.13.1、numpy）；无新增依赖；参考代码 `third-party/mPOD-DeepONet/`、`third-party/physicsnemo/`（只读参考，不作依赖）。

**Spec:** `docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md`（定稿，commit 3e1301c）

## Global Constraints

- 所有命令在仓库根目录 `/home/siqi/CLionProjects/DeepSDF` 下执行；Python 一律用 `.venv/bin/python`。
- 单元测试脚本放 `/tmp`（仓库惯例，不入库），文件头部统一 `sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")`；测试为纯 assert 脚本（不依赖 pytest）。
- **网络结构严格参考 mPOD-DeepONet、物理约束严格参考 PhysicsNeMo**（对照表见 spec §9）；仅有意偏离：SiLU（PhysicsNeMo FullyConnected 默认）、trunk 输入含 (d, ∇d, h)、学习基 + bias trunk。
- 无量纲约定：快照存**有量纲速度**（scale ~U）与 **Cp**；训练管线加载时 `u /= U`；POD/网络/物理残差全在无量纲空间（p 通道即 Cp，动量方程压力梯度系数为 ½）。
- DeepSDF decoder 全程冻结（`requires_grad=False`）；trunk 物理分支的 d 经 decoder 带图评估、∇d 特征 detached。
- GPU 8GB，DrivAerNet 训练在跑（~5.7GB 占用）：新训练任务全部小模型 + 分块，显存预算 ≤ 2GB；启动前 `nvidia-smi` 确认余量，OOM 则减小 ScenesPerBatch/chunk。
- 每个任务结束 git commit（只 add 本任务文件）；代码风格同现有模块：英文 docstring、最少行内注释、标准化 buffer 恒等默认 + setter + clamp 1e-8。
- 真实 CFD 数据接口：npz 快照契约（`fields (G,4)` / `bc (4,)` / `shape`）保持不变。

---

### Task 1: 快照 IO 上移到 volume.py（现有 POD 代码的行为不变重构）

**Files:**
- Modify: `deep_sdf/cfd/volume.py`（顶部 import 加一行；`make_reference_grid` 之后新增三个函数）
- Modify: `deep_sdf/cfd/__init__.py`（导出列表追加）
- Modify: `train_volume_rom.py`（删除三个本地定义，改为 import）
- Test: `/tmp/test_snapshot_io.py`

**Interfaces:**
- Consumes: 现有 `deep_sdf/cfd/labels.py: BC_FIELDS`、`train_volume_rom.py: save_snapshot/load_snapshot/snapshot_filename`（被移动的实现）。
- Produces: `deep_sdf.cfd.volume.save_snapshot(path, fields, bc, shape_name)`、`load_snapshot(path, expected_points=None) -> dict(fields (G,4) float32, bc (4,) float32, shape str)`、`snapshot_filename(npz, case_idx) -> str`；`train_volume_rom.py` 行为不变。

- [ ] **Step 1: 写失败测试** `/tmp/test_snapshot_io.py`

```python
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")
from deep_sdf.cfd.volume import load_snapshot, save_snapshot, snapshot_filename


def test_filename():
    assert snapshot_filename("ellipsoids/ellipsoid/e_a0.5.npz", 3) == \
        "ellipsoids_ellipsoid_e_a0.5_case003.npz"


def test_roundtrip():
    fields = torch.rand(64, 4)
    bc = [15.0, 1.0, 0.0, 0.0]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sub", "case.npz")  # 目录不存在也能写
        save_snapshot(p, fields, bc, "a/b/c.npz")
        snap = load_snapshot(p, expected_points=64)
        assert snap["fields"].dtype == torch.float32
        assert torch.allclose(snap["fields"], fields)
        assert torch.allclose(snap["bc"], torch.tensor(bc, dtype=torch.float32))
        assert snap["shape"] == "a/b/c.npz"


def test_validation():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.npz")
        np.savez(p, fields=np.zeros((4, 4), dtype=np.float32))  # 缺 bc/shape
        try:
            load_snapshot(p)
            raise AssertionError("missing-key snapshot should raise ValueError")
        except ValueError:
            pass
        save_snapshot(p, torch.rand(4, 4), [1.0, 1.0, 0.0, 0.0], "s.npz")
        try:
            load_snapshot(p, expected_points=8)  # grid mismatch
            raise AssertionError("grid mismatch should raise ValueError")
        except ValueError:
            pass


if __name__ == "__main__":
    test_filename()
    test_roundtrip()
    test_validation()
    print("test_snapshot_io: ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_snapshot_io.py`
Expected: FAIL，`ImportError: cannot import name 'save_snapshot' from 'deep_sdf.cfd.volume'`

- [ ] **Step 3: 实现——volume.py 顶部 import 区追加**

```python
from deep_sdf.cfd.labels import BC_FIELDS
```

并在 `make_reference_grid` 定义之后插入（实现从 `train_volume_rom.py` 原样上移，仅把 `deep_sdf.cfd.BC_FIELDS` 改为直接引用的 `BC_FIELDS`）：

```python
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
```

`deep_sdf/cfd/__init__.py` 的 volume 导入块改为：

```python
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
```

`train_volume_rom.py`：删除本地的 `snapshot_filename`/`save_snapshot`/`load_snapshot` 三个定义（原 45–96 行），在 `from train_pressure_surrogate import (...)` 之后加：

```python
from deep_sdf.cfd.volume import load_snapshot, save_snapshot, snapshot_filename
```

- [ ] **Step 4: 运行测试确认通过 + 回归**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_snapshot_io.py && .venv/bin/python -c "import train_volume_rom; print('import ok')"`
Expected: `test_snapshot_io: ALL PASS` + `import ok`

- [ ] **Step 5: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add deep_sdf/cfd/volume.py deep_sdf/cfd/__init__.py train_volume_rom.py
git commit -m "Hoist snapshot IO from train_volume_rom into deep_sdf.cfd.volume"
```

---

### Task 2: 物理一致合成场 `deep_sdf/cfd/flow_synth.py`

**Files:**
- Create: `deep_sdf/cfd/flow_synth.py`
- Test: `/tmp/test_flow_synth.py`

**Interfaces:**
- Consumes: Task 1 的 `save_snapshot`/`load_snapshot`（测试用）；npz 形状命名约定 `ellipsoid_a{A}_b{B}_c{C}.npz`。
- Produces: `parse_ellipsoid_axes(npz_name) -> (a, b, c)`；`sphere_potential_flow(xi (N,3), direction (3,), U) -> (N,3)`；`potential_flow_field(axes, center, bc, grid_points, wake_amp=0.15, wake_sigma=0.5, wake_offset=1.5) -> (G,4) float32 [u,v,w,Cp]`（**有量纲**速度；Task 6 训练管线按 U 归一）。全部函数对 `grid_points` 可微（physics 残差真值测试依赖这点），dtype 跟随输入。

- [ ] **Step 1: 写失败测试** `/tmp/test_flow_synth.py`

```python
import sys

import torch

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")
from deep_sdf.cfd.flow_synth import (
    parse_ellipsoid_axes,
    potential_flow_field,
    sphere_potential_flow,
)


def test_parse_axes():
    assert parse_ellipsoid_axes(
        "ellipsoids/ellipsoid/ellipsoid_a0.5_b0.7_c0.9.npz"
    ) == (0.5, 0.7, 0.9)
    try:
        parse_ellipsoid_axes("chair_001.npz")
        raise AssertionError("non-ellipsoid name should raise ValueError")
    except ValueError:
        pass


def test_far_field():
    # 远场 r~10：r^-3 修正 ~1e-3 量级，速度应回到 U*dir，Cp 回 0
    pts = torch.tensor([[10.0, 0.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 9.0, 5.0]])
    f = potential_flow_field((0.5, 0.7, 0.9), (0, 0, 0), [15.0, 1, 0, 0], pts,
                             wake_amp=0.0)
    assert ((f[:, :3] - torch.tensor([15.0, 0, 0])).norm(dim=1) / 15 < 2e-3).all()
    assert f[:, 3].abs().max() < 1e-2


def test_slip_wall_exact():
    # 椭球面上 u·n = 0（wake 关闭）；法向 n ∝ (x-c)/axes^2
    axes = (0.5, 0.7, 0.9)
    g = torch.Generator().manual_seed(0)
    d = torch.randn(2000, 3, generator=g)
    d = d / d.norm(dim=1, keepdim=True)
    a = torch.tensor(axes)
    pts = d * a  # 椭球面点
    normals = pts / (a ** 2)
    normals = normals / normals.norm(dim=1, keepdim=True)
    f = potential_flow_field(axes, (0, 0, 0), [15.0, 0.6, 0.48, 0.64], pts,
                             wake_amp=0.0)
    un = (f[:, :3] * normals).sum(dim=1).abs() / 15.0
    assert un.max() < 1e-5, "slip violation {}".format(un.max())


def test_interior_extension():
    g = torch.Generator().manual_seed(1)
    pts = torch.rand(500, 3, generator=g) * 0.6 - 0.3  # 椭球内部（min axis 0.5）
    f = potential_flow_field((0.7, 0.7, 0.7), (0, 0, 0), [12.0, 1, 0, 0], pts,
                             wake_amp=0.15)
    assert f[:, :3].abs().max() == 0.0
    assert (f[:, 3] == 1.0).all()


def _fd_divergence(vel, n, spacing):
    """vel (N,N,N,3) 中心差分散度。"""
    du = (vel[2:, 1:-1, 1:-1, 0] - vel[:-2, 1:-1, 1:-1, 0]) / (2 * spacing)
    dv = (vel[1:-1, 2:, 1:-1, 1] - vel[1:-1, :-2, 1:-1, 1]) / (2 * spacing)
    dw = (vel[1:-1, 1:-1, 2:, 2] - vel[1:-1, 1:-1, :-2, 2]) / (2 * spacing)
    return du + dv + dw  # (N-2)^3


def test_divergence_free_base_flow():
    # 基流（wake 关）FD 散度 ≈ 0：纯 FD 截断误差
    n = 64
    lin = torch.linspace(-1.5, 1.5, n)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], 1)
    axes = (0.5, 0.7, 0.9)
    f = potential_flow_field(axes, (0, 0, 0), [15.0, 0.6, 0.48, 0.64], pts,
                             wake_amp=0.0)
    vel = f[:, :3].reshape(n, n, n, 3)
    spacing = 3.0 / (n - 1)
    div = _fd_divergence(vel, n, spacing)
    # 流体内部点（去边界两层 + 去体内/近壁：|xi|>1.15）
    a = torch.tensor(axes)
    xi = (pts / a).norm(dim=1).reshape(n, n, n)
    fluid = (xi[1:-1, 1:-1, 1:-1] > 1.15)
    assert div[fluid].abs().max() < 1e-2, \
        "base-flow FD divergence {}".format(div[fluid].abs().max())


def test_wake_divergence_free_relative():
    # wake 开：FD 散度应远小于速度梯度量级（否则散度来自构造错误而非 FD 截断）
    n = 64
    lin = torch.linspace(-1.5, 1.5, n)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], 1)
    f = potential_flow_field((0.5, 0.7, 0.9), (0, 0, 0), [15.0, 0.6, 0.48, 0.64],
                             pts, wake_amp=0.15)
    vel = f[:, :3].reshape(n, n, n, 3)
    spacing = 3.0 / (n - 1)
    div = _fd_divergence(vel, n, spacing)
    gx = (vel[2:, 1:-1, 1:-1] - vel[:-2, 1:-1, 1:-1]) / (2 * spacing)
    grad_scale = gx.norm(dim=-1).max()
    assert div.abs().max() < 0.05 * grad_scale, \
        "wake div {} vs grad scale {}".format(div.abs().max(), grad_scale)


def test_sphere_momentum_exact_autodiff():
    # 球体 + wake 关：用 torch autodiff 直接对解析场求稳态不可压 NS 残差，
    # 连续性与动量都应为 0（float64 机器精度量级）。这一步同时验证
    # potential_flow_field 对 grid_points 可微（physics 真值测试的前提）。
    g = torch.Generator().manual_seed(2)
    pts = (torch.rand(2000, 3, generator=g, dtype=torch.float64) * 2 - 1)
    pts = pts[pts.norm(dim=1) > 1.05].requires_grad_(True)  # 单位球外流区
    f = potential_flow_field((1.0, 1.0, 1.0), (0, 0, 0), [10.0, 1, 0, 0], pts,
                             wake_amp=0.0)
    u, cp = f[:, 0:1], f[:, 3:4]
    U = 10.0
    # 三个速度分量的梯度
    grads = [
        torch.autograd.grad(f[:, c].sum(), pts, create_graph=True)[0]
        for c in range(3)
    ]
    # 散度 div u* = (du/dx + dv/dy + dw/dz) / U
    div = sum(grads[c][:, c] for c in range(3)) / U
    # 动量 x 分量: (u*·∇)u*_x + 0.5 dCp/dx - (1/Re) ∇²u*_x
    gu = grads[0]  # ∂u_x/∂(x,y,z)
    convect = sum(f[:, j:j + 1] / U * gu[:, j:j + 1] / U for j in range(3))
    dcp = torch.autograd.grad(cp.sum(), pts, create_graph=True)[0][:, 0:1]
    lap = sum(
        torch.autograd.grad(gu[:, c].sum(), pts, create_graph=True)[0][:, c:c + 1]
        for c in range(3)
    ) / U
    mom = convect + 0.5 * dcp - (1.0 / 1e4) * lap
    assert div.abs().max() < 1e-8, "continuity residual {}".format(div.abs().max())
    assert mom.abs().max() < 1e-6, "momentum residual {}".format(mom.abs().max())


if __name__ == "__main__":
    test_parse_axes()
    test_far_field()
    test_slip_wall_exact()
    test_interior_extension()
    test_divergence_free_base_flow()
    test_wake_divergence_free_relative()
    test_sphere_momentum_exact_autodiff()
    print("test_flow_synth: ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_flow_synth.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'deep_sdf.cfd.flow_synth'`

- [ ] **Step 3: 实现** `deep_sdf/cfd/flow_synth.py`

```python
#!/usr/bin/env python3
"""Physics-consistent synthetic volume fields (design doc
docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md section 4).

Base flow: potential flow past a unit sphere, anisotropically mapped to the
ellipsoid with semi-axes (a, b, c): xi = M (x - c) with M = diag(1/a, 1/b,
1/c), u(x) = M^-1 u_hat(xi). For constant diagonal M this is exactly
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

    u = sphere_potential_flow(xi, d, U) * axes_t  # M^-1 u_hat

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
```

（注意用 `torch.where` 而非原位索引赋值，保证对 `grid_points` 的可微性——测试 `test_sphere_momentum_exact_autodiff` 依赖这一点。）

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_flow_synth.py`
Expected: `test_flow_synth: ALL PASS`

- [ ] **Step 5: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add deep_sdf/cfd/flow_synth.py
git commit -m "Add physics-consistent synthetic flow field generator"
```

---

### Task 3: 椭球数据集生成 + decoder 训练（`generate_ellipsoid_dataset.py`）

**Files:**
- Create: `generate_ellipsoid_dataset.py`（根目录）
- Test: `/tmp/test_ellipsoid_dataset.py`、`/tmp/test_ellipsoid_decoder.py`

**Interfaces:**
- Consumes: `deep_sdf.data` 的 npz 约定（`pos`/`neg` (N,4) float32 [x,y,z,sdf]）、`examples/chairs/specs.json` 的 specs 键集合、`train_deep_sdf.py`。
- Produces: `data/ellipsoids/SdfSamples/ellipsoids/ellipsoid/ellipsoid_a{A}_b{B}_c{C}.npz`（27 个）；`examples/ellipsoids/split.json`（`{"ellipsoids": {"ellipsoid": [...27 names...]}}`）；`examples/ellipsoids/specs.json`；训练后产出 `examples/ellipsoids/LatentCodes/latest.pth` 与 `ModelParameters/latest.pth`（Task 6 经 `load_or_fit_latent` 惯例使用）。

- [ ] **Step 1: 写失败测试** `/tmp/test_ellipsoid_dataset.py`

```python
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")

REPO = "/home/siqi/CLionProjects/DeepSDF"


def test_generate(tmp_out):
    subprocess.run(
        [os.path.join(REPO, ".venv/bin/python"),
         os.path.join(REPO, "generate_ellipsoid_dataset.py"),
         "--out", tmp_out, "--samples_per_shape", "2000", "--seed", "0"],
        check=True, cwd=REPO,
    )
    sdf_dir = os.path.join(tmp_out, "data/ellipsoids/SdfSamples/ellipsoids/ellipsoid")
    names = sorted(os.listdir(sdf_dir))
    assert len(names) == 27
    assert "ellipsoid_a0.5_b0.7_c0.9.npz" in names
    d = np.load(os.path.join(sdf_dir, names[0]))
    assert set(d.files) == {"pos", "neg"}
    for k in ("pos", "neg"):
        assert d[k].ndim == 2 and d[k].shape[1] == 4
        assert d[k].dtype == np.float32
    # pos 近表面：|sdf| 应基本在噪声 3σ（0.15）+ 近似误差范围内
    assert np.abs(d["pos"][:, 3]).max() < 0.25
    # 符号混合（内外都有点）
    assert (d["pos"][:, 3] > 0).any() and (d["pos"][:, 3] < 0).any()
    specs = json.load(open(os.path.join(tmp_out, "examples/ellipsoids/specs.json")))
    assert specs["CodeLength"] == 16
    assert specs["NetworkArch"] == "deep_sdf_decoder"
    split = json.load(open(os.path.join(tmp_out, "examples/ellipsoids/split.json")))
    assert len(split["ellipsoids"]["ellipsoid"]) == 27


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        test_generate(t)
    print("test_ellipsoid_dataset: ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_ellipsoid_dataset.py`
Expected: FAIL，`FileNotFoundError: generate_ellipsoid_dataset.py`（脚本不存在）

- [ ] **Step 3: 实现** `generate_ellipsoid_dataset.py`

```python
#!/usr/bin/env python3
"""Generate the 27-ellipsoid DeepSDF dataset + experiment specs (design doc
docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md section 6).

Writes, under --out (default: the repo root):
- data/ellipsoids/SdfSamples/ellipsoids/ellipsoid/ellipsoid_a{A}_b{B}_c{C}.npz
  for axes in {0.5, 0.7, 0.9}^3: DeepSDF-convention pos/neg (N, 4) float32
  [x, y, z, sdf] samples (pos = near-surface with sigma 0.005 / 0.05 noise,
  neg = uniform volume);
- examples/ellipsoids/split.json  ({"ellipsoids": {"ellipsoid": [...]}});
- examples/ellipsoids/specs.json  (small decoder: dims [128]*4, latent 16).

The SDF uses iq's ellipsoid approximation k0*(k0-1)/k1 (accurate near the
surface, which is what the clamped DeepSDF loss weights). Data files are
generated on demand and not committed to git.
"""

import argparse
import json
import logging
import os

import numpy as np

AXES_VALUES = (0.5, 0.7, 0.9)
SURFACE_NOISE = (0.005, 0.05)
POS_FRACTIONS = (0.5, 0.25)  # per noise level; remainder is uniform neg


def ellipsoid_sdf(points, axes):
    """iq's ellipsoid SDF approximation: k0 = |p/r|, k1 = |p/r^2|,
    sd = k0 (k0 - 1) / k1."""
    p = points / np.asarray(axes, dtype=np.float64)
    k0 = np.linalg.norm(p, axis=1)
    k1 = np.linalg.norm(points / np.asarray(axes, dtype=np.float64) ** 2,
                        axis=1).clip(1e-12)
    return (k0 * (k0 - 1.0) / k1).astype(np.float32)


def sample_shape(axes, n_samples, rng):
    """pos/neg (N, 4) samples for one ellipsoid (semi-axes, centered origin)."""
    n1 = int(n_samples * POS_FRACTIONS[0])
    n2 = int(n_samples * POS_FRACTIONS[1])
    n3 = n_samples - n1 - n2
    # uniform directions on the unit sphere -> ellipsoid surface points
    dirs = rng.normal(size=(n1 + n2, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    surface = dirs * np.asarray(axes)
    pts1 = surface[:n1] + rng.normal(scale=SURFACE_NOISE[0], size=(n1, 3))
    pts2 = surface[n1:] + rng.normal(scale=SURFACE_NOISE[1], size=(n2, 3))
    pts3 = rng.uniform(-1.25, 1.25, size=(n3, 3))
    pos = np.concatenate([pts1, pts2], 0).astype(np.float32)
    neg = pts3.astype(np.float32)
    pos = np.concatenate([pos, ellipsoid_sdf(pos, axes)[:, None]], 1)
    neg = np.concatenate([neg, ellipsoid_sdf(neg, axes)[:, None]], 1)
    return pos.astype(np.float32), neg.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.dirname(
        os.path.abspath(__file__)))
    parser.add_argument("--samples_per_shape", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(args.seed)

    data_dir = os.path.join(
        args.out, "data/ellipsoids/SdfSamples/ellipsoids/ellipsoid")
    exp_dir = os.path.join(args.out, "examples/ellipsoids")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    names = []
    for a in AXES_VALUES:
        for b in AXES_VALUES:
            for c in AXES_VALUES:
                name = "ellipsoid_a{}_b{}_c{}".format(a, b, c)
                pos, neg = sample_shape((a, b, c), args.samples_per_shape, rng)
                np.savez(os.path.join(data_dir, name + ".npz"),
                         pos=pos, neg=neg)
                names.append(name)
    logging.info("wrote %d shapes to %s", len(names), data_dir)

    split = {"ellipsoids": {"ellipsoid": names}}
    with open(os.path.join(exp_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    specs = {
        "Description": "DeepSDF autodecoder on 27 analytic ellipsoids "
                       "(PIPOD-DeepONet validation family)",
        "DataSource": os.path.join(os.path.abspath(args.out),
                                   "data/ellipsoids"),
        "TrainSplit": os.path.join(os.path.abspath(args.out),
                                   "examples/ellipsoids/split.json"),
        "NetworkArch": "deep_sdf_decoder",
        "NetworkSpecs": {
            "dims": [128, 128, 128, 128],
            "dropout": [],
            "dropout_prob": 0.0,
            "norm_layers": [],
            "latent_in": [2],
            "xyz_in_all": False,
            "use_tanh": False,
            "latent_dropout": False,
            "weight_norm": False,
        },
        "CodeLength": 16,
        "NumEpochs": 1001,
        "SnapshotFrequency": 200,
        "AdditionalSnapshots": [100, 500],
        "LearningRateSchedule": [
            {"Type": "Step", "Initial": 0.0005, "Interval": 500,
             "Factor": 0.5},
            {"Type": "Step", "Initial": 0.001, "Interval": 500,
             "Factor": 0.5},
        ],
        "SamplesPerScene": 16384,
        "ScenesPerBatch": 27,
        "DataLoaderThreads": 4,
        "ClampingDistance": 0.1,
        "CodeRegularization": True,
        "CodeRegularizationLambda": 0.0001,
        "CodeBound": 1.0,
    }
    with open(os.path.join(exp_dir, "specs.json"), "w") as f:
        json.dump(specs, f, indent=2)
    logging.info("wrote split.json + specs.json to %s", exp_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过（小样本人检）**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_ellipsoid_dataset.py`
Expected: `test_ellipsoid_dataset: ALL PASS`

- [ ] **Step 5: 生成正式数据（全量 10 万样本/形状）**

```bash
cd /home/siqi/CLionProjects/DeepSDF
.venv/bin/python generate_ellipsoid_dataset.py
ls data/ellipsoids/SdfSamples/ellipsoids/ellipsoid/ | wc -l   # 期望 27
```

- [ ] **Step 6: 训练椭球 decoder（后台，先确认 GPU 余量）**

```bash
cd /home/siqi/CLionProjects/DeepSDF
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
# 余量 < 2GB 时把 specs.json 的 ScenesPerBatch 降到 8 再启动
nohup .venv/bin/python train_deep_sdf.py -e examples/ellipsoids --batch_split 1 \
    > /tmp/ellipsoid_train.log 2>&1 &
```

验证训练正常推进（几分钟后）：

```bash
grep -E "epoch" /tmp/ellipsoid_train.log | tail -3
```

Expected: loss 随 epoch 下降；`examples/ellipsoids/ModelParameters/` 出现 snapshot。

- [ ] **Step 7: decoder 精度验证**（训练完成后）`/tmp/test_ellipsoid_decoder.py`

```python
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")
import deep_sdf
import deep_sdf.data
import deep_sdf.utils
import deep_sdf.workspace as ws
from generate_ellipsoid_dataset import ellipsoid_sdf

REPO = "/home/siqi/CLionProjects/DeepSDF"
EXP = os.path.join(REPO, "examples/ellipsoids")

specs = json.load(open(os.path.join(EXP, "specs.json")))
arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
latent_size = specs["CodeLength"]
decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
decoder = torch.nn.DataParallel(decoder)
saved = torch.load(os.path.join(EXP, ws.model_params_subdir, "latest.pth"),
                   map_location="cpu")
decoder.load_state_dict(saved["model_state_dict"])
decoder = decoder.module.cuda().eval()

with open(os.path.join(EXP, "split.json")) as f:
    split = json.load(f)
npz_files = deep_sdf.data.get_instance_filenames(specs["DataSource"], split)
latents = torch.load(os.path.join(EXP, ws.latent_codes_subdir, "latest.pth"),
                     map_location="cpu")["latent_codes"]
if isinstance(latents, dict):
    latents = latents["weight"]
elif hasattr(latents, "weight"):
    latents = latents.weight
latents = latents.float().cuda()

rng = np.random.default_rng(0)
errs = []
for i, npz in enumerate(npz_files):
    name = npz.rsplit("/", 1)[-1][:-4]
    axes = tuple(float(t[1:]) for t in name.split("_") if t[0] in "abc")
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = dirs * np.asarray(axes)  # 表面点
    pts = pts + rng.normal(scale=0.02, size=pts.shape)  # 近表面扰动
    sdf_true = ellipsoid_sdf(pts.astype(np.float32), axes)
    pts_t = torch.from_numpy(pts).float().cuda()
    with torch.no_grad():
        sdf_pred = deep_sdf.utils.decode_sdf(
            decoder, latents[i].reshape(1, -1), pts_t).squeeze(1)
    # 只比较 clamp 内的近表面点
    m = np.abs(sdf_true) < specs["ClampingDistance"]
    errs.append(np.abs(sdf_pred[m].cpu().numpy() - sdf_true[m]).mean())
    print("{}: mean |sdf err| = {:.5f}".format(name, errs[-1]))
mean_err = float(np.mean(errs))
print("mean over shapes: {:.5f}".format(mean_err))
assert mean_err < 0.02, "decoder SDF accuracy insufficient"
print("test_ellipsoid_decoder: ALL PASS")
```

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_ellipsoid_decoder.py`
Expected: 每形状 mean |sdf err| ≲ 0.02，`ALL PASS`。

- [ ] **Step 8: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add generate_ellipsoid_dataset.py examples/ellipsoids/specs.json examples/ellipsoids/split.json
git commit -m "Add 27-ellipsoid dataset generator and experiment specs"
```

（`data/ellipsoids/` 与训练 checkpoint 不入库。）

---

### Task 4: DeepONet 模块 `deep_sdf/cfd/deeponet.py`

**Files:**
- Create: `deep_sdf/cfd/deeponet.py`
- Modify: `deep_sdf/cfd/__init__.py`（追加导出）
- Test: `/tmp/test_deeponet.py`

**Interfaces:**
- Consumes: 无上游代码依赖（独立 nn.Module）；风格对齐 `volume.py: VolumeCoefficientRegressor`（标准化 buffer 惯例）与 `third-party/mPOD-DeepONet/models/DeepONet.py: MultiTrunkDeepONet`（组合模式）。
- Produces:
  - `BranchNet(latent_size, bc_dim=4, rank, n_outputs=4, hidden=256, num_layers=4, bc_hidden=64)`；`forward_normalized(latent, bc) -> (B, n_outputs, rank)`（标准化空间，供 L_POD）；`forward(latent, bc) -> (B, n_outputs, rank)`（物理系数）；`set_z/bc/coef_normalization(mean, std)`（coef 统计量形状 (n_outputs, rank)）。
  - `TrunkNet(in_dim=8, rank, n_outputs=4, hidden_sizes=(256, 512))`；`basis(features (N,8)) -> (n_outputs, N, rank)`；`bias(features) -> (N, n_outputs)`。
  - `PODDeepONet(branch, trunk)`；`forward(latent, bc, features) -> (N, 4)`（单次调用一个 (shape, case)，B=1）；`coefficients(latent, bc) -> (4, rank)`；`save(path)`/`load(path, device)` 类方法（存 branch/trunk kwargs + state_dict）。

- [ ] **Step 1: 写失败测试** `/tmp/test_deeponet.py`

```python
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")
from deep_sdf.cfd.deeponet import BranchNet, PODDeepONet, TrunkNet

L, BC, R, C, N = 16, 4, 12, 4, 257


def make_model():
    branch = BranchNet(L, bc_dim=BC, rank=R, n_outputs=C, hidden=64,
                       num_layers=3, bc_hidden=32)
    trunk = TrunkNet(in_dim=8, rank=R, n_outputs=C, hidden_sizes=(64, 96))
    return PODDeepONet(branch, trunk)


def test_shapes_and_einsum():
    torch.manual_seed(0)
    m = make_model()
    z = torch.randn(1, L)
    bc = torch.tensor([[15.0, 1.0, 0.0, 0.0]])
    feats = torch.randn(N, 8)
    a = m.branch(z, bc)
    assert a.shape == (1, C, R)
    phi = m.trunk.basis(feats)
    assert phi.shape == (C, N, R)
    q = m(z, bc, feats)
    assert q.shape == (N, C)
    # einsum 与显式双重循环一致
    q_loop = m.trunk.bias(feats).clone()
    for c in range(C):
        for n in range(N):
            q_loop[n, c] += torch.dot(a[0, c], phi[c, n])
    assert (q - q_loop).abs().max() < 1e-4


def test_normalization_roundtrip():
    torch.manual_seed(1)
    b = BranchNet(L, bc_dim=BC, rank=R, n_outputs=C)
    b.set_z_normalization(torch.full((L,), 0.5), torch.full((L,), 2.0))
    b.set_bc_normalization(torch.full((BC,), 10.0), torch.full((BC,), 3.0))
    b.set_coef_normalization(torch.full((C, R), 1.5), torch.full((C, R), 0.25))
    assert (b.z_mean - 0.5).abs().max() == 0.0
    assert (b.coef_std - 0.25).abs().max() == 0.0
    z = torch.randn(2, L)
    bc = torch.rand(2, BC) * 20
    phys = b(z, bc)
    norm = b.forward_normalized(z, bc)
    assert (phys - (norm * b.coef_std + b.coef_mean)).abs().max() < 1e-5
    try:
        b.set_z_normalization(torch.zeros(3), torch.ones(3))
        raise AssertionError("size mismatch should raise ValueError")
    except ValueError:
        pass


def test_smooth_derivatives():
    # SiLU trunk：对输入坐标的一阶/二阶导与中心差分一致（float64）
    torch.manual_seed(2)
    m = make_model().double()
    z = torch.randn(1, L, dtype=torch.float64)
    bc = torch.tensor([[15.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    x0 = torch.randn(5, 8, dtype=torch.float64)
    x0[:, 3] = torch.rand(5, dtype=torch.float64) + 0.1  # sdf > 0
    x = x0.clone().requires_grad_(True)
    q = m(z, bc, x)
    eps = 1e-5
    for c in range(q.shape[1]):
        g_auto = torch.autograd.grad(
            q[:, c].sum(), x, create_graph=True)[0]
        for j in range(3):  # 只对 xyz 分量核对
            fd = (m(z, bc, x0 + torch.eye(8, dtype=torch.float64)[j] * eps)[:, c]
                  - m(z, bc, x0 - torch.eye(8, dtype=torch.float64)[j] * eps)[:, c]) / (2 * eps)
            rel = (g_auto[:, j] - fd).abs() / fd.abs().clamp_min(1e-8)
            assert rel.max() < 1e-3, "first-derivative mismatch"
        # 二阶导（x 方向）
        d2_auto = torch.autograd.grad(
            g_auto[:, 0].sum(), x, create_graph=True)[0][:, 0]
        e0 = torch.eye(8, dtype=torch.float64)[0] * eps
        fd2 = (m(z, bc, x0 + e0)[:, c] - 2 * m(z, bc, x0)[:, c]
               + m(z, bc, x0 - e0)[:, c]) / eps ** 2
        rel2 = (d2_auto - fd2).abs() / fd2.abs().clamp_min(1e-6)
        assert rel2.max() < 5e-3, "second-derivative mismatch"


def test_save_load(tmp=None):
    torch.manual_seed(3)
    m = make_model()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pth")
        m.save(p)
        m2 = PODDeepONet.load(p)
        z = torch.randn(1, L)
        bc = torch.rand(1, BC) * 20
        feats = torch.randn(33, 8)
        assert (m(z, bc, feats) - m2(z, bc, feats)).abs().max() == 0.0


def test_single_case_check():
    m = make_model()
    try:
        m(torch.randn(2, L), torch.rand(2, BC), torch.randn(9, 8))
        raise AssertionError("B>1 should raise ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    test_shapes_and_einsum()
    test_normalization_roundtrip()
    test_smooth_derivatives()
    test_save_load()
    test_single_case_check()
    print("test_deeponet: ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_deeponet.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'deep_sdf.cfd.deeponet'`

- [ ] **Step 3: 实现** `deep_sdf/cfd/deeponet.py`

```python
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
```

`deep_sdf/cfd/__init__.py` 追加：

```python
from deep_sdf.cfd.deeponet import (
    BranchNet,
    PODDeepONet,
    TrunkNet,
)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_deeponet.py`
Expected: `test_deeponet: ALL PASS`

- [ ] **Step 5: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add deep_sdf/cfd/deeponet.py deep_sdf/cfd/__init__.py
git commit -m "Add geometry-conditioned POD-DeepONet (branch/trunk/bias, MultiTrunkDeepONet-style)"
```

---

### Task 5: 物理约束模块 `deep_sdf/cfd/physics.py`

**Files:**
- Create: `deep_sdf/cfd/physics.py`
- Modify: `deep_sdf/cfd/__init__.py`（追加导出）
- Test: `/tmp/test_physics.py`

**Interfaces:**
- Consumes: Task 2 的 `potential_flow_field`（真值残差测试用，float64）；sympy 1.13.1；torch autograd。
- Produces:
  - `PDE`（最小基类，`self.equations` dict）；
  - `IncompressibleNS(re=1e4)` → equations `{"continuity", "momentum_u", "momentum_v", "momentum_w"}`（无量纲，Cp 约定）；
  - `PDEInformer(equations, grad_method="autodiff")`，`forward({"coordinates": (N,3) requires_grad, "u"/"v"/"w"/"cp": (N,1)}) -> {name: (N,1)}`（PhysicsNeMo PhysicsInformer 同接口；所有导数 `create_graph=True`，物理 loss 可反传到网络权重）；
  - `fluid_mask(sdf, margin) -> BoolTensor`；`wall_slip_loss(q, normals)`；`noslip_loss(q)`；`farfield_loss(q, u_inf)`；`physics_weight_schedule(progress) -> float`（0/0.01/0.05/0.1 四段）；
  - `FluidMaskEmpty(RuntimeError)`；`CollocationSampler(fractions=(0.3,0.3,0.2,0.2), near_factor=10.0, wake_xi_min=0.3, wake_rho_max=1.0, margin=2.0, grad_quantile=0.9)`，`sample(grid_points, grid_shape, sdf, fields, bc, n_points, generator) -> dict{"collocation", "wall", "far"}`（LongTensor 索引）。

- [ ] **Step 1: 写失败测试** `/tmp/test_physics.py`

```python
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")
from deep_sdf.cfd.flow_synth import potential_flow_field
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


def make_net(seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(3, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 4),
    ).to(dtype)


def hand_residuals(q, pts, re):
    """手写 autodiff 残差（对照组）。"""
    def grad(t):
        return torch.autograd.grad(t.sum(), pts, create_graph=True)[0]
    gu, gv, gw, gp = (grad(q[:, i:i+1]) for i in range(4))
    cont = gu[:, 0:1] + gv[:, 1:2] + gw[:, 2:3]
    moms = []
    for i, (u_i, g_i) in enumerate(zip((0, 1, 2), (gu, gv, gw))):
        conv = (q[:, 0:1] * g_i[:, 0:1] + q[:, 1:2] * g_i[:, 1:2]
                + q[:, 2:3] * g_i[:, 2:3])
        lap = sum(grad(g_i[:, c:c+1])[:, c:c+1] for c in range(3))
        moms.append(conv + 0.5 * gp[:, i:i+1] - (1.0 / re) * lap)
    return {"continuity": cont, "momentum_u": moms[0],
            "momentum_v": moms[1], "momentum_w": moms[2]}


def test_informer_vs_handcoded():
    net = make_net()
    pts = torch.randn(97, 3, dtype=torch.float64).requires_grad_(True)
    q = net(pts)
    pde = IncompressibleNS(re=1e4)
    informer = PDEInformer(pde.equations)
    res = informer({"coordinates": pts, "u": q[:, 0:1], "v": q[:, 1:2],
                    "w": q[:, 2:3], "cp": q[:, 3:4]})
    ref = hand_residuals(q, pts, 1e4)
    for k in ref:
        diff = (res[k] - ref[k]).abs().max().item()
        assert diff < 1e-6, "{} mismatch {}".format(k, diff)
    # 物理 loss 必须能反传到网络权重
    loss = sum((r ** 2).mean() for r in res.values())
    loss.backward()
    assert net[0].weight.grad is not None and net[0].weight.grad.abs().sum() > 0


def test_ground_truth_sphere():
    # 球体势流（wake 关）经 informer 的残差应为 0（float64）
    pts = torch.rand(1500, 3, dtype=torch.float64) * 2 - 1
    keep = pts.norm(dim=1) > 1.05
    pts = pts[keep].detach().requires_grad_(True)
    f = potential_flow_field((1.0, 1.0, 1.0), (0, 0, 0), [10.0, 1, 0, 0], pts,
                             wake_amp=0.0)
    q = torch.cat([f[:, :3] / 10.0, f[:, 3:4]], dim=1)  # 无量纲化
    informer = PDEInformer(IncompressibleNS(re=1e4).equations)
    res = informer({"coordinates": pts, "u": q[:, 0:1], "v": q[:, 1:2],
                    "w": q[:, 2:3], "cp": q[:, 3:4]})
    assert res["continuity"].abs().max() < 1e-8
    for k in ("momentum_u", "momentum_v", "momentum_w"):
        assert res[k].abs().max() < 1e-5, k


def test_losses_and_mask():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0]])
    n = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert wall_slip_loss(q, n).item() == 0.0
    assert noslip_loss(q).item() == 2.0
    assert farfield_loss(q, torch.tensor([1.0, 0.0, 0.0])).item() == 0.5
    sdf = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    assert fluid_mask(sdf, margin=0.1).tolist() == [False, False, True, True]


def test_sampler():
    n = 16
    lin = torch.linspace(-1.5, 1.5, n)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], 1)
    sdf = pts.norm(dim=1) - 0.6  # 球
    fields = torch.rand(n ** 3, 4)
    bc = torch.tensor([15.0, 1.0, 0.0, 0.0])
    sampler = CollocationSampler()
    g1 = torch.Generator().manual_seed(0)
    out1 = sampler.sample(pts, (n, n, n), sdf, fields, bc, 1000, g1)
    assert out1["collocation"].numel() == 1000
    assert set(out1) == {"collocation", "wall", "far"}
    g2 = torch.Generator().manual_seed(0)
    out2 = sampler.sample(pts, (n, n, n), sdf, fields, bc, 1000, g2)
    assert (out1["collocation"] == out2["collocation"]).all()  # 同 seed 可复现
    # collocation 全部位于流体区（sdf > margin*h）
    h = 3.0 / (n - 1)
    assert (sdf[out1["collocation"]] > 2 * h).all()
    # 全在体内 -> FluidMaskEmpty
    try:
        sampler.sample(pts, (n, n, n), torch.full((n ** 3,), -1.0), fields, bc,
                       100, torch.Generator().manual_seed(0))
        raise AssertionError("empty fluid mask should raise FluidMaskEmpty")
    except FluidMaskEmpty:
        pass


def test_schedule():
    assert physics_weight_schedule(0.0) == 0.0
    assert physics_weight_schedule(0.19) == 0.0
    assert physics_weight_schedule(0.2) == 0.01
    assert physics_weight_schedule(0.49) == 0.01
    assert physics_weight_schedule(0.5) == 0.05
    assert physics_weight_schedule(0.79) == 0.05
    assert physics_weight_schedule(0.8) == 0.1
    assert physics_weight_schedule(1.0) == 0.1


if __name__ == "__main__":
    test_informer_vs_handcoded()
    test_ground_truth_sphere()
    test_losses_and_mask()
    test_sampler()
    test_schedule()
    print("test_physics: ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_physics.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'deep_sdf.cfd.physics'`

- [ ] **Step 3: 实现** `deep_sdf/cfd/physics.py`

```python
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

    def _pools(self, grid_points, grid_shape, sdf, fields, bc):
        n = grid_shape[0]
        spacing = float(grid_points[:, 0].max() - grid_points[:, 0].min())
        spacing /= max(n - 1, 1)
        device = grid_points.device
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
        vel = fields[:, :3].reshape(n, n, n, 3)
        gx = torch.zeros_like(vel)
        gy = torch.zeros_like(vel)
        gz = torch.zeros_like(vel)
        gx[1:-1] = (vel[2:] - vel[:-2]) / (2 * spacing)
        gy[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / (2 * spacing)
        gz[:, :, 1:-1] = (vel[:, :, 2:] - vel[:, :, :-2]) / (2 * spacing)
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
               generator):
        """-> {"collocation": (n_points,), "wall": (n_points//8,),
        "far": (n_points//8,)} index tensors."""
        pools = self._pools(grid_points, grid_shape, sdf, fields, bc)
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
```

`deep_sdf/cfd/__init__.py` 追加：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_physics.py`
Expected: `test_physics: ALL PASS`

- [ ] **Step 5: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add deep_sdf/cfd/physics.py deep_sdf/cfd/__init__.py
git commit -m "Add PhysicsInformer-pattern PDE residuals, BC losses and collocation sampler"
```

---

### Task 6: 训练脚本 `train_pipod_deeponet.py`（数据管线 + stage 1）

**Files:**
- Create: `train_pipod_deeponet.py`（根目录）
- Test: `/tmp/test_pipod_stage1.py`

**Interfaces:**
- Consumes: Task 1 快照 IO、Task 2 `potential_flow_field`/`parse_ellipsoid_axes`、Task 4 `BranchNet`、Task 3 的椭球实验（decoder + split）；`train_pressure_surrogate.load_or_fit_latent/make_bc/sample_flow_direction`；`deep_sdf.cfd.volume.pod_fit/PODBasis/make_reference_grid`。
- Produces: `train_pipod_deeponet.py --stage 1` 完整可跑；输出目录 `<experiment>/PipodONet/`：`pod_basis_{u,v,w,p}.pth`（逐变量 POD 基）、`stage1.pth`（best-on-val checkpoint，含 model_kwargs / bc_fields / grid_resolution / rank / val 指标 / seed）。后续任务复用本任务的 `build_shapes()`（含几何特征缓存）、`fit_pod_bases()`、`set_normalizations()`、`evaluate()`。

约定：case dict = `{"bc": (4,), "fields": (G,4) float32 已无量纲, "coef": (4,r) 真值投影系数, "target": (4,r) 标准化系数, "case_id": str}`；shape dict = `{"name", "latent": (1,L), "cases": [...], "sdf": (G,), "sdf_grad": (G,3), "h": float}`（几何特征惰性缓存）。

- [ ] **Step 1: 写失败测试（冒烟）** `/tmp/test_pipod_stage1.py`

```python
import os
import re
import subprocess
import sys

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")
REPO = "/home/siqi/CLionProjects/DeepSDF"
EXP = os.path.join(REPO, "examples/ellipsoids")

CMD = [
    os.path.join(REPO, ".venv/bin/python"),
    os.path.join(REPO, "train_pipod_deeponet.py"),
    "-e", EXP, "-d", os.path.join(REPO, "data/ellipsoids"),
    "-s", os.path.join(EXP, "split.json"),
    "--synthetic", "--stage", "1",
    "--grid_resolution", "16", "--cases_per_shape", "2",
    "--iters", "400", "--lr", "1e-3", "--val_fraction", "0.2",
    "--hidden", "64", "--num_layers", "2", "--seed", "0",
]


def run():
    p = subprocess.run(CMD, cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout + p.stderr


def test_stage1_smoke():
    out = run()
    assert "train projection error" in out  # 每个变量一行（4 次）
    assert out.count("train projection error") == 4
    ckpt = os.path.join(EXP, "PipodONet", "stage1.pth")
    assert os.path.isfile(ckpt)
    import torch
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    for key in ("model_type", "model_state_dict", "model_kwargs", "rank",
                    "pod_basis_files", "val_rel_l2", "seed"):
        assert key in state, key
    assert state["model_type"] == "pipod_deeponet_stage1"
    # 拟合能力 sanity：val rel L2 应明显优于均值场基线（100% 量级）
    assert state["val_rel_l2"] < 0.8 * state["val_mean_field_rel_l2"]


def test_seed_reproducibility():
    out = run()  # 同配置重跑
    m = re.findall(r"iter 0 mse: ([0-9.eE+-]+)", out)
    ckpt = os.path.join(EXP, "PipodONet", "stage1.pth")
    import torch
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert m and state["train_coef_mse"] > 0  # 与上轮逐位一致由重载验证
    out2 = run()
    m2 = re.findall(r"iter 0 mse: ([0-9.eE+-]+)", out2)
    assert m[0] == m2[0], "same seed must reproduce iter-0 loss"


if __name__ == "__main__":
    test_stage1_smoke()
    test_seed_reproducibility()
    print("test_pipod_stage1: ALL PASS")
```

注：该测试依赖 Task 3 的 decoder 已训练完成（`examples/ellipsoids/ModelParameters/latest.pth` 存在）。

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_pipod_stage1.py`
Expected: FAIL，`assert p.returncode == 0`（`train_pipod_deeponet.py` 不存在 → subprocess 报文件不存在或退出码非零）

- [ ] **Step 3: 实现** `train_pipod_deeponet.py`（stage 2/3 的 flag 先声明、入口显式报错，Task 7/8 填实现）

```python
#!/usr/bin/env python3
"""Train the physics-informed POD-DeepONet volume-field operator (design doc
docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md).

--stage 1: branch-only coefficient regression (L_POD);
--stage 2: branch + trunk field training (L_POD + lambda_f * L_field);
--stage 3: physics-informed fine-tuning from a stage-2 checkpoint
           (+ lambda_phys(progress) * (L_c + L_m + L_wall + L_ff)).

Snapshots follow the npz contract of deep_sdf.cfd.volume (fields (G, 4)
[u, v, w, Cp] on the shared reference grid, bc (4,) = [U, dir], shape) and
are nondimensionalized at load (u /= U). POD is fit per variable (cPOD) on
the TRAINING shapes only; the common rank is max of the per-variable
energy-truncated ranks unless --pod_rank fixes it. With --synthetic the
physics-consistent potential-flow fields (deep_sdf.cfd.flow_synth) are
written to <experiment>/PipodONet/snapshots/ and read back through the same
npz path as real data.
"""

import argparse
import glob
import json
import logging
import os
import random
import time

import numpy as np
import torch

import deep_sdf
import deep_sdf.cfd
import deep_sdf.data
import deep_sdf.utils
import deep_sdf.workspace as ws
from deep_sdf.cfd.deeponet import BranchNet
from deep_sdf.cfd.flow_synth import parse_ellipsoid_axes, potential_flow_field
from deep_sdf.cfd.volume import (
    load_snapshot,
    make_reference_grid,
    pod_fit,
    save_snapshot,
    snapshot_filename,
)
from train_pressure_surrogate import (
    load_or_fit_latent,
    make_bc,
    sample_flow_direction,
)

FIELD_NAMES = ["u", "v", "w", "p"]


def build_shapes(args, decoder, latent_size, saved_model_epoch, npz_filenames,
                 grid_points, grid_shape, num_points, snapshots_dir, rng,
                 device):
    """Per shape: latent (loaded or fitted, train_volume_rom convention) +
    per-case snapshots (nondimensionalized at load: velocity /= U)."""
    snap_index = {}
    if args.snapshots:
        for path in sorted(glob.glob(os.path.join(snapshots_dir, "*.npz"))):
            data = np.load(path, allow_pickle=False)
            snap_index.setdefault(str(data["shape"]), []).append(path)

    shapes = []
    for npz in npz_filenames:
        if "npz" not in npz:
            continue
        logging.info("processing {}".format(npz))
        latent = load_or_fit_latent(
            args, decoder, latent_size, saved_model_epoch, npz
        )
        cases = []
        if args.synthetic:
            axes = parse_ellipsoid_axes(npz)
            case_paths = []
            for case_idx in range(args.cases_per_shape):
                direction = sample_flow_direction(rng, args.dir_cone_deg)
                velocity = rng.uniform(args.u_range[0], args.u_range[1])
                bc = make_bc(direction, velocity)
                path = os.path.join(
                    snapshots_dir, snapshot_filename(npz, case_idx))
                regenerate = True
                if os.path.isfile(path):
                    try:
                        old = load_snapshot(path, num_points)
                        regenerate = not (
                            old["shape"] == npz
                            and np.allclose(old["bc"].numpy(), bc, atol=1e-5)
                        )
                    except (ValueError, KeyError):
                        regenerate = True
                if regenerate:
                    fields = potential_flow_field(
                        axes, (0.0, 0.0, 0.0), bc, grid_points,
                        wake_amp=args.wake_amp, wake_sigma=args.wake_sigma,
                    )
                    save_snapshot(path, fields, bc, npz)
                case_paths.append(path)
            for path in case_paths:
                snap = load_snapshot(path, num_points)
                if snap["shape"] != npz:
                    raise RuntimeError(
                        "snapshot {} belongs to shape {}, expected {}".format(
                            path, snap["shape"], npz))
                cases.append({"bc": snap["bc"].to(device),
                              "fields": snap["fields"].to(device),
                              "case_id": os.path.basename(path)})
        else:
            paths = snap_index.get(npz)
            if not paths:
                raise RuntimeError(
                    "no snapshots found for shape {} in {}".format(
                        npz, snapshots_dir))
            for path in paths:
                snap = load_snapshot(path, num_points)
                cases.append({"bc": snap["bc"].to(device),
                              "fields": snap["fields"].to(device),
                              "case_id": os.path.basename(path)})
        for c in cases:
            c["fields"] = c["fields"].clone()
            c["fields"][:, :3] /= c["bc"][0]  # nondimensionalize velocity
        shapes.append({"name": npz, "latent": latent.detach(),
                       "cases": cases})
    if not shapes:
        raise RuntimeError("no shapes produced snapshots; nothing to train on")
    return shapes


def fit_pod_bases(train_cases, energy, rank, device):
    """Per-variable cPOD on the training snapshots. Returns (bases, r) with
    r the common rank (max of energy-truncated per-variable ranks unless
    ``rank`` fixes it)."""
    S = torch.stack([c["fields"] for _, c in train_cases])  # (N, G, 4)
    bases = [pod_fit(S[:, :, v], energy=energy, rank=None, device=device)
             for v in range(4)]
    r = max(b.rank for b in bases) if rank is None else rank
    bases = [pod_fit(S[:, :, v], energy=energy, rank=r, device=device)
             for v in range(4)]
    return bases, r


def set_targets(bases, model_cases):
    """True projection coefficients per case: coef (4, r)."""
    for s, c in model_cases:
        c["coef"] = torch.stack([
            bases[v].project(c["fields"][:, v].unsqueeze(0))[0]
            for v in range(4)
        ]).detach()


def set_normalizations(branch, train_shapes, train_cases, val_cases):
    train_z = torch.cat([s["latent"] for s in train_shapes], 0)
    branch.set_z_normalization(train_z.mean(0), train_z.std(0))
    train_bcs = torch.stack([c["bc"] for _, c in train_cases])
    branch.set_bc_normalization(train_bcs.mean(0), train_bcs.std(0))
    train_coefs = torch.stack([c["coef"] for _, c in train_cases])  # (N,4,r)
    branch.set_coef_normalization(train_coefs.mean(0), train_coefs.std(0))
    # 标准化目标对 train 与 val 案例都要设置（val 评估需要 target）
    for _, c in train_cases + val_cases:
        c["target"] = ((c["coef"] - branch.coef_mean)
                       / branch.coef_std).detach()


def evaluate_stage1(branch, bases, flat_cases):
    """Mean over cases: standardized coef MSE, per-variable reconstructed
    relative L2 (mean), projection lower bound, mean-field baseline."""
    loss_fn = torch.nn.MSELoss()
    coef_mses, rel_l2s, projs, baselines = [], [], [], []
    with torch.no_grad():
        for s, c in flat_cases:
            pred_n = branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0))
            coef_mses.append(loss_fn(pred_n, c["target"].unsqueeze(0)).item())
            a_pred = branch(s["latent"], c["bc"].unsqueeze(0))  # (1,4,r)
            errs, ps, bs = [], [], []
            for v in range(4):
                Y = c["fields"][:, v].unsqueeze(0)
                errs.append(bases[v].relative_error(Y, a_pred[:, v])[0].item())
                ps.append(bases[v].projection_error(Y)[0].item())
                bs.append(bases[v].relative_error(
                    Y, torch.zeros_like(a_pred[:, v]))[0].item())
            rel_l2s.append(float(np.mean(errs)))
            projs.append(float(np.mean(ps)))
            baselines.append(float(np.mean(bs)))
    return {"coef_mse": float(np.mean(coef_mses)),
            "rel_l2": float(np.mean(rel_l2s)),
            "proj": float(np.mean(projs)),
            "baseline": float(np.mean(baselines))}


def train_stage1(args, branch, branch_kwargs, bases, train_shapes,
                 train_cases, val_cases, out_dir, rng):
    optimizer = torch.optim.Adam(branch.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    start = time.time()
    loss_num = 0.0
    best = None
    ckpt_path = os.path.join(out_dir, "stage1.pth")

    def save(val_metrics=None):
        checkpoint = {
            "model_type": "pipod_deeponet_stage1",
            "model_state_dict": branch.state_dict(),
            "model_kwargs": branch_kwargs,
            "rank": bases[0].rank,
            "pod_basis_files": ["pod_basis_{}.pth".format(v)
                                for v in FIELD_NAMES],
            "bc_fields": list(deep_sdf.cfd.BC_FIELDS),
            "grid_resolution": args.grid_resolution,
            "u_range": list(args.u_range),
            "dir_cone_deg": args.dir_cone_deg,
            "wake_amp": args.wake_amp,
            "train_coef_mse": loss_num,
            "seed": args.seed,
        }
        if val_metrics is not None:
            checkpoint.update({
                "val_coef_mse": val_metrics["coef_mse"],
                "val_rel_l2": val_metrics["rel_l2"],
                "val_projection_error": val_metrics["proj"],
                "val_mean_field_rel_l2": val_metrics["baseline"],
            })
        torch.save(checkpoint, ckpt_path)

    for e in range(int(args.iterations)):
        optimizer.zero_grad()
        s, c = rng.choice(train_cases)
        pred = branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0))
        loss = loss_fn(pred, c["target"].unsqueeze(0))
        loss.backward()
        optimizer.step()
        loss_num = loss.item()
        if e % 200 == 0:
            if val_cases:
                metrics = evaluate_stage1(branch, bases, val_cases)
                logging.info(
                    "iter {} mse: {:.6e} | val coef mse: {:.6e} rel L2: "
                    "{:.6e} proj bound: {:.6e} mean-field: {:.6e}".format(
                        e, loss_num, metrics["coef_mse"], metrics["rel_l2"],
                        metrics["proj"], metrics["baseline"]))
                if best is None or metrics["rel_l2"] < best:
                    best = metrics["rel_l2"]
                    save(metrics)
            else:
                logging.info("iter {} mse: {:.6e}".format(e, loss_num))
    if val_cases and best is None:
        metrics = evaluate_stage1(branch, bases, val_cases)
        save(metrics)
    elif not val_cases:
        save()
    logging.info("stage-1 training time: {:.2f}s; saved {}".format(
        time.time() - start, ckpt_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the physics-informed POD-DeepONet volume operator "
        "(staged: 1 coefficients, 2 field, 3 physics fine-tuning)."
    )
    parser.add_argument("--experiment", "-e", dest="experiment_directory",
                        required=True)
    parser.add_argument("--checkpoint", "-c", dest="checkpoint",
                        default="latest")
    parser.add_argument("--data", "-d", dest="data_source", required=True)
    parser.add_argument("--split", "-s", dest="split_filename", required=True)
    parser.add_argument("--snapshots", dest="snapshots", default=None)
    parser.add_argument("--synthetic", dest="synthetic", action="store_true")
    parser.add_argument("--grid_resolution", type=int, default=64)
    parser.add_argument("--pod_energy", type=float, default=0.999)
    parser.add_argument("--pod_rank", type=int, default=None)
    parser.add_argument("--cases_per_shape", type=int, default=4)
    parser.add_argument("--u_range", type=float, nargs=2, default=[10.0, 20.0])
    parser.add_argument("--dir_cone_deg", type=float, default=180.0)
    parser.add_argument("--wake_amp", type=float, default=0.15)
    parser.add_argument("--wake_sigma", type=float, default=0.5)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--init_from", default=None,
                        help="checkpoint to initialize branch (stage 2) or "
                        "the full operator (stage 3) from")
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--bc_hidden", type=int, default=64)
    parser.add_argument("--trunk_hidden", type=int, nargs="+",
                        default=[256, 512])
    parser.add_argument("--lambda_pod", type=float, default=1.0)
    parser.add_argument("--lambda_field", type=float, default=1.0)
    parser.add_argument("--lambda_phys", type=float, default=None,
                        help="fixed physics weight (default: the 0/0.01/0.05/"
                        "0.1 progress schedule of the design doc)")
    parser.add_argument("--re", type=float, default=1e4)
    parser.add_argument("--n_field", type=int, default=16384)
    parser.add_argument("--n_collocation", type=int, default=4096)
    parser.add_argument("--phys_chunk", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--wall_bc", choices=["slip", "noslip"],
                        default="slip")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--reconstruct_iters", dest="reconstruct_iterations",
                        type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    deep_sdf.add_common_args(parser)
    args = parser.parse_args()
    deep_sdf.configure_logging(args)

    if bool(args.snapshots) == bool(args.synthetic):
        raise RuntimeError("pass exactly one of --snapshots <dir> or --synthetic")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs = json.load(open(os.path.join(
        args.experiment_directory, "specs.json")))
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(os.path.join(
        args.experiment_directory, ws.model_params_subdir,
        args.checkpoint + ".pth"))
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda().eval()
    for param in decoder.parameters():
        param.requires_grad = False

    with open(args.split_filename) as f:
        split = json.load(f)
    npz_filenames = deep_sdf.data.get_instance_filenames(
        args.data_source, split)

    device = torch.device("cuda")
    out_dir = os.path.join(args.experiment_directory, "PipodONet")
    os.makedirs(out_dir, exist_ok=True)
    snapshots_dir = (args.snapshots if args.snapshots
                     else os.path.join(out_dir, "snapshots"))
    if args.synthetic:
        os.makedirs(snapshots_dir, exist_ok=True)

    grid_points, grid_shape = make_reference_grid(args.grid_resolution)
    grid_points = grid_points.to(device)
    num_points = grid_points.shape[0]

    rng = random.Random(args.seed)
    shapes = build_shapes(args, decoder, latent_size, saved_model_epoch,
                          npz_filenames, grid_points, grid_shape, num_points,
                          snapshots_dir, rng, device)

    indices = list(range(len(shapes)))
    rng.shuffle(indices)
    num_val = 0
    if args.val_fraction > 0.0 and len(shapes) >= 2:
        num_val = max(1, min(len(shapes) - 1,
                             int(round(args.val_fraction * len(shapes)))))
    val_idx = set(indices[:num_val])
    train_shapes = [s for i, s in enumerate(shapes) if i not in val_idx]
    val_shapes = [s for i, s in enumerate(shapes) if i in val_idx]
    if not val_shapes:
        train_shapes = shapes
    train_cases = [(s, c) for s in train_shapes for c in s["cases"]]
    val_cases = [(s, c) for s in val_shapes for c in s["cases"]]
    logging.info("snapshot ensemble: %d train / %d val cases",
                 len(train_cases), len(val_cases))

    bases, rank = fit_pod_bases(train_cases, args.pod_energy, args.pod_rank,
                                device)
    for v, basis in zip(FIELD_NAMES, bases):
        basis.save(os.path.join(out_dir, "pod_basis_{}.pth".format(v)))
        logging.info(
            "train projection error ({}): {:.6e}".format(
                v, basis.projection_error(torch.stack(
                    [c["fields"][:, FIELD_NAMES.index(v)]
                     for _, c in train_cases])).mean().item()))
    logging.info("common POD rank r = %d", rank)
    set_targets(bases, train_cases + val_cases)

    branch_kwargs = {
        "latent_size": latent_size, "bc_dim": len(deep_sdf.cfd.BC_FIELDS),
        "rank": rank, "n_outputs": 4, "hidden": args.hidden,
        "num_layers": args.num_layers, "bc_hidden": args.bc_hidden,
    }
    branch = BranchNet(**branch_kwargs).to(device)
    set_normalizations(branch, train_shapes, train_cases, val_cases)

    if args.stage == 1:
        train_stage1(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, out_dir, rng)
    else:
        raise SystemExit(
            "stage {} is implemented in a later task of the plan".format(
                args.stage))
```

（stage 2/3 的实现在 Task 7/8 中追加；届时该分发块被替换为三分支调用。）

- [ ] **Step 4: 运行测试确认通过**

前置：Task 3 的 decoder 已训练。先 `nvidia-smi` 确认余量。

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_pipod_stage1.py`
Expected: `test_pipod_stage1: ALL PASS`（16³ 网格 + 400 iter，分钟级）

- [ ] **Step 5: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add train_pipod_deeponet.py
git commit -m "Add PIPOD-DeepONet training script: data pipeline, per-variable POD, stage 1"
```

---

### Task 7: stage 2——trunk 接入 + 场监督（`train_pipod_deeponet.py` 扩展）

**Files:**
- Modify: `train_pipod_deeponet.py`（import 区、`main` 的 stage 分发、新增四个函数）
- Test: `/tmp/test_pipod_stage2.py`

**Interfaces:**
- Consumes: Task 6 的 `build_shapes/fit_pod_bases/set_normalizations`、Task 4 的 `TrunkNet/PODDeepONet`、`deep_sdf.differentiable_mesh.compute_sdf_gradients`、`deep_sdf.utils.decode_sdf`。
- Produces: `build_shape_geometry(decoder, shapes, grid_points, grid_shape, max_batch)`（就地给 shape 加 `sdf (G,)`、`sdf_grad (G,3)`、`h (float)` 缓存）；`make_features(grid_points, shape, idx) -> (n, 8)`；`predict_field(model, shape, bc, grid_points, chunk=2**16) -> (G, 4)`；`train_stage2(...)`；`evaluate_field(model, bases, flat_cases, grid_points) -> dict`（含 per-variable rel L2）。stage-2 checkpoint：`PipodONet/stage2.pth`，`model_type="pipod_deeponet_stage2"`。

- [ ] **Step 1: 写失败测试** `/tmp/test_pipod_stage2.py`

```python
import os
import re
import subprocess
import sys

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")
REPO = "/home/siqi/CLionProjects/DeepSDF"
EXP = os.path.join(REPO, "examples/ellipsoids")
BASE = [
    os.path.join(REPO, ".venv/bin/python"),
    os.path.join(REPO, "train_pipod_deeponet.py"),
    "-e", EXP, "-d", os.path.join(REPO, "data/ellipsoids"),
    "-s", os.path.join(EXP, "split.json"),
    "--synthetic", "--grid_resolution", "16", "--cases_per_shape", "2",
    "--lr", "1e-3", "--val_fraction", "0.2", "--hidden", "64",
    "--num_layers", "2", "--trunk_hidden", "64", "96", "--seed", "0",
]


def run(extra):
    p = subprocess.run(BASE + extra, cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout + p.stderr


def test_stage2_smoke():
    import torch
    run(["--stage", "1", "--iters", "300"])  # 先产出 stage1.pth
    out = run(["--stage", "2", "--iters", "600", "--n_field", "2048",
               "--init_from", os.path.join(EXP, "PipodONet", "stage1.pth")])
    ckpt = os.path.join(EXP, "PipodONet", "stage2.pth")
    assert os.path.isfile(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert state["model_type"] == "pipod_deeponet_stage2"
    for key in ("trunk_kwargs", "branch_kwargs", "val_rel_l2",
                "val_rel_l2_per_var", "val_projection_error",
                "val_mean_field_rel_l2"):
        assert key in state, key
    assert len(state["val_rel_l2_per_var"]) == 4
    # 场重构应明显优于均值场基线
    assert state["val_rel_l2"] < 0.8 * state["val_mean_field_rel_l2"]


def test_seed_reproducibility():
    out1 = run(["--stage", "2", "--iters", "600", "--n_field", "2048",
                "--init_from", os.path.join(EXP, "PipodONet", "stage1.pth")])
    out2 = run(["--stage", "2", "--iters", "600", "--n_field", "2048",
                "--init_from", os.path.join(EXP, "PipodONet", "stage1.pth")])
    m1 = re.findall(r"iter 0 loss: ([0-9.eE+-]+)", out1)
    m2 = re.findall(r"iter 0 loss: ([0-9.eE+-]+)", out2)
    assert m1 and m1 == m2


if __name__ == "__main__":
    test_stage2_smoke()
    test_seed_reproducibility()
    print("test_pipod_stage2: ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_pipod_stage2.py`
Expected: FAIL（stage 2 分支还是 `SystemExit` → returncode != 0）

- [ ] **Step 3: 实现**——`train_pipod_deeponet.py` 追加/修改

import 区追加：

```python
from deep_sdf.cfd.deeponet import PODDeepONet, TrunkNet
from deep_sdf.differentiable_mesh import compute_sdf_gradients
```

`evaluate_stage1` 之后新增：

```python
def build_shape_geometry(decoder, shapes, grid_points, grid_shape,
                         max_batch=2 ** 18):
    """Per-shape detached SDF + SDF-gradient features on the reference grid
    (data-loss branch), cached on the shape dict."""
    spacing = float(grid_points[:, 0].max() - grid_points[:, 0].min())
    spacing /= max(grid_shape[0] - 1, 1)
    for s in shapes:
        if "sdf" in s:
            continue
        sds = []
        with torch.no_grad():
            head = 0
            while head < grid_points.shape[0]:
                chunk = grid_points[head:head + max_batch]
                sds.append(deep_sdf.utils.decode_sdf(
                    decoder, s["latent"], chunk).squeeze(1).float())
                head += max_batch
        s["sdf"] = torch.cat(sds, 0)
        s["sdf_grad"] = compute_sdf_gradients(
            decoder, s["latent"], grid_points, max_batch)
        s["h"] = spacing


def make_features(grid_points, shape, idx):
    """(n, 8) trunk features [x, y, z, sdf, dsdf/dx, dsdf/dy, dsdf/dz, h]."""
    n = idx.numel()
    return torch.cat([
        grid_points[idx],
        shape["sdf"][idx].unsqueeze(1),
        shape["sdf_grad"][idx],
        torch.full((n, 1), shape["h"], device=grid_points.device),
    ], dim=1)


def predict_field(model, shape, bc, grid_points, chunk=2 ** 16):
    """Full-grid field prediction (no_grad, chunked) -> (G, 4)."""
    outs = []
    with torch.no_grad():
        for i in range(0, grid_points.shape[0], chunk):
            idx = torch.arange(i, min(i + chunk, grid_points.shape[0]),
                               device=grid_points.device)
            outs.append(model(shape["latent"], bc.unsqueeze(0),
                              make_features(grid_points, shape, idx)))
    return torch.cat(outs, 0)


def evaluate_field(model, bases, flat_cases, grid_points):
    """Per-variable and mean relative L2 of the full-grid prediction, plus
    the projection lower bound and the mean-field baseline (both from the
    POD bases) and the standardized coefficient MSE."""
    loss_fn = torch.nn.MSELoss()
    rel_l2_v, projs, baselines, coef_mses = [], [], [], []
    for s, c in flat_cases:
        pred = predict_field(model, s, c["bc"], grid_points)
        truth = c["fields"]
        per_var = ((pred - truth).pow(2).sum(0)
                   / truth.pow(2).sum(0).clamp_min(1e-30)).sqrt()
        rel_l2_v.append(per_var.cpu())
        coef_mses.append(loss_fn(
            model.branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0)),
            c["target"].unsqueeze(0)).item())
        ps, bs = [], []
        for v in range(4):
            Y = truth[:, v].unsqueeze(0)
            ps.append(bases[v].projection_error(Y)[0].item())
            bs.append(bases[v].relative_error(
                Y, torch.zeros(1, bases[v].rank, device=Y.device))[0].item())
        projs.append(float(np.mean(ps)))
        baselines.append(float(np.mean(bs)))
    per_var = torch.stack(rel_l2_v).mean(0)
    return {"coef_mse": float(np.mean(coef_mses)),
            "rel_l2": float(per_var.mean()),
            "rel_l2_per_var": [float(x) for x in per_var],
            "proj": float(np.mean(projs)),
            "baseline": float(np.mean(baselines))}


def load_branch_from_checkpoint(branch, path):
    """Initialize branch weights from a stage-1 checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_type") != "pipod_deeponet_stage1":
        raise ValueError("stage-2 --init_from expects a stage-1 checkpoint, "
                         "got {}".format(state.get("model_type")))
    branch.load_state_dict(state["model_state_dict"])


def train_stage2(args, branch, branch_kwargs, bases, train_shapes,
                 train_cases, val_cases, shapes, decoder, grid_points,
                 grid_shape, out_dir, rng, device):
    """Branch + trunk field training: L = lambda_pod * L_POD + lambda_field *
    L_field (per-variable pointwise MSE summed over variables, the
    ChannelwiseMSE convention of mPOD-DeepONet)."""
    if args.init_from:
        load_branch_from_checkpoint(branch, args.init_from)
        logging.info("initialized branch from {}".format(args.init_from))
    trunk = TrunkNet(in_dim=8, rank=branch_kwargs["rank"], n_outputs=4,
                     hidden_sizes=tuple(args.trunk_hidden))
    model = PODDeepONet(branch, trunk).to(device)
    build_shape_geometry(decoder, shapes, grid_points, grid_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(args.seed)
    num_points = grid_points.shape[0]
    start = time.time()
    loss_num = 0.0
    best = None
    ckpt_path = os.path.join(out_dir, "stage2.pth")

    def save(val_metrics=None):
        checkpoint = {
            "model_type": "pipod_deeponet_stage2",
            "model_state_dict": model.state_dict(),
            "branch_kwargs": branch_kwargs,
            "trunk_kwargs": {"in_dim": 8, "rank": branch_kwargs["rank"],
                             "n_outputs": 4,
                             "hidden_sizes": list(args.trunk_hidden)},
            "rank": branch_kwargs["rank"],
            "pod_basis_files": ["pod_basis_{}.pth".format(v)
                                for v in FIELD_NAMES],
            "bc_fields": list(deep_sdf.cfd.BC_FIELDS),
            "grid_resolution": args.grid_resolution,
            "train_loss": loss_num,
            "seed": args.seed,
        }
        if val_metrics is not None:
            checkpoint.update({
                "val_coef_mse": val_metrics["coef_mse"],
                "val_rel_l2": val_metrics["rel_l2"],
                "val_rel_l2_per_var": val_metrics["rel_l2_per_var"],
                "val_projection_error": val_metrics["proj"],
                "val_mean_field_rel_l2": val_metrics["baseline"],
            })
        torch.save(checkpoint, ckpt_path)

    for e in range(int(args.iterations)):
        optimizer.zero_grad()
        s, c = rng.choice(train_cases)
        idx = torch.randint(num_points, (args.n_field,),
                            generator=gen).to(device)
        q = model(s["latent"], c["bc"].unsqueeze(0),
                  make_features(grid_points, s, idx))
        loss_field = ((q - c["fields"][idx]) ** 2).mean(dim=0).sum()
        loss_pod = loss_fn(
            model.branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0)),
            c["target"].unsqueeze(0))
        loss = args.lambda_pod * loss_pod + args.lambda_field * loss_field
        loss.backward()
        optimizer.step()
        loss_num = loss.item()
        if e % 200 == 0:
            if val_cases:
                metrics = evaluate_field(model, bases, val_cases, grid_points)
                logging.info(
                    "iter {} loss: {:.6e} (pod {:.6e} field {:.6e}) | val rel "
                    "L2: {:.6e} per-var {} proj: {:.6e} mean-field: "
                    "{:.6e}".format(e, loss_num, loss_pod.item(),
                                    loss_field.item(), metrics["rel_l2"],
                                    ["%.3f" % v for v in
                                     metrics["rel_l2_per_var"]],
                                    metrics["proj"], metrics["baseline"]))
                if best is None or metrics["rel_l2"] < best:
                    best = metrics["rel_l2"]
                    save(metrics)
            else:
                logging.info("iter {} loss: {:.6e}".format(e, loss_num))
    if val_cases and best is None:
        save(evaluate_field(model, bases, val_cases, grid_points))
    elif not val_cases:
        save()
    logging.info("stage-2 training time: {:.2f}s; saved {}".format(
        time.time() - start, ckpt_path))
```

`main` 的分发块替换为：

```python
    if args.stage == 1:
        train_stage1(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, out_dir, rng)
    elif args.stage == 2:
        train_stage2(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, shapes, decoder, grid_points,
                     grid_shape, out_dir, rng, device)
    else:
        raise SystemExit("stage 3 is implemented in Task 8 of the plan")
```

（注意：`train_stage1` 的签名与调用在本任务统一为带 `branch_kwargs` 参数的形式——若 Task 6 实现时已是闭包引用，本步一并改成显式参数。）

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_pipod_stage2.py`
Expected: `test_pipod_stage2: ALL PASS`

- [ ] **Step 5: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add train_pipod_deeponet.py
git commit -m "Add stage-2 field training (trunk + L_field) to PIPOD-DeepONet"
```

---

### Task 8: stage 3——physics-informed 微调

**Files:**
- Modify: `train_pipod_deeponet.py`（import 区、stage 分发、新增三个函数）
- Test: `/tmp/test_pipod_stage3.py`

**Interfaces:**
- Consumes: Task 5 的 `IncompressibleNS/PDEInformer/CollocationSampler/FluidMaskEmpty/wall_slip_loss/noslip_loss/farfield_loss/physics_weight_schedule`、Task 7 的全部函数。
- Produces: `physics_losses(model, informer, decoder, latent, bc, points, h, max_batch) -> (L_c, L_m)`；`boundary_losses(model, shape, bc, grid_points, idx_wall, idx_far, wall_bc) -> (L_wall, L_far)`；`train_stage3(...)`；`evaluate_physics(...) -> dict`（在 stage-2 指标上增加 `val_continuity`、`val_momentum`、`val_wall`）。stage-3 checkpoint：`PipodONet/stage3.pth`，`model_type="pipod_deeponet_stage3"`。

几何特征微图约定（spec §Global/设计决策）：physics 分支的 **d 经冻结 decoder 带图评估**（d 对 x 的一阶导有物理意义），**∇d 特征 detached**（ReLU decoder 二阶导几乎处处为 0，带图无增益），∇²u 的导数链只经 trunk 的显式 x 输入与 d 路径。

- [ ] **Step 1: 写失败测试** `/tmp/test_pipod_stage3.py`

```python
import os
import re
import subprocess
import sys

sys.path.insert(0, "/home/siqi/CLionProjects/DeepSDF")
REPO = "/home/siqi/CLionProjects/DeepSDF"
EXP = os.path.join(REPO, "examples/ellipsoids")
BASE = [
    os.path.join(REPO, ".venv/bin/python"),
    os.path.join(REPO, "train_pipod_deeponet.py"),
    "-e", EXP, "-d", os.path.join(REPO, "data/ellipsoids"),
    "-s", os.path.join(EXP, "split.json"),
    "--synthetic", "--grid_resolution", "16", "--cases_per_shape", "2",
    "--lr", "1e-3", "--val_fraction", "0.2", "--hidden", "64",
    "--num_layers", "2", "--trunk_hidden", "64", "96", "--seed", "0",
]


def run(extra):
    p = subprocess.run(BASE + extra, cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout + p.stderr


def test_stage3_physics_improves_continuity():
    import torch
    run(["--stage", "1", "--iters", "300"])
    run(["--stage", "2", "--iters", "600", "--n_field", "2048",
         "--init_from", os.path.join(EXP, "PipodONet", "stage1.pth")])
    # lambda_phys 固定 0.05，确保 800 iter 内 physics 全程参与
    out = run(["--stage", "3", "--iters", "800", "--n_field", "2048",
               "--n_collocation", "512", "--phys_chunk", "256",
               "--lambda_phys", "0.05",
               "--init_from", os.path.join(EXP, "PipodONet", "stage2.pth")])
    ckpt = os.path.join(EXP, "PipodONet", "stage3.pth")
    assert os.path.isfile(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert state["model_type"] == "pipod_deeponet_stage3"
    for key in ("val_continuity", "val_momentum", "val_wall", "val_rel_l2"):
        assert key in state, key
    # 训练日志中 val 连续性残差应下降（首末对比）
    cont = [float(x) for x in
            re.findall(r"val cont: ([0-9.eE+-]+)", out)]
    assert len(cont) >= 2, out[-2000:]
    assert cont[-1] < cont[0], \
        "continuity residual did not decrease: {}".format(cont)
    # 场精度不显著劣化（容忍 20%）
    s2 = torch.load(os.path.join(EXP, "PipodONet", "stage2.pth"),
                    map_location="cpu", weights_only=True)
    assert state["val_rel_l2"] < 1.2 * s2["val_rel_l2"]


if __name__ == "__main__":
    test_stage3_physics_improves_continuity()
    print("test_pipod_stage3: ALL PASS")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_pipod_stage3.py`
Expected: FAIL（stage 3 分支还是 `SystemExit`）

- [ ] **Step 3: 实现**——`train_pipod_deeponet.py` 追加/修改

import 区追加：

```python
from deep_sdf.cfd.physics import (
    CollocationSampler,
    FluidMaskEmpty,
    IncompressibleNS,
    PDEInformer,
    farfield_loss,
    noslip_loss,
    physics_weight_schedule,
    wall_slip_loss,
)
```

`train_stage2` 之后新增：

```python
def physics_losses(model, informer, decoder, latent, bc, points, h,
                   max_batch):
    """Continuity + momentum residuals at collocation points (second-order
    autodiff). d is evaluated through the frozen decoder WITH the graph
    attached; grad d is detached (the ReLU decoder's second derivative
    vanishes a.e.). Losses accumulate over chunks, normalized by count."""
    device = points.device
    lc = torch.zeros((), device=device)
    lm = torch.zeros((), device=device)
    n_tot = 0
    for chunk in points.split(max_batch):
        p = chunk.detach().requires_grad_(True)
        with torch.enable_grad():
            d = deep_sdf.utils.decode_sdf(decoder, latent, p)
            g = torch.autograd.grad(d.sum(), p, create_graph=True)[0]
            feats = torch.cat(
                [p, d, g.detach(), torch.full_like(d, h)], dim=1)
            q = model(latent, bc, feats)
            res = informer({"coordinates": p, "u": q[:, 0:1], "v": q[:, 1:2],
                            "w": q[:, 2:3], "cp": q[:, 3:4]})
            lc = lc + (res["continuity"] ** 2).sum()
            lm = lm + sum((res["momentum_" + k] ** 2).sum() for k in "uvw")
        n_tot += p.shape[0]
    return lc / max(n_tot, 1), lm / max(n_tot, 1)


def boundary_losses(model, shape, bc, grid_points, idx_wall, idx_far,
                    wall_bc):
    """Wall loss on the near-wall band (slip: (u.n)^2 with n from grad SDF;
    noslip: |u|^2) + far-field loss (u -> unit flow direction)."""
    l_wall = torch.zeros((), device=grid_points.device)
    l_far = torch.zeros((), device=grid_points.device)
    if idx_wall.numel():
        q_w = model(shape["latent"], bc.unsqueeze(0),
                    make_features(grid_points, shape, idx_wall))
        if wall_bc == "slip":
            n_w = shape["sdf_grad"][idx_wall]
            n_w = n_w / n_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
            l_wall = wall_slip_loss(q_w, n_w)
        else:
            l_wall = noslip_loss(q_w)
    if idx_far.numel():
        q_f = model(shape["latent"], bc.unsqueeze(0),
                    make_features(grid_points, shape, idx_far))
        d = bc[1:4]
        l_far = farfield_loss(q_f, d / d.norm().clamp_min(1e-12))
    return l_wall, l_far


def load_operator_from_checkpoint(model, path):
    """Initialize the full PODDeepONet from a stage-2 checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_type") != "pipod_deeponet_stage2":
        raise ValueError("stage-3 --init_from expects a stage-2 checkpoint, "
                         "got {}".format(state.get("model_type")))
    model.load_state_dict(state["model_state_dict"])


def evaluate_physics(model, informer, decoder, bases, val_cases, grid_points,
                     grid_shape, sampler, args, device):
    """evaluate_field + physics residuals on a fixed-seed val collocation
    set + wall violation."""
    metrics = evaluate_field(model, bases, val_cases, grid_points)
    conts, moms, walls = [], [], []
    gen = torch.Generator().manual_seed(args.seed + 12345)  # 固定评估集
    for s, c in val_cases:
        try:
            picks = sampler.sample(grid_points, grid_shape, s["sdf"],
                                   c["fields"], c["bc"], args.n_collocation,
                                   gen)
        except FluidMaskEmpty:
            continue
        with torch.enable_grad():
            lc, lm = physics_losses(model, informer, decoder, s["latent"],
                                    c["bc"].unsqueeze(0),
                                    grid_points[picks["collocation"]], s["h"],
                                    args.phys_chunk)
        conts.append(lc.sqrt().item())
        moms.append(lm.sqrt().item())
        if picks["wall"].numel():
            with torch.no_grad():
                q_w = model(s["latent"], c["bc"].unsqueeze(0),
                            make_features(grid_points, s, picks["wall"]))
                n_w = s["sdf_grad"][picks["wall"]]
                n_w = n_w / n_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
                walls.append(
                    ((q_w[:, :3] * n_w).sum(1).abs()
                     / q_w[:, :3].norm(dim=1).clamp_min(1e-12)
                     ).mean().item())
    metrics["continuity"] = float(np.mean(conts)) if conts else float("nan")
    metrics["momentum"] = float(np.mean(moms)) if moms else float("nan")
    metrics["wall"] = float(np.mean(walls)) if walls else float("nan")
    return metrics


def train_stage3(args, branch, branch_kwargs, bases, train_shapes,
                 train_cases, val_cases, shapes, decoder, grid_points,
                 grid_shape, out_dir, rng, device):
    """Physics-informed fine-tuning from a stage-2 checkpoint:
    L = lambda_pod L_POD + lambda_field L_field
        + lambda_phys(progress) * (L_c + L_m + L_wall + L_ff)."""
    trunk = TrunkNet(in_dim=8, rank=branch_kwargs["rank"], n_outputs=4,
                     hidden_sizes=tuple(args.trunk_hidden))
    model = PODDeepONet(branch, trunk).to(device)
    if not args.init_from:
        raise SystemExit("stage 3 requires --init_from <stage2.pth>")
    load_operator_from_checkpoint(model, args.init_from)
    logging.info("initialized operator from {}".format(args.init_from))
    build_shape_geometry(decoder, shapes, grid_points, grid_shape)
    informer = PDEInformer(IncompressibleNS(re=args.re).equations)
    sampler = CollocationSampler(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(args.seed)
    num_points = grid_points.shape[0]
    start = time.time()
    loss_num = 0.0
    best = None
    ckpt_path = os.path.join(out_dir, "stage3.pth")

    def save(val_metrics=None):
        checkpoint = {
            "model_type": "pipod_deeponet_stage3",
            "model_state_dict": model.state_dict(),
            "branch_kwargs": branch_kwargs,
            "trunk_kwargs": {"in_dim": 8, "rank": branch_kwargs["rank"],
                             "n_outputs": 4,
                             "hidden_sizes": list(args.trunk_hidden)},
            "rank": branch_kwargs["rank"],
            "re": args.re,
            "pod_basis_files": ["pod_basis_{}.pth".format(v)
                                for v in FIELD_NAMES],
            "bc_fields": list(deep_sdf.cfd.BC_FIELDS),
            "grid_resolution": args.grid_resolution,
            "train_loss": loss_num,
            "seed": args.seed,
        }
        if val_metrics is not None:
            checkpoint.update({
                "val_rel_l2": val_metrics["rel_l2"],
                "val_rel_l2_per_var": val_metrics["rel_l2_per_var"],
                "val_projection_error": val_metrics["proj"],
                "val_mean_field_rel_l2": val_metrics["baseline"],
                "val_continuity": val_metrics["continuity"],
                "val_momentum": val_metrics["momentum"],
                "val_wall": val_metrics["wall"],
            })
        torch.save(checkpoint, ckpt_path)

    for e in range(int(args.iterations)):
        progress = e / max(int(args.iterations), 1)
        lam = (args.lambda_phys if args.lambda_phys is not None
               else physics_weight_schedule(progress))
        optimizer.zero_grad()
        s, c = rng.choice(train_cases)
        idx = torch.randint(num_points, (args.n_field,),
                            generator=gen).to(device)
        q = model(s["latent"], c["bc"].unsqueeze(0),
                  make_features(grid_points, s, idx))
        loss_field = ((q - c["fields"][idx]) ** 2).mean(dim=0).sum()
        loss_pod = loss_fn(
            model.branch.forward_normalized(s["latent"], c["bc"].unsqueeze(0)),
            c["target"].unsqueeze(0))
        loss = args.lambda_pod * loss_pod + args.lambda_field * loss_field
        phys_terms = {}
        if lam > 0.0:
            try:
                picks = sampler.sample(grid_points, grid_shape, s["sdf"],
                                       c["fields"], c["bc"],
                                       args.n_collocation, gen)
            except FluidMaskEmpty:
                logging.warning("case {} has no fluid points; skipping "
                                "physics".format(c["case_id"]))
                picks = None
            if picks is not None:
                lc, lm = physics_losses(
                    model, informer, decoder, s["latent"],
                    c["bc"].unsqueeze(0),
                    grid_points[picks["collocation"]], s["h"],
                    args.phys_chunk)
                lw, lf = boundary_losses(model, s, c["bc"], grid_points,
                                         picks["wall"], picks["far"],
                                         args.wall_bc)
                loss = loss + lam * (lc + lm + lw + lf)
                phys_terms = {"cont": lc.item(), "mom": lm.item(),
                              "wall": lw.item(), "far": lf.item()}
        loss.backward()
        optimizer.step()
        loss_num = loss.item()
        if e % 200 == 0:
            if val_cases:
                metrics = evaluate_physics(model, informer, decoder, bases,
                                           val_cases, grid_points,
                                           grid_shape, sampler, args, device)
                logging.info(
                    "iter {} loss: {:.6e} lam: {:.3g} {} | val rel L2: "
                    "{:.6e} val cont: {:.6e} mom: {:.6e} wall: "
                    "{:.6e}".format(e, loss_num, lam,
                                    " ".join("{}={:.2e}".format(k, v)
                                             for k, v in phys_terms.items()),
                                    metrics["rel_l2"], metrics["continuity"],
                                    metrics["momentum"], metrics["wall"]))
                if best is None or metrics["rel_l2"] < best:
                    best = metrics["rel_l2"]
                    save(metrics)
            else:
                logging.info("iter {} loss: {:.6e} lam: {:.3g}".format(
                    e, loss_num, lam))
    if val_cases and best is None:
        save(evaluate_physics(model, informer, decoder, bases, val_cases,
                              grid_points, grid_shape, sampler, args, device))
    elif not val_cases:
        save()
    logging.info("stage-3 training time: {:.2f}s; saved {}".format(
        time.time() - start, ckpt_path))
```

`main` 的分发块替换为：

```python
    if args.stage == 1:
        train_stage1(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, out_dir, rng)
    elif args.stage == 2:
        train_stage2(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, shapes, decoder, grid_points,
                     grid_shape, out_dir, rng, device)
    else:
        train_stage3(args, branch, branch_kwargs, bases, train_shapes,
                     train_cases, val_cases, shapes, decoder, grid_points,
                     grid_shape, out_dir, rng, device)
```

注意：stage 3 的 best-on-val 仍以场误差（val_rel_l2）为准——physics 微调的首要目标是场精度不退化前提下的物理一致性提升（spec §7 成功判据）。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/siqi/CLionProjects/DeepSDF && .venv/bin/python /tmp/test_pipod_stage3.py`
Expected: `test_pipod_stage3: ALL PASS`

- [ ] **Step 5: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add train_pipod_deeponet.py
git commit -m "Add stage-3 physics-informed fine-tuning (continuity/momentum/wall/far-field)"
```

---

### Task 9: 端到端验证（64³ 全量三阶段）+ 文档更新

**Files:**
- Modify: `DEEPMESH.md`（新增 PIPOD+DeepONet 一节 + 文件结构更新）

**Interfaces:**
- Consumes: Task 1–8 全部产物。

- [ ] **Step 1: 全量三阶段训练**（先 `nvidia-smi` 确认 GPU 余量；每个 stage 依序执行）

```bash
cd /home/siqi/CLionProjects/DeepSDF
.venv/bin/python train_pipod_deeponet.py -e examples/ellipsoids -d data/ellipsoids \
    -s examples/ellipsoids/split.json --synthetic --stage 1 \
    --grid_resolution 64 --cases_per_shape 4 --u_range 10 20 --dir_cone_deg 180 \
    --iters 20000 --lr 1e-3 --val_fraction 0.2 --seed 0
.venv/bin/python train_pipod_deeponet.py -e examples/ellipsoids -d data/ellipsoids \
    -s examples/ellipsoids/split.json --synthetic --stage 2 \
    --grid_resolution 64 --cases_per_shape 4 --u_range 10 20 --dir_cone_deg 180 \
    --iters 20000 --lr 1e-3 --val_fraction 0.2 --seed 0 \
    --init_from examples/ellipsoids/PipodONet/stage1.pth
.venv/bin/python train_pipod_deeponet.py -e examples/ellipsoids -d data/ellipsoids \
    -s examples/ellipsoids/split.json --synthetic --stage 3 \
    --grid_resolution 64 --cases_per_shape 4 --u_range 10 20 --dir_cone_deg 180 \
    --iters 20000 --lr 1e-4 --val_fraction 0.2 --seed 0 \
    --init_from examples/ellipsoids/PipodONet/stage2.pth
```

（同 seed 下三次运行的合成快照经 bc 匹配检查自动复用；stage 3 学习率降到 1e-4 做微调。）

- [ ] **Step 2: 成功判据核对**（spec §7）

从三个 checkpoint/日志提取并对照：

- stage-2：val rel L2（逐变量 + 均值）显著优于 mean-field 基线、并向 proj 下界靠拢（记录数值，不作硬性阈值；与旧 ROM 在旧合成场的 30.7% vs 6.6% 不直接可比）；
- stage-3：val continuity/momentum 残差较 stage-2 同点评估明显下降，val rel L2 变化在 ±5% 内；
- 记录单次前向耗时（`predict_field` 全网格计时）。

- [ ] **Step 3: 全部 /tmp 测试回归**

```bash
cd /home/siqi/CLionProjects/DeepSDF
for t in test_snapshot_io test_flow_synth test_deeponet test_physics \
         test_pipod_stage1 test_pipod_stage2 test_pipod_stage3; do
    .venv/bin/python /tmp/$t.py || echo "FAIL: $t"
done
```

Expected: 全部 ALL PASS。

- [ ] **Step 4: 更新 DEEPMESH.md**（新增一节，含：架构公式、三阶段流程与命令、stage-1/2/3 验证数值表、physics 真值基准（球体残差≈0）、与参考实现的对照说明、文件结构更新；风格与既有章节一致）

- [ ] **Step 5: Commit**

```bash
cd /home/siqi/CLionProjects/DeepSDF
git add DEEPMESH.md
git commit -m "Document PIPOD-DeepONet volume operator (stages 1-3) with validation results"
```

---

## 追加：OpenFOAM 真实数据线（Task 10–12，2026-09-12 用户指令）

用户指令：OpenFOAM 14 已安装（`source /opt/openfoam14/etc/bashrc`；20 核 CPU / 62GB RAM）。构造简单外流场案例训练测试：**bounding box ≥ 10× 最长轴**；**snappyHexMesh 创建边界层网格**；**基于椭球 latent code 做 Latin Hypercube Sampling 生成小训练样本集并快速 CFD**。流态裁定：**层流 simpleFoam**（ν=0.1，Re≈U·L/ν≈270——层流速度场严格散度自由、NS 动量与 physics loss 同构，无 Reynolds 应力项）。规模裁定：**先 2 案例端到端验证管道，再 60 LHS 形状 × 4 BC = 240 案例**。

与合成场线的关系：追加对照；合成场 e2e（stage 1/2/3）照常完成并如实记录（含 stage-2 在 64³ 上 val rel L2 0.934 ≈ 均值基线的弱泛化结果——数据量瓶颈正是 LHS 数据线要解决的）。

### Task 10: OpenFOAM 案例管线（`deep_sdf/cfd/openfoam_runner.py` + `generate_openfoam_snapshots.py`）+ 2 案例端到端验证

**Files:**
- Create: `deep_sdf/cfd/openfoam_runner.py`
- Create: `generate_openfoam_snapshots.py`（根目录）
- Test: `/tmp/test_openfoam_pipeline.py`

**Interfaces:**
- Consumes: `deep_sdf.cfd.volume.save_snapshot/load_snapshot/make_reference_grid`、`deep_sdf.cfd.flow_synth.parse_ellipsoid_axes`、`deep_sdf.cfd.labels.export_stl`、`deep_sdf.differentiable_mesh.extract_differentiable_mesh`（latent→STL 路径）、trimesh（.venv 已有）。
- Produces:
  - `write_stl_from_geometry(path, axes=None, decoder=None, latent=None, resolution=63)`：椭球走 trimesh icosphere(subdivisions=3) 按 axes 缩放（解析精确）；latent 走 decoder→提取→导出；
  - `make_case(case_dir, stl_path, bc, nu=0.1, domain_half=18.0, n_base=36, layers=6, layer_ratio=1.25, first_layer=0.011, surface_level=3, end_time=2000, residual_p=1e-5, residual_u=1e-6)`：写完整 OpenFOAM 14 case；
  - `run_case(case_dir)`：blockMesh → snappyHexMesh → simpleFoam（OF14 下若 simpleFoam 兼容封装不可用则用 `foamRun -solver incompressibleFluid`）→ 末端采样；
  - `sample_to_snapshot(case_dir, grid_points, shape_name, bc, out_path)`：在 64³ 参考网格点采样 (U,p)，Cp = p/(0.5U²)，体内（SDF<0）u=0，写 npz 契约。

**案例设置（需求硬值）：**

- 域：`[-D, D]³`，D = 18.0（≥10× 最长全轴 1.8）；外边界全部 freestream（`freestreamVelocity`/`freestreamPressure`，值 = U·dir——任意来流方向无需旋转几何）；
- 壁面 noSlip（粘性数据；训练侧对应 `--wall_bc noslip`）；
- blockMesh 基础网格 36³（1 单元/单位长）；snappyHexMesh：表面加密 level 3（单元 ~0.125）、尾迹 refinementBox（下游延伸到 x+6，level 2）、snap、**addLayers 6 层 expansionRatio 1.25 首层 ~0.011**（层流边界层 δ≈L/√Re≈0.11 内有 6+ 单元）；
- 求解：`simulationType laminar`（OF14 为 constant/momentumTransport），SIMPLE + consistent，relaxation U 0.7 / p 0.3，GAMG 求解 p；endTime 2000 迭代或残差达标即止；
- 采样：probes/sampledSet cloud 在参考网格全部 G=262144 点（C-order，x 慢 z 快）取 (U, p)，点序即输入序；
- 单位制与坐标：全部在 DeepSDF 归一化坐标系（几何 ~[-1,1]³，域 [-18,18]³）。

**验证（2 案例端到端，/tmp/test_openfoam_pipeline.py）：**

1. 椭球 a=0.9,b=0.7,c=0.5，bc=[15, 1, 0, 0]：checkMesh 通过；simpleFoam 残差达标收敛；采样文件 262144 点齐全；
2. 远场点（|x|>12）|u−U·dir|/U < 5%（域足够大、阻塞可忽略）；驻点附近 max Cp ∈ [0.8, 1.3]（层流势流近似量级）；尾迹区有速度亏损；
3. npz 经 load_snapshot 校验通过；fields (G,4) float32；
4. 第二案例不同方向（bc=[12, 0.48, 0.64, 0.6] 归一）验证 freestream 任意方向。

**Step 结构**：TDD（先写契约级测试：npz 格式/掩码/函数签名，管道函数先用 dry-run 桩过测试，再真跑 2 案例验收上述物理判据）→ commit（只 add 两个新文件）。

### Task 11: LHS 形状采样 + 全量 240 案例

**Files:**
- Modify: `generate_openfoam_snapshots.py`（加 `--lhs N` 模式）
- Test: `/tmp/test_lhs_shapes.py`

**要点：**
- 从 `examples/ellipsoids/LatentCodes/latest.pth` 读 27 个训练 latent（16 维），逐维取 [min, max] 外扩 10% 为 LHS 边界；`--lhs 60` 采样 60 个 z（numpy LHS 实现，`--seed` 可复现）；
- 每个 z：decoder → SDF → extract_differentiable_mesh（resolution 63）→ 合法性检查（面数 > 500、顶点范围在 [-1.2, 1.2]³ 内、法向一致）→ STL；不合格样本跳过并记录；
- 形状命名：`lhs/shape_XXX.npz` 风格占位——注意 npz 契约的 shape 字段须能被训练侧映射到 latent：快照 npz 的 `shape` 直接存 latent 文件相对路径不行（split 中没有）；处理：`--snapshots` 模式下训练脚本目前按 split 名字索引 latent。**裁定**：为 LHS 形状生成 manifest `lhs_latents.npz`（shape 名 → z 向量），训练脚本加 `--latent_manifest` 覆盖 load_or_fit_latent（新增小功能，含测试）；
- 并发：4 路并行案例（各串行 simpleFoam），240 案例预计 4–8h；失败案例（网格/收敛）跳过并汇总报告（成功率应 > 90%）。

### Task 12: 真实数据三阶段训练 + 对比 + 文档

**要点：**
- `train_pipod_deeponet.py --snapshots data/openfoam/ellipsoids/snapshots --latent_manifest ... --stage 1/2/3 --wall_bc noslip --re 270 --grid_resolution 64`（层流 Re=270 与 CFD 一致；λ 日程不变）；
- 与合成场结果并排对比（val rel L2 / 物理残差 / 前向耗时）；DEEPMESH.md 更新 OpenFOAM 数据线一节（含网格/边界层参数、收敛统计、成功率）；
- commit 只含文档与（如有）训练脚本的 latent_manifest 小功能。

# DeepSDF + PhysicsNeMo DeepONet MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按路线图 §63/§38 净室实现 DeepSDF latent → PhysicsNeMo xDeepONet → 体积流场 [u,v,w,Cp] 的 surrogate，在 ellipsoids_u10 数据（281 形状、固定 BC）上训练评估。

**Architecture:** 冻结 DeepSDF decoder（16 维 latent）提供 SDF/法向特征；physicsnemo `FullyConnected` branch（[z,bc] 20→512×4→256）+ trunk（[x,y,z,sdf,n]+Fourier 43→512×6→256）+ `xDeepONet DeepONet`（Hadamard + mlp decoder 128×2 → 4 通道）。latent k-means 簇级几何划分 70/15/15；channelwise MSE（标准化空间）；best-on-val。

**Tech Stack:** torch 2.5.1+cu121（不动）、vendored PhysicsNeMo 2.3.0a0（sys.path 引用，源码不改）、scipy（kmeans2）、numpy。规格：`docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md`。

## Global Constraints

- **不升级 torch**（保持 2.5.1+cu121）；**不修改** `third-party/physicsnemo/` 任何文件；任何 `import physicsnemo...` 之前必须先 `import deep_sdf.cfd.physicsnemo_compat`。
- 仓库**无 pytest 测试体系**（既定约定：用可执行探针 `python -c` / 训练器 `--smoke` 模式验证）。每个 Task 的验证步骤给出确切命令与期望输出，不新增测试框架。
- Python 一律用 `.venv/bin/python`；工作目录 `/home/siqi/CLionProjects/DeepSDF`。
- 训练产物（`examples/ellipsoids/RoadmapONet*/`、`data/openfoam/ellipsoids_u10/sdf_cache/`）**不入 git**；只提交代码文件。
- 关键数据事实（实现者须知）：
  - manifest：`data/openfoam/ellipsoids_u10/lhs_latents.npz`（`names (281,)` str、`latents (281,16)` f32；名字形如 `ellipsoids/ellipsoid/ellipsoid_a0.5_b0.5_c0.5.npz` 与 `lhs/shape_XXX.npz`）。
  - 快照：`snapshots/*.npz`（282 个文件，含 1 个 `_of4_view` 须排除），内部 `shape` 字段 == manifest 名；`fields (1442897,4)` f32 = **[u_raw(m/s)×3, Cp]**（体内点已置 u=0/Cp=1，无哨兵）；加载时 `fields[:,:3] /= bc[0]` 无量纲化。
  - 参考网格：`deep_sdf.cfd.volume.make_stretched_grid()` 默认参数 → `grid_points (1442897,3)` f32，**域 [-1.5,1.5]³**（不是 [-9,9]³）；`domain_half=1.5`。
  - decoder：specs 在 `examples/ellipsoids_of4/specs.json`（CodeLength=16，dims=[128]×4，latent_in=[2]，use_tanh=false，但 forward 末尾恒有 `self.th=nn.Tanh()` → 输出∈(-1,1)，近壁带阈值 0.15 作用在该 tanh 空间）；权重 `examples/ellipsoids/ModelParameters/latest.pth`，state_dict 键带 `module.` 前缀（训练时 DataParallel 保存）。
  - xDeepONet core 模式语义：`model(x_branch (B,20), x_trunk (T,43)) -> (B,T,4)`，**同一组 trunk 查询点施加于 batch 内每个 branch**；各 case 查询点不同（SDF/法向不同），故前向必须**逐 case**（B=1）调用。

---

### Task 1: PhysicsNeMo 兼容 shim

**Files:**
- Create: `deep_sdf/cfd/physicsnemo_compat.py`

**Interfaces:**
- Produces: 副作用模块。导入后 `physicsnemo` 可 import。后续所有任务依赖。

- [ ] **Step 1: 写 shim 模块**

```python
#!/usr/bin/env python3
"""Runtime compatibility shim for the vendored PhysicsNeMo 2.3.0a0 tree on
torch 2.5 (spec
docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md
section 3). Import this module before any ``import physicsnemo...``.

Effects: (1) inserts ``third-party/physicsnemo`` into ``sys.path``;
(2) aliases ``torch.Tag.cudagraph_unsafe`` (absent before torch 2.10) to
``nondeterministic_bitwise`` so module-level attribute reads succeed;
(3) wraps ``torch.library.custom_op`` to drop the ``tags`` kwarg that
torch 2.5 does not accept. The vendored source stays unmodified.
"""

import inspect
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PHYSICSNEMO = os.path.join(_REPO_ROOT, "third-party", "physicsnemo")
if os.path.isdir(_PHYSICSNEMO) and _PHYSICSNEMO not in sys.path:
    sys.path.insert(0, _PHYSICSNEMO)

if not hasattr(torch.Tag, "cudagraph_unsafe"):
    torch.Tag.cudagraph_unsafe = torch.Tag.nondeterministic_bitwise

if "tags" not in inspect.signature(torch.library.custom_op).parameters:
    _orig_custom_op = torch.library.custom_op

    def _custom_op_drop_tags(*args, **kwargs):
        kwargs.pop("tags", None)
        return _orig_custom_op(*args, **kwargs)

    torch.library.custom_op = _custom_op_drop_tags
```

- [ ] **Step 2: 验证导入与 GPU 前向**

Run:
```bash
.venv/bin/python - <<'EOF'
import deep_sdf.cfd.physicsnemo_compat  # noqa: F401
import torch
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
b = FullyConnected(in_features=20, layer_size=64, num_layers=2, out_features=32)
t = FullyConnected(in_features=43, layer_size=64, num_layers=2, out_features=32)
m = DeepONet(b, trunk=t, dimension=3, width=32, out_channels=4,
             decoder_type="mlp", decoder_width=32, decoder_layers=1).cuda()
y = m(torch.randn(2, 20, device="cuda"), torch.randn(128, 43, device="cuda"))
y.square().mean().backward()
assert y.shape == (2, 128, 4), y.shape
print("SHIM-OK", tuple(y.shape))
EOF
```
Expected: 末尾打印 `SHIM-OK (2, 128, 4)`（ExperimentalFeatureWarning 属正常）。

- [ ] **Step 3: Commit**

```bash
git add deep_sdf/cfd/physicsnemo_compat.py
git commit -m "Add PhysicsNeMo torch-2.5 compat shim (sys.path + Tag/custom_op patches)"
```

---

### Task 2: 模型构建与特征编码（roadmap_deeponet.py 第一部分）

**Files:**
- Create: `deep_sdf/cfd/roadmap_deeponet.py`（本任务只含配置/特征/模型构建，后续任务追加数据与评估）

**Interfaces:**
- Consumes: Task 1 的 shim。
- Produces:
  - `DEFAULT_CFG: dict`（全部超参，键名见代码）
  - `fourier_encode(x: Tensor (...,3), n_bands: int) -> Tensor (..., 3*2*n_bands)`
  - `trunk_features(xyz (N,3), sdf (N,1), normal (N,3), n_bands int, domain_half float) -> Tensor (N,43)`（bands=6 时）
  - `build_model(cfg: dict) -> DeepONet`（cuda() 之前返回 CPU 模型）
  - `predict_normalized(model, latent (16,), bc (4,), xyz (N,3), sdf (N,1), normal (N,3), stats dict, cfg, amp bool) -> Tensor (N,4)`（标准化空间预测）

- [ ] **Step 1: 写模块**

```python
#!/usr/bin/env python3
"""DeepSDF + PhysicsNeMo DeepONet MVP library (roadmap section 63; spec
docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md).

Model: [z(16), bc(4)] -> branch; [x,y,z,sdf,normal] + Fourier -> trunk;
xDeepONet Hadamard + mlp decoder -> [u, v, w, Cp] (normalized space).
Data/geometry/split/eval helpers are appended by later tasks.
"""

import numpy as np
import torch

from deep_sdf.cfd import physicsnemo_compat as _physicsnemo_compat  # noqa: F401

DEFAULT_CFG = {
    # model
    "latent_size": 16, "bc_dim": 4, "width": 256,
    "branch_hidden": 512, "branch_layers": 4,
    "trunk_hidden": 512, "trunk_layers": 6,
    "decoder_hidden": 128, "decoder_layers": 2,
    "fourier_bands": 6, "domain_half": 1.5,
    # geometry cache / sampling
    "near_band": 0.15, "near_frac": 0.3,
    "batch_cases": 4, "n_points": 8192,
    # training
    "iters": 20000, "lr": 1e-3, "lr_min": 1e-5, "weight_decay": 1e-5,
    "amp": True, "eval_every": 500, "metrics_every": 50,
    "eval_points": 16384, "final_eval_points": 65536,
    # split / stats
    "n_clusters": 12, "seed": 0, "stats_sample": 4096,
}

OUT_VARS = ("u", "v", "w", "p")


def fourier_encode(x, n_bands):
    """x (..., d) -> (..., d * 2 * n_bands): [sin(2^k pi x), cos(2^k pi x)]
    concatenated per band pair (roadmap section 21)."""
    outs = []
    for k in range(n_bands):
        w = (2.0 ** k) * np.pi
        outs.append(torch.sin(w * x))
        outs.append(torch.cos(w * x))
    return torch.cat(outs, dim=-1)


def trunk_features(xyz, sdf, normal, n_bands, domain_half):
    """(N,3),(N,1),(N,3) -> (N, 3+1+3+3*2*n_bands): normalized coords,
    raw (tanh-space) SDF, unit normal, Fourier features of the coords."""
    xn = xyz / domain_half
    return torch.cat([xn, sdf, normal, fourier_encode(xn, n_bands)], dim=-1)


def build_model(cfg):
    """Assemble the xDeepONet (CPU; caller moves to device). Branch:
    [z, bc] -> width; trunk: 43 -> width; mlp decoder -> 4 channels."""
    from physicsnemo.models.mlp import FullyConnected
    from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet

    trunk_in = 3 + 1 + 3 + 3 * 2 * cfg["fourier_bands"]
    branch = FullyConnected(
        in_features=cfg["latent_size"] + cfg["bc_dim"],
        layer_size=cfg["branch_hidden"], num_layers=cfg["branch_layers"],
        out_features=cfg["width"], activation_fn="silu")
    trunk = FullyConnected(
        in_features=trunk_in, layer_size=cfg["trunk_hidden"],
        num_layers=cfg["trunk_layers"], out_features=cfg["width"],
        activation_fn="silu")
    return DeepONet(
        branch, trunk=trunk, dimension=3, width=cfg["width"],
        out_channels=len(OUT_VARS), decoder_type="mlp",
        decoder_width=cfg["decoder_hidden"],
        decoder_layers=cfg["decoder_layers"], decoder_activation_fn="silu")


def predict_normalized(model, latent, bc, xyz, sdf, normal, stats, cfg,
                       amp=False):
    """Single-case forward (xDeepONet core mode pairs one branch with one
    query set). latent (16,) / bc (4,) CPU-or-GPU; xyz/sdf/normal (N,..)
    on the model device. Returns (N, 4) normalized-space prediction
    (float32)."""
    z = (latent - stats["z_mean"]) / stats["z_std"]
    b = (bc - stats["bc_mean"]) / stats["bc_std"]
    xb = torch.cat([z, b]).unsqueeze(0)                      # (1, 20)
    xt = trunk_features(xyz, sdf, normal, cfg["fourier_bands"],
                        cfg["domain_half"])                   # (N, 43)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        y = model(xb, xt)[0]                                  # (N, 4)
    return y.float()
```

- [ ] **Step 2: 验证构建、前向/反向、checkpoint round-trip**

Run:
```bash
.venv/bin/python - <<'EOF'
import torch
from deep_sdf.cfd.roadmap_deeponet import (DEFAULT_CFG, build_model,
    fourier_encode, trunk_features, predict_normalized)
cfg = dict(DEFAULT_CFG)
assert fourier_encode(torch.zeros(5, 3), 6).shape == (5, 36)
xt = trunk_features(torch.randn(7, 3), torch.randn(7, 1),
                    torch.randn(7, 3), 6, 1.5)
assert xt.shape == (7, 43), xt.shape
model = build_model(cfg).cuda()
stats = {"z_mean": torch.zeros(16, device="cuda"), "z_std": torch.ones(16, device="cuda"),
         "bc_mean": torch.zeros(4, device="cuda"), "bc_std": torch.ones(4, device="cuda")}
y = predict_normalized(model, torch.randn(16, device="cuda"),
                       torch.randn(4, device="cuda"),
                       torch.randn(1000, 3, device="cuda"),
                       torch.randn(1000, 1, device="cuda"),
                       torch.randn(1000, 3, device="cuda"),
                       stats, cfg, amp=True)
assert y.shape == (1000, 4) and y.dtype == torch.float32 and torch.isfinite(y).all()
y.square().mean().backward()
n_params = sum(p.numel() for p in model.parameters())
assert 2_000_000 < n_params < 3_500_000, n_params
model.save("/tmp/_task2.mdlus")
from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
m2 = DeepONet.from_checkpoint("/tmp/_task2.mdlus")
assert sum(p.numel() for p in m2.parameters()) == n_params
print("MODEL-OK params=%d" % n_params)
EOF
```
Expected: 末尾打印 `MODEL-OK params=24xxxxx` 左右（245 万上下）。

- [ ] **Step 3: Commit**

```bash
git add deep_sdf/cfd/roadmap_deeponet.py
git commit -m "Add roadmap DeepONet model builder + Fourier trunk features (xDeepONet)"
```

---

### Task 3: 冻结 decoder 加载 + SDF/法向几何缓存

**Files:**
- Modify: `deep_sdf/cfd/roadmap_deeponet.py`（追加）

**Interfaces:**
- Consumes: `deep_sdf.utils.decode_sdf(decoder, latent (1,16), queries (N,3)) -> (N,1)`；`deep_sdf.cfd.volume.make_stretched_grid()`。
- Produces:
  - `load_frozen_decoder(specs_path: str, experiment_dir: str, checkpoint: str = "latest") -> (nn.Module, int latent_size)`（cuda/eval/requires_grad=False，剥离 `module.` 前缀）
  - `cache_key(name: str) -> str`（`lhs/shape_001.npz` → `lhs_shape_001`）
  - `build_geometry_cache(decoder, name_latents: list[tuple[str, Tensor (16,)]], grid_points (G,3) CPU, cache_dir str, near_band float, chunk int) -> None`（跳过已存在）
  - `load_geometry_cache(cache_dir: str, name: str) -> dict(sdf (G,) f32, normal (G,3) f32, fluid_idx int64, near_idx int64)`（CPU tensors）

- [ ] **Step 1: 追加实现**

向 `deep_sdf/cfd/roadmap_deeponet.py` 追加（顶部 import 区补 `import json, os` 与 `from deep_sdf.utils import decode_sdf`）：

```python
def load_frozen_decoder(specs_path, experiment_dir, checkpoint="latest"):
    """Rebuild the DeepSDF autodecoder from specs.json + ModelParameters
    checkpoint (trained with DataParallel -> strip the 'module.' prefix).
    Returns (decoder cuda/eval/frozen, latent_size)."""
    specs = json.load(open(specs_path))
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(specs["CodeLength"], **specs["NetworkSpecs"])
    path = os.path.join(experiment_dir, "ModelParameters", checkpoint + ".pth")
    saved = torch.load(path, map_location="cpu")
    state = saved["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    decoder.load_state_dict(state)
    decoder = decoder.cuda().eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder, specs["CodeLength"]


def cache_key(name):
    """Manifest shape name -> cache file stem ('lhs/shape_001.npz' ->
    'lhs_shape_001'; mirrors volume.snapshot_filename's '/'->'_')."""
    return name[:-4].replace("/", "_") if name.endswith(".npz") \
        else name.replace("/", "_")


def build_geometry_cache(decoder, name_latents, grid_points, cache_dir,
                         near_band=0.15, chunk=2 ** 18):
    """Per shape: chunked frozen-decoder SDF + autograd gradient on the
    shared grid; unit normal; fluid mask (sdf>0) and near-wall mask
    (|sdf|<near_band, tanh space). Writes sdf_cache/<cache_key>.npz with
    keys sdf (G,)f32 / normal (G,3)f32 / fluid_idx / near_idx (int64).
    Existing files are skipped."""
    os.makedirs(cache_dir, exist_ok=True)
    gp = grid_points.cuda()
    n_total = grid_points.shape[0]
    for i, (name, latent) in enumerate(name_latents):
        out = os.path.join(cache_dir, cache_key(name) + ".npz")
        if os.path.isfile(out):
            continue
        lat = latent.reshape(1, -1).float().cuda()
        sdf_chunks, grad_chunks = [], []
        for head in range(0, n_total, chunk):
            q = gp[head:head + chunk].clone().requires_grad_(True)
            d = decode_sdf(decoder, lat, q)
            g = torch.autograd.grad(d.sum(), q)[0]
            sdf_chunks.append(d.detach().squeeze(1).float().cpu())
            grad_chunks.append(g.detach().float().cpu())
        sdf = torch.cat(sdf_chunks)
        grad = torch.cat(grad_chunks)
        normal = grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-8)
        fluid = torch.nonzero(sdf > 0).squeeze(1)
        near = torch.nonzero(sdf.abs() < near_band).squeeze(1)
        if fluid.numel() == 0:
            raise RuntimeError("no fluid points for shape {}".format(name))
        if near.numel() == 0:
            near = fluid
        np.savez(out, sdf=sdf.numpy().astype(np.float32),
                 normal=normal.numpy().astype(np.float32),
                 fluid_idx=fluid.numpy().astype(np.int64),
                 near_idx=near.numpy().astype(np.int64))
        print("[cache] %d/%d %s (fluid %d, near %d)" % (
            i + 1, len(name_latents), name, fluid.numel(), near.numel()))


def load_geometry_cache(cache_dir, name):
    path = os.path.join(cache_dir, cache_key(name) + ".npz")
    data = np.load(path)
    return {"sdf": torch.from_numpy(data["sdf"]),
            "normal": torch.from_numpy(data["normal"]),
            "fluid_idx": torch.from_numpy(data["fluid_idx"]),
            "near_idx": torch.from_numpy(data["near_idx"])}
```

- [ ] **Step 2: 验证（真实 decoder + 全网格、2 个形状）**

Run:
```bash
.venv/bin/python - <<'EOF'
import numpy as np, torch
from deep_sdf.cfd.roadmap_deeponet import (load_frozen_decoder,
    build_geometry_cache, load_geometry_cache, cache_key)
from deep_sdf.cfd.volume import make_stretched_grid
from generate_openfoam_snapshots import load_manifest

decoder, latent_size = load_frozen_decoder(
    "examples/ellipsoids_of4/specs.json", "examples/ellipsoids")
assert latent_size == 16
n = sum(p.numel() for p in decoder.parameters())
assert n == 49774, n
grid_points, grid_shape, _ = make_stretched_grid()
assert grid_points.shape == (1442897, 3), grid_points.shape

names, latents = load_manifest("data/openfoam/ellipsoids_u10/lhs_latents.npz")
assert len(names) == 281 and latents.shape == (281, 16)
sel = [(names[0], torch.from_numpy(latents[0])),
       (names[-1], torch.from_numpy(latents[-1]))]
build_geometry_cache(decoder, sel, grid_points, "/tmp/_sdf_cache_test",
                     near_band=0.15)
for name, _ in sel:
    c = load_geometry_cache("/tmp/_sdf_cache_test", name)
    G = 1442897
    assert c["sdf"].shape == (G,) and c["normal"].shape == (G, 3)
    frac = c["fluid_idx"].numel() / G
    assert 0.90 < frac < 1.0, frac          # 椭球体占域体积 <10%
    assert c["near_idx"].numel() > 0
    norms = c["normal"][c["fluid_idx"][:10000]].norm(dim=1)
    assert (norms - 1).abs().max() < 1e-4
    print("CACHE-OK", name, "fluid=%.4f" % frac, "near=%d" % c["near_idx"].numel())
# 原点在任何椭球内部 -> sdf < 0
c0 = load_geometry_cache("/tmp/_sdf_cache_test", names[0])
center = int(np.argmin(grid_points.abs().sum(1).numpy()))
assert c0["sdf"][center] < 0, c0["sdf"][center]
print("SDF-SIGN-OK")
EOF
```
Expected: 打印 decoder 参数 49774 校验通过、两行 `CACHE-OK ... fluid=0.99x near=...`、`SDF-SIGN-OK`。

- [ ] **Step 3: Commit**

```bash
git add deep_sdf/cfd/roadmap_deeponet.py
git commit -m "Add frozen DeepSDF decoder loader + SDF/normal geometry cache"
```

---

### Task 4: 快照索引、形状加载、cluster 划分、统计

**Files:**
- Modify: `deep_sdf/cfd/roadmap_deeponet.py`（追加）

**Interfaces:**
- Consumes: Task 3 的 cache 接口；`deep_sdf.cfd.volume.load_snapshot(path, expected_points) -> dict(fields, bc, shape)`。
- Produces:
  - `snapshot_index(snapshots_dir: str) -> dict[str, str]`（shape 名 → 路径，自动排除 `_of4_view` 等不在 manifest 的文件由调用方交集处理）
  - `load_shapes(names list[str], latents (S,16) np, snap_idx dict, cache_dir str, expected_points int) -> list[dict]`；每个 dict：`{"name", "latent" (16,)f32, "bc" (4,)f32, "fields" (G,4)f32（已 /U 无量纲）, "sdf" (G,), "normal" (G,3), "fluid_idx", "near_idx"}`，全部 CPU
  - `cluster_split(names, latents (S,16) np, n_clusters int, seed int) -> (dict["train"/"val"/"test" -> list[str]], labels np (S,))`（z-score → kmeans2(minit="++") → 簇 shuffle 后 8/2/2 分）
  - `compute_stats(shapes list[dict], n_sample int, seed int) -> dict`：`z_mean/z_std (16,)`、`bc_mean/bc_std (4,)`、`y_mean/y_std (4,)`（CPU tensors，std 钳 1e-8；y 统计=每形状随机 n_sample 流体点聚合）

- [ ] **Step 1: 追加实现**

```python
def snapshot_index(snapshots_dir):
    """Glob snapshot npz files and map internal 'shape' field -> path."""
    import glob
    idx = {}
    for path in sorted(glob.glob(os.path.join(snapshots_dir, "*.npz"))):
        data = np.load(path, allow_pickle=False)
        if "shape" not in data.files:
            continue
        idx[str(data["shape"])] = path
    return idx


def load_shapes(names, latents, snap_idx, cache_dir, expected_points):
    """Join manifest latents with snapshots (nondim u /= U at load) and the
    geometry cache. All tensors CPU-resident."""
    from deep_sdf.cfd.volume import load_snapshot
    shapes = []
    for i, name in enumerate(names):
        if name not in snap_idx:
            raise RuntimeError("no snapshot for manifest shape {}".format(name))
        snap = load_snapshot(snap_idx[name], expected_points)
        fields = snap["fields"].clone()
        fields[:, :3] /= snap["bc"][0]
        if not torch.isfinite(fields).all():
            raise RuntimeError("non-finite fields in {}".format(snap_idx[name]))
        geom = load_geometry_cache(cache_dir, name)
        shapes.append({"name": name,
                       "latent": torch.from_numpy(latents[i]).float(),
                       "bc": snap["bc"].float(), "fields": fields, **geom})
    return shapes


def cluster_split(names, latents, n_clusters=12, seed=0):
    """Roadmap section 30: k-means on z-scored latents, whole clusters
    assigned 8/2/2 to train/val/test (cluster order shuffled by seed)."""
    from scipy.cluster.vq import kmeans2
    X = latents.astype(np.float64)
    X = (X - X.mean(0)) / X.std(0).clip(1e-8)
    _, labels = kmeans2(X, n_clusters, seed=seed, minit="++")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_clusters)
    groups = {"train": perm[:8], "val": perm[8:10], "test": perm[10:]}
    out = {}
    for key, clusters in groups.items():
        member = set(int(c) for c in clusters)
        out[key] = [n for n, lab in zip(names, labels) if int(lab) in member]
    return out, labels


def compute_stats(shapes, n_sample=4096, seed=0):
    """z/bc/output-channel z-score statistics from the TRAIN shapes; the
    output stats aggregate n_sample random fluid points per shape."""
    z = torch.stack([s["latent"] for s in shapes])
    bc = torch.stack([s["bc"] for s in shapes])
    gen = torch.Generator().manual_seed(seed)
    ys = []
    for s in shapes:
        pool = s["fluid_idx"]
        pick = pool[torch.randint(0, pool.numel(),
                                  (min(n_sample, pool.numel()),),
                                  generator=gen)]
        ys.append(s["fields"][pick])
    Y = torch.cat(ys)
    return {"z_mean": z.mean(0), "z_std": z.std(0).clamp_min(1e-8),
            "bc_mean": bc.mean(0), "bc_std": bc.std(0).clamp_min(1e-8),
            "y_mean": Y.mean(0), "y_std": Y.std(0).clamp_min(1e-8)}
```

- [ ] **Step 2: 验证（真实数据，281 形状全量）**

先确保全量缓存存在（约 281 形状一次性构建，几分钟；若 Task 5 前不想全建，本步可先只对划分后 train 子集验证加载——但推荐本步直接全建，后续任务都依赖它）：

Run（全量缓存构建，~5-10 min，已存在自动跳过）:
```bash
.venv/bin/python - <<'EOF'
import torch
from deep_sdf.cfd.roadmap_deeponet import (load_frozen_decoder,
    build_geometry_cache)
from deep_sdf.cfd.volume import make_stretched_grid
from generate_openfoam_snapshots import load_manifest
decoder, _ = load_frozen_decoder("examples/ellipsoids_of4/specs.json",
                                 "examples/ellipsoids")
grid_points, _, _ = make_stretched_grid()
names, latents = load_manifest("data/openfoam/ellipsoids_u10/lhs_latents.npz")
build_geometry_cache(decoder,
    [(n, torch.from_numpy(latents[i])) for i, n in enumerate(names)],
    grid_points, "data/openfoam/ellipsoids_u10/sdf_cache", near_band=0.15)
print("FULL-CACHE-OK")
EOF
```
Expected: 末尾 `FULL-CACHE-OK`（中间逐形状 `[cache] i/281 ...` 进度）。

Run（加载/划分/统计验证）:
```bash
.venv/bin/python - <<'EOF'
import numpy as np
from deep_sdf.cfd.roadmap_deeponet import (snapshot_index, load_shapes,
    cluster_split, compute_stats)
from generate_openfoam_snapshots import load_manifest

names, latents = load_manifest("data/openfoam/ellipsoids_u10/lhs_latents.npz")
idx = snapshot_index("data/openfoam/ellipsoids_u10/snapshots")
missing = [n for n in names if n not in idx]
assert not missing, missing[:3]

split, labels = cluster_split(names, latents, n_clusters=12, seed=0)
sizes = {k: len(v) for k, v in split.items()}
assert sum(sizes.values()) == 281
assert not (set(split["train"]) & set(split["val"])) and \
       not (set(split["train"]) & set(split["test"])) and \
       not (set(split["val"]) & set(split["test"]))
assert 150 < sizes["train"] < 240 and sizes["val"] > 10 and sizes["test"] > 10
# 确定性
split2, _ = cluster_split(names, latents, n_clusters=12, seed=0)
assert split == split2
print("SPLIT-OK", sizes)

shapes = load_shapes(split["train"][:4], latents[[names.index(n) for n in split["train"][:4]]],
                     idx, "data/openfoam/ellipsoids_u10/sdf_cache", 1442897)
stats = compute_stats(shapes, n_sample=4096, seed=0)
assert stats["z_mean"].shape == (16,) and stats["y_mean"].shape == (4,)
assert (stats["y_std"] > 0).all()
assert abs(float(stats["bc_mean"][0]) - 10.0) < 1e-4   # 固定 U=10
print("STATS-OK y_mean=%s y_std=%s" % (stats["y_mean"].numpy().round(3),
                                       stats["y_std"].numpy().round(3)))
EOF
```
Expected: `SPLIT-OK {'train': ~19x, 'val': ~4x, 'test': ~4x}` 与 `STATS-OK ...`。

- [ ] **Step 3: Commit**

```bash
git add deep_sdf/cfd/roadmap_deeponet.py
git commit -m "Add snapshot index, shape loader, latent cluster split, z-score stats"
```

---

### Task 5: 采样、评估与训练器 CLI

**Files:**
- Modify: `deep_sdf/cfd/roadmap_deeponet.py`（追加采样/评估/全场推理）
- Create: `train_roadmap_deeponet.py`

**Interfaces:**
- Consumes: Tasks 1-4 全部接口。
- Produces（roadmap_deeponet.py 追加）:
  - `sample_case_batch(shape dict, grid_points (G,3), n_points int, near_frac float, gen torch.Generator, device) -> dict`：`{"latent"(16,), "bc"(4,), "xyz"(N,3), "sdf"(N,1), "normal"(N,3), "y"(N,4)}`，全在 device 上；近壁比例 near_frac（near_idx 空则全 fluid）
  - `evaluate(model, shapes list, grid_points, stats, cfg, n_points int, seed int, chunk int) -> dict`：`{"rel_l2" float, "per_var" [4], "baseline" float, "per_case" {name: float}}`；物理空间 rel L2（预测经 y_mean/y_std 反标准化），baseline = 预测 y_mean 的 rel L2；每形状固定种子
  - `predict_field(model, shape dict, grid_points, stats, cfg, chunk int) -> Tensor (G,4)`（物理空间，分块，no_grad，fp32）
- Produces（train_roadmap_deeponet.py）: CLI `main()`，输出 `<experiment>/RoadmapONet/`（`--smoke` 时 `RoadmapONet_smoke/`）：`config.json`、`split.json`、`stats.pth`、`metrics.jsonl`、`best.mdlus`、`train_state.pth`、`eval.json`。

- [ ] **Step 1: 追加采样/评估/推理到 roadmap_deeponet.py**

```python
def sample_case_batch(shape, grid_points, n_points, near_frac, gen, device):
    """Draw n_points from one shape: near_frac from the near-wall pool, the
    rest uniform fluid; returns GPU tensors ready for predict_normalized."""
    n_near = int(round(n_points * near_frac))
    pool_near, pool_fluid = shape["near_idx"], shape["fluid_idx"]
    take_near = min(n_near, pool_near.numel())
    take_fluid = n_points - take_near
    idx = torch.cat([
        pool_near[torch.randint(0, pool_near.numel(), (take_near,),
                                generator=gen)],
        pool_fluid[torch.randint(0, pool_fluid.numel(), (take_fluid,),
                                 generator=gen)]])
    return {"latent": shape["latent"].to(device),
            "bc": shape["bc"].to(device),
            "xyz": grid_points[idx].to(device),
            "sdf": shape["sdf"][idx].unsqueeze(1).to(device),
            "normal": shape["normal"][idx].to(device),
            "y": shape["fields"][idx].to(device)}


@torch.no_grad()
def evaluate(model, shapes, grid_points, stats, cfg, n_points, seed,
             chunk=2 ** 18):
    """Physical-space per-variable rel L2 on a fixed-seed fluid subsample
    per shape; baseline predicts the train channel mean (y_mean)."""
    import zlib
    device = next(model.parameters()).device
    stats_g = {k: v.to(device) for k, v in stats.items()}
    per_var, per_case, baselines = [], {}, []
    for s in shapes:
        gen = torch.Generator().manual_seed(
            seed + zlib.crc32(s["name"].encode()))
        pool = s["fluid_idx"]
        idx = pool[torch.randperm(pool.numel(), generator=gen)[:n_points]]
        xyz = grid_points[idx].to(device)
        sdf = s["sdf"][idx].unsqueeze(1).to(device)
        normal = s["normal"][idx].to(device)
        y = s["fields"][idx].to(device)
        preds = []
        for head in range(0, idx.numel(), chunk):
            sl = slice(head, min(head + chunk, idx.numel()))
            pn = predict_normalized(
                model, s["latent"].to(device), s["bc"].to(device),
                xyz[sl], sdf[sl], normal[sl], stats_g, cfg, amp=False)
            preds.append(pn * stats_g["y_std"] + stats_g["y_mean"])
        pred = torch.cat(preds)
        rel_v = [((pred[:, v] - y[:, v]).norm() /
                  y[:, v].norm().clamp_min(1e-12)).item() for v in range(4)]
        per_var.append(rel_v)
        per_case[s["name"]] = float(np.mean(rel_v))
        baselines.append([((stats_g["y_mean"][v] - y[:, v]).norm() /
                           y[:, v].norm().clamp_min(1e-12)).item()
                          for v in range(4)])
    return {"rel_l2": float(np.mean([np.mean(r) for r in per_var])),
            "per_var": [float(np.mean([r[v] for r in per_var]))
                        for v in range(4)],
            "baseline": float(np.mean([np.mean(b) for b in baselines])),
            "per_case": per_case}


@torch.no_grad()
def predict_field(model, shape, grid_points, stats, cfg, chunk=2 ** 18):
    """Full-grid physical-space prediction (G,4), fp32, chunked."""
    device = next(model.parameters()).device
    stats_g = {k: v.to(device) for k, v in stats.items()}
    lat = shape["latent"].to(device)
    bc = shape["bc"].to(device)
    out = []
    for head in range(0, grid_points.shape[0], chunk):
        sl = slice(head, min(head + chunk, grid_points.shape[0]))
        xyz = grid_points[sl].to(device)
        sdf = shape["sdf"][sl].unsqueeze(1).to(device)
        normal = shape["normal"][sl].to(device)
        pn = predict_normalized(model, lat, bc, xyz, sdf, normal,
                                stats_g, cfg, amp=False)
        out.append((pn * stats_g["y_std"] + stats_g["y_mean"]).cpu())
    return torch.cat(out)
```

- [ ] **Step 2: 写训练器 `train_roadmap_deeponet.py`**

```python
#!/usr/bin/env python3
"""DeepSDF + PhysicsNeMo DeepONet MVP trainer (roadmap section 63; spec
docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md).

    .venv/bin/python train_roadmap_deeponet.py --smoke
    .venv/bin/python train_roadmap_deeponet.py --iters 20000
    .venv/bin/python train_roadmap_deeponet.py --eval_only --resume

Outputs under <experiment>/RoadmapONet/ (RoadmapONet_smoke/ with --smoke):
config.json, split.json, stats.pth, metrics.jsonl, best.mdlus,
train_state.pth, eval.json.
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import deep_sdf.cfd.physicsnemo_compat  # noqa: F401 (before physicsnemo)
from deep_sdf.cfd import roadmap_deeponet as rd
from deep_sdf.cfd.volume import make_stretched_grid
from generate_openfoam_snapshots import load_manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_directory", default="examples/ellipsoids")
    p.add_argument("--specs", default="examples/ellipsoids_of4/specs.json")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--data_root", default="data/openfoam/ellipsoids_u10")
    p.add_argument("--config", default=None,
                   help="JSON with DEFAULT_CFG overrides")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--iters", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = dict(rd.DEFAULT_CFG)
    if args.config:
        cfg.update(json.load(open(args.config)))
    if args.iters is not None:
        cfg["iters"] = args.iters
    if args.smoke:
        cfg.update(iters=500, eval_every=100, eval_points=4096)

    out_dir = os.path.join(args.experiment_directory,
                           "RoadmapONet_smoke" if args.smoke else "RoadmapONet")
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")

    # --- data -------------------------------------------------------------
    decoder, latent_size = rd.load_frozen_decoder(
        args.specs, args.experiment_directory, args.checkpoint)
    assert latent_size == cfg["latent_size"]
    grid_points, grid_shape, _ = make_stretched_grid()
    n_grid = grid_points.shape[0]
    names, latents = load_manifest(
        os.path.join(args.data_root, "lhs_latents.npz"))
    cache_dir = os.path.join(args.data_root, "sdf_cache")
    rd.build_geometry_cache(
        decoder, [(n, torch.from_numpy(latents[i]))
                  for i, n in enumerate(names)],
        grid_points, cache_dir, near_band=cfg["near_band"])
    del decoder
    torch.cuda.empty_cache()

    snap_idx = rd.snapshot_index(os.path.join(args.data_root, "snapshots"))
    split, labels = rd.cluster_split(names, latents,
                                     n_clusters=cfg["n_clusters"], seed=seed)
    logging.info("split sizes: %s",
                 {k: len(v) for k, v in split.items()})

    def load(name_list):
        sel = [names.index(n) for n in name_list]
        return rd.load_shapes(name_list, latents[sel], snap_idx, cache_dir,
                              n_grid)

    train_shapes = load(split["train"])
    stats = rd.compute_stats(train_shapes, n_sample=cfg["stats_sample"],
                             seed=seed)

    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({**split,
                   "labels": {n: int(l) for n, l in zip(names, labels)}},
                  f, indent=1)
    torch.save(stats, os.path.join(out_dir, "stats.pth"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    # --- model / optimizer -------------------------------------------------
    model = rd.build_model(cfg).to(device)
    stats_g = {k: v.to(device) for k, v in stats.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    start_iter, best = 0, float("inf")
    state_path = os.path.join(out_dir, "train_state.pth")
    if args.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=device)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        start_iter, best = st["iter"], st["best"]
        logging.info("resumed from iter %d (best %.4f)", start_iter, best)

    def lr_at(it):
        t = min(it / max(cfg["iters"], 1), 1.0)
        return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (
            1 + np.cos(np.pi * t))

    val_shapes = load(split["val"])
    if args.smoke:
        val_shapes = val_shapes[:8]

    def run_eval(n_points):
        model.eval()
        res = rd.evaluate(model, val_shapes, grid_points, stats_g, cfg,
                          n_points=n_points, seed=12345)
        model.train()
        return res

    if args.eval_only:
        from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
        model = DeepONet.from_checkpoint(
            os.path.join(out_dir, "best.mdlus")).to(device).eval()
        test_shapes = load(split["test"])
        res = {"val": rd.evaluate(model, val_shapes, grid_points, stats_g,
                                  cfg, cfg["final_eval_points"], 777),
               "test": rd.evaluate(model, test_shapes, grid_points, stats_g,
                                   cfg, cfg["final_eval_points"], 999)}
        shape0 = test_shapes[0]
        t0 = time.time()
        rd.predict_field(model, shape0, grid_points, stats_g, cfg)
        res["infer_sec_per_case"] = time.time() - t0
        res["speedup_vs_cfd_100s"] = 100.0 / res["infer_sec_per_case"]
        with open(os.path.join(out_dir, "eval.json"), "w") as f:
            json.dump(res, f, indent=1)
        logging.info("eval: %s", json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in res.items() if k != "val" and k != "test"}))
        logging.info("val rel_l2 %.4f (baseline %.4f) | test rel_l2 %.4f "
                     "(baseline %.4f) | infer %.2fs/case (~%.0fx)",
                     res["val"]["rel_l2"], res["val"]["baseline"],
                     res["test"]["rel_l2"], res["test"]["baseline"],
                     res["infer_sec_per_case"], res["speedup_vs_cfd_100s"])
        return

    # --- train --------------------------------------------------------------
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    gen = torch.Generator().manual_seed(seed)
    model.train()
    t0 = time.time()
    for it in range(start_iter, cfg["iters"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        cases = [train_shapes[int(torch.randint(0, len(train_shapes), (1,),
                                              generator=gen))]
                 for _ in range(cfg["batch_cases"])]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for c in cases:
            b = rd.sample_case_batch(c, grid_points, cfg["n_points"],
                                     cfg["near_frac"], gen, device)
            pred = rd.predict_normalized(model, b["latent"], b["bc"],
                                         b["xyz"], b["sdf"], b["normal"],
                                         stats_g, cfg, amp=cfg["amp"])
            yn = (b["y"] - stats_g["y_mean"]) / stats_g["y_std"]
            loss = loss + sum(F.mse_loss(pred[:, v], yn[:, v])
                              for v in range(4))
        loss = loss / len(cases)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (it + 1) % cfg["metrics_every"] == 0 or it == start_iter:
            rec = {"iter": it + 1, "train_loss": float(loss.item()),
                   "lr": lr_at(it),
                   "sec_per_iter": (time.time() - t0) / (it + 1 - start_iter)}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if (it + 1) % cfg["eval_every"] == 0 or it + 1 == cfg["iters"]:
            res = run_eval(cfg["eval_points"])
            rec = {"iter": it + 1, "val_rel_l2": res["rel_l2"],
                   "val_u": res["per_var"][0], "val_v": res["per_var"][1],
                   "val_w": res["per_var"][2], "val_p": res["per_var"][3],
                   "val_baseline": res["baseline"]}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            logging.info("iter %d loss %.4f val %.4f (baseline %.4f)",
                         it + 1, float(loss.item()), res["rel_l2"],
                         res["baseline"])
            if res["rel_l2"] < best:
                best = res["rel_l2"]
                model.save(os.path.join(out_dir, "best.mdlus"))
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "iter": it + 1, "best": best}, state_path)

    logging.info("done. best val rel_l2 %.4f", best)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: smoke 验证**

Run:
```bash
.venv/bin/python train_roadmap_deeponet.py --smoke
```
Expected（缓存已建时约 2-5 min）：
- 无异常；日志出现 `split sizes:`、每 100 iter 的 `iter ... loss ... val ... (baseline ...)`；
- `examples/ellipsoids/RoadmapONet_smoke/` 下生成 `config.json / split.json / stats.pth / metrics.jsonl / best.mdlus / train_state.pth`；
- `metrics.jsonl` 中 val_rel_l2 有限且 < baseline（smoke 规模下不强求大幅低于）。

随后验证 checkpoint 可重载：
```bash
.venv/bin/python - <<'EOF'
import deep_sdf.cfd.physicsnemo_compat  # noqa: F401
from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
m = DeepONet.from_checkpoint("examples/ellipsoids/RoadmapONet_smoke/best.mdlus")
print("RELOAD-OK params=%d" % sum(p.numel() for p in m.parameters()))
EOF
```
Expected: `RELOAD-OK params=24xxxxx`。

- [ ] **Step 4: Commit**

```bash
git add deep_sdf/cfd/roadmap_deeponet.py train_roadmap_deeponet.py
git commit -m "Add roadmap DeepONet trainer (sampling/eval/train loop/smoke/resume)"
```

---

### Task 6: 正式训练 + 最终评估 + 测速

**Files:**
- 无新代码；产出 `examples/ellipsoids/RoadmapONet/`（不入 git）

**Interfaces:**
- Consumes: Task 5 训练器。
- Produces: `RoadmapONet/{metrics.jsonl, best.mdlus, train_state.pth, eval.json}`。

- [ ] **Step 1: 正式训练（后台运行，预计 15-45 min）**

Run:
```bash
.venv/bin/python train_roadmap_deeponet.py --iters 20000
```
Expected: 每 500 iter 打印 val 指标；val_rel_l2 总体下行；结束后 `best.mdlus` 存在。若显存溢出（RTX 3070 8GB）：把 `--config` 写 `{"batch_cases": 2}` 或 `{"n_points": 4096}` 重跑（`--resume` 续训）。

- [ ] **Step 2: 最终评估 + 测速**

Run:
```bash
.venv/bin/python train_roadmap_deeponet.py --eval_only --resume
```
Expected: 写 `eval.json`；日志打印 val/test rel_l2、各自 baseline、每 case 推理秒数与加速比。

- [ ] **Step 3: 对照验收标准记录结果**

逐项核对规格 §7：
1. smoke 全链路无错 + round-trip（Task 5 已验证）；
2. `eval.json` 中 `val.rel_l2 < 0.5 * val.baseline`；
3. test per-variable 指标已报告（`eval.json` 的 `test.per_var`）；
4. 推理加速比已报告（`speedup_vs_cfd_100s`）；
5. 在与用户汇报中注明与 PiPOD 0.34~0.39 的口径差异（POD 系数域 vs 点态场）。

若验收 2 未达成：检查 metrics.jsonl 训练曲线（loss 是否下降、val 是否震荡），把曲线、`eval.json`、可能原因（如 lr/points/宽度）如实汇报给用户，由用户决定调参续训还是接受现状——不自行扩大范围。

---

## Self-Review 记录

- 规格覆盖：§3 shim→T1；§4 模型/特征→T2；§5.1 缓存→T3；§5.2 掩码→T3（随缓存）；§5.3 划分→T4；§4.2/§5.4 统计与采样→T4/T5；§6 训练/resume/smoke→T5；§7 评估/测速/验收→T6。✅
- 口径修正（相较规格）：参考网格域为 [-1.5,1.5]³ → `domain_half=1.5`（规格写成坐标除 9 是按 CFD 域 [-9,9] 误解，特此更正）；快照保存时速度为原始 m/s、加载时 /U（与 PiPOD 一致）；xDeepONet core 模式不支持逐 case 不同查询点 → 逐 case 前向。
- 类型一致性：`predict_normalized` 在 T2 定义、T5 采样/评估/训练复用，签名一致；stats 键（z_mean/z_std/bc_mean/bc_std/y_mean/y_std）贯穿一致；cache 键名（sdf/normal/fluid_idx/near_idx）T3 写、T4 读一致。

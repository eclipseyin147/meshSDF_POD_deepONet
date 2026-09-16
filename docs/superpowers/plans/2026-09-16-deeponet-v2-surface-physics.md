# DeepONet v2（动态采样加权 / Surface 头 + Cd/Cl / PDE 微调）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在第一轮 MVP（field stage，val 0.2513）基础上增加：W1 误差分析+形状级动态采样加权、W2 surface Cp/Cf 头+Cd/Cl 双路预测（含 OpenFOAM 表面场重导出）、W3 PDE 物理微调。

**Architecture:** 三条工作流共用 MVP 的 split/stats/缓存体系；surface/physics 作为独立 stage 链式 init_from field best。表面真值由 OpenFOAM 重跑全 281 形状导出（wallShearStress FO + surfaces FO → pyvista 解析 → 每面心 Cp/Cf/decoder 法向/面积 + cd_gt/cl_gt）。PDE 微调复用 deep_sdf/cfd/physics.py 的 PDEInformer/IncompressibleNS(re=100)/CollocationSampler。

**Tech Stack:** torch 2.5.1（不动）、vendored PhysicsNeMo（经 physicsnemo_compat）、OpenFOAM 14（/opt/openfoam14）、pyvista（新装，仅导出用）、scipy。规格：`docs/superpowers/specs/2026-09-16-deeponet-v2-surface-physics-design.md`。

## Global Constraints

- **不升级 torch**（2.5.1+cu121）；**不修改** `third-party/physicsnemo/`；任何 `import physicsnemo...` 之前先 `import deep_sdf.cfd.physicsnemo_compat`。
- 仓库**无 pytest**：验证用可执行探针（`.venv/bin/python` heredoc）与训练器 `--smoke`，不新增测试框架。
- Python 一律 `.venv/bin/python`；工作目录 `/home/siqi/CLionProjects/DeepSDF`。
- 产物不入 git：`data/openfoam/ellipsoids_u10/{surface,cases}/`、`examples/ellipsoids/RoadmapONet*/` 只留盘不提交。
- **PiPOD 线只读复用**（physics.py 的类被 import，不改其文件；train_pipod_deeponet.py 不动）。
- MVP 已有产物 `examples/ellipsoids/RoadmapONet/best.mdlus` 是 surface/physics 的 init_from 源，**不得被覆盖**：field 增强 run 一律用 `--out_name RoadmapONet_adaptive`。
- 关键数据事实（实现者须知）：
  - 表面真值 npz 契约（本计划 T1 产出）：`data/openfoam/ellipsoids_u10/surface/<cache_key>.npz`，键 `centers (F,3)` / `normals (F,3)`（decoder SDF 梯度归一化，物体外向）/ `areas (F,)` / `cp (F,)` / `cf (F,3)` 全 f32 + 标量 `cd_gt` / `cl_gt`；`cache_key(name)` 见 roadmap_deeponet.py（`lhs/shape_001.npz`→`lhs_shape_001`）。
  - **退化形状排除（2026-09-16 用户裁定）**：5 个 latent 超界形状（`lhs/shape_{007,087,174,253,254}.npz`，|z|>1.36 超 CodeBound=1.0，snappy body patch 塌缩 <1000 面）从 surface 阶段排除；排除清单 `data/openfoam/ellipsoids_u10/surface/excluded_shapes.json`（`load_surface_exclusions` 读取）；其 npz 存在但无效（面数过少），`load_surface` 必须经 `validate_surface_npz` 校验拒绝之。field 线不受影响。surface 阶段有效形状数 = 276。
  - 力积分约定（GT 与预测共用，`openfoam_runner.integrate_cd_cl` 唯一实现）：`F = −Σ cp·n·A + Σ cf·A`；`cd = F·dir/A_ref`、`cl = F_z/A_ref`；`A_ref = Σ max(0, −n·dir)·A`。
  - 快照 `bc` = `[10,1,0,0]` 全部 281 case；`bc[1:4]` = 来流方向 = +x。
  - 体积快照 `snapshots/`、几何缓存 `sdf_cache/`、manifest `lhs_latents.npz`（281×16）沿用 MVP；网格 `make_stretched_grid()` 默认（(1442897,3)，域 [-1.5,1.5]³，`domain_half=1.5`）。
  - STL 缓存命名：`_stl_cache_path(stl_dir, name)` = `stls/<name 去 .npz、/→_>.stl`（281 个全部已存在）。
  - xDeepONet core 模式：`model(x_branch (1,20), x_trunk (N,D)) -> (1,N,C)`，逐 case 调用；`DeepONet` 实例把 branch 存为 `self.branch1`（共享 branch = 把 `field_model.branch1` 传给新 DeepONet）。
  - decoder 经 `roadmap_deeponet.load_frozen_decoder("examples/ellipsoids_of4/specs.json", "examples/ellipsoids")` 加载（generate_openfoam_snapshots.py 自带的 `load_decoder` 读 `<exp>/specs.json`，须用 `--experiment examples/ellipsoids_of4`）。

---

### Task 1: OpenFOAM 表面场导出（openfoam_runner 扩展 + 单 case 冒烟）

**Files:**
- Modify: `deep_sdf/cfd/openfoam_runner.py`（追加函数，不改既有函数）

**Interfaces:**
- Consumes: 既有 `_run(cmd, case_dir, log_name)`、`_write(path, text)`、`_foam_header(name)`/`_foam_footer()`、`_parse_bc(bc) -> (U, direction)`、`make_case_from_template`（在 generate_openfoam_snapshots.py）、`run_case(case_dir, n_procs=1)`。
- Produces:
  - `decoder_normals(decoder, latent, points, max_batch=2**17, eps=1e-8) -> np (F,3) f32`（冻结 decoder SDF 梯度归一化；decoder 在哪个 device 就用哪个）
  - `reference_area_from_faces(normals, areas, direction) -> torch scalar`
  - `integrate_cd_cl(cp, cf, normals, areas, direction) -> (cd, cl) torch scalars`（array-like 输入，内部 `torch.as_tensor`）
  - `export_surface(case_dir, out_path, bc, decoder, latent) -> out_path`
  - `validate_surface_npz(path) -> None`（不合格抛 RuntimeError）

- [ ] **Step 1: 安装 pyvista**

```bash
.venv/bin/pip install pyvista 2>&1 | tail -2
.venv/bin/python -c "import pyvista; print('pyvista', pyvista.__version__)"
```
Expected: 正常打印版本号（vtk 轮子约百 MB，属预期）。

- [ ] **Step 2: 追加导出函数到 openfoam_runner.py**

顶部 import 区补 `import glob`（若无）。文件末尾追加：

```python
# --------------------------------------------------------------------------
# Surface field export (DeepONet v2 spec section 3)
# --------------------------------------------------------------------------

def decoder_normals(decoder, latent, points, max_batch=2 ** 17, eps=1e-8):
    """Unit outward SDF normals at query points from a frozen DeepSDF
    decoder (autograd gradient of decode_sdf, normalized)."""
    import deep_sdf.utils
    device = next(decoder.parameters()).device
    lat = torch.as_tensor(latent, dtype=torch.float32).reshape(1, -1).to(device)
    pts = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3)
    outs = []
    for lo in range(0, pts.shape[0], max_batch):
        q = pts[lo:lo + max_batch].to(device).requires_grad_(True)
        d = deep_sdf.utils.decode_sdf(decoder, lat, q)
        g = torch.autograd.grad(d.sum(), q)[0]
        outs.append((g / g.norm(dim=1, keepdim=True).clamp_min(eps)).cpu())
    return torch.cat(outs).numpy().astype(np.float32)


def reference_area_from_faces(normals, areas, direction):
    """A_ref = sum_f max(0, -n_f . dir) * A_f (labels.reference_area 约定)."""
    n = torch.as_tensor(normals, dtype=torch.float32).reshape(-1, 3)
    a = torch.as_tensor(areas, dtype=torch.float32).reshape(-1)
    v = torch.as_tensor(direction, dtype=torch.float32).reshape(3)
    v = v / v.norm()
    return (torch.clamp(-(n @ v), min=0.0) * a).sum()


def integrate_cd_cl(cp, cf, normals, areas, direction):
    """F = -sum cp*n*A + sum cf*A (n = outward unit normal); returns
    (cd, cl) = (F.dir/A_ref, F_z/A_ref) as torch float32 scalars."""
    cp = torch.as_tensor(cp, dtype=torch.float32).reshape(-1)
    cf = torch.as_tensor(cf, dtype=torch.float32).reshape(-1, 3)
    n = torch.as_tensor(normals, dtype=torch.float32).reshape(-1, 3)
    a = torch.as_tensor(areas, dtype=torch.float32).reshape(-1)
    v = torch.as_tensor(direction, dtype=torch.float32).reshape(3)
    v = v / v.norm()
    force = -(cp.unsqueeze(1) * n * a.unsqueeze(1)).sum(0) \
        + (cf * a.unsqueeze(1)).sum(0)
    a_ref = reference_area_from_faces(n, a, v)
    return (force @ v) / a_ref, force[2] / a_ref


def _surface_dict(name="surfaces", patch_regex=".*body.*"):
    out = _foam_header(name)
    out += """%s
{
    type                surfaces;
    libs                ("libsampling.so");
    writeControl        timeStep;
    writeInterval       1;
    interpolationScheme cell;
    surfaceFormat       vtp;
    fields              (p wallShearStress);
    surfaces
    {
        wall
        {
            type        patch;
            patches     ( "%s" );
            triangulate true;
        }
    }
}
""" % (name, patch_regex)
    return out + _foam_footer()


def export_surface(case_dir, out_path, bc, decoder, latent):
    """Post-process a finished case into a wall-surface npz:
    foamPostProcess wallShearStress -> surfaces FO (vtp, patch body.*) ->
    per-face centers/areas + Cp = p/(0.5 U^2) / cf = tau_w/(0.5 U^2) +
    decoder normals -> GT Cd/Cl -> npz
    (centers/normals/areas/cp/cf f32, cd_gt/cl_gt scalars)."""
    U, direction = _parse_bc(bc)
    _run(["foamPostProcess", "-func", "wallShearStress", "-latestTime"],
         case_dir, "log.wallShearStress")
    _write(os.path.join(case_dir, "system", "surfaceDict"), _surface_dict())
    _run(["foamPostProcess", "-latestTime", "-dict",
          os.path.join("system", "surfaceDict")], case_dir, "log.surfaces")
    hits = sorted(glob.glob(os.path.join(
        case_dir, "postProcessing", "surfaces", "*", "wall.vtp")))
    if not hits:
        raise RuntimeError("no surfaces output in {}".format(case_dir))
    import pyvista as pv
    mesh = pv.read(hits[-1])
    if mesh.n_cells < 1:
        raise RuntimeError("empty wall surface in {}".format(hits[-1]))
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:4]
    tri = np.asarray(mesh.points, dtype=np.float64)[faces]      # (F,3,3)
    centers = tri.mean(axis=1)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cr, axis=1)
    p = np.asarray(mesh.cell_data["p"], dtype=np.float64).reshape(-1)
    wss = np.asarray(
        mesh.cell_data["wallShearStress"], dtype=np.float64).reshape(-1, 3)
    if p.shape[0] != centers.shape[0] or wss.shape[0] != centers.shape[0]:
        raise RuntimeError("field/cell count mismatch in {}".format(hits[-1]))
    q_inf = 0.5 * U * U
    cp = (p / q_inf).astype(np.float32)
    cf = (wss / q_inf).astype(np.float32)
    normals = decoder_normals(decoder, latent, centers)
    cd, cl = integrate_cd_cl(cp, cf, normals, areas, direction)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.savez(out_path,
             centers=centers.astype(np.float32), normals=normals,
             areas=areas.astype(np.float32), cp=cp, cf=cf,
             cd_gt=np.asarray(float(cd), dtype=np.float32),
             cl_gt=np.asarray(float(cl), dtype=np.float32))
    return out_path


def validate_surface_npz(path):
    """Hard checks on an exported surface npz (spec section 3.3)."""
    data = np.load(path)
    for key in ("centers", "normals", "areas", "cp", "cf", "cd_gt", "cl_gt"):
        if key not in data.files:
            raise RuntimeError("surface npz {} missing '{}'".format(path, key))
    n_faces = data["centers"].shape[0]
    if n_faces < 1000:
        raise RuntimeError("surface npz {} has only {} faces".format(
            path, n_faces))
    for key in ("centers", "normals", "areas", "cp", "cf"):
        if not np.isfinite(data[key]).all():
            raise RuntimeError("surface npz {} has non-finite {}".format(
                path, key))
    norms = np.linalg.norm(data["normals"], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("surface npz {} normals not unit".format(path))
    cd = float(data["cd_gt"])
    if not (1e-3 < cd < 2.0):
        raise RuntimeError("surface npz {} cd_gt={} out of range".format(
            path, cd))
```

- [ ] **Step 3: 单 case 冒烟（球 a0.5b0.5c0.5，端到端约 2-4 min）**

用后台或加长超时运行：

```bash
.venv/bin/python - <<'EOF'
import os, numpy as np, torch
import deep_sdf.cfd.openfoam_runner as ofr
from generate_openfoam_snapshots import (make_case_from_template,
    _stl_cache_path, load_manifest)
from deep_sdf.cfd.roadmap_deeponet import load_frozen_decoder, cache_key

root = "data/openfoam/ellipsoids_u10"
name = "ellipsoids/ellipsoid/ellipsoid_a0.5_b0.5_c0.5.npz"
names, latents = load_manifest(os.path.join(root, "lhs_latents.npz"))
i = names.index(name)
bc = np.load(os.path.join(root, "snapshots",
    "ellipsoids_ellipsoid_ellipsoid_a0.5_b0.5_c0.5_case000.npz"))["bc"]
decoder, _ = load_frozen_decoder("examples/ellipsoids_of4/specs.json",
                                 "examples/ellipsoids")

case_dir = os.path.join(root, "cases", "smoke_surface_sphere")
make_case_from_template(os.path.join(root, "template_case"), case_dir,
                        _stl_cache_path(os.path.join(root, "stls"), name), bc)
ofr.run_case(case_dir, n_procs=1)
out = os.path.join(root, "surface", cache_key(name) + ".npz")
ofr.export_surface(case_dir, out, bc, decoder, latents[i])
ofr.validate_surface_npz(out)

d = np.load(out)
F = d["centers"].shape[0]
cp_max = float(d["cp"].max()); cd = float(d["cd_gt"]); cl = float(d["cl_gt"])
print("faces=%d cp_max=%.3f cd=%.4f cl=%.4f" % (F, cp_max, cd, cl))
assert 0.5 < cp_max < 1.5          # 驻点 Cp ≈ 1
assert 0.3 < cd < 2.0              # 球 Re=100 量级先验（文献 ~1.0）
assert abs(cl) < 0.2               # 球几乎零升力
assert abs(float(np.linalg.norm(d["normals"], axis=1).mean()) - 1) < 1e-3
import shutil; shutil.rmtree(case_dir, ignore_errors=True)
print("SURFACE-SMOKE-OK")
EOF
```
Expected: 末尾 `SURFACE-SMOKE-OK`（faces 数万级、cp_max≈1、cd 0.3-2、|cl| 小）。若 `wallShearStress` 或 surfaces FO 调用形式在 OF14 有差异，读 `case_dir/log.wallShearStress`、`log.surfaces` 定位并修正（修正后把差异记入报告）。

- [ ] **Step 4: Commit**

```bash
git add deep_sdf/cfd/openfoam_runner.py
git commit -m "Add wall-surface export (wallShearStress + surfaces FO -> Cp/Cf/Cd/Cl npz)"
```

---

### Task 2: 批量表面场重跑（--surface_only）并后台启动全量 281

**Files:**
- Modify: `generate_openfoam_snapshots.py`（追加 `--surface_only` 分支与两个函数）

**Interfaces:**
- Consumes: Task 1 的 `export_surface`/`validate_surface_npz`；既有 `load_manifest`、`load_decoder`、`_stl_cache_path`、`make_case_from_template`、`snapshot_filename`、`ensure_template_case`；roadmap_deeponet 的 `cache_key`。
- Produces:
  - CLI `--surface_only`（与 `--skip_existing`、`--jobs`、`--keep_cases` 组合）
  - `run_surface_case(spec, template_dir, decoder, keep_case=False) -> dict`（never raises）
  - `run_surface_batch(args) -> None`（写 `<root>/surface/surface_summary.json`）
  - 后台批跑产物：`data/openfoam/ellipsoids_u10/surface/*.npz`（281 个）

- [ ] **Step 1: 修改 generate_openfoam_snapshots.py**

parser 区（`parser.add_argument("--skip_existing", ...)` 附近）加：

```python
    parser.add_argument("--surface_only", action="store_true",
                        help="re-run cases only to export wall surface "
                             "fields (DeepONet v2 W2a)")
```

`main()` 开头（`args = parser.parse_args()` 与 logging 初始化之后、网格/decoder 加载之前）加：

```python
    if args.surface_only:
        run_surface_batch(args)
        return
```

文件末尾追加：

```python
# --------------------------------------------------------------------------
# DeepONet v2 W2a: wall-surface field re-export
# --------------------------------------------------------------------------

def run_surface_case(spec, template_dir, decoder, keep_case=False):
    """One surface case: clone template -> run -> export surface npz ->
    validate. Never raises; returns a result dict."""
    result = {"shape": spec["shape"]}
    t0 = time.time()
    stage = "case setup"
    try:
        make_case_from_template(
            template_dir, spec["case_dir"], spec["stl"], spec["bc"])
        stage = "meshing/solver"
        openfoam_runner.run_case(spec["case_dir"], n_procs=1)
        stage = "surface export"
        openfoam_runner.export_surface(
            spec["case_dir"], spec["out_path"], spec["bc"], decoder,
            spec["latent"])
        stage = "validation"
        openfoam_runner.validate_surface_npz(spec["out_path"])
    except Exception as e:
        result.update(status="failed", stage=stage,
                      error="{}: {}".format(type(e).__name__, e))
    else:
        if not keep_case:
            shutil.rmtree(spec["case_dir"], ignore_errors=True)
        result.update(status="ok")
    result["elapsed"] = round(time.time() - t0, 2)
    return result


def run_surface_batch(args):
    """Re-run all manifest shapes at their existing snapshot BC and export
    wall surface fields to <root>/surface/<cache_key>.npz."""
    from deep_sdf.cfd.roadmap_deeponet import cache_key
    root = args.root
    stl_dir = os.path.join(root, "stls")
    surface_root = os.path.join(root, "surface")
    cases_root = os.path.join(root, "cases")
    snapshots_root = os.path.join(root, "snapshots")
    os.makedirs(surface_root, exist_ok=True)
    os.makedirs(cases_root, exist_ok=True)
    names, latents = load_manifest(os.path.join(root, "lhs_latents.npz"))
    decoder, _ = load_decoder(args.experiment, args.checkpoint)
    decoder = decoder.cuda()
    template_dir = os.path.join(root, "template_case")
    ensure_template_case(template_dir)
    specs = []
    skipped = 0
    for i, name in enumerate(names):
        out_path = os.path.join(surface_root, cache_key(name) + ".npz")
        if args.skip_existing and os.path.isfile(out_path):
            try:
                openfoam_runner.validate_surface_npz(out_path)
                skipped += 1
                continue
            except RuntimeError:
                pass
        stl = _stl_cache_path(stl_dir, name)
        if not os.path.isfile(stl):
            raise RuntimeError("missing STL for shape {}: {}".format(
                name, stl))
        snap_path = os.path.join(
            snapshots_root, snapshot_filename(name, 0))
        bc = np.load(snap_path)["bc"]
        specs.append({"shape": name, "latent": latents[i], "stl": stl,
                      "bc": bc,
                      "case_dir": os.path.join(
                          cases_root, cache_key(name) + "_surface"),
                      "out_path": out_path})
    logging.info("surface batch: %d to run, %d skipped (valid existing)",
                 len(specs), skipped)
    results, t0, done = [], time.time(), 0
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(args.jobs)) as pool:
        futures = [pool.submit(run_surface_case, s, template_dir, decoder,
                               args.keep_cases) for s in specs]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if res["status"] == "ok":
                logging.info("surface %d/%d %s: ok in %.1fs", done,
                             len(specs), res["shape"], res["elapsed"])
            else:
                logging.warning("surface %d/%d %s: FAILED at %s: %s",
                                done, len(specs), res["shape"],
                                res["stage"], res["error"])
    ok = sum(1 for r in results if r["status"] == "ok")
    summary = {"total": len(specs) + skipped, "ran": len(specs),
               "ok": ok, "skipped": skipped,
               "failed": [r for r in results if r["status"] != "ok"],
               "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(surface_root, "surface_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    logging.info("surface batch done: %d ok / %d failed / %d skipped",
                 ok, len(specs) - ok, skipped)
    if ok + skipped != len(names):
        raise SystemExit("surface batch incomplete: {}".format(
            os.path.join(surface_root, "surface_summary.json")))
```

- [ ] **Step 2: 后台启动全量批跑**

```bash
cd /home/siqi/CLionProjects/DeepSDF
nohup .venv/bin/python generate_openfoam_snapshots.py \
    --root data/openfoam/ellipsoids_u10 \
    --experiment examples/ellipsoids_of4 \
    --surface_only --skip_existing --jobs 12 \
    > data/openfoam/ellipsoids_u10/surface_batch.log 2>&1 &
echo "launched pid $!"
```
（Task 1 冒烟已导出球体 npz，--skip_existing 会跳过它。预计 1.5~2.5 小时。本任务**不等待其完成**：启动后观察前 2 个 case 成功即算本步通过——`sleep 400 && tail -5 data/openfoam/ellipsoids_u10/surface_batch.log` 应见 `surface 1/... ok` / `surface 2/... ok` 且无 FAILED。全量完成的门禁在 Task 6。）

- [ ] **Step 3: Commit**

```bash
git add generate_openfoam_snapshots.py
git commit -m "Add --surface_only batch re-export of wall Cp/Cf fields"
```

---

### Task 3: 共享数据管线 helper + 误差分析 --analyze

**Files:**
- Modify: `deep_sdf/cfd/roadmap_deeponet.py`（追加 prepare_experiment / evaluate_detailed）
- Modify: `train_roadmap_deeponet.py`（parse_args 加 `--analyze`/`--out_name`；main 加 analyze 分支）

**Interfaces:**
- Consumes: MVP 已有接口（load_frozen_decoder/build_geometry_cache/snapshot_index/load_shapes/cluster_split/compute_stats/evaluate/predict_normalized）。
- Produces:
  - `prepare_experiment(experiment_dir, specs_path, checkpoint, data_root, cfg) -> dict`，键：`grid_points (G,3) cpu`、`grid_shape`、`axis`、`names` (list[str])、`latents` (S,16) np、`split` (dict train/val/test -> names)、`labels` (S,) np、`snap_idx`、`cache_dir`、`train_shapes`、`stats`、`load(name_list) -> list[dict]`（闭包）
  - `evaluate_detailed(model, shapes, grid_points, stats, cfg, n_points, seed, near_band=0.15, chunk=2**18) -> dict`：`{"cases": {name: {"rel_l2", "per_var" [4], "near_rel", "far_rel"}}}`（near = 采样点中 |sdf|<near_band 的子集 rel L2，4 通道平均；far 同理其余点；子集为空则 None）
  - 训练器 CLI：`--analyze`（读 out_dir/best.mdlus 写 analysis.json）、`--out_name <str>`（默认 RoadmapONet，--smoke 时 RoadmapONet_smoke；替换原硬编码目录名）

- [ ] **Step 1: roadmap_deeponet.py 追加**

DEFAULT_CFG 增三个键（放 "# split / stats" 注释行之后）：

```python
    # analysis / adaptive sampling
    "analyze_points": 65536, "adaptive_points": 4096, "adaptive_eps": 0.1,
```

文件末尾追加：

```python
def prepare_experiment(experiment_dir, specs_path, checkpoint, data_root,
                       cfg):
    """Shared data pipeline for analysis / surface / physics stages: frozen
    decoder geometry cache (built if missing), snapshot index, cluster
    split, train shapes + stats, and a lazy shape loader. Returns the dict
    documented in the plan (grid_points/grid_shape/axis/names/latents/
    split/labels/snap_idx/cache_dir/train_shapes/stats/load)."""
    from deep_sdf.cfd.volume import make_stretched_grid
    from generate_openfoam_snapshots import load_manifest
    decoder, _ = load_frozen_decoder(specs_path, experiment_dir, checkpoint)
    grid_points, grid_shape, axis = make_stretched_grid()
    names, latents = load_manifest(
        os.path.join(data_root, "lhs_latents.npz"))
    cache_dir = os.path.join(data_root, "sdf_cache")
    build_geometry_cache(
        decoder, [(n, torch.from_numpy(latents[i]))
                  for i, n in enumerate(names)],
        grid_points, cache_dir, near_band=cfg["near_band"])
    del decoder
    torch.cuda.empty_cache()
    snap_idx = snapshot_index(os.path.join(data_root, "snapshots"))
    split, labels = cluster_split(names, latents,
                                  n_clusters=cfg["n_clusters"],
                                  seed=cfg["seed"])
    n_grid = grid_points.shape[0]

    def load(name_list):
        sel = [names.index(n) for n in name_list]
        return load_shapes(name_list, latents[sel], snap_idx, cache_dir,
                           n_grid)

    train_shapes = load(split["train"])
    stats = compute_stats(train_shapes, n_sample=cfg["stats_sample"],
                          seed=cfg["seed"])
    return {"grid_points": grid_points, "grid_shape": grid_shape,
            "axis": axis, "names": names, "latents": latents,
            "split": split, "labels": labels, "snap_idx": snap_idx,
            "cache_dir": cache_dir, "train_shapes": train_shapes,
            "stats": stats, "load": load}


@torch.no_grad()
def evaluate_detailed(model, shapes, grid_points, stats, cfg, n_points, seed,
                      near_band=0.15, chunk=2 ** 18):
    """evaluate() plus per-case near-wall (|sdf|<near_band) vs far masked
    rel L2. Returns {"cases": {name: {"rel_l2", "per_var", "near_rel",
    "far_rel"}}}; a mask subset that is empty yields None for that entry."""
    import zlib
    device = next(model.parameters()).device
    stats_g = {k: v.to(device) for k, v in stats.items()}
    cases = {}
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

        def masked_rel(mask):
            if int(mask.sum()) == 0:
                return None
            return float(np.mean([
                ((pred[mask, v] - y[mask, v]).norm() /
                 y[mask, v].norm().clamp_min(1e-12)).item()
                for v in range(4)]))

        near = sdf.squeeze(1).abs() < near_band
        cases[s["name"]] = {
            "rel_l2": float(np.mean(rel_v)),
            "per_var": [float(r) for r in rel_v],
            "near_rel": masked_rel(near),
            "far_rel": masked_rel(~near),
        }
    return {"cases": cases}
```

- [ ] **Step 2: 训练器加 --out_name / --analyze**

`train_roadmap_deeponet.py` 的 parse_args 加（`--iters` 行之后）：

```python
    p.add_argument("--out_name", default=None,
                   help="output dir under <experiment>/ (default "
                        "RoadmapONet, or RoadmapONet_smoke with --smoke)")
    p.add_argument("--analyze", action="store_true",
                   help="write analysis.json (near/far masked per-case "
                        "errors, worst-10) from <out_name>/best.mdlus")
```

main() 中 out_dir 一行（`out_dir = os.path.join(args.experiment_directory, "RoadmapONet_smoke" if args.smoke else "RoadmapONet")`）替换为：

```python
    out_name = args.out_name or (
        "RoadmapONet_smoke" if args.smoke else "RoadmapONet")
    out_dir = os.path.join(args.experiment_directory, out_name)
```

在 `if args.eval_only:` 分支之前插入 analyze 分支：

```python
    if args.analyze:
        from physicsnemo.experimental.models.xdeeponet.deeponet import (
            DeepONet)
        model = DeepONet.from_checkpoint(
            os.path.join(out_dir, "best.mdlus")).to(device).eval()
        val_shapes = load(split["val"])
        test_shapes = load(split["test"])
        lat_t = torch.from_numpy(latents).float()
        label_map = {n: int(l) for n, l in zip(names, labels)}
        # 簇心只用 train 形状的 latent
        train_idx = [names.index(n) for n in split["train"]]
        centroids = {}
        for c in np.unique(labels[train_idx]):
            rows = [i for i in train_idx if labels[i] == c]
            centroids[int(c)] = lat_t[rows].mean(0)

        out = {"cases": []}
        for split_name, shapes in (("val", val_shapes),
                                   ("test", test_shapes)):
            det = rd.evaluate_detailed(
                model, shapes, grid_points, stats_g, cfg,
                n_points=cfg["analyze_points"], seed=777)
            for nm, rec in det["cases"].items():
                zi = lat_t[names.index(nm)]
                cen = centroids[label_map[nm]]
                rec = dict(rec)
                rec.update(split=split_name, name=nm,
                           cluster=label_map[nm],
                           dist_to_centroid=float((zi - cen).norm()))
                out["cases"].append(rec)
        out["cases"].sort(key=lambda r: -r["rel_l2"])
        out["worst10"] = [r["name"] for r in out["cases"][:10]]
        with open(os.path.join(out_dir, "analysis.json"), "w") as f:
            json.dump(out, f, indent=1)
        logging.info("worst-10 by rel_l2:")
        for r in out["cases"][:10]:
            logging.info("  %-58s %s rel %.4f near %s far %s cluster %d "
                         "dist %.3f", r["name"], r["split"], r["rel_l2"],
                         "%.4f" % r["near_rel"] if r["near_rel"] is not None
                         else "n/a",
                         "%.4f" % r["far_rel"] if r["far_rel"] is not None
                         else "n/a",
                         r["cluster"], r["dist_to_centroid"])
        return
```

注意：analyze 分支用到 `names`/`labels`/`latents`/`load`/`stats_g`——这些名字在 main() 现有代码里已存在（names/latents 来自 load_manifest，load 是闭包，stats_g 在模型构建段）；`labels` 当前 main() 里 cluster_split 已返回（`split, labels = rd.cluster_split(...)`）。若个别变量名与现状不符，以 main() 实际变量名为准并在报告中注明。

- [ ] **Step 3: 验证（对 MVP best 跑 analyze，约 1-2 min）**

```bash
.venv/bin/python train_roadmap_deeponet.py --analyze
```
Expected：无异常；日志打印 worst-10 表格（含 near/far/cluster/dist 列）；`examples/ellipsoids/RoadmapONet/analysis.json` 存在且 `cases` 长度 = 93（36 val + 57 test）、`worst10` 长度 10。快速断言：

```bash
.venv/bin/python -c "
import json
d = json.load(open('examples/ellipsoids/RoadmapONet/analysis.json'))
assert len(d['cases']) == 93 and len(d['worst10']) == 10
r = d['cases'][0]
assert r['rel_l2'] > 0 and r['near_rel'] is not None and r['far_rel'] is not None
print('ANALYZE-OK worst=%s rel=%.4f near=%.4f far=%.4f' % (
    r['name'], r['rel_l2'], r['near_rel'], r['far_rel']))
"
```

- [ ] **Step 4: Commit**

```bash
git add deep_sdf/cfd/roadmap_deeponet.py train_roadmap_deeponet.py
git commit -m "Add prepare_experiment, detailed eval and --analyze/--out_name"
```

---

### Task 4: 动态误差自适应采样 + field 增强重训

**Files:**
- Modify: `train_roadmap_deeponet.py`（parse_args 加 `--adaptive_sampling`；训练循环加权重逻辑）

**Interfaces:**
- Consumes: Task 3 的 main() 现状；`rd.evaluate(...)` 的 `per_case` 返回；cfg 新键 `adaptive_points`/`adaptive_eps`（Task 3 已加入 DEFAULT_CFG）。
- Produces: `--adaptive_sampling`；产物 `examples/ellipsoids/RoadmapONet_adaptive/`（训练 + eval.json，不入 git）。

- [ ] **Step 1: 修改训练器**

parse_args 加：

```python
    p.add_argument("--adaptive_sampling", action="store_true",
                   help="shape-level error-adaptive case sampling "
                        "(spec section 4.2)")
```

训练循环段：在 `metrics_path = ...` 之前（`gen = torch.Generator()...` 之后）加：

```python
    shape_weights = None
    if args.adaptive_sampling:
        shape_weights = torch.ones(len(train_shapes))
```

case 选取两行（`cases = [train_shapes[int(torch.randint(...))] for _ in ...]`）替换为：

```python
        if shape_weights is not None:
            picks = torch.multinomial(shape_weights, cfg["batch_cases"],
                                      replacement=True,
                                      generator=gen).tolist()
            cases = [train_shapes[i] for i in picks]
        else:
            cases = [train_shapes[int(torch.randint(0, len(train_shapes),
                                                    (1,), generator=gen))]
                     for _ in range(cfg["batch_cases"])]
```

评估块（`if (it + 1) % cfg["eval_every"] == 0 or it + 1 == cfg["iters"]:` 内，`if res["rel_l2"] < best:` 之后）追加：

```python
            if args.adaptive_sampling:
                model.eval()
                quick = rd.evaluate(model, train_shapes, grid_points,
                                    stats_g, cfg,
                                    n_points=cfg["adaptive_points"],
                                    seed=1000 + it)
                model.train()
                errs = torch.tensor([quick["per_case"][s["name"]]
                                     for s in train_shapes])
                shape_weights = errs / errs.mean().clamp_min(1e-12) \
                    + cfg["adaptive_eps"]
                top = torch.argsort(shape_weights, descending=True)[:5]
                logging.info("adaptive top-5: %s",
                             [train_shapes[i]["name"] for i in top])
                rec = {"iter": it + 1, "adaptive_mean_err":
                       float(errs.mean()),
                       "adaptive_worst_err": float(errs.max())}
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
```

- [ ] **Step 2: 验证（field 增强正式 run，约 8-10 min，后台）**

```bash
.venv/bin/python train_roadmap_deeponet.py --adaptive_sampling \
    --out_name RoadmapONet_adaptive --iters 20000
.venv/bin/python train_roadmap_deeponet.py --eval_only --resume \
    --out_name RoadmapONet_adaptive
```
Expected：`RoadmapONet_adaptive/` 全套产物；metrics.jsonl 含 `adaptive_*` 记录与 adaptive top-5 日志；eval.json 中 `val.rel_l2 <= 0.27`（MVP 基准 0.2513 允许波动）。随后核对 MVP 与 adaptive 两 run 的 worst-10（验收第 1 条）：

```bash
.venv/bin/python - <<'EOF'
import json
mvp = json.load(open("examples/ellipsoids/RoadmapONet/eval.json"))
adp = json.load(open("examples/ellipsoids/RoadmapONet_adaptive/eval.json"))
ana = json.load(open("examples/ellipsoids/RoadmapONet/analysis.json"))
w10 = ana["worst10"]
mvp_val = mvp["val"]["per_case"]; adp_val = adp["val"]["per_case"]
both = [n for n in w10 if n in mvp_val and n in adp_val]
m_mvp = sum(mvp_val[n] for n in both) / max(len(both), 1)
m_adp = sum(adp_val[n] for n in both) / max(len(both), 1)
print("WORST10-MEAN mvp=%.4f adaptive=%.4f (n=%d)" % (m_mvp, m_adp, len(both)))
print("VAL mvp=%.4f adaptive=%.4f | TEST mvp=%.4f adaptive=%.4f" % (
    mvp["val"]["rel_l2"], adp["val"]["rel_l2"],
    mvp["test"]["rel_l2"], adp["test"]["rel_l2"]))
EOF
```
Expected: 打印两行对比；若 `VAL adaptive > 0.27`，状态 DONE_WITH_CONCERNS 并附 metrics 曲线特征，不自行调参。

- [ ] **Step 3: Commit**

```bash
git add train_roadmap_deeponet.py
git commit -m "Add error-adaptive shape sampling (--adaptive_sampling)"
```

---

### Task 5: roadmap_surface.py 库（surface 模型/统计/积分/评估 + stage runner + smoke）

**Files:**
- Create: `deep_sdf/cfd/roadmap_surface.py`
- Modify: `deep_sdf/cfd/roadmap_deeponet.py`（DEFAULT_CFG 增 surface 键）

**Interfaces:**
- Consumes: Task 1 的 surface npz 契约与 `integrate_cd_cl`；MVP 的 `predict_normalized`/trunk 特征/`build_model` 模式；`DeepONet.branch1` 共享；Task 3 的 `prepare_experiment`。
- Produces:
  - `SURF_VARS = ("cp", "cfx", "cfy", "cfz")`
  - `load_surface(surface_dir, name) -> dict(centers (F,3), normals (F,3), areas (F,), cp (F,), cf (F,3), cd_gt float, cl_gt float)`（CPU tensors；加载时经 `validate_surface_npz` 校验，无效 npz 抛 RuntimeError）
  - `load_surface_exclusions(surface_dir) -> set[str]`（读 `excluded_shapes.json`，不存在则空集）
  - `attach_surface(shapes, surface_dir) -> shapes`（每 shape 增 `"surf"` 键；排除清单中的形状静默跳过、不挂 "surf"——调用方须在 attach 后过滤 `[s for s in shapes if "surf" in s]`）
  - `compute_surface_stats(train_shapes) -> dict`：`surf_mean/surf_std (4,)`、`force_mean/force_std (2,)`（force_std 下限 `0.1*cd_std`）
  - `surface_features(xyz (N,3), normal (N,3), n_bands, domain_half) -> (N, 42)`
  - `build_surface_model(field_model, cfg) -> (surface_model, force_head)`
  - `predict_surface_normalized(surface_model, latent, bc, xyz, normal, stats, cfg, amp=False) -> (N,4)`
  - `predict_force_normalized(force_head, latent, bc, stats) -> (2,)`
  - `integrate_cd_cl_mc(cp, cf, normals, area_total, direction) -> (cd, cl)`（面积加权 MC）
  - `evaluate_surface(surface_model, force_head, shapes, stats, cfg, chunk=2**18) -> dict`：`{"cp_rel", "cf_rel" [3], "cd_head", "cd_int", "cl_head", "cl_int"（各 split 形状的中位相对误差）, "per_case" {name: {...}}, "worst5" [...]}`；相对误差分母 `max(|gt|, 0.05)`
  - `run_surface_stage(args, cfg) -> None`（完整 stage：数据→训练→best 保存→eval_surface.json；支持 args.smoke）

- [ ] **Step 1: DEFAULT_CFG 增键（roadmap_deeponet.py）**

```python
    # surface stage
    "surface_trunk_hidden": 512, "surface_trunk_layers": 4,
    "force_hidden": 256, "force_layers": 2,
    "surface_points": 4096, "lambda_force": 1.0,
    "lambda_consistency": 0.1, "branch_lr_scale": 0.1,
    "surface_iters": 10000,
```

- [ ] **Step 2: 写 deep_sdf/cfd/roadmap_surface.py**

```python
#!/usr/bin/env python3
"""Surface stage (DeepONet v2 spec section 5): surface Cp/Cf DeepONet head
sharing the field branch, dual-path Cd/Cl (surface integration + direct
force head) with consistency loss, and the surface stage runner."""

import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from deep_sdf.cfd import physicsnemo_compat as _compat  # noqa: F401
from deep_sdf.cfd.roadmap_deeponet import (cache_key, fourier_encode,
                                           prepare_experiment)
from deep_sdf.cfd.openfoam_runner import integrate_cd_cl

SURF_VARS = ("cp", "cfx", "cfy", "cfz")


def load_surface_exclusions(surface_dir):
    """Read surface/excluded_shapes.json (degenerate latent-extrapolated
    shapes, plan Global Constraints); returns an empty set when absent."""
    path = os.path.join(surface_dir, "excluded_shapes.json")
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return set(json.load(f)["excluded"])


def load_surface(surface_dir, name):
    path = os.path.join(surface_dir, cache_key(name) + ".npz")
    from deep_sdf.cfd.openfoam_runner import validate_surface_npz
    validate_surface_npz(path)
    data = np.load(path)
    out = {k: torch.from_numpy(np.asarray(data[k], dtype=np.float32))
           for k in ("centers", "normals", "areas", "cp", "cf")}
    out["cd_gt"] = float(np.asarray(data["cd_gt"]))
    out["cl_gt"] = float(np.asarray(data["cl_gt"]))
    return out


def attach_surface(shapes, surface_dir):
    """Attach the 'surf' dict to each shape. Shapes on the exclusion list
    are skipped (no 'surf' key) - callers must filter them out afterwards:
    ``shapes = [s for s in shapes if "surf" in s]``."""
    excluded = load_surface_exclusions(surface_dir)
    for s in list(shapes):
        if s["name"] in excluded:
            logging.info("surface: excluding degenerate shape %s",
                         s["name"])
            continue
        if "surf" not in s:
            s["surf"] = load_surface(surface_dir, s["name"])
    return shapes


def compute_surface_stats(train_shapes):
    """z-score over all train surface points; force std floored at
    0.1*std_Cd per component (spec section 5.1)."""
    cp = torch.cat([s["surf"]["cp"] for s in train_shapes])
    cf = torch.cat([s["surf"]["cf"] for s in train_shapes])
    y = torch.cat([cp.unsqueeze(1), cf], dim=1)
    cd = torch.tensor([s["surf"]["cd_gt"] for s in train_shapes])
    cl = torch.tensor([s["surf"]["cl_gt"] for s in train_shapes])
    fstd = torch.stack([cd.std(), cl.std()])
    fstd = fstd.clamp_min(0.1 * float(cd.std()))
    return {"surf_mean": y.mean(0), "surf_std": y.std(0).clamp_min(1e-8),
            "force_mean": torch.stack([cd.mean(), cl.mean()]),
            "force_std": fstd}


def surface_features(xyz, normal, n_bands, domain_half):
    """(N,3),(N,3) -> (N, 6 + 3*2*n_bands): normalized coords + face
    normal + Fourier(coords)."""
    xn = xyz / domain_half
    return torch.cat([xn, normal, fourier_encode(xn, n_bands)], dim=-1)


def build_surface_model(field_model, cfg):
    """Surface DeepONet sharing field_model.branch1; plus the direct force
    head (z,bc)->(Cd,Cl). Returns (surface_model, force_head)."""
    from physicsnemo.models.mlp import FullyConnected
    from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
    trunk_in = 3 + 3 + 3 * 2 * cfg["fourier_bands"]
    surface_trunk = FullyConnected(
        in_features=trunk_in, layer_size=cfg["surface_trunk_hidden"],
        num_layers=cfg["surface_trunk_layers"], out_features=cfg["width"],
        activation_fn="silu")
    surface_model = DeepONet(
        field_model.branch1, trunk=surface_trunk, dimension=3,
        width=cfg["width"], out_channels=len(SURF_VARS),
        decoder_type="mlp", decoder_width=cfg["decoder_hidden"],
        decoder_layers=cfg["decoder_layers"], decoder_activation_fn="silu")
    force_head = FullyConnected(
        in_features=cfg["latent_size"] + cfg["bc_dim"],
        layer_size=cfg["force_hidden"], num_layers=cfg["force_layers"],
        out_features=2, activation_fn="silu")
    return surface_model, force_head


def _branch_input(latent, bc, stats):
    z = (latent - stats["z_mean"]) / stats["z_std"]
    b = (bc - stats["bc_mean"]) / stats["bc_std"]
    return torch.cat([z, b]).unsqueeze(0)                    # (1, 20)


def predict_surface_normalized(surface_model, latent, bc, xyz, normal,
                               stats, cfg, amp=False):
    xt = surface_features(xyz, normal, cfg["fourier_bands"],
                          cfg["domain_half"])
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        y = surface_model(_branch_input(latent, bc, stats), xt)[0]
    return y.float()                                         # (N, 4)


def predict_force_normalized(force_head, latent, bc, stats):
    return force_head(_branch_input(latent, bc, stats))[0]   # (2,)


def integrate_cd_cl_mc(cp, cf, normals, area_total, direction):
    """Area-weighted Monte-Carlo estimate of the force integral from
    sampled faces: each sample represents area_total / n of surface."""
    n_pts = cp.shape[0]
    a = torch.full((n_pts,), float(area_total) / n_pts,
                   device=cp.device, dtype=cp.dtype)
    return integrate_cd_cl(cp, cf, normals, a, direction)


def _rel_err(pred, gt, floor=0.05):
    return float(abs(float(pred) - float(gt)) / max(abs(float(gt)), floor))


@torch.no_grad()
def evaluate_surface(surface_model, force_head, shapes, stats, cfg,
                     chunk=2 ** 18):
    """Full-surface eval per shape: Cp rel L2, per-component Cf rel L2,
    Cd/Cl relative error for the head and integral paths. Returns the dict
    documented in the plan."""
    device = next(surface_model.parameters()).device
    stats_g = {k: v.to(device) for k, v in stats.items()}
    per_case = {}
    for s in shapes:
        surf = s["surf"]
        n_faces = surf["centers"].shape[0]
        lat = s["latent"].to(device)
        bc = s["bc"].to(device)
        preds = []
        for head in range(0, n_faces, chunk):
            sl = slice(head, min(head + chunk, n_faces))
            pn = predict_surface_normalized(
                surface_model, lat, bc, surf["centers"][sl].to(device),
                surf["normals"][sl].to(device), stats_g, cfg, amp=False)
            preds.append(pn * stats_g["surf_std"] + stats_g["surf_mean"])
        pred = torch.cat(preds)                                # (F, 4)
        cp_gt = surf["cp"].to(device)
        cf_gt = surf["cf"].to(device)
        cp_rel = ((pred[:, 0] - cp_gt).norm()
                  / cp_gt.norm().clamp_min(1e-12)).item()
        cf_rel = [((pred[:, 1 + v] - cf_gt[:, v]).norm() /
                   cf_gt[:, v].norm().clamp_min(1e-12)).item()
                  for v in range(3)]
        direction = bc[1:4]
        cd_int, cl_int = integrate_cd_cl(pred[:, 0], pred[:, 1:4],
                                         surf["normals"].to(device),
                                         surf["areas"].to(device),
                                         direction)
        fh = predict_force_normalized(force_head, lat, bc, stats_g)
        fh = fh * stats_g["force_std"] + stats_g["force_mean"]
        per_case[s["name"]] = {
            "cp_rel": float(cp_rel),
            "cf_rel": [float(r) for r in cf_rel],
            "cd_head": _rel_err(fh[0], surf["cd_gt"]),
            "cd_int": _rel_err(cd_int, surf["cd_gt"]),
            "cl_head": _rel_err(fh[1], surf["cl_gt"]),
            "cl_int": _rel_err(cl_int, surf["cl_gt"]),
            "cd_gt": surf["cd_gt"], "cl_gt": surf["cl_gt"],
            "cd_int_val": float(cd_int), "cd_head_val": float(fh[0]),
        }

    def median(key):
        vals = [r[key] for r in per_case.values()]
        return float(np.median(vals)) if vals else None

    worst = sorted(per_case.items(), key=lambda kv: -kv[1]["cd_int"])[:5]
    return {"cp_rel": median("cp_rel"),
            "cf_rel": [float(np.median([r["cf_rel"][v]
                                        for r in per_case.values()]))
                       for v in range(3)],
            "cd_head": median("cd_head"), "cd_int": median("cd_int"),
            "cl_head": median("cl_head"), "cl_int": median("cl_int"),
            "per_case": per_case,
            "worst5": [{"name": n, "cd_int": r["cd_int"],
                        "cd_gt": r["cd_gt"], "cd_int_val": r["cd_int_val"]}
                       for n, r in worst]}


def run_surface_stage(args, cfg):
    """Surface stage runner: init branch from the field best checkpoint,
    train surface trunk/decoder + force head (branch at 0.1x lr), save
    best_surface.mdlus + best_surface_force.pth + metrics_surface.jsonl,
    then write eval_surface.json."""
    from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
    out_dir = args._out_dir
    device = torch.device("cuda")
    init_from = args.init_from or os.path.join(
        args.experiment_directory, "RoadmapONet", "best.mdlus")
    field_model = DeepONet.from_checkpoint(init_from)
    surface_model, force_head = build_surface_model(field_model, cfg)
    surface_model.to(device)
    force_head.to(device)
    del field_model

    prep = prepare_experiment(args.experiment_directory, args.specs,
                              args.checkpoint, args.data_root, cfg)
    surface_dir = os.path.join(args.data_root, "surface")
    grid_points = prep["grid_points"]
    split = prep["split"]
    train_shapes = [s for s in attach_surface(prep["train_shapes"],
                                              surface_dir)
                    if "surf" in s]
    if not train_shapes:
        raise RuntimeError("no train shapes with valid surface data")
    stats = prep["stats"]
    stats.update(compute_surface_stats(train_shapes))
    stats_g = {k: v.to(device) for k, v in stats.items()}
    torch.save(stats, os.path.join(out_dir, "stats_surface.pth"))
    with open(os.path.join(out_dir, "config_surface.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    iters = 500 if args.smoke else cfg["surface_iters"]
    eval_every = 100 if args.smoke else cfg["eval_every"]
    param_groups = [
        {"params": surface_model.branch1.parameters(),
         "lr": cfg["lr"] * cfg["branch_lr_scale"]},
        {"params": [p for n, p in surface_model.named_parameters()
                    if not n.startswith("branch1.")], "lr": cfg["lr"]},
        {"params": force_head.parameters(), "lr": cfg["lr"]},
    ]
    opt = torch.optim.AdamW(param_groups,
                            weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    def lr_at(it):
        t = min(it / max(iters, 1), 1.0)
        return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (
            1 + np.cos(np.pi * t))

    val_shapes = [s for s in attach_surface(prep["load"](split["val"]),
                                            surface_dir) if "surf" in s]
    if args.smoke:
        val_shapes = val_shapes[:8]
    metrics_path = os.path.join(out_dir, "metrics_surface.jsonl")
    state_path = os.path.join(out_dir, "train_state_surface.pth")
    gen = torch.Generator().manual_seed(cfg["seed"])
    start_iter, best = 0, float("inf")
    if args.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=device)
        surface_model.load_state_dict(st["model_state_dict"])
        force_head.load_state_dict(st["force_head_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        start_iter, best = st["iter"], st["best"]
        logging.info("surface resumed from iter %d (best %.4f)",
                     start_iter, best)

    def val_metric():
        surface_model.eval(); force_head.eval()
        res = evaluate_surface(surface_model, force_head, val_shapes,
                               stats_g, cfg)
        surface_model.train(); force_head.train()
        return res

    t0 = time.time()
    for it in range(start_iter, iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it) * (
                cfg["branch_lr_scale"]
                if g is param_groups[0] else 1.0)
        cases = [train_shapes[int(torch.randint(0, len(train_shapes),
                                                (1,), generator=gen))]
                 for _ in range(cfg["batch_cases"])]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for c in cases:
            surf = c["surf"]
            probs = surf["areas"]
            idx = torch.multinomial(probs, cfg["surface_points"],
                                    replacement=True, generator=gen)
            xyz = surf["centers"][idx].to(device)
            nrm = surf["normals"][idx].to(device)
            y = torch.cat([surf["cp"][idx].unsqueeze(1),
                           surf["cf"][idx]], dim=1).to(device)
            lat = c["latent"].to(device)
            bc = c["bc"].to(device)
            pred = predict_surface_normalized(
                surface_model, lat, bc, xyz, nrm, stats_g, cfg,
                amp=cfg["amp"])
            yn = (y - stats_g["surf_mean"]) / stats_g["surf_std"]
            l_surf = sum(F.mse_loss(pred[:, v], yn[:, v]) for v in range(4))
            pred_phys = pred * stats_g["surf_std"] + stats_g["surf_mean"]
            cd_int, cl_int = integrate_cd_cl_mc(
                pred_phys[:, 0], pred_phys[:, 1:4], nrm,
                surf["areas"].sum(), bc[1:4])
            fh = predict_force_normalized(force_head, lat, bc, stats_g)
            gt = torch.stack([
                (torch.as_tensor(c["surf"]["cd_gt"], device=device)
                 - stats_g["force_mean"][0]) / stats_g["force_std"][0],
                (torch.as_tensor(c["surf"]["cl_gt"], device=device)
                 - stats_g["force_mean"][1]) / stats_g["force_std"][1]])
            ints = torch.stack([
                (cd_int - stats_g["force_mean"][0])
                / stats_g["force_std"][0],
                (cl_int - stats_g["force_mean"][1])
                / stats_g["force_std"][1]])
            l_force = (ints - gt).abs().sum() + (fh - gt).abs().sum()
            l_cons = ((fh - ints) ** 2).sum()
            loss = loss + l_surf \
                + cfg["lambda_force"] * l_force \
                + cfg["lambda_consistency"] * l_cons
        loss = loss / len(cases)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (it + 1) % cfg["metrics_every"] == 0 or it == start_iter:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iter": it + 1, "train_loss": float(loss.item()),
                    "lr": lr_at(it),
                    "sec_per_iter": (time.time() - t0)
                    / (it + 1 - start_iter)}) + "\n")

        if (it + 1) % eval_every == 0 or it + 1 == iters:
            res = val_metric()
            crit = res["cp_rel"] + res["cd_int"]
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iter": it + 1, "val_cp_rel": res["cp_rel"],
                    "val_cf_rel": res["cf_rel"],
                    "val_cd_int": res["cd_int"],
                    "val_cd_head": res["cd_head"]}) + "\n")
            logging.info("surface iter %d loss %.4f cp_rel %.4f "
                         "cd_int %.4f cd_head %.4f",
                         it + 1, float(loss.item()), res["cp_rel"],
                         res["cd_int"], res["cd_head"])
            if crit < best:
                best = crit
                surface_model.save(os.path.join(out_dir,
                                                "best_surface.mdlus"))
                torch.save(force_head.state_dict(),
                           os.path.join(out_dir, "best_surface_force.pth"))
            torch.save({"model_state_dict": surface_model.state_dict(),
                        "force_head_state_dict": force_head.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "iter": it + 1, "best": best}, state_path)

    if not args.smoke:
        surface_model.eval(); force_head.eval()
        test_shapes = [s for s in attach_surface(prep["load"](split["test"]),
                                                 surface_dir)
                       if "surf" in s]
        res = {"val": evaluate_surface(surface_model, force_head,
                                       val_shapes, stats_g, cfg),
               "test": evaluate_surface(surface_model, force_head,
                                        test_shapes, stats_g, cfg)}
        with open(os.path.join(out_dir, "eval_surface.json"), "w") as f:
            json.dump(res, f, indent=1)
        logging.info("surface test: cp_rel %.4f cd_int %.4f cd_head %.4f",
                     res["test"]["cp_rel"], res["test"]["cd_int"],
                     res["test"]["cd_head"])
    logging.info("surface stage done. best crit %.4f", best)
```

注意：`run_surface_stage(args, cfg)` 期望 `args._out_dir` 已由训练器 main() 设置（Task 6 接线）。

- [ ] **Step 3: 库级探针验证（用 Task 1 冒烟的球体 npz，不训练）**

```bash
.venv/bin/python - <<'EOF'
import torch
from deep_sdf.cfd import roadmap_deeponet as rd
from deep_sdf.cfd import roadmap_surface as rs
from deep_sdf.cfd.roadmap_deeponet import DEFAULT_CFG
import numpy as np, os

cfg = dict(DEFAULT_CFG)
surf = rs.load_surface("data/openfoam/ellipsoids_u10/surface",
                       "ellipsoids/ellipsoid/ellipsoid_a0.5_b0.5_c0.5.npz")
F = surf["centers"].shape[0]
assert F > 1000 and surf["normals"].shape == (F, 3)
xt = rs.surface_features(torch.randn(9, 3), torch.randn(9, 3), 6, 1.5)
assert xt.shape == (9, 42), xt.shape

# GT 积分自洽：npz 的 cd_gt 与 integrate_cd_cl 重算一致
cd, cl = rs.integrate_cd_cl(surf["cp"], surf["cf"], surf["normals"],
                            surf["areas"], torch.tensor([1.0, 0, 0]))
assert abs(float(cd) - surf["cd_gt"]) < 1e-4, (float(cd), surf["cd_gt"])

# MC 积分收敛性：面积加权抽 4096 面，多次均值应逼近精确值（10% 内）
gen = torch.Generator().manual_seed(0)
cds = []
for _ in range(20):
    idx = torch.multinomial(surf["areas"], 4096, replacement=True, generator=gen)
    c, _ = rs.integrate_cd_cl_mc(surf["cp"][idx], surf["cf"][idx],
                                 surf["normals"][idx],
                                 surf["areas"].sum(), torch.tensor([1., 0, 0]))
    cds.append(float(c))
assert abs(np.mean(cds) - float(cd)) / abs(float(cd)) < 0.1, np.mean(cds)

# 共享 branch 的模型构建与前向
import deep_sdf.cfd.physicsnemo_compat  # noqa
from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
field = rd.build_model(cfg)
sm, fh = rs.build_surface_model(field, cfg)
assert sm.branch1 is field.branch1
sm.cuda(); fh.cuda()
stats = {"z_mean": torch.zeros(16, device="cuda"), "z_std": torch.ones(16, device="cuda"),
         "bc_mean": torch.zeros(4, device="cuda"), "bc_std": torch.ones(4, device="cuda")}
y = rs.predict_surface_normalized(sm, torch.randn(16, device="cuda"),
                                  torch.tensor([10., 1, 0, 0], device="cuda"),
                                  torch.randn(500, 3, device="cuda"),
                                  torch.randn(500, 3, device="cuda"),
                                  stats, cfg, amp=True)
assert y.shape == (500, 4) and torch.isfinite(y).all()
f = rs.predict_force_normalized(fh, torch.randn(16, device="cuda"),
                                torch.tensor([10., 1, 0, 0], device="cuda"), stats)
assert f.shape == (2,)
y.square().mean().backward()
g1 = [p.grad is not None for p in field.branch1.parameters()]
assert all(g1)  # 共享 branch 收到梯度
print("SURFACE-LIB-OK faces=%d cd=%.4f" % (F, float(cd)))
EOF
```
Expected: `SURFACE-LIB-OK faces=... cd=...`。

- [ ] **Step 4: Commit**

```bash
git add deep_sdf/cfd/roadmap_surface.py deep_sdf/cfd/roadmap_deeponet.py
git commit -m "Add surface stage library (shared-branch surface DeepONet + force head)"
```

---

### Task 6: 训练器接线 --stage surface + 全量表面数据门禁 + 正式训练

**Files:**
- Modify: `train_roadmap_deeponet.py`（parse_args 加 `--stage`/`--init_from`；main 开头加 stage dispatch 与 `args._out_dir`）

**Interfaces:**
- Consumes: Task 5 的 `run_surface_stage`；Task 2 的批量产物。
- Produces: `RoadmapONet{,_smoke}/` 下 `best_surface.mdlus`、`best_surface_force.pth`、`train_state_surface.pth`、`metrics_surface.jsonl`、`stats_surface.pth`、`config_surface.json`、`eval_surface.json`。

- [ ] **Step 1: 训练器接线**

parse_args 加：

```python
    p.add_argument("--stage", default="field",
                   choices=("field", "surface", "physics"))
    p.add_argument("--init_from", default=None,
                   help="explicit checkpoint path for non-field stages "
                        "(default <experiment>/RoadmapONet/best.mdlus)")
```

main() 中 `os.makedirs(out_dir, exist_ok=True)` 之后、`# --- data ---` 段之前插入：

```python
    args._out_dir = out_dir
    if args.stage == "surface":
        from deep_sdf.cfd import roadmap_surface
        roadmap_surface.run_surface_stage(args, cfg)
        return
    if args.stage == "physics":
        raise SystemExit("physics stage is added by a later task")
```

- [ ] **Step 2: 表面数据门禁（Task 2 批跑必须已完成；276 有效 + 5 排除，见 Global Constraints 排除条款）**

```bash
.venv/bin/python - <<'EOF'
import json, os, glob
s = json.load(open("data/openfoam/ellipsoids_u10/surface/surface_summary.json"))
exc = json.load(open("data/openfoam/ellipsoids_u10/surface/excluded_shapes.json"))
assert s["ok"] + s["skipped"] == 276, s
assert len(s["failed"]) == 5, s["failed"]
assert sorted(f["shape"] for f in s["failed"]) == sorted(exc["excluded"])
from deep_sdf.cfd.openfoam_runner import validate_surface_npz
valid = 0
for p in glob.glob("data/openfoam/ellipsoids_u10/surface/*.npz"):
    try:
        validate_surface_npz(p)
        valid += 1
    except RuntimeError:
        pass
assert valid == 276, valid
print("SURFACE-DATA-OK 276 valid, 5 excluded")
EOF
```
Expected: `SURFACE-DATA-OK 276 valid, 5 excluded`。若批跑仍在进行（surface_summary.json 不存在）：等待其完成（`tail -f` 批跑日志）再继续，不要在本步启动第二次批跑。

- [ ] **Step 3: smoke + 正式训练 + 评估**

```bash
.venv/bin/python train_roadmap_deeponet.py --stage surface --smoke
.venv/bin/python train_roadmap_deeponet.py --stage surface
```
Expected：smoke 无异常、产物齐（无 eval_surface.json 属预期——smoke 跳过最终评估）；正式 run（10000 iters，估 10-20 min）日志每 500 iter 打印 `surface iter ... cp_rel ... cd_int ...`；结束后 eval_surface.json 存在且 `test.cd_int` 中位相对误差 **< 0.10**（验收第 3 条）。若未达标：DONE_WITH_CONCERNS 附 metrics 与 eval_surface.json 要点，不自行调参。

- [ ] **Step 4: Commit**

```bash
git add train_roadmap_deeponet.py
git commit -m "Wire --stage surface into the roadmap trainer"
```

---

### Task 7: roadmap_physics.py + --stage physics + 训练评估

**Files:**
- Create: `deep_sdf/cfd/roadmap_physics.py`
- Modify: `deep_sdf/cfd/roadmap_deeponet.py`（DEFAULT_CFG 增 physics 键）
- Modify: `train_roadmap_deeponet.py`（physics dispatch 替换占位 SystemExit；parse_args 加 `--lambda_phys`）

**Interfaces:**
- Consumes: `deep_sdf/cfd/physics.py` 的 `PDEInformer`、`IncompressibleNS(re)`、`CollocationSampler()`、`physics_weight_schedule(progress)`、`noslip_loss(q)` 语义；`deep_sdf.utils.decode_sdf`；MVP 的 `trunk_features`/`predict_normalized`/`sample_case_batch`/`evaluate`；Task 3 的 `prepare_experiment`。
- Produces:
  - `predict_physical(model, latent, bc, xyz, sdf, normal, stats, cfg, amp=False) -> (N,4)`（反标准化）
  - `physics_losses(model, decoder, latent, bc, points, stats, cfg, informer, max_batch=256, backward_scale=None) -> (lc, lm)`（逐 chunk backward；latent 为**原始** latent，分支输入内部再做 z-score）
  - `boundary_losses(model, shape, bc, grid_points, idx_wall, idx_far, stats, cfg) -> (l_wall, l_far)`（noslip |u|²；far u→dir 且 cp→0）
  - `eval_residuals(model, decoder, shapes, grid_points, grid_shape, stats, cfg, informer, sampler, n_points, seed, device) -> dict`（`{"continuity", "momentum", "wall", "far"}` 跨形状均值，backward_scale=None）
  - `run_physics_stage(args, cfg) -> None`（init field best → data+physics 微调 → best_physics.mdlus / metrics_physics.jsonl / eval_physics.json；支持 smoke）
  - 训练器：`--stage physics` + `--lambda_phys <float|None>`

- [ ] **Step 1: DEFAULT_CFG 增键（roadmap_deeponet.py）**

```python
    # physics stage
    "n_collocation": 4096, "phys_chunk": 256, "lambda_phys": None,
    "physics_iters": 10000, "re": 100.0, "phys_lr_scale": 0.1,
```

- [ ] **Step 2: 写 deep_sdf/cfd/roadmap_physics.py**

```python
#!/usr/bin/env python3
"""Physics stage (DeepONet v2 spec section 6): PDE fine-tuning of the
field model with steady incompressible NS residuals (PDEInformer), wall /
far-field boundary losses, and the physics stage runner. Follows the
PiPOD train_pipod_deeponet.physics_losses pattern: SDF through the frozen
decoder WITH graph, SDF gradient detached, per-chunk backward, AMP off
for the residual path."""

import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from deep_sdf.cfd.roadmap_deeponet import (trunk_features,
                                           predict_normalized,
                                           sample_case_batch, evaluate,
                                           prepare_experiment)
from deep_sdf.utils import decode_sdf


def predict_physical(model, latent, bc, xyz, sdf, normal, stats, cfg,
                     amp=False):
    pn = predict_normalized(model, latent, bc, xyz, sdf, normal, stats,
                            cfg, amp=amp)
    return pn * stats["y_std"] + stats["y_mean"]


def physics_losses(model, decoder, latent, bc, points, stats, cfg,
                   informer, max_batch=256, backward_scale=None):
    """Continuity + momentum residuals at collocation points (second-order
    autodiff). ``latent`` is the RAW latent (decoder input); the branch
    z-score is applied inside. With ``backward_scale`` each chunk's loss
    is backwarded immediately (scaled) and detached scalars returned."""
    device = points.device
    n_tot = points.shape[0]
    lc = torch.zeros((), device=device)
    lm = torch.zeros((), device=device)
    z = (latent.reshape(-1) - stats["z_mean"]) / stats["z_std"]
    b = (bc.reshape(-1) - stats["bc_mean"]) / stats["bc_std"]
    xb = torch.cat([z, b]).unsqueeze(0)                        # (1, 20)
    for lo in range(0, n_tot, max_batch):
        chunk = points[lo:lo + max_batch]
        p = chunk.detach().requires_grad_(True)
        with torch.enable_grad():
            d = decode_sdf(decoder, latent.reshape(1, -1), p)
            g = torch.autograd.grad(d.sum(), p, create_graph=True)[0]
            n = (g / g.norm(dim=1, keepdim=True).clamp_min(1e-8)).detach()
            xt = trunk_features(p, d, n, cfg["fourier_bands"],
                                cfg["domain_half"])
            with torch.amp.autocast("cuda", enabled=False):
                qn = model(xb, xt)[0].float()
                q = qn * stats["y_std"] + stats["y_mean"]
                res = informer({"coordinates": p, "u": q[:, 0:1],
                                "v": q[:, 1:2], "w": q[:, 2:3],
                                "cp": q[:, 3:4]})
                l_c = (res["continuity"] ** 2).sum()
                l_m = sum((res["momentum_" + k] ** 2).sum()
                          for k in "uvw")
            if backward_scale is not None:
                (backward_scale * (l_c + l_m) / max(n_tot, 1)).backward()
        lc = lc + l_c.detach()
        lm = lm + l_m.detach()
    return lc / max(n_tot, 1), lm / max(n_tot, 1)


def boundary_losses(model, shape, bc, grid_points, idx_wall, idx_far,
                    stats, cfg):
    """noslip |u|^2 on the wall band + far-field u->dir, cp->0 (PiPOD
    boundary_losses pattern; features from the geometry cache)."""
    device = grid_points.device
    l_wall = torch.zeros((), device=device)
    l_far = torch.zeros((), device=device)
    direction = bc[1:4] / bc[1:4].norm()
    if idx_wall.numel():
        qw = predict_physical(
            model, shape["latent"].to(device), bc, grid_points[idx_wall],
            shape["sdf"][idx_wall].unsqueeze(1).to(device),
            shape["normal"][idx_wall].to(device), stats, cfg, amp=False)
        l_wall = (qw[:, :3] ** 2).sum(1).mean()
    if idx_far.numel():
        qf = predict_physical(
            model, shape["latent"].to(device), bc, grid_points[idx_far],
            shape["sdf"][idx_far].unsqueeze(1).to(device),
            shape["normal"][idx_far].to(device), stats, cfg, amp=False)
        tgt = torch.zeros_like(qf)
        tgt[:, :3] = direction.to(device)
        l_far = ((qf - tgt) ** 2).sum(1).mean()
    return l_wall, l_far


@torch.no_grad()
def eval_residuals(model, decoder, shapes, grid_points, grid_shape, stats,
                   cfg, informer, sampler, n_points, seed, device):
    """Mean continuity/momentum/wall/far metrics over shapes on a fixed
    collocation sample per shape (no backward)."""
    from deep_sdf.cfd.physics import CollocationSampler  # noqa: F401
    grid_g = grid_points.to(device)
    gen = torch.Generator().manual_seed(seed)
    out = {k: [] for k in ("continuity", "momentum", "wall", "far")}
    for s in shapes:
        picks = sampler.sample(
            grid_g, grid_shape, s["sdf"].to(device),
            s["fields"].to(device), s["bc"], n_points, gen)
        lc, lm = physics_losses(
            model, decoder, s["latent"].to(device),
            s["bc"].to(device), grid_g[picks["collocation"]], stats, cfg,
            informer, max_batch=cfg["phys_chunk"], backward_scale=None)
        lw, lf = boundary_losses(model, s, s["bc"].to(device), grid_g,
                                 picks["wall"], picks["far"], stats, cfg)
        out["continuity"].append(float(lc))
        out["momentum"].append(float(lm))
        out["wall"].append(float(lw))
        out["far"].append(float(lf))
    return {k: float(np.mean(v)) for k, v in out.items()}


def run_physics_stage(args, cfg):
    """Physics stage runner: init from the field best, fine-tune with
    L_field + lambda_phys(t) * (L_cont + L_mom + L_wall + L_far)."""
    from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet
    from deep_sdf.cfd.physics import (PDEInformer, IncompressibleNS,
                                      CollocationSampler,
                                      physics_weight_schedule)
    from deep_sdf.cfd.roadmap_deeponet import (build_model,
                                               load_frozen_decoder)
    out_dir = args._out_dir
    device = torch.device("cuda")
    init_from = args.init_from or os.path.join(
        args.experiment_directory, "RoadmapONet", "best.mdlus")
    model = DeepONet.from_checkpoint(init_from).to(device)
    decoder, _ = load_frozen_decoder(args.specs,
                                     args.experiment_directory,
                                     args.checkpoint)

    prep = prepare_experiment(args.experiment_directory, args.specs,
                              args.checkpoint, args.data_root, cfg)
    grid_points, grid_shape = prep["grid_points"], prep["grid_shape"]
    train_shapes = prep["train_shapes"]
    stats = prep["stats"]
    stats_g = {k: v.to(device) for k, v in stats.items()}
    with open(os.path.join(out_dir, "config_physics.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    iters = 500 if args.smoke else cfg["physics_iters"]
    eval_every = 100 if args.smoke else cfg["eval_every"]
    informer = PDEInformer(IncompressibleNS(re=cfg["re"]).equations)
    sampler = CollocationSampler()
    grid_g = grid_points.to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["lr"] * cfg["phys_lr_scale"],
                            weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    lambda_fixed = args.lambda_phys

    def lr_at(it):
        base = cfg["lr"] * cfg["phys_lr_scale"]
        lo = cfg["lr_min"] * cfg["phys_lr_scale"]
        t = min(it / max(iters, 1), 1.0)
        return lo + 0.5 * (base - lo) * (1 + np.cos(np.pi * t))

    val_shapes = prep["load"](prep["split"]["val"])
    if args.smoke:
        val_shapes = val_shapes[:8]
    metrics_path = os.path.join(out_dir, "metrics_physics.jsonl")
    state_path = os.path.join(out_dir, "train_state_physics.pth")
    gen = torch.Generator().manual_seed(cfg["seed"])
    start_iter, best = 0, float("inf")
    if args.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location=device)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        start_iter, best = st["iter"], st["best"]
        logging.info("physics resumed from iter %d (best %.4f)",
                     start_iter, best)

    def run_eval(n_points):
        model.eval()
        res = evaluate(model, val_shapes, grid_points, stats_g, cfg,
                       n_points=n_points, seed=12345)
        model.train()
        return res

    t0 = time.time()
    model.train()
    for it in range(start_iter, iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        progress = it / max(iters, 1)
        lam = (lambda_fixed if lambda_fixed is not None
               else physics_weight_schedule(progress))
        cases = [train_shapes[int(torch.randint(0, len(train_shapes),
                                                (1,), generator=gen))]
                 for _ in range(cfg["batch_cases"])]
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        phys_terms = {}
        for c in cases:
            b = sample_case_batch(c, grid_points, cfg["n_points"],
                                  cfg["near_frac"], gen, device)
            pred = predict_normalized(model, b["latent"], b["bc"],
                                      b["xyz"], b["sdf"], b["normal"],
                                      stats_g, cfg, amp=cfg["amp"])
            yn = (b["y"] - stats_g["y_mean"]) / stats_g["y_std"]
            loss = loss + sum(F.mse_loss(pred[:, v], yn[:, v])
                              for v in range(4))
            if lam > 0.0:
                picks = sampler.sample(
                    grid_g, grid_shape, c["sdf"].to(device),
                    c["fields"].to(device), c["bc"],
                    cfg["n_collocation"], gen)
                lc, lm = physics_losses(
                    model, decoder, c["latent"].to(device),
                    c["bc"].to(device), grid_g[picks["collocation"]],
                    stats_g, cfg, informer,
                    max_batch=cfg["phys_chunk"], backward_scale=lam)
                lw, lf = boundary_losses(
                    model, c, c["bc"].to(device), grid_g, picks["wall"],
                    picks["far"], stats_g, cfg)
                loss = loss + lam * (lw + lf)
                phys_terms = {"cont": float(lc), "mom": float(lm),
                              "wall": float(lw), "far": float(lf),
                              "lambda_phys": lam}
        loss = loss / len(cases)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (it + 1) % cfg["metrics_every"] == 0 or it == start_iter:
            rec = {"iter": it + 1, "train_loss": float(loss.item()),
                   "lr": lr_at(it),
                   "sec_per_iter": (time.time() - t0)
                   / (it + 1 - start_iter)}
            rec.update(phys_terms)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if (it + 1) % eval_every == 0 or it + 1 == iters:
            res = run_eval(cfg["eval_points"])
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iter": it + 1, "val_rel_l2": res["rel_l2"],
                    "val_u": res["per_var"][0], "val_v": res["per_var"][1],
                    "val_w": res["per_var"][2], "val_p": res["per_var"][3],
                    "val_baseline": res["baseline"]}) + "\n")
            logging.info("physics iter %d loss %.4f val %.4f lam %.3g",
                         it + 1, float(loss.item()), res["rel_l2"], lam)
            if res["rel_l2"] < best:
                best = res["rel_l2"]
                model.save(os.path.join(out_dir, "best_physics.mdlus"))
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "iter": it + 1, "best": best}, state_path)

    if not args.smoke:
        model.eval()
        best_model = DeepONet.from_checkpoint(
            os.path.join(out_dir, "best_physics.mdlus")).to(device).eval()
        field_model = DeepONet.from_checkpoint(init_from).to(device).eval()
        res = {
            "val": evaluate(best_model, val_shapes, grid_points, stats_g,
                            cfg, cfg["final_eval_points"], 777),
            "residuals_physics": eval_residuals(
                best_model, decoder, val_shapes, grid_points, grid_shape,
                stats_g, cfg, informer, sampler, cfg["n_collocation"],
                4242, device),
            "residuals_field_only": eval_residuals(
                field_model, decoder, val_shapes, grid_points, grid_shape,
                stats_g, cfg, informer, sampler, cfg["n_collocation"],
                4242, device),
        }
        with open(os.path.join(out_dir, "eval_physics.json"), "w") as f:
            json.dump(res, f, indent=1)
        logging.info("physics eval: val %.4f | cont %.4g -> %.4g | "
                     "mom %.4g -> %.4g",
                     res["val"]["rel_l2"],
                     res["residuals_field_only"]["continuity"],
                     res["residuals_physics"]["continuity"],
                     res["residuals_field_only"]["momentum"],
                     res["residuals_physics"]["momentum"])
    logging.info("physics stage done. best val %.4f", best)
```

- [ ] **Step 3: 训练器接线**

parse_args 加：

```python
    p.add_argument("--lambda_phys", type=float, default=None,
                   help="fixed physics loss weight (default: schedule)")
```

main() 中 physics 占位（`raise SystemExit("physics stage is added by a later task")`）替换为：

```python
    if args.stage == "physics":
        from deep_sdf.cfd import roadmap_physics
        roadmap_physics.run_physics_stage(args, cfg)
        return
```

同时 parse_args 的 `--stage` choices 已含 "physics"（Task 6 接线时已列入），若 Task 6 未列入则补上。

- [ ] **Step 4: smoke + 正式训练 + 评估**

```bash
.venv/bin/python train_roadmap_deeponet.py --stage physics --smoke
.venv/bin/python train_roadmap_deeponet.py --stage physics
```
Expected：smoke 无异常（注意 λ 日程在 500 iters 的 20% 处才起物理项，前 100 iter 为纯数据）；正式 run（10000 iters；物理路径 fp32 二阶导较慢，估 30-90 min）结束后 eval_physics.json 满足验收第 4 条：`residuals_physics.continuity` 与 `momentum` 均低于 `residuals_field_only`，且 `val.rel_l2` ≤ field best（0.2500）+ 0.03。若未达标：DONE_WITH_CONCERNS 附数字，不自行调参。

- [ ] **Step 5: Commit**

```bash
git add deep_sdf/cfd/roadmap_physics.py deep_sdf/cfd/roadmap_deeponet.py train_roadmap_deeponet.py
git commit -m "Add physics fine-tune stage (PDEInformer NS residuals + boundary losses)"
```

---

## Self-Review 记录

- 规格覆盖：§3 W2a→T1/T2；§4.1 analyze→T3；§4.2 动态加权→T4；§5 surface→T5/T6；§6 physics→T7；§7 编排→T3(--out_name)/T6/T7 接线；§8 验收→T2 门禁（2 条）、T4 Step2（1）、T6 Step3（3）、T7 Step4（4）、各 smoke（5）。✅
- 依赖序：T1→T2（批跑后台）→T6 门禁；T3→T4；T5→T6；T7 独立（仅需 MVP best）。执行序 T1,T2,T3,T4,T5,T7,T6 可让 T6 前的批跑有最大完成时间；**实施顺序按此**（T7 先于 T6）。
- 类型一致性：surface npz 键（T1 写 = T5 读）；stats 键 surf_mean/surf_std/force_mean/force_std（T5 内一致）；`args._out_dir`（T6 设置 = T5/T7 使用）；`run_surface_stage(args, cfg)`/`run_physics_stage(args, cfg)` 签名一致；`integrate_cd_cl`（T1 定义 = T5 import）。
- 已知取舍：physics stage 的 boundary/PDE 特征用缓存 SDF（tanh 空间）与 CollocationSampler 默认参数，与 PiPOD of4 stage-3 同口径；far 池（sdf>1.0 tanh 空间恒空）回退 uniform，为既有行为。
- smoke 模式 surface/physics 跳最终评估（省时间），正式 run 才写 eval_surface/eval_physics.json——验收点都在正式 run。

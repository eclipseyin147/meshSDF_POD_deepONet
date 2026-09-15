# ellipsoid_u10 LHS-300 + OpenFOAM 批量生成 + PiPOD-DeepONet stage 1-3 流水线

日期：2026-09-15
状态：已批准（用户确认）

## 目标

基于 `generate_openfoam_snapshots.py` 的 `--lhs` 批量模式，对 DeepSDF latent code
**全部重新采样**约 300 个 LHS 训练形状；每个形状以固定边界条件 `[U=10, dir=(1,0,0)]`
生成 1 个 OpenFOAM case（12 并发：建 case → blockMesh/snappyHexMesh/simpleFoam →
foamPostProcess 探针采样 → npz 快照）；然后运行 `train_pipod_deeponet.py` 的
stage 1 → 2 → 3。

## 现状（调研结论）

- `data/openfoam/ellipsoids_u10/lhs_latents.npz`：今天 10:15 由旧 seed 生成的
  manifest，27 椭球 + 254 LHS = 281 形状。**本次全部弃用重采**（备份后重建）。
- `snapshots/` 中仅剩 `ellipsoid_a0.5_b0.5_c0.5_of4_view.npz`（stretched grid，
  1,442,897 点，bc=[10,1,0,0]，23MB）。上一个批次（244 case 全 ok）的快照已删除。
  该文件与新批次快照文件名不冲突，保留。
- `cases/lhs_shape_000..003_case000`：10:21 被中途 kill 的 4 个半成品（停在
  snappyHexMesh，非真正网格失败），删除。
- `examples/ellipsoids_of4/pipod_config.json` 已指向本 root 的 manifest 与
  snapshots，训练侧无需改路径。
- 环境：OpenFOAM 14（/opt/openfoam14）、2×V100-32GB、256 核、251GB RAM、193GB 磁盘空闲。

## 设计

### 交付物

仓库根目录新增一个 bash 驱动脚本 `run_u10_pipeline.sh`，5 个阶段顺序执行，
全流程日志 tee 到 `data/openfoam/ellipsoids_u10/pipeline_<时间戳>.log`。
脚本内 Python 一律用项目内 `.env/bin/python`（Stage 0 第 0 步创建）。

### Stage 0 — 环境准备 + 预备 + 冒烟测试

0. **Python 环境**：本机系统 python3.12 无 torch/foamlib，且磁盘上不存在任何含这些
   包的虚拟环境（已全盘搜索确认）。用 uv（`/home/s/paraview/bin/uv`，已确认可用）
   在项目根创建 `.env` venv：torch==2.6.0（cu126 wheel，须含 sm_70 以支持 V100）、
   foamlib、plyfile、scikit-image、trimesh、sympy、numpy、scipy。
   验证：`torch.cuda.is_available()` 为 True 且 `get_arch_list()` 含 sm_70，
   并能在 `.env` 下 import `generate_openfoam_snapshots` / `train_pipod_deeponet` /
   `foamlib`。
1. `lhs_latents.npz` → `lhs_latents.npz.pre300.bak`（移走，避免 run_batch 把旧
   254 个 LHS 形状合并进新 manifest）；删除 4 个残留 case 目录。
2. 冒烟测试：临时 root（`data/openfoam/_smoke_u10`）+
   `--lhs 4 --jobs 2 --fixed_bc 10 1 0 0 --grid_stretch` 跑通 ≥1 个 case 全链路，
   确认 mesh/solver/postProcess 正常后删除临时 root。失败则中止整个流水线。

### Stage 1 — LHS 重采样 + OpenFOAM 批量生成

```
source /opt/openfoam14/etc/bashrc
python3 generate_openfoam_snapshots.py \
  --root data/openfoam/ellipsoids_u10 \
  --lhs 310 --seed 3 \
  --fixed_bc 10 1 0 0 \
  --jobs 12 --grid_stretch \
  --experiment examples/ellipsoids_of4 \
  --data_source data/ellipsoids
```

- 全新 manifest：27 训练椭球 + 通过网格有效性检查的 LHS 样本（310 采样预留 ~2%
  拒绝余量，历史上 256 采 254 过）。
- 每个形状 1 个 case（`--fixed_bc` 覆盖 cases_per_shape/u_range），共 ~332 个
  case；12 并发、每 case 单核。快照写 stretched grid，与训练配置一致。
- 产出：`snapshots/` 下 ~332 个新 npz（× 23MB ≈ 7.6GB，外加保留的 of4_view）、
  `lhs_latents.npz`、
  `batch_summary.json`。预计 2.5~4 小时。
- 幂等：已存在且 bc 匹配的快照自动跳过，失败 case 保留目录可重跑补齐。

### Stage 2 — 训练配置更新

`examples/ellipsoids_of4/pipod_config.json` 仅改两处：

- `iters`: 40000 → 20000（stage 1/2/3 共用顶层值）
- `n_field`: 16384 → 60000（stage 2/3 每场损失采样点数）

其余不动：grid_stretch、cosine lr、stage1/2 lr=1e-3、stage3 lr=1e-4 +
wall_bc=noslip + re=100、val_fraction=0.2、field_near_frac=0.3、phys_chunk=256。

### Stage 3 — stage 1→2→3 链式训练

依次执行（脚本自动从上一 stage checkpoint 链式 init）：

```
python3 train_pipod_deeponet.py --config examples/ellipsoids_of4/pipod_config.json --stage 1
python3 train_pipod_deeponet.py --config examples/ellipsoids_of4/pipod_config.json --stage 2
python3 train_pipod_deeponet.py --config examples/ellipsoids_of4/pipod_config.json --stage 3
```

产出：`examples/ellipsoids_of4/PipodONet/stage{1,2,3}.pth`、
`metrics_stage{1,2,3}.jsonl`。

### 错误处理

- Stage 1 结束后解析 `batch_summary.json`：若 `cases.failed` 非空，打印失败列表并
  **中止**后续训练（`build_shapes` 要求 manifest 中每个形状至少有一个快照，否则
  抛错）。用户可重跑同一命令补齐后再次启动。
- 任一阶段命令非零退出即中止流水线（`set -euo pipefail`）。

### 不做的事（YAGNI）

- 不修改 `generate_openfoam_snapshots.py` / `train_pipod_deeponet.py` 任何 Python 代码。
- 不做自动监控/自动推进下一阶段的循环逻辑。
- 不清理 `of4_view.npz`、不动 `examples/ellipsoids_of4` 的 DeepSDF 权重。

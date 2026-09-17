# RoadmapONet：训练数据生成与 DeepONet 训练操作手册

> 适用代码：分支 `roadmap-deeponet-mvp` 上的 `generate_openfoam_snapshots.py` + `deep_sdf/cfd/roadmap_deeponet.py` + `train_roadmap_deeponet.py`（+ `roadmap_surface.py` / `roadmap_physics.py`）。
> 设计依据：`DeepSDF_PhysicsNeMo_DeepONet_CFD_Roadmap.md`；规格见 `docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md` 与 `2026-09-16-deeponet-v2-surface-physics-design.md`。

## 0. 架构一句话

冻结的 DeepSDF decoder（16 维 latent）提供几何全局编码与 SDF/法向特征；PhysicsNeMo xDeepONet 的 branch 吃 `[z, bc]`、trunk 吃 `[x,y,z,SDF,法向]+Fourier`，输出体积场 `[u,v,w,Cp]`；surface stage 加壁面 `[Cp,Cf]` 头与 Cd/Cl 双路预测；physics stage 用 NS 残差微调。

## 1. 前置条件

| 项 | 位置/命令 | 说明 |
|---|---|---|
| Python 环境 | `.venv/bin/python` | torch 2.5.1+cu121；physicsnemo 经 `deep_sdf/cfd/physicsnemo_compat.py` 免安装加载（sys.path + torch shim），**不要** pip install physicsnemo |
| OpenFOAM 14 | `/opt/openfoam14` | 无需手动 source（`openfoam_runner._run` 自动 fallback source bashrc） |
| 冻结 DeepSDF | `examples/ellipsoids/ModelParameters/latest.pth` + `examples/ellipsoids_of4/specs.json` | 训练脚本从 of4 读 specs（`examples/ellipsoids/specs.json` 已删除，勿用 `--experiment examples/ellipsoids` 跑数据生成） |
| 27 解析锚点 | `data/openfoam/ellipsoids_u10/analytic27_latents.npz` | 27 个解析椭球的 names+latents（从旧 manifest 提取的保留件） |

## 2. 训练数据生成

数据根目录：`data/openfoam/ellipsoids_u10/`（下称 `$ROOT`）。

### 2.1 一条命令（完整批跑）

```bash
cd /home/siqi/CLionProjects/DeepSDF

# 首次/重建：用 27 锚点做 manifest 种子
cp $ROOT/analytic27_latents.npz $ROOT/lhs_latents.npz

nohup .venv/bin/python generate_openfoam_snapshots.py \
    --root $ROOT \
    --experiment examples/ellipsoids_of4 \
    --lhs 500 --latent_max_norm 1.0 \
    --fixed_bc 10 1 0 0 --cases_per_shape 1 \
    --export_surface --grid_stretch \
    --jobs 12 \
    > $ROOT/batch.log 2>&1 &
```

参数说明：

| 参数 | 作用 |
|---|---|
| `--lhs 500` | 在 27 个训练 latent 的逐维 [min,max]（外扩 10%）内 LHS 采 500 个新形状 |
| `--latent_max_norm 1.0` | **关键**：拒绝 ‖z‖>1.0（DeepSDF CodeBound）的候选并补采。不过滤时约 85% 样本超界，decoder 外推出退化微小体（上轮 5 个表面网格塌缩形状的根因） |
| `--fixed_bc 10 1 0 0` | 固定来流 U=10、方向 +x；每形状 1 个 case（只采形状，不采工况） |
| `--export_surface` | 同一次求解内同时导出体积场与壁面场（比"先体积后表面两遍跑"省一半求解） |
| `--grid_stretch` | **关键**：113³ 拉伸网格（1442897 点，域 [-1.5,1.5]³）。漏掉会退化为 64³ 均匀网格，与训练管线不兼容 |
| `--jobs 12` | 12 个 case 并发（RTX 3070 + 62GB 内存实测安全） |
| `--skip_existing` | （可选）断点续跑：已有且校验通过的产物跳过 |

### 2.2 批跑内部流程（每形状自动完成）

```text
latent z ──decoder→marching cubes──> STL（合法性校验：>500 面、顶点∈[-1.2,1.2]³、无 NaN）
解析椭球名 ──解析 STL（按半轴缩放 icosphere）──> STL
        │
        ▼
make_case（模板克隆 + foamlib 写入 bc/尾迹加密盒）
        ▼
run_case：blockMesh 36³ → snappyHexMesh（表面 level 4 + 6 层边界层 + 尾迹加密）→ potentialFoam → simpleFoam（层流 ν=0.1，Re=100）
        ▼
体积：probes 在共享 113³ 拉伸网格采样 → Cp=p/(½U²)、体内 u=0/Cp=1 → snapshots/<name>_case000.npz
表面：wallShearStress FO（solver 内嵌 + 重启 1 迭代）→ surfaces FO（patch body.*, vtk）→ 面心/面积 + decoder 法向 + Cp/Cf + 积分 cd_gt/cl_gt → surface/<key>.npz
        ▼
校验（load_snapshot / validate_surface_npz）→ 删 case 目录
```

### 2.3 数据契约

**体积快照** `snapshots/<shape名 / 转 _>_case000.npz`：

| 键 | 形状/类型 | 说明 |
|---|---|---|
| `fields` | (1442897, 4) f32 | `[u, v, w, Cp]`；**速度为原始 m/s**，训练加载时 `/bc[0]` 无量纲化；Cp 已无量纲 |
| `bc` | (4,) f32 | `[U, dir_x, dir_y, dir_z]` = `[10,1,0,0]` |
| `shape` | str | manifest 形状名（`ellipsoids/ellipsoid/...npz` 或 `lhs/shape_XXX.npz`） |

**表面** `surface/<cache_key>.npz`（cache_key = 名去 `.npz`、`/`→`_`）：

| 键 | 形状/类型 | 说明 |
|---|---|---|
| `centers` | (F,3) f32 | 壁面三角面心 |
| `normals` | (F,3) f32 | decoder SDF 梯度归一化（物体外向，**不是** patch 法向） |
| `areas` | (F,) f32 | 面元面积 |
| `cp` / `cf` | (F,) / (F,3) f32 | `p/(½U²)`、`−wallShearStress/(½U²)`（取负因 OF 输出为流体侧应力，已经三重验证对齐 forces FO） |
| `cd_gt` / `cl_gt` | 标量 | GT 积分气动力系数（`F=−ΣCp·n·A+ΣCf·A`，`A_ref=Σmax(0,−n·dir)·A`） |

**其他**：`lhs_latents.npz`（names+latents manifest，批跑就地合并扩写）；`surface/excluded_shapes.json`（退化形状排除清单，`attach_surface` 读取跳过；新一轮 max_norm≤1.0 采样正常应为空集）；`batch_summary.json`（批跑总结）。

**SDF 缓存** `$ROOT/sdf_cache/<key>.npz`（`sdf (G,)`、`normal (G,3)`、`fluid_idx`、`near_idx`）：由训练器首次运行时自动逐形状构建（冻结 decoder 在 113³ 网格上算，~33MB/形状），已存在自动跳过，可删可重建。

## 3. 训练 DeepONet（train_roadmap_deeponet.py）

### 3.1 三个 stage

```bash
# field（体积场，MVP 配置）
.venv/bin/python train_roadmap_deeponet.py --out_name RoadmapONet_r4

# surface（壁面 Cp/Cf + Cd/Cl；从 field best 接力）
.venv/bin/python train_roadmap_deeponet.py --stage surface --out_name RoadmapONet_r4

# physics（NS 残差微调；从 field best 接力）
.venv/bin/python train_roadmap_deeponet.py --stage physics --out_name RoadmapONet_r4
```

其他常用模式：

```bash
--smoke                 # 500 iters 快速全链路验证（写 RoadmapONet_smoke/）
--resume                # 从 train_state*.pth 续训
--eval_only             # 用 best*.mdlus 做最终评估（val+test 65536 点/形状 + 推理测速）
--analyze               # 产 analysis.json（近壁/远域误差分解、worst-10、latent 簇距离）
--adaptive_sampling     # field 阶段：按 train 形状误差动态加权采样（P∝err/mean+0.1）
--config <json>         # 覆盖任意超参（键见下表）
--init_from <path>      # surface/physics 的初始 checkpoint（默认 RoadmapONet/best.mdlus）
```

注意：`--eval_only`/`--analyze` 不带 `--config` 时自动读 `<out_dir>/config.json` 存档（修复后的行为），训练时才重写 config.json。

### 3.2 输出产物（`<out_dir>/`，默认 `examples/ellipsoids/RoadmapONet/`）

| 文件 | stage | 内容 |
|---|---|---|
| `best.mdlus` | field | val 最优模型（physicsnemo .mdlus，`DeepONet.from_checkpoint` 可载） |
| `train_state.pth` | field | model/optimizer/iter/best（--resume 用） |
| `metrics.jsonl` | field | 每 50 iter train_loss/lr + 每 500 iter val_rel_l2/分变量 |
| `eval.json` | field | val/test rel L2 + 分变量 + 均值基线 + 推理加速比 |
| `analysis.json` | field | `--analyze` 的逐 case 近壁/远域误差 + worst-10 |
| `best_surface.mdlus` + `best_surface_force.pth` | surface | 最优 surface 模型与力头（按 cp_rel+cd_int 判据） |
| `metrics_surface.jsonl` / `eval_surface.json` | surface | Cp/Cf rel L2、Cd/Cl 双路（积分/直接头）误差 |
| `best_physics.mdlus` / `metrics_physics.jsonl` / `eval_physics.json` | physics | 含 cont/mom/wall/far 残差与 field-only 对照 |

### 3.3 关键超参（DEFAULT_CFG，可 `--config` 覆盖）

| 键 | 默认 | 说明 |
|---|---|---|
| `fourier_bands` | 6 | 坐标 Fourier 频带数 |
| `feature_set` | `v1` | `v1`=43 维 `[x̃,sdf,n]+γ(x̃)`；`v2`=72 维（+scaled_sdf/inside/sdf·n，v3 消融显示 val 改善但 test 不泛化） |
| `loss_type` | `zmse` | `zmse`（z-score MSE，MVP 行为）/`relmse`（相对 MSE）/`huber` |
| `amp_dtype` | `fp16` | `bf16` 时自动不用 GradScaler |
| `early_stop_patience` | 0 | >0 时按评估周期数早停（如 8 = 4000 iters） |
| `batch_cases` / `n_points` | 4 / 8192 | 每 iter 4 形状 × 8192 点（30% 近壁带） |
| `iters` / `lr` / `lr_min` | 20000 / 1e-3 / 1e-5 | cosine 衰减 |
| `surface_iters` / `physics_iters` | 10000 / 10000 | 注意 `--iters` 只对 field stage 有效 |
| `re` | 100 | physics stage 的雷诺数（ν=0.1, U=10, L=1） |
| `n_collocation` / `phys_chunk` | 4096 / 256 | PDE 配点数 / 二阶导图分块（显存保护） |
| `n_clusters` / `seed` | 12 / 0 | latent k-means 簇级 70/15/15 几何划分 |

### 3.4 绘图

```bash
.venv/bin/python plot_roadmap_metrics.py -d examples/ellipsoids/RoadmapONet_r4           # field/surface/physics 三 panel
.venv/bin/python plot_roadmap_metrics.py -d .../RoadmapONet_r4 --overlay .../RoadmapONet # 叠加对照 run
```

## 4. 已踩过的坑（排障索引）

1. **网格必须 113³ 拉伸**：漏 `--grid_stretch` 会产 64³ 快照，训练器点数校验直接报错。
2. **latent 必须限模长**：`--latent_max_norm 1.0`，否则 ~85% LHS 超 CodeBound，产生退化形状（上轮 shape_007/087/174/253/254，表面网格塌缩 + 模型最差 case）。
3. **杀后台批跑要杀 python 真身**：`kill $!` 只杀 nohup 壳；用 `pgrep -f generate_openfoam_snapshots` 确认。两个批跑进程同时写同一数据根目录会互相污染（快照分辨率混杂），重启前务必清 snapshots/surface/cases。
4. **eval/分析用存档 config**：v2 特征等 run 的 `--eval_only` 不需要手动带 `--config`（自动读 config.json）。
5. **physics stage 不能用 GradScaler**（与逐 chunk PDE backward 的 unscale 冲突会稀释物理梯度）——代码已修为无 scaler，勿改回。
6. **surface 终评必须重载 best checkpoint**（已修：`--stage surface` 评估的是交付工件本身）。
7. **OF14 的 wallShearStress 只能 solver 内嵌 FO 算**（`foamPostProcess -func` 在 laminar 下缺 turbulence model 报错）；`surfaceFormat` 只有 `vtk` 没有 `vtp`。

## 5. 一轮完整参考耗时（RTX 3070 8GB）

| 阶段 | 耗时 |
|---|---|
| 527 STL 生成 | ~15 min |
| 527 case（求解+体积+表面，12 并发） | ~4-4.5 h |
| SDF 缓存构建（527 形状） | ~10-15 min（首次训练时自动） |
| field 20000 iters | ~7-15 min |
| surface 10000 iters | ~6 min |
| physics 10000 iters | ~35-90 min |

## 6. 参考结果（281 形状旧数据，供新 run 对照）

- field MVP：val rel L2 0.2513 / test 0.2640 / 推理 0.53 s·case⁻¹（≈189× vs simpleFoam）
- field+adaptive：val 0.2491（worst-10 均值 0.899→0.884）
- surface：test Cd 中位相对误差 5.3%（积分路）/ 5.1%（直接头）
- physics：continuity −34%、momentum −44%，val 0.2623（不退化上限 0.28）

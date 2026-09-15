# DeepSDF + PhysicsNeMo DeepONet MVP 设计文档

日期：2026-09-15
状态：设计已确认（brainstorming 流程产出）
上游文档：`DeepSDF_PhysicsNeMo_DeepONet_CFD_Roadmap.md`（§38 配置、§63 MVP、§30 几何划分、§21 Fourier 特征、§15 归一化）

## 1. 目标与范围

按路线图 §63 第一版 MVP 净室实现一个 **DeepSDF latent → DeepONet → 体积流场** 的 surrogate：

```text
[z_g(16), bc(4)] → branch
[x, y, z, SDF, normal] → trunk（Fourier 特征）
                    → [u, v, w, Cp]
```

- 训练数据：`data/openfoam/ellipsoids_u10/`（281 形状 × 1 固定 BC `[10,1,0,0]`，113³ 拉伸网格快照）。
- 模型层直接使用 vendored PhysicsNeMo v2.3.0a0（免安装加载，见 §3）。
- **本轮唯一自变量是几何外形**（BC 常量零方差，branch 保留 bc 通道保持 BC-ready）。
- 显式排除（后续版本）：POD 监督、PDE/physics loss、surface/force head、cut-cell 特征、多工况、DoMINO/GeoTransolver benchmark。
- 与现有 PiPOD（`train_pipod_deeponet.py`）并行存在、互不修改，用于同数据对标。

## 2. 现状基础（实现直接复用）

| 资产 | 位置 | 说明 |
|---|---|---|
| DeepSDF decoder（冻结） | `examples/ellipsoids/ModelParameters/latest.pth` | epoch 1000，4×128，latent_in=[2]，tanh 输出，49,774 参数 |
| latent 清单 | `data/openfoam/ellipsoids_u10/lhs_latents.npz` | `names (281,)` + `latents (281,16)`；27 解析椭球 + 254 LHS |
| 体积快照 | `data/openfoam/ellipsoids_u10/snapshots/<shape>_case000.npz` | `fields (1442897,4) float32 [u,v,w,Cp]`（u 已 /U，p 已 Cp 化）、`bc (4,)`、`shape`；体内/无效点为 -1e30 哨兵，体内约定 u=0/Cp=1 |
| 网格坐标 | `deep_sdf.cfd.volume.make_stretched_grid` 默认参数 | 113³ = 1,442,897 点，域 [-9,9]³，hi=1.5/dense_half=1.1/h_fine=0.022/growth=1.35 |
| 快照 IO | `deep_sdf.cfd.volume.load_snapshot` | 强制点数校验 |
| decoder 批量查询 | `deep_sdf.utils.decode_sdf` | latent 拼接 xyz 过 decoder |
| SDF 梯度参考实现 | `train_pipod_deeponet.py` 的 `compute_sdf_gradients` / `build_shape_geometry` | autograd 求 ∇SDF 的模式照抄 |

注意：`examples/ellipsoids/{specs.json,split.json}` 已被删除，本实现**不依赖**它们——latent↔形状对应完全走 `lhs_latents.npz` manifest。

## 3. PhysicsNeMo 免安装接入（已实测验证）

vendored `third-party/physicsnemo`（v2.3.0a0）硬要求 torch≥2.10，当前 .venv 为 torch 2.5.1+cu121（RTX 3070 8GB）。**不升级 torch**，采用 sys.path + 运行时 shim，vendored 源码保持未修改（git status 干净）。

### 3.1 已装依赖（pip）

`requests fsspec einops hydra-core omegaconf jaxtyping nvtx termcolor tqdm s3fs GitPython cftime h5py tensordict treelib safetensors huggingface_hub pyyaml`，加 `--no-deps` 安装的 `timm 1.0.29`、`torchvision 0.20.1`（匹配 torch 2.5.1）。

### 3.2 shim（`deep_sdf/cfd/physicsnemo_compat.py`）

任何 `import physicsnemo...` 之前必须 `import deep_sdf.cfd.physicsnemo_compat`。它做三件事：

1. `sys.path` 插入 `third-party/physicsnemo`（仓库根相对定位）。
2. `torch.Tag.cudagraph_unsafe` 缺失时设别名为 `torch.Tag.nondeterministic_bitwise`（torch 2.5 的 pybind11 枚举可 setattr；该 tag 仅在自定义算子注册处被读取，随后被第 3 条丢弃）。
3. `torch.library.custom_op` 无 `tags` 形参时包一层丢弃 `tags` kwarg。

实测通过（2026-09-15）：`physicsnemo.core.Module`、`physicsnemo.models.mlp.FullyConnected`、`physicsnemo.experimental.models.xdeeponet.deeponet.DeepONet` 导入；GPU 前向 `(3,20)+(4096,D)→(3,4096,4)`、反向、`save`/`from_checkpoint`（.mdlus）round-trip。

## 4. 模型架构（physicsnemo 构件）

```python
branch = FullyConnected(in_features=20, layer_size=512, num_layers=4,
                        out_features=256, activation_fn="silu")   # [z(16), bc(4)]
trunk  = FullyConnected(in_features=43, layer_size=512, num_layers=6,
                        out_features=256, activation_fn="silu")   # 见 §4.1
model  = DeepONet(branch, trunk=trunk, dimension=3, width=256,
                  out_channels=4, decoder_type="mlp",
                  decoder_width=128, decoder_layers=2,
                  decoder_activation_fn="silu")
# core 模式: model(x_branch (B,20), x_trunk (T,43)) -> (B, T, 4)
```

参数量约 245 万。组合方式为 branch×trunk Hadamard 积 + decoder MLP（xDeepONet 标准形态，路线图 §26 einsum 收缩的超集）。

### 4.1 trunk 输入特征（43 维）

| 特征 | 维数 | 说明 |
|---|---|---|
| `x/9, y/9, z/9` | 3 | 原始坐标 / L_ref（L_ref=1）后再除 domain 半长 9 → [-1,1]，保持几何 aspect（§4.1 单一 L_ref） |
| `sdf` | 1 | 冻结 decoder 的 SDF（未 clamp） |
| `n_x, n_y, n_z` | 3 | `∇sdf / (|∇sdf| + 1e-8)`，由冻结 decoder autograd 得到（用户指定的 MVP 增量） |
| `γ(x/9)` | 36 | 坐标 Fourier 特征，6 bands：`[sin(2^k π x̃), cos(2^k π x̃)], k=0..5`，逐坐标拼接（路线图 §21） |

Fourier 只作用于归一化坐标；SDF/法向不编码。

### 4.2 归一化

- latent：train 集 mean/std z-score（buffer 存模型外 stats.json）。
- bc：常量 → 标准化后≈0（std 钳位 1e-8）。
- 输出 4 通道：train 集逐通道 z-score（**采样统计**：每 train 形状随机 4096 流体点聚合，避免全场 1.44M×197 的开销）。loss 与评估均在标准化空间计算 MSE，rel L2 在物理空间计算（标准化可解析反变换）。

## 5. 数据管线

### 5.1 SDF/法向缓存

启动时逐形状用冻结 decoder 在 113³ 网格点分块（≤2¹⁸/块）计算 SDF 与 ∇SDF（autograd），法向归一化后写盘：

```text
data/openfoam/ellipsoids_u10/sdf_cache/<shape>.npz   # keys: sdf (N,)f32, normal (N,3)f32
```

每形状约 23 MB，281 形状共约 6.5 GB；已存在则跳过。计算约 281 × (decoder 两次遍历 1.44M 点) 分钟级。

### 5.2 可采样点掩码

每形状：`sdf > 0`（流体域）且 `fields` 四通道均 > -1e20（哨兵过滤）且有限。掩码索引与近壁子索引（`|sdf| < 0.15`）随缓存一并预计算存 `sdf_cache/<shape>.npz`（keys: `fluid_idx`, `near_idx`，int32）。

### 5.3 几何划分（路线图 §30 cluster split）

`scipy.cluster.vq.kmeans2` 对 281×16 latent（z-score 后）聚 12 簇（seed 固定），整簇随机分到 train/val/test = 70/15/15，得约 197/42/42 形状。清单写 `RoadmapONet/split.json`（含簇号）。同一形状永不跨集。

### 5.4 训练采样

每 iter：`B_case=4` 形状 × `N_pts=8192` 点。每形状内：30% 从 `near_idx`、70% 从 `fluid_idx` 均匀抽取（近壁为空则全均匀）。返回：

```python
{
  "latent": (B,16) f32, "bc": (B,4) f32,
  "xyz": (B,N,3) f32, "sdf": (B,N,1), "normal": (B,N,3),
  "fields": (B,N,4) f32,          # [u,v,w,Cp] 物理值（已无量纲）
}
```

快照 fields（每形状 5.8 MB f32）CPU 常驻，按 batch 索引拷贝到 GPU；SDF/法向同。全部 281 形状约 6.5+1.6+6.5 GB 磁盘，CPU 内存 62 GB 可容纳训练子集常驻。

## 6. 训练

- 优化：AdamW lr 1e-3、wd 1e-5，cosine 衰减到 1e-5；`--iters` 默认 20000。
- AMP：前向 fp16（autocast），loss/评估 fp32（路线图 §35）。
- 损失：标准化空间 channelwise MSE（4 通道 MSE 求和，与 PiPOD/mPOD-DeepONet 口径一致）。
- 每 `--eval_every` 500 iter：val 形状每形状固定种子抽 16384 流体点，算物理空间 per-variable rel L2（`‖pred−gt‖/‖gt‖`）与四通道平均，best-on-val（平均 rel L2）保存 `best.mdlus` + 优化器状态 `train_state.pth`（含 iter/best，支持 `--resume` 续训：模型/优化器/iter/best 恢复，LR 日程按 `--iters` 重建快进——沿用 PiPOD commit 0a17a5e 约定）。
- `--smoke`：500 iters、val 缩到 8 形状 × 4096 点，验证全链路 + checkpoint round-trip。
- 指标日志 `metrics.jsonl`（每 50 iter 一条：iter/train_loss/val_rel_l2/分变量/lr/耗时），字段命名对齐 `plot_pipod_metrics.py` 以便复用绘图。

## 7. 评估与验收

训练结束后（或 `--eval_only`）在 val+test 上：每形状 65536 固定种子流体点 → per-variable rel L2、平均、均值基线（预测 train 均值场）rel L2；推理测速：单形状全 113³ 分块（2¹⁸/块）前向耗时，对照 simpleFoam ~100 s/case 给出加速比。结果写 `eval.json` 并打印汇总。

验收标准：

1. smoke 500 iters 全链路无错，`best.mdlus` 可 `DeepONet.from_checkpoint` 重载且输出一致。
2. 正式训练 val 平均 rel L2 显著低于均值基线（目标 < 0.5 × 基线）。
3. test 集（未见几何）per-variable rel L2 报告。
4. 推理加速比报告。
5. 口径说明：PiPOD 的 0.34~0.39 为 POD 系数/重建口径，本 MVP 为点态场口径，并排报告但不直接比大小。

## 8. 交付物与文件布局

| 文件 | 职责 |
|---|---|
| `deep_sdf/cfd/physicsnemo_compat.py` | §3.2 shim + sys.path |
| `deep_sdf/cfd/roadmap_deeponet.py` | Fourier 编码、模型构建（§4）、数据集/采样（§5.4）、SDF 缓存（§5.1）、cluster split（§5.3）、统计（§4.2）、评估（§7） |
| `train_roadmap_deeponet.py` | CLI（--config/--smoke/--iters/--resume/--eval_only 等）+ 训练循环（§6） |
| `examples/ellipsoids/RoadmapONet/` | config.json、split.json、stats.json、metrics.jsonl、best.mdlus、train_state.pth、eval.json |

配置单文件 JSON（`--config`，全部超参含 §4/§5/§6 数值），启动时把解析后全量配置写 `RoadmapONet/config.json` 存档（沿用 of4 的 config_stage 契约风格）。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| physicsnemo experimental API 变动 | 版本固定在 vendored 2.3.0a0；shim 集中于单一 compat 模块 |
| 8GB 显存（32k 点 × 512 宽 fp16） | 实测 smoke 显存；必要时 `B_case`/`N_pts` 降档（2×8192 或 4×4096） |
| SDF 缓存 6.5GB 磁盘 | 磁盘余量 469GB；缓存可删可重建 |
| 法向在远场数值小/方向噪声 | 归一化带 eps；MVP 不依赖法向幅值 |
| 281 形状单工况、几何方差有限 | 验收用"相对均值基线"归一口径；test 集看未见几何泛化 |

## 10. 后续路线（本轮不做）

第二版：∇SDF 幅值/h/cut-cell 特征、surface head（Cp/Cf）与 Cd/Cl/Cm 积分头（需 OpenFOAM 表面场重导出）；第三版：PDE/FVM physics loss（复用 `deep_sdf/cfd/physics.py` PDEInformer，λ 日程 0→0.01→0.1）；第四版：latent 空间 shape optimization（`optimize_drag.py` 路线对接）；以及 DoMINO/GeoTransolver benchmark（路线图 §31）。

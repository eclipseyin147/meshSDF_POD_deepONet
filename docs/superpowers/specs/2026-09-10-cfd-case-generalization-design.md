# CFD 案例预测一般化与鲁棒性升级设计

日期：2026-09-10
关联：`DEEPMESH.md`（当前实现）、`DeepSDF_DeepMesh_GeoFNO_CFD_Architecture.md`（目标架构）

## 1. 背景与差距

当前 CFD 代理（`deep_sdf/cfd/`）实现了 DeepMesh §4.3 的单一场景版本：固定来流方向/速度、固定代理参数下训练 `PressureNeuralOperator`（branch=z，trunk=逐面片几何特征，FiLM 条件化）。对照架构文档，差距如下：

| 差距 | 架构文档出处 | 后果 |
|---|---|---|
| 无边界条件（BC）编码：flow_dir/velocity 训练时固定 | §8、§9 | 换个来流方向/速度（新 CFD 案例）即失效，无"案例预测"能力 |
| 标签是有量纲"压力"而非 Cp | §21（第一阶段即预测 Cp） | 跨速度/密度不泛化；量纲随 U² 变化，训练不稳定 |
| trunk 缺少 SDF 场特征（d, ∥∇d∥） | §5、§6 | 丢失 geometry-aware 坐标信息；无法向离面点查询扩展 |
| 面片特征纯逐点，无邻域上下文 | §12（MeshGraphNet） | 对 latent 外推、噪声网格鲁棒性差 |
| 无验证集 / 最优 checkpoint / 特征归一化 | 工程鲁棒性 | 训练过拟合不可见；κ、logA 量程差异大 |
| optimize_drag 无法指定案例 BC，drag 未归一为 Cd | §25（闭环需 C_D） | 优化结果不可与 CFD 惯例对照 |

## 2. 目标

在**不破坏既有接口与旧 checkpoint** 的前提下（所有新能力默认关闭或由 checkpoint 元数据驱动），把 surface 压力代理升级为：

> **(z, BC) + 逐面片几何/场特征 → Cp 的可微算子**，训练覆盖多形状 × 多案例，支持留出形状与留出来流方向的泛化评估与降阻优化。

Volume Geo-FNO 分支（架构文档 §10/§13）不在本次范围；本次落在其 §21 定义的"第一阶段"的 surface 侧，并为 volume 侧预留 SDF 特征接口。

## 3. 具体修改

### 3.1 `deep_sdf/cfd/labels.py` — Cp 标签与 BC 约定

- BC 向量约定：`bc = [U, dx, dy, dz]`（来流速度大小 + 单位方向），`BC_FIELDS = ["U", "dir_x", "dir_y", "dir_z"]` 常量导出，存入 checkpoint。
- 新增 `proxy_pressure_coefficient(verts, faces, flow_dir, base_suction=0.3, beta=1.0)`：返回无量纲压强系数 Cp。现有 proxy 公式本身就是 Cp 形式（驻点 Cp=1），函数体即现 `proxy_pressure`，改名并注明 Cp 语义；保留 `proxy_pressure` 作为向后兼容别名。
- 新增 `pressure_to_cp(p, U)`：OpenFOAM 不可压（运动压力，ρ≡1）下 `Cp = p / (0.5·U²)`。
- 新增 `reference_area(verts, faces, flow_dir)`：迎风投影面积 `A_ref = Σ_f max(0, −n_f·v̂)·A_f`，可微。

### 3.2 `deep_sdf/cfd/surrogate.py` — 算子升级

`PressureNeuralOperator(latent_size, hidden=128, num_layers=3, use_curvature=True, bc_dim=0, use_sdf_features=False, gnn_layers=0)`：

- **BC 条件化**（`bc_dim>0`）：`bc_encoder = MLP(bc_dim→hidden)`；branch 输入改为 `[z, bc_encoder(bc)]`；FiLM 仍逐层调制 trunk（对应架构文档 §8/§9 的 h₀=Fusion(z, h_BC) + 多层条件化）。forward 增加 `bc=None` 参数；模型带 bc_dim 而 bc 为 None 时按零向量处理并 warning 一次。
- **SDF 场特征**（`use_sdf_features=True`）：trunk 输入追加 `[sdf(c), ‖∇sdf(c)‖]`（面质心 c 处由 decoder 评估，detached；∇d 方向信息与法向重复，只取模长，对应架构文档 §6 的 [d, ∇d, |∇d|] 中不冗余的分量）。`face_features` 增加可选参数 `sdf_values, sdf_grad_norm`；operator forward 增加同名可选参数。训练/优化脚本中由 `compute_sdf_gradients` 与一次 batched decode 生成。
- **邻域 message passing**（`gnn_layers>0`）：在 FiLM trunk 之前，对面特征做 N 层图消息传递——面邻接图由 faces 共享边构建（纯 torch：边表 → scatter mean 聚合 → MLP 更新，每层 `h_i ← MLP([h_i, mean_j h_j])`）。给面片局部邻域上下文（MeshGraphNet 思想的轻量版），提升对噪声网格/latent 外推的鲁棒性。邻接表由 faces 现算（优化循环中 faces 每轮变化，计算量为 O(F)）。
- **输入标准化**：模型内 `register_buffer("feat_mean"/"feat_std", ...)`，forward 内对 trunk 输入特征标准化；提供 `set_feature_normalization(mean, std)`。默认恒等（mean=0,std=1），保证不设置时行为同旧模型。
- 所有旧参数默认值不变 → 旧式构造 `PressureNeuralOperator(latent_size)` 行为与今完全一致。
- 新增 `drag_coefficient(verts, faces, cp, flow_dir)`：返回 `(Cd, force_integral)`，`Cd = Σ cp·(n_into·v̂)·A / A_ref`（n_into 为体内法向，沿用 `drag_from_pressure` 约定）。
- `deep_sdf/cfd/__init__.py` 导出新符号。

### 3.3 `train_pressure_surrogate.py` — 多案例训练 + 泛化评估

新参数（默认值保持旧行为之外的合理值）：

- `--cases_per_shape N`（默认 4）：每个形状采样 N 个案例；方向在球面均匀采样（`--dir_cone_deg` 可限制为绕 +x 的锥角，默认 180=全球面）；U 在 `--u_range lo hi`（默认 10 20）均匀采样。
- `--val_fraction`（默认 0.2）：按形状留出验证集。
- `--gnn_layers`（默认 0）、`--sdf_features`（默认关）、`--no_curvature`。
- 每个 (shape, case) 样本：提取一次网格（每形状一次，共享给该形状所有案例），按案例 BC 生成 Cp 标签；训练集面孔汇总计算特征 mean/std 写入模型 buffer。
- 训练循环：每 iter 随机抽一个 (shape, case)；每 200 iter 评估 val MSE 与 val Cd 相对误差（整面积分对比）；保存 **val 最优** checkpoint。
- `--openfoam` 路径：用 `pressure_to_cp(p, U)` 转 Cp 后训练（逐案例 velocity=U）。
- checkpoint 增加：`model_kwargs`（含 bc_dim/use_sdf_features/gnn_layers 等全部构造参数）、`bc_fields`、`val_mse`、`val_cd_rel_err`。旧格式加载不受影响。

### 3.4 `optimize_drag.py` — 案例指定与 Cd 报告

- 用 checkpoint 的 `model_kwargs` 重建模型，`load_state_dict(..., strict=False)` 兼容旧 checkpoint（缺 normalization buffer 时补恒等值并提示）。
- 新参数 `--alpha`（度，来流在 xy 平面内相对 +x 的迎角）与 `--velocity`；模型带 BC 而未指定时用零迎角 + 训练 U 中值，并打印所用 BC。direction = (cos α, sin α, 0)。
- 损失改用 Cd：`loss = Cd_pred + reg_lambda·‖z−z₀‖²`（Cd 经 A_ref 归一，量级 O(1)，使 reg_lambda 语义跨案例稳定）；报告初始/最终 Cd 及相对改进。

### 3.5 文档

`DEEPMESH.md`：更新 §2.6（架构升级说明）、§3 文件结构、§4.4 用法（新命令行）、§5.4 验证结果、§7 局限（移除已解决项，保留 OpenFOAM 未实测等）。

## 4. 验证方案（无 OpenFOAM，全部实际运行）

验证脚本放 /tmp，不入库。

1. **单元级**：新旧构造路径前向输出 shape 一致；bc/sdf/gnn 各 flag 组合前向正确；面邻接构建在小网格（四面体）上手算核对；标准化 buffer 生效（设 mean/std 后输出变化符合预期）；`strict=False` 加载旧格式 checkpoint 成功。
2. **端到端（合成椭球族，沿用 DEEPMESH.md §5.4 路线）**：`generate_training_meshes.py`/等价脚本生成 27 椭球 {0.5,0.7,0.9}³ 的 SDF npz → 训练小 decoder（小 dims、短 epoch，GPU 上分钟级）→
   - 新算子多案例训练（22 训练 / 5 留出形状 × 每形状 4 案例）；
   - 泛化指标：留出形状 × 留出新方向 的 Cp MSE 与 Cd 相对误差（目标：proxy 标签下 Cd 相对误差 ≲ 5%，且优于 bc_dim=0 基线在新方向上的失效对照）；
   - `optimize_drag.py --alpha 20` 运行 150 轮，Cd 单调下降趋势且网格非空。
3. **回归**：`--model operator` 不带任何新 flag 的训练/优化路径行为同旧实现。

## 5. 明确不做

- Volume Geo-FNO、[u,v,w,p,k,ω] 全场预测、wall shear/τw、壁面一致性损失（架构文档 §10/§11/§13，属第二、三阶段）。
- DeepONet+Geo-FNO hybrid（§16）。
- OpenFOAM 实测（本机无求解器，沿用既有标注）。

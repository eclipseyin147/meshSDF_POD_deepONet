# Physics-Informed POD-DeepONet（PIPOD+DeepONet）体积场架构设计

日期：2026-09-11
关联：`Physics_Informed_POD_DeepONet_CFD_Implementation_Roadmap.md`（AI_CFD_SDF 仓库，总体路线图）、`docs/superpowers/specs/2026-09-10-volume-pod-rom-design.md`（前作：固定 POD 基 + 系数回归 ROM）、`docs/superpowers/specs/2026-09-10-cfd-case-generalization-design.md`（surface 分支）

## 1. 目标与范围

在现有「POD（固定基）+ (z, BC) → 系数回归」ROM 之上，搭建路线图 §25 最终形式的 **physics-informed POD-DeepONet**：

```
q_v(x; z, μ) = b_v(x) + Σ_{k=1..r} a^v_k(z, μ) · φ^v_k(x, d, ∇d, h),   v ∈ {u, v, w, p}
```

- **trunk 基 φ 完全学习**（方案 A）：任意点查询、分辨率无关、对 x 二阶可微（动量方程需要 ∇²u）；
- **POD 的角色**（路线图 §24）：只用 train 快照做 `pod_fit`，确定 rank r、提供系数监督目标 `a^CFD = basis.project(y)` 和均值/投影下界参考；**POD 基不进入前向传播**（避免网格插值破坏二阶导）；
- **physics-informed**：连续性 + 动量残差经 autodiff 计算（PhysicsNeMo `darcy_physics_informed_deeponet.py` 的模式：`coords.requires_grad_(True)` → autodiff → `loss = loss_data + λ_phys · loss_pde`），加上壁面/远场 BC loss。

**本次范围 = 体积场 Stage 1–3**。Surface head（Cp/Cf/τw）与 Cd/Cl 积分暂不新建，后续复用已有 `PressureNeuralOperator` 接入。

明确不做：RANS k-ω（路线图 V2+）、end-to-end DeepSDF 微调（DeepSDF 全程冻结）、FNO/Geo-FNO 对照（另立路线）。

## 2. 验证数据与几何

- **几何**：27 椭球族（axes ∈ {0.5, 0.7, 0.9}³）。此前该实验的 decoder 与数据都在 /tmp，已丢失；本次补一个 in-repo 生成脚本并重新训练 decoder（见 §6）。
- **流场**：**物理一致合成场**（`flow_synth.py`，见 §4）——连续性严格成立，physics loss 与 data loss 不冲突，能真实验证物理正则化的价值。npz 快照契约与现有 `volume.py` 完全一致，真实 CFD 快照将来经 `--snapshots` 直接接入。

## 3. 网络结构（`deep_sdf/cfd/deeponet.py`）

cPOD 风格逐变量输出（路线图 §3.1 推荐；对应 mPOD-DeepONet 的 `MultiTrunkDeepONet` 组合模式 `einsum('bcp, cnp -> bcn')`，但他们的基固定、我们的基学习——有意的扩展，见 §9 对照）。

### 3.1 BranchNet

```python
BranchNet(latent_size, bc_dim=4, rank=r, n_outputs=4, hidden=256, num_layers=4)
```

- `h_bc = bc_encoder(bc)`（4→hidden 的 SiLU MLP），`cat([z_norm, h_bc])` → MLP → 线性输出 `(n_outputs, rank)`；
- 标准化 buffer 沿用现有惯例（恒等默认 + setter + clamp 1e-8）：`z_mean/z_std`、`bc_mean/bc_std`、`coef_mean/coef_std`（逐变量逐模态，统计量来自 train 集 POD 投影系数）；
- 与 `VolumeCoefficientRegressor` 分层一致：`forward_normalized(z, bc)` 返回标准化系数（供 L_POD），`forward(z, bc)` 在内部反标准化后返回物理系数（供场重构）；
- DeepSDF 冻结：z 从实验 `LatentCodes/latest.pth` 读取，`requires_grad=False`。

### 3.2 TrunkNet

```python
TrunkNet(in_dim=8, rank=r, n_outputs=4, hidden=256, num_layers=4)
```

- 输入：`(x, y, z, d, ∂d/∂x, ∂d/∂y, ∂d/∂z, h)`（8 维）。`d, ∇d` 由冻结 decoder 在查询点现场评估（沿用现有 `max_batch` 分块）；`h` 为局部网格尺寸，规则参考网格上为常数通道（`h = Δ/L_ref`），接口为 octree 数据预留；
- **激活函数 SiLU**（C^∞）。参考实现（mPOD-DeepONet、PhysicsNeMo MLP）默认 ReLU——分段线性场二阶导恒为零，无法做 ∇²u，此处是有意偏离；
- 输出 `n_outputs * rank + n_outputs`：前 `4r` 为逐变量基 `φ^v_k`（各变量独立切片），后 4 为逐变量偏置 `b_v`（学习均值场，替代 POD 均值在前向中的位置——保证整场对 x 光滑可微）。

### 3.3 PODDeepONet

```python
PODDeepONet(branch, trunk)
forward(z, bc, x, sdf, grad_sdf, h) -> (N, 4)        # q_v = b_v + Σ_k a^v_k φ^v_k
coefficients(z, bc) -> (4, r)                        # 供 L_POD
set_*_normalization(mean, std) / save / load         # 风格同 PressureNeuralOperator
```

## 4. 物理一致合成场（`deep_sdf/cfd/flow_synth.py`）

```python
potential_flow_field(axes, center, bc, grid_points, wake_amp=0.15) -> (G, 4)
```

- **基流**：球体势流解析解经各向异性映射到椭球：`ξ = M(x−c)`（M=diag(1/a,1/b,1/c)），`u(x) = M⁻¹ · û(ξ)`，û 为球体势流速度场。性质：**连续性严格成立**（常对角映射下 `div_x u = Σ_i M⁻¹_ii·M_ii·∂û_i/∂ξ_i = div_ξ û = 0`）；**椭球面滑移壁面严格成立**（u·n ∝ û·ξ = 0）；
- **压力**：Bernoulli `Cp = 1 − |u|²/U²`。球体子集（a=b=c）下连续性+动量残差真值都严格为 0——用作 physics 管线的单元测试基准；**非球椭球动量残差真值非零但有界**（映射场非无旋），文档与测试断言中明确标注，不期望其为 0；
- **尾迹**（默认开，`wake_amp>0`）：`u += ∇×A`，`A = a₀·U·exp(−‖x−x_w‖²/σ_w²)·ĉ`，ĉ ⊥ dir 为常向量，x_w 为尾迹区中心 → 严格保连续性，避免数据集退化为平凡低维；
- **体内延拓**：沿用现有约定 d<0 处 u≡0；physics loss 只在 `fluid_mask`（d > margin）上计算，延拓方式不影响 physics loss；
- 输出写标准 npz 契约（`fields (G,4)` / `bc (4,)` / `shape`），p 通道 = Cp，与真实 CFD 导出同构。

## 5. 损失与分阶段训练（`train_pipod_deeponet.py`）

无量纲化约定（与现有数据契约一致：速度已按 U 缩放、长度按 L_ref、p 通道为 Cp = (p−p∞)/(½ρU²)）：

```
连续性:  R_c = ∇·u
动量:    R_m = (u·∇)u + ½∇Cp − (1/Re)∇²u      （稳态不可压；Re 为配置项，默认 1e4）
壁面:    L_wall = mean ‖u·n‖²   （d≈0 采样点；势流场下滑移为精确 BC。no-slip 留 --wall_bc noslip 给真实粘性数据）
远场:    L_ff   = mean ‖u − U·dir‖² （域边界/远场采样点）
```

physics loss 只在 `fluid_mask = (sdf > margin)` 上计算（路线图 §15），margin 默认 `2h`。

| Stage | 命令 | 训练内容 | Loss |
|---|---|---|---|
| 1 | `--stage 1` | 只训 BranchNet（对标现有 ROM 的 sanity） | `L_POD`（标准化空间系数 MSE，目标系数来自逐变量 POD 投影） |
| 2 | `--stage 2` | Branch+Trunk 联合（可 `--init_from` stage-1 checkpoint） | `L_POD + λ_f·L_field`（λ_f 默认 1；L_field 为无量纲场 MSE） |
| 3 | `--stage 3` | 从 stage-2 checkpoint 续训，加物理正则 | `+ λ_phys(t)·(L_c + L_m + L_wall + L_ff)` |

- **λ_phys 日程**（路线图 §23）：训练进度 0–20% 取 0，20–50% 取 0.01，50–80% 取 0.05，80–100% 取 0.1；`--lambda_phys` 可固定覆盖。λ_phys=0 时完全跳过 autodiff 建图（零显存开销）。
- **collocation 采样**（`physics.py: CollocationSampler`，与 CFD 监督点分离）：30% 近壁（|d|<10h）、30% 尾迹（下游锥）、20% 高梯度（用真值快照 |∇u| 分位点筛选——合成/真实数据都有真值可用）、20% 均匀远场；每案例每步采 `--n_collocation`（默认 4096）点，二阶 autodiff 显存经分块（chunk 默认 1024）控制。
- **POD 拟合**：逐变量 cPOD——对 u/v/w/p 各调一次现有 `pod_fit`（energy=0.999），公共 rank `r = max(r_u, r_v, r_w, r_p)`，`--pod_rank` 可固定覆盖；POD 只用 train 形状快照（val 不得参与建基，沿用现有纪律）。
- **训练纪律**（沿用仓库惯例）：`--seed` 播种全部随机源；best-on-val checkpoint；标准化统计量只取 train 集；checkpoint 含 model_kwargs / bc_fields / grid_resolution / pod 基文件 / val 指标 / seed。

## 6. 椭球族数据与 decoder 重建（`generate_ellipsoid_dataset.py`）

此前 27 椭球实验的数据与 decoder 均在 /tmp（已丢失）。新增根目录脚本一次性生成：

- `data/ellipsoids/`：27 个形状的 DeepSDF 训练 npz（pos/neg 表面采样，解析椭球 SDF——数据生成脚本入库、数据本身不入库）；
- `examples/ellipsoids/specs.json`：CodeLength 16（3 参数形状族足够）等；
- 用既有 `train_deep_sdf.py -e examples/ellipsoids` 训练 decoder，产出 `LatentCodes/latest.pth` 供 PIPOD 使用。

## 7. 验证（全部实跑；单元测试脚本放 /tmp，按仓库惯例不入库）

**单元测试**

1. SiLU trunk 对 x 的一阶/二阶 autodiff vs 中心有限差分（rel err < 1e-3）；
2. 球体子集（a=b=c）且 `wake_amp=0` 的真值场上：连续性残差、动量残差 ≈ 0（机器精度外的截断误差有界）——验证残差管线正确性；
3. stage-1 BranchNet sanity：与现有 `VolumeCoefficientRegressor` 在相同快照数据上对照（注意现有 ROM 是 u,v,w,p 堆叠的联合 POD、本设计为逐变量 cPOD，基不同——只要求留出集重构相对 L2 同量级，不苛求逐位一致）；
4. 标准化 setter 精确生效（max diff 0.0）；checkpoint save/load 往返逐位一致；同 seed 两次短训逐位一致。

**端到端指标**（并列对照，沿用现有日志风格）

- 每变量相对 L2（u/v/w/p 分列）、系数 MSE/R²；
- **POD 投影下界**（不可约）与**均值场基线**两条参考线；
- 连续性/动量残差范数、壁面违约度（val 集）；
- 单次前向耗时（对照现有 ROM 的 ~1ms 量级）。

**成功判据**

- stage-2：留出形状+新方向 val 重构相对 L2 显著优于均值基线，并接近投影下界（基线参照：旧合成场上 ROM 为 30.7% vs 下界 6.6%；新场数字重测，不作跨数据集直接对比）；
- stage-3：val 连续性残差较 stage-2 明显下降，且场误差不劣化（容忍 ±5%）。

## 8. 边界情况与风险

- fluid_mask 全空（查询点全在体内）→ 跳过 physics loss 并 warning；
- 标准化 std < 1e-8 → clamp（现有惯例）；
- decoder 评估延续现有 `max_batch` 分块；
- 旧格式 checkpoint strict=False 兼容加载；
- 非球椭球动量残差真值非零（§4 已标注）——physics loss 权重日程从 0 爬升即为缓解；
- GPU：RTX 3070 8GB，DrivAerNet 训练在跑；椭球族实验显存需求小（decoder 冻结 + 小 MLP），错峰或共存均可；
- 参考仓库已入库 `third-party/mPOD-DeepONet/`、`third-party/physicsnemo/`（tarball 快照，仅作代码参考，不作为依赖）。

## 9. 与参考实现的对照

| 项 | mPOD-DeepONet / PhysicsNeMo | 本设计 | 理由 |
|---|---|---|---|
| 基函数 | 固定 POD 基（`ChannelPodONet` 无学习 trunk） | 学习式 trunk 基 + POD 系数监督 | 任意点查询、几何感知、physics 需要 |
| 逐变量 | `MultiTrunkDeepONet` einsum('bcp,cnp->bcn') | 同模式 | 直接借用 |
| 激活 | ReLU | **SiLU** | ReLU 二阶导恒零，∇²u 不可用 |
| trunk 输入 | (x, y) | (x, d, ∇d, h) | 路线图 §8：几何感知坐标 |
| physics | PhysicsInformer(autodiff) | 纯 `torch.autograd.grad` | 不引入 PhysicsNeMo 依赖，风格与仓库一致 |
| PDE 边界处理 | PDE 残差剔除边界点（pad 清零） | 边界条件单独成 loss 项 | 更细粒度（wall/far-field 分列） |

## 10. 文件清单

```
deep_sdf/cfd/deeponet.py        # 新增：BranchNet / TrunkNet / PODDeepONet
deep_sdf/cfd/physics.py         # 新增：autodiff 残差、BC loss、fluid_mask、CollocationSampler
deep_sdf/cfd/flow_synth.py      # 新增：物理一致合成场
deep_sdf/cfd/__init__.py        # 追加导出（既有条目不动）
generate_ellipsoid_dataset.py   # 新增（根目录）
train_pipod_deeponet.py         # 新增（根目录，--stage 1/2/3）
DEEPMESH.md                     # 实施后新增 PIPOD+DeepONet 一节
volume.py / surrogate.py / differentiable_mesh.py   # 零改动
```

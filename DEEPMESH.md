# DeepMesh + DrivAerNet++：可微等值面提取与气动优化流水线

基于 DeepSDF 仓库实现的 DeepMesh（*DeepMesh: Differentiable Iso-Surface Extraction*, Guillard et al., arXiv:2106.11795v2）完整流程：可微等值面提取 → 网格级重建/剪影精修 → CFD 快速预测与气动降阻优化，并在 DrivAerNet++ 真实汽车数据（含 OpenFOAM 压力场标签）上落地。

设计文档：`docs/superpowers/specs/2026-09-10-deepmesh-design.md`（核心）、`docs/superpowers/specs/2026-09-10-deepmesh-pipeline-design.md`（流水线扩展）、`docs/superpowers/specs/2026-09-10-cfd-case-generalization-design.md`（CFD 案例一般化升级）、`docs/superpowers/specs/2026-09-10-volume-pod-rom-design.md`（体积场 POD-ROM）。

**许可证注意**：DrivAerNet++ 数据采用 CC BY-NC 4.0，仅限非商业研究与教学使用。

## 目录

1. [DeepSDF 与 DeepMesh 的差距](#1-deepsdf-与-deepmesh-的差距)
2. [核心原理](#2-核心原理)
3. [代码结构](#3-代码结构)
4. [DrivAerNet++ 数据](#4-drivaernet-数据)
5. [使用流程](#5-使用流程)
6. [验证结果汇总](#6-验证结果汇总)
7. [实现要点与坑](#7-实现要点与坑)
8. [局限与遗留](#8-局限与遗留)

## 1. DeepSDF 与 DeepMesh 的差距

两者的网络结构完全相同（MLP `f_Θ(z, x) → SDF/OCC`），差距集中在「隐式场 → 显式网格」环节：

| 环节 | 原 DeepSDF | DeepMesh（本实现） |
|---|---|---|
| 网格提取 | 256³ 密集均匀采样 + skimage marching cubes（`deep_sdf/mesh.py`） | 分层 octree 采样（32³ 起步、近表面细分，论文 Algorithm 1）+ 可选 GPU marching cubes |
| 可微性 | MC 顶点由线性插值 `s_i/(s_i−s_j)` 确定，`s_i=s_j` 处不连续，拓扑变化无法反传（论文 §6.1） | 隐函数定理（Theorem 1）：`∂x/∂c = −(n/‖n‖²)·∂f/∂c`，`n=∇f(x)`，梯度在连续场上计算，拓扑可随梯度改变 |
| 反向目标 | 仅隐式域（对 SDF 采样值做 L1） | 网格顶点级损失（Chamfer、剪影、CFD drag）可回传到 latent `z` 或网络权重 `Θ` |
| 场类型 | 仅 SDF | SDF + occupancy（BCE 训练 + logit 放大提取，论文 §6.3） |

## 2. 核心原理

### 2.1 可微等值面提取（论文 Theorem 1 + Algorithm 1/2）

前向：照常提取网格（拓扑每轮重新确定，天然可变）；反向：不求导 MC，而是用隐函数定理把顶点梯度转化为隐式场梯度：

```
∂L/∂c = − Σ_v (∂L/∂v) · (∇f(v)/‖∇f(v)‖²) · ∂f(z,v)/∂c        (论文 Eq. 5)
```

本实现用**重连恒等式**而非自定义 autograd Function：

```python
verts = verts - (n / ||n||²) * (f(z, verts) - f(z, verts).detach())
```

该式数值上恒等于 `verts`（括号内为 0），梯度恰为 Theorem 1 的结果；普通 autograd 即可同时支持 `c = z` 与 `c = [z|Θ]`（论文 §4.5 端到端训练）。相比 MeshSDF 官方实现的手写 backward + 单位法向近似（利用 SDF 的 ‖∇f‖≈1），本实现保留完整 `n/‖n‖²` 项，对非理想 SDF 更严格。

### 2.2 分层 octree 采样（Algorithm 1）

- SDF：体素 8 角点 `min|f| < threshold_factor × voxel_size`（默认 2.0，即体素对角线判据）则细分；
- occupancy：8 角点不全部位于 0.5 同侧则细分；
- 分辨率链 `N → 2N−1`（粗格点是细格子集，已有点不重复评估）；未评估的细格点用粗网格三线性插值填充——由 SDF 的 1-Lipschitz 性可证明：未细分体素角点必同号，凸组合保号且保下界，不会引入伪零交叉。

### 2.3 GPU marching cubes

纯 PyTorch 向量化实现：逐边生成顶点（x/y/z 三方向边网格分别计算符号交叉 + 线性插值，cumsum 压缩为全局顶点 id，相邻体素天然共享顶点，输出 welded 网格），逐体素查经典 Lorensen 256 case 表（复用 skimage 序列化 LUT，含旧版 THE_LUTS 兼容路径）生成三角形。case 表与 skimage 实测逐顶点一致（Chamfer 2e-8）。

### 2.4 Occupancy 支持（§6.3）

- 训练：decoder 末层 tanh 输出 ∈(−1,1)，映射 `occ = (pred+1)/2 ∈ (0,1)`，BCE 损失（Eq. 3），标签 `occ_gt = (sdf < 0)`（直接从 SDF 样本符号派生，无需额外数据）；
- 提取：iso = 0.5，MC 前对网格做 logit 放大 `logit(clamp(g, 1e-6, 1−1e-6))`，近似 SDF 的线性区域、消除 occupancy 直接网格化的 artifacts；
- 法向仍对原始网络输出求解析梯度；重连恒等式中用 `f(v) − 0.5`。
- 开启方式：specs.json 加 `"FieldType": "occupancy"`（见 `drivaernet/experiments/specs_occ.json`）；decoder 输出天然有界（末层恒有 tanh），无需 `use_tanh`。
- 限制：occupancy 场不含 SDF 值，CFD 环节（`--sdf_features`、曲率特征、volume ROM 合成场）不可用，主要用于重建/剪影实验或与 occupancy 系网络对接。

### 2.5 可微剪影渲染（§4.2, Eq. 14）

纯 PyTorch SoftRas 风格软光栅化（无 neural_renderer/pytorch3d 依赖）：正交投影到 [−1,1]²，像素到投影三角形的有向距离 `D`（内部用重心符号、外部用点-线段距离），软覆盖 `A = sigmoid(D/σ)`，对数域稳定聚合 `S = 1 − exp(Σ_f logsigmoid(−D_f/σ))`；面片分块 + checkpoint 控制显存。剪影损失 `L = ‖S(M(z)) − T‖₁` 对顶点可微，经 IFT 回传 latent。

### 2.6 CFD 代理与降阻（§4.3, Eq. 16）

```
L_task(M) = Cd(M) + λ‖z − z_0‖²,   Cd = Σ_f Cp_f·(n_f·v̂)·A_f / A_ref
```

2026-09-10 升级（设计文档 `docs/superpowers/specs/2026-09-10-cfd-case-generalization-design.md`）：从「单一固定来流场景」升级为「**多形状 × 多边界条件（案例）一般化**」——算子学习 (z, BC) + 逐面片特征 → Cp 的映射，支持留出形状与留出来流方向的泛化评估。

- 压力代理采用**物理神经算子** `PressureNeuralOperator`（DeepONet 风格 branch/trunk + FiLM）：branch 吃形状 latent z 与 **BC 编码**（`bc=[U, dir_x, dir_y, dir_z]`，见 `BC_FIELDS`；`bc_dim>0` 时经 `bc_encoder` 后与 z 拼接，原始 bc 先经 `bc_mean`/`bc_std` buffer 标准化——U∈O(10) 与方向分量 ∈O(1) 量级悬殊，不标准化会使条件化病态），trunk 吃逐面片物理特征 [质心, 外法向, 平均曲率 κ, sdf, ‖∇sdf‖, log面积]（κ 与 sdf 特征可选）。性质：分辨率无关、形状/来流条件化、物理特征驱动。trunk 输入用训练集统计量标准化（`feat_mean`/`feat_std` buffer）；可选 `gnn_layers` 层在 FiLM trunk 前做面级 message passing（共享边邻接 + mean 聚合，轻量 MeshGraphNet），引入局部邻域上下文。保留局部 MLP `PressureSurrogate` 作为基线对照。
- 标签统一为**无量纲压强系数 Cp**（驻点 Cp=1）：几何代理 `proxy_pressure_coefficient`（保留 `proxy_pressure` 兼容别名）；OpenFOAM 运动压力经 `pressure_to_cp(p, U) = p/(0.5·U²)` 转换；DrivAerNet++ 真实压力数据经 `--cp_data` 直接接入（见 §4.3）。
- 平均曲率 κ = div(∇f/‖∇f‖)/2 由 SDF 场二阶导解析求得；注意 ReLU decoder 的场分段线性导致解析 κ≡0，实现带有限差分回退（`method="auto"` 自动检测）。
- drag 积分 `Σ_f Cp_f·(n_f·v̂)·A_f` 中法向/面积由顶点现算，梯度同时经 branch 直连路径与 IFT 几何路径回传 latent；`drag_coefficient` 以迎风投影面积 `reference_area = Σ max(0,−n·v̂)·A` 归一为 Cd，量级 O(1)、跨来流速度/案例可比。
- 压力标签：有 OpenFOAM 时 `run_openfoam`（simpleFoam）；无求解器时用几何代理 `Cp = max(0,−n·v̂)²·(1+β·D_yz/L_x) − base_suction·max(0,n·v̂)`（含非局部阻塞因子，明确标注为流程验证代理，非物理模拟）。

### 2.7 体积场 POD-ROM（(z, BC) → 体积流场）

2026-09-10 新增（设计文档 `docs/superpowers/specs/2026-09-10-volume-pod-rom-design.md`）：在表面 Cp 算子之外，新增**体积流场 (u,v,w,p) 的降阶预测**。所有案例共享归一化坐标系下的规则参考网格（默认 64³、域 [−1.5,1.5]³，形状外留来流/尾迹余量），快照矩阵 `S` (N_cases, G·4) 因此天然对齐；POD（proper orthogonal decomposition）在 **GPU 上用 randomized SVD** 完成（range finder：Ω∼N(0,1)、Y=SΩ、QR、B=QᵀS、SVD(B)），实现上对 Gram 矩阵 S Sᵀ 做精确特征分解定能量谱与截断 rank（`--pod_energy 0.999`），幂迭代与 B Bᵀ=Qᵀ(S Sᵀ)Q 均经该 Gram 恒等求值，全程按特征维分块、显存有界（N_cases ~2000、D ~1M 不爆显存）。NN 只做**低维系数回归**：`h_bc=bc_encoder(bc)`，`cat([z_norm, h_bc]) → MLP → r 维线性输出`；三组恒等默认标准化 buffer（z/bc/coef，setter 与 surface 算子同风格），其中 coef 标准化关键——POD 系数幅值随奇异值衰减数个量级，不标准化会导致高阶模态欠拟合。重构 ŷ = mean + basis·a_pred；评估指标为重构全场相对 L2，并与**真值投影的不可约下界**（projection_error，基截断决定的下限）及「直接预测均值场」基线并列对照。POD 只用 train 形状的快照拟合，val 形状的场不参与建基。数据接口（每 (shape, case) 一个 npz：`fields` (G,4)、`bc` (4,)、`shape`）与真实 CFD 快照兼容；无求解器时用 `synthetic_volume_field` 从 decoder SDF 现场生成合成场（近壁 sigmoid 衰减 + 尾迹亏损（幅度 ∝ 阻塞度 D_yz/L_x）+ Bernoulli Cp，体内 u≡0；**非物理模拟，仅供流程验证**）。

### 2.8 Physics-informed POD-DeepONet 体积算子

2026-09-11/12 新增（设计文档 `docs/superpowers/specs/2026-09-11-pipod-deeponet-design.md`，实施计划 `docs/superpowers/plans/2026-09-11-pipod-deeponet.md`）：在 §2.7 的「POD 压缩 + 系数回归」之上升级为**几何条件化、物理正则的算子**。数学形式：

```
q(x; G, μ) = q̄(x) + Σ_k a_k(z, μ) · φ_k(x, d(x), ∇d(x), h)
```

- **cPOD**：每变量（u,v,w,p）独立拟合 POD 基（Gram 精确谱定能量谱，randomized SVD 取基），公共 rank 取各变量能量截断的最大值；均值场与基按变量存储（`pod_basis_{u,v,w,p}.pth`）；
- **Branch**：`[z_norm, bc_encoder(bc)] → MLP → 4r` 标准化系数（z/bc/coef 三组标准化 buffer，与 §2.6/2.7 同风格）；
- **Trunk**：输入 8 维特征 `[x, y, z, sdf, ∂sdf/∂x, ∂sdf/∂y, ∂sdf/∂z, h]`（MultiTrunk 风格，4 变量共享 trunk、各自取基）；sdf/∇sdf 由 decoder 在参考网格逐点求得，h 为网格间距——算子因此感知「离壁面多远、壁面法向朝哪、局部尺度」；
- **三阶段训练**（损失逐级叠加，参考 PhysicsNeMo autodiff 模式）：
  - Stage 1：仅 branch，`L = L_POD`（系数 MSE），验证 (z, μ)→a 可预测性；
  - Stage 2：branch + trunk 全场训练，`L = L_POD + λ_f·L_field`（逐点场 MSE）；
  - Stage 3：物理微调（lr 降至 1e-4），`L += λ_phys(t)·(L_cont + L_mom + L_wall + L_ff)`；λ_phys 日程 0→0.1（前 20% iter 为 0）；PDE 残差由 autodiff 在 collocation 点求值（不可压缩连续方程 + 稳态 NS 动量，特征 Re 由 `--re` 指定；流体掩码排除固体内部；近壁加密采样）；壁面支持 `--wall_bc slip|noslip`，远场约束 `q→(dir, 0)`。

## 3. 代码结构

### 3.1 DeepMesh 核心（仓库根 + `deep_sdf/`）

```
deep_sdf/
├── differentiable_mesh.py      # 核心：octree 采样 + 可微网格提取 + ply 保存
│   ├── sample_sdf_octree(decoder, latent_vec, resolution, max_batch,
│   │                      initial_resolution=32, threshold_factor=2.0,
│   │                      field_type="sdf", bounds=None) -> (sdf (N,N,N), N)
│   ├── compute_sdf_gradients(decoder, latent_vec, points, max_batch) -> (M,3)
│   ├── extract_differentiable_mesh(decoder, latent_vec, resolution=128,
│   │                               initial_resolution=32, max_batch=2**18,
│   │                               threshold_factor=2.0, eps=1e-8,
│   │                               field_type="sdf",        # "sdf"|"occupancy"
│   │                               mc_backend="skimage",    # "skimage"|"torch"
│   │                               bounds=None)             # 默认 [-1,1]³
│   │         -> (verts (M,3) 可微, faces (F,3) int64, normals (M,3) detached)
│   └── save_mesh(verts, faces, filename)
├── marching_cubes_torch.py     # GPU marching cubes
│   └── marching_cubes_torch(sdf_grid, level=0.0, voxel_size=1.0)
│             -> (verts (M,3), faces (F,3)) 与输入同设备
├── soft_renderer.py            # 可微剪影渲染
│   ├── look_at(eyes, center, up) -> (K,3,3)
│   ├── render_silhouette(verts, faces, eyes, image_size=64, sigma=1e-4,
│   │                     dist_thresh=0.1, face_chunk=1024) -> (K,H,W)∈[0,1]
│   └── save_silhouette(silhouette, filename)
└── cfd/
    ├── labels.py               # Cp 标签与 BC 约定
    │   ├── BC_FIELDS = ["U", "dir_x", "dir_y", "dir_z"]
    │   ├── export_stl(verts, faces, path)
    │   ├── openfoam_available() -> bool
    │   ├── run_openfoam(stl_path, case_dir, velocity=15.0) -> per-face pressure
    │   ├── proxy_pressure_coefficient(verts, faces, flow_dir=(1,0,0),
    │   │                            base_suction=0.3, beta=1.0) -> (F,) Cp
    │   ├── proxy_pressure            # proxy_pressure_coefficient 的兼容别名
    │   ├── pressure_to_cp(p, U)      # OpenFOAM 运动压力 -> Cp = p/(0.5 U²)
    │   └── reference_area(verts, faces, flow_dir)  # 迎风投影面积，可微
    ├── surrogate.py            # 压力代理 + drag 泛函
    │   ├── compute_mean_curvature(decoder, latent_vec, points, method="auto")
    │   ├── face_features(verts, faces, mean_curvature=None,
    │   │                 sdf_values=None, sdf_grad_norm=None) -> (F,7..10)
    │   ├── PressureSurrogate(hidden=128, num_layers=3)          # 局部 MLP 基线
    │   ├── PressureNeuralOperator(latent_size, hidden=128,      # 物理神经算子
    │   │                          num_layers=3, use_curvature=True,
    │   │                          bc_dim=0, use_sdf_features=False,
    │   │                          gnn_layers=0)
    │   │         forward(latent, verts, faces, mean_curvature, bc,
    │   │                 sdf_values, sdf_grad_norm) -> (F,) Cp
    │   │         set_feature_normalization(mean, std)
    │   │         set_bc_normalization(mean, std)     # bc 向量标准化（bc_dim>0）
    │   ├── drag_from_pressure(verts, faces, pressure, flow_dir=(1,0,0))
    │   └── drag_coefficient(verts, faces, cp, flow_dir) -> (Cd, force_integral)
    └── volume.py               # 体积场 POD + 系数回归 ROM
        ├── make_reference_grid(resolution=64, domain=(-1.5,1.5))
        │         -> (grid_points (G,3) float32 C-order, shape (N,N,N))
        ├── pod_fit(S, energy=0.999, rank=None, randomized=True,
        │           oversampling=10, niter=2, device="cuda") -> PODBasis
        │         # S (N_cases, D) float32；GPU randomized SVD（range finder
        │         # Ω~N(0,1) → Y=SΩ → QR → B=QᵀS → SVD(B)，幂迭代/B Bᵀ 经精确
        │         # Gram 恒等求值）；特征维分块，显存有界；rank=None 按 energy
        │         # 截断；打印前 20 个奇异值的精确累积能量谱
        ├── PODBasis(mean (D,), basis (D,r), singular_values (r,), energy)
        │         # nn.Module buffer；project(Y)->(N,r)、reconstruct(A)->(N,D)、
        │         # relative_error(Y, coef=None)->(N,) 全场相对 L2（coef=0 即
        │         # 均值场基线）、projection_error(Y)->(N,) 真值投影下界、
        │         # save/load（torch.save 张量 dict）
        ├── VolumeCoefficientRegressor(latent_size, rank, hidden=256,
        │                              num_layers=4, bc_dim=4)
        │         # h_bc=bc_encoder(bc)，cat([z_norm,h_bc])->MLP->r 线性输出
        │         # forward(latent, bc)->物理系数；forward_normalized->标准化空间
        │         # set_z/bc/coef_normalization(mean, std)（恒等默认，clamp 1e-8）
        ├── synthetic_volume_field(decoder, latent, grid_points, shape, bc,
        │                            delta=0.05) -> (G,4)  # 非物理，流程验证用
        ├── deeponet.py             # 几何条件化 POD-DeepONet（§2.8）
        │   ├── BranchNet(latent_size, bc_dim, rank, n_outputs, hidden=256,
        │   │             num_layers=4, bc_hidden=64)
        │   │         # forward(latent, bc)->(B, n_outputs·r) 物理系数；
        │   │         # forward_normalized->标准化空间；set_z/bc/coef_normalization
        │   ├── TrunkNet(in_dim=8, rank, n_outputs, hidden_sizes=[256,512])
        │   │         # [x, sdf, ∇sdf, h] -> (B, n_outputs·r) 空间基
        │   └── PODDeepONet(branch, trunk)
        │             forward(latent, bc, features)->(B,4)；predict = mean + Σ a_k φ_k
        ├── physics.py              # PDE 残差与 BC 损失（PhysicsNeMo autodiff 模式）
        │   ├── IncompressibleNS(re)  # continuity/momentum 残差（autodiff）
        │   ├── PDEInformer           # 分批求值封装
        │   ├── CollocationSampler    # collocation 点采样（近壁加密、流体掩码）
        │   ├── noslip_loss / wall_slip_loss / farfield_loss
        │   └── physics_weight_schedule(progress)  # λ_phys 0→0.1 日程
        ├── flow_synth.py           # 物理一致的合成势流场（--synthetic 数据源）
        │   ├── parse_ellipsoid_axes(shape_name) -> (a,b,c)
        │   └── potential_flow_field(...) -> (G,4)  # 散度自由、NS 残差≈0
        └── openfoam_runner.py      # OpenFOAM 14 案例生成/运行/采样
            ├── write_stl_from_geometry(path, axes=None, decoder=None,
            │                           latent=None, resolution=63)
            ├── make_case(case_dir, stl_path, bc, nu=0.1, domain_half=18.0,
            │             n_base=36, layers=6, layer_ratio=1.25,
            │             first_layer=0.011, surface_level=3, ...)
            ├── run_case(case_dir)    # blockMesh→snappyHexMesh→simpleFoam→采样
            └── sample_to_snapshot(case_dir, grid_points, shape_name, bc, out_path)

train_deep_sdf.py               # 修改：specs 增加可选 "FieldType": "occupancy"
reconstruct_mesh.py             # 新增：网格级 Chamfer 重建（--field_type/--mc_backend/--init/--bounds）
refine_silhouette.py            # 新增：剪影精修（--view_index/--image_size/--sigma/--field_type）
train_pressure_surrogate.py     # 新增：压力代理训练（--model local|operator/--openfoam/--cp_data）
train_volume_rom.py             # 新增：体积场 POD-ROM 训练（--snapshots|--synthetic）
train_pipod_deeponet.py         # 新增：PIPOD-DeepONet 三阶段训练（--config JSON 或
                                #   CLI：--stage 1|2|3、--snapshots/--synthetic、
                                #   --latent_manifest、--wall_bc、--re、--lambda_phys；
                                #   JSONL 指标日志）
generate_openfoam_snapshots.py  # 新增：OpenFOAM 批量快照（--lhs N 形状采样、
                                #   4 路并行、batch_summary.json 汇总）
plot_pipod_metrics.py           # 新增：PIPOD 三阶段指标曲线（读 metrics_stageN.jsonl，
                                #   --overlay 双实验对比、-w 训练中实时刷新）
optimize_drag.py                # 新增：气动降阻优化（--latent/--reg_lambda/--bounds）
server.py                       # 新增：FastAPI + WebSocket 实时 latent 探索服务
web/                            # 新增：three.js 前端（index.html + app.js，CDN 引入无构建）
```

既有文件行为不变：`train_deep_sdf.py` 的 occupancy 模式由 specs 可选项开启；`deep_sdf/mesh.py`、`reconstruct.py`、`networks/deep_sdf_decoder.py` 未改动。

### 3.2 DrivAerNet++ 数据与脚本（`drivaernet/`）

```
drivaernet/
├── data/
│   ├── raw/F_D_WM_WW_1.zip          # 原始下载（Harvard Dataverse, ~10.4GB）
│   ├── stl/                         # 500 个原始 STL（未缩放，单位米，车长约 5m）
│   ├── rawPressureData/vtk/         # 500 个 OpenFOAM sampleSurface VTK
│   └── processed/
│       ├── SdfSamples/DrivAerNet/Cars/*.npz   # DeepSDF 训练输入
│       ├── Pressure/DrivAerNet/Cars/*.npz     # 逐面 Cp 标签（与 STL 面 1:1）
│       ├── preprocess_log.json                # 逐网格处理记录
│       └── pressure_preprocess_log.json       # 逐车 Cd(p) 质量检查
├── scripts/
│   ├── preprocess_sdfsample.py      # STL -> SDF npz 采样（替代官方 C++ 工具）
│   ├── preprocess_pressure.py       # VTK 压力 -> STL 逐面 Cp（支持断点续跑）
│   ├── make_splits.py               # 生成 train/val/test split
│   ├── train.sh                     # 训练启动脚本（SDF 场）
│   ├── train_occ.sh                 # 训练启动脚本（occupancy 场变体）
│   └── reconstruct_mesh.sh          # 阶段 2 网格级重建
├── experiments/
│   ├── drivaernet_f1/specs.json     # SDF 实验配置
│   ├── drivaernet_f1_occ/           # occupancy 实验（train_occ.sh 自动创建）
│   └── specs_occ.json               # occupancy 配置模板
└── splits/{train,val,test}.json     # 400/50/50，seed 42
```

## 4. DrivAerNet++ 数据

基于 DrivAerNet++（[GitHub](https://github.com/Mohamedelrefaie/DrivAerNet) / [论文](https://arxiv.org/abs/2406.09624)）的 500 辆 fastback 汽车 STL 网格及配套 OpenFOAM 压力场。

### 4.1 数据来源与含义

- 数据集页面：https://doi.org/10.7910/DVN/OYU2FG （3D Meshes 子集）
- `F_D_WM_WW_1`：**F**astback（快背）/ **D**etailed underbody（详细底盘）/ **W**ith **W**heels（含车轮）/ **W**ith **M**irrors（含后视镜），第 1 批共 500 辆。
- 其余 14 个 zip（共 443GB，命名规则相同：F/N/E × D/S × WW/WWC/WWS）可按需追加下载后放 `data/stl/` 重新运行预处理，scripts 会自动跳过已处理文件。
- 每辆车 20.8 万顶点 / 41.7 万三角面，拓扑一致（参数化变形族），尺寸都在同一数量级（车长 ~5m），**未做缩放**。

**关键：所有坐标是米制原尺寸**，因此 `specs.json` 里 `ClampingDistance: 0.2` 的单位是米（0.2m 截断），不要和 ShapeNet 单位球配置的数值含义混淆；同理所有网格提取必须传 bounds（见 §5.1 注意事项）。

### 4.2 网格水密性（未修复）

原始 STL 非严格水密：每辆车约 33 条非流形边，位于车身内部传动通道处的微型双层面片（面积约 1e-5 m²），外部几何不受影响。尝试过 pymeshfix 修复，但它会把 4 个独立车轮组件连同底盘细网格一起删掉（几何损失约 2%，`remove_smallest_components=False` 也无法避免），**得不偿失，故保留原始网格**。SDF 符号判定改用"最近面法向投票"（与 DeepSDF 原文做法等价），对非水密网格稳健，测试确认正确。

### 4.3 预处理（已按此流程执行）

```bash
source .venv/bin/activate
# 1) STL -> SDF npz（500k 点/车：94% 近表面高斯扰动 σ=5mm + 6% 包围盒外扩 10% 均匀采样）
python drivaernet/scripts/preprocess_sdfsample.py \
    --src drivaernet/data/stl --out drivaernet/data/processed \
    --num-samples 500000 --near-ratio 0.94 --noise-std 0.005 --workers 4
# 2) split（400/50/50）
python drivaernet/scripts/make_splits.py --processed drivaernet/data/processed \
    --out drivaernet/splits --ratios 0.8 0.1 0.1 --seed 42
```

输出 npz 格式与官方 `PreprocessMesh` 完全一致：`pos` (N,4) sdf>0（形状外部），`neg` (M,4) sdf<0（形状内部），float32，坐标为**原始米制坐标**。

### 4.4 压力数据（真实 CFD 标签，已预处理）

- 来源：DrivAerNet++ Pressure 子集 `F_D_WM_WW_1.zip`（500 个 OpenFOAM `sampleSurface` VTK，legacy ASCII，POINT_DATA 场 `p` 为表压 Pa），解压于 `data/rawPressureData/vtk/`。
- CFD 工况（论文 Appendix A.3）：OpenFOAM v11，k-ω SST，u∞ = 30 m/s，ρ = 1.184 kg/m³，q∞ = 532.8 Pa，出口表压 0。Cp = p / q∞。
- 处理：VTK 壁面（四/五/六边形混合）扇形三角化后，将 `p` 按最近面重心坐标插值回 STL 三角网格，得到与 STL 面 1:1 对应的逐面压力。
- 输出：`data/processed/Pressure/DrivAerNet/Cars/<name>.npz`，含 `verts / faces / face_p / face_cp / transfer_dist`（插值距离，质量指标，中位数 ~0.2mm）。
- 脚本：`scripts/preprocess_pressure.py`（支持断点续跑）。每车同时积分压差阻力 Cd(p) 写入 `pressure_preprocess_log.json` 作质量检查。

`train_pressure_surrogate.py` 的 `--cp_data` 参数支持真实 Cp 训练（默认自动探测 `<data_source>/Pressure`）。有此数据的 shape 直接**使用压力 npz 里的 STL 网格**训练（face_cp 与面 1:1，无映射误差；注意 marching-cubes 网格不与 STL 面对应，故真实 Cp 模式不能用提取网格）。真实标签是论文的单工况（u=30 m/s，+x），每个 shape 只有一个 case；缺压力文件的 shape 自动回退几何代理标签并在日志中统计。

### 4.5 环境

- Python 依赖：项目内 `.venv`（torch 2.5.1+cu121、scikit-image、plyfile、`trimesh`（含 `rtree`）、`pymeshfix`（备用）、`scipy`、`fastapi`、`uvicorn[standard]`）。
- OpenFOAM：机器上 `~/openSrcPkg/OF12/OpenFOAM-12`（**OpenFOAM 12**，debug 编译）可用，`source ~/openSrcPkg/OF12/OpenFOAM-12/etc/bashrc` 后 `blockMesh`/`snappyHexMesh`/`foamRun` 均正常。**未安装 OpenFOAM 14**（现有版本足够）。
- 官方 C++ 工具（`bin/PreprocessMesh`）：**未编译**。原因：`libpangolin-dev` 不在 Ubuntu 24.04 源里（Pangolin 无官方包），且该工具运行时依赖 GL 上下文做光栅化，无显示环境下难以运行。SDF 采样因此使用等效的 `scripts/preprocess_sdfsample.py`（输出格式与官方 npz 完全一致）。如确需官方工具：`git submodule update --init third-party/cnpy` 后从源码编译 [Pangolin](https://github.com/stevenlovegrove/Pangolin)，再 `cmake -B build && cmake --build build`。

## 5. 使用流程

### 5.1 训练隐式场（SDF 或 occupancy）

```bash
# SDF（DrivAerNet 主实验）
bash drivaernet/scripts/train.sh
# occupancy 变体（实验目录独立为 drivaernet_f1_occ，checkpoint 互不覆盖）
bash drivaernet/scripts/train_occ.sh

# ShapeNet 示例（原流程，不变）
.venv/bin/python train_deep_sdf.py -e examples/chairs
```

注意事项：

- **米制数据的提取域**：`extract_differentiable_mesh` 默认采样 [-1,1]³ 单位立方体，覆盖不了米制车形（x∈[-1.03,3.97]）。所有提取命令必须传 `--bounds -1.2 -1.2 -0.2 4.2 1.2 1.9`（已写入两个 specs.json 的 `ExtractBounds`，`server.py` 自动读取），否则只能得到被截断的残缺网格。
- **occupancy 下游**：重建/精修命令需显式带 `--field_type occupancy`（`server.py` 从 specs.json 自动读取）；CFD 环节在 occupancy 模式下不可用（见 §2.4）。
- FieldType 改变训练目标本身，不能中途切换，必须从头训。

### 5.2 网格级重建（Chamfer，拓扑可变）

```bash
# DrivAerNet 封装脚本
bash drivaernet/scripts/reconstruct_mesh.sh train

# 原始命令
.venv/bin/python reconstruct_mesh.py -e <experiment> -d <data> -s <split> \
    --iters 300 --resolution 128 --mc_backend torch \
    --init sdf            # 或 rand：纯网格级优化，演示拓扑变化
    # --field_type occupancy  若 decoder 是 occupancy 模式训练的
    # --bounds xmin ymin zmin xmax ymax zmax  米制数据必传
```

输出：`<experiment>/Reconstructions_mesh/<epoch>/Meshes|Codes`。其中的 latent code 即 `train_pressure_surrogate.py` 做形状条件时复用的码。

### 5.3 剪影精修（2D 监督）

```bash
.venv/bin/python refine_silhouette.py -e <experiment> -d <data> -s <split> \
    --iters 200 --resolution 128 --view_index 0 --image_size 64
```

输出：`<experiment>/ReconstructionsSilhouette/<epoch>/Meshes|Codes`（含目标剪影 `_target.npy` 与中间网格）。

### 5.4 CFD：训练压力代理 → 快速预测 → 降阻优化

```bash
# 1) 训练压力代理（多案例：每形状 4 个 BC 案例、留出 20% 形状验证、
#    来流方向在绕 +x 的 60° 锥内均匀采样、U∈[10,20]、sdf 特征 + 2 层 GNN）
.venv/bin/python train_pressure_surrogate.py -e <experiment> -d <data> -s <split> \
    --model operator --resolution 63 --iters 20000 --lr 3e-4 \
    --cases_per_shape 4 --val_fraction 0.2 --dir_cone_deg 60 --u_range 10 20 \
    --sdf_features --gnn_layers 2
# --seed（默认 0）播种全部随机源（案例采样/val 划分/训练循环），同配置完全可复现
# --cp_data 用 DrivAerNet++ 真实 Cp 标签（默认自动探测 <data>/Pressure）：
.venv/bin/python train_pressure_surrogate.py \
    -e drivaernet/experiments/drivaernet_f1 \
    -d drivaernet/data/processed \
    -s drivaernet/splits/train.json \
    --cases_per_shape 1 --val_fraction 0.2
# 旧行为（单案例、无 BC 条件化、按 final loss 保存）：
#   --cases_per_shape 1 --val_fraction 0 --flow_dir 1 0 0
# --openfoam 使用现场 OpenFOAM 标签（逐案例 velocity=U，经 pressure_to_cp 转 Cp）

# 2) 降阻优化（BC 条件化模型可指定案例：--alpha 为 xy 平面内迎角（度），
#    --velocity 缺省取训练 u_range 中值；损失 = Cd + λ‖z−z₀‖²）
.venv/bin/python optimize_drag.py -e <experiment> --latent <path/to/latent.pth> \
    --alpha 20 --iters 150 --reg_lambda 2.0 --resolution 63
# 旧格式 checkpoint（无 model_kwargs）自动回退：raw drag 损失 + 固定 flow_dir
```

输出：surrogate 存 `<experiment>/CfdSurrogate/latest.pth`（含 model_kwargs / bc_fields / val_mse / val_cd_rel_err / u_range / dir_cone_deg，val 最优保存）；优化结果存 `<experiment>/DragOptimization/`（初始/优化网格 ply + latent）。

### 5.5 体积场 POD-ROM：(z, BC) → 体积流场

```bash
# 合成场全链条（无 CFD 求解器时的流程验证；现场生成快照后走统一 npz 加载路径）
.venv/bin/python train_volume_rom.py -e <experiment> -d <data> -s <split> \
    --synthetic --grid_resolution 64 --pod_energy 0.999 \
    --cases_per_shape 4 --u_range 10 20 --dir_cone_deg 180 \
    --iters 20000 --lr 1e-3 --val_fraction 0.2 --seed 0

# 真实/预生成快照（每 (shape, case) 一个 npz：fields (G,4) float32 [u,v,w,p]
# 按参考网格 C-order 展平、bc (4,) [U,dir_x,dir_y,dir_z]、shape 为 SDF 样本 npz 名）
.venv/bin/python train_volume_rom.py -e <experiment> -d <data> -s <split> \
    --snapshots <dir> --grid_resolution 64 --pod_energy 0.999
# --snapshots 与 --synthetic 互斥；--pod_rank r 可固定秩（覆盖能量截断）；
# --seed 播种全部随机源（案例采样/val 划分/POD 随机起始/训练循环）
```

输出：`<experiment>/VolumeROM/`：`pod_basis.pth`（POD 基 + 均值场 + 奇异值 + 捕获能量，拟合只用 train 形状快照；日志打印前 20 个奇异值的精确累积能量谱与实际截断 r）、`latest.pth`（best-on-val 回归器，含 model_kwargs / bc_fields / u_range / dir_cone_deg / grid_resolution / pod_basis_file / val 指标 / seed）；`--synthetic` 的快照存 `<experiment>/VolumeROM/snapshots/`。训练日志每 200 iter 并列打印 val 系数 MSE、重构全场相对 L2、真值 projection_error 下界与均值场基线。

#### 快照 npz 格式约定（真实 CFD 数据接入契约）

权威出处：`deep_sdf/cfd/volume.py` 模块 docstring、`train_volume_rom.py` 的 `save_snapshot`（写）/ `load_snapshot`（读 + 校验）。

**文件组织**：每个 (shape, case) 一个 `.npz`，全部放 `--snapshots <dir>`；**文件名随意**（加载按文件内容的 `shape` 字段建索引而非文件名）；目录中多余快照被忽略，split 中某形状无快照则报错。

**npz 内部**（`np.savez` 写出；加载 `allow_pickle=False`，必须为纯数组）：

| key | 形状 | dtype | 含义 |
|---|---|---|---|
| `fields` | (G, 4) | float（加载转 float32） | 参考网格逐点 **[u, v, w, p]**，列序固定 |
| `bc` | (4,) | 同上 | **[U, dir_x, dir_y, dir_z]**（`BC_FIELDS`） |
| `shape` | 0 维字符串标量 | numpy 字符串 | split 中的 SDF 样本名（如 `car/shape_000.npz`），决定挂到哪个 latent |

**代码强制校验**（`load_snapshot`）：三个 key 齐全；`fields` 二维且第 2 列 = 4；`bc` 恰 4 元素；`fields.shape[0]` = `grid_resolution³`（不符报 grid mismatch）。

**网格行序**：`fields` 第 g 行对应 `make_reference_grid(N, domain)` 的第 g 个点——C-order 展平、**x 最慢 z 最快**：`g = i·N² + j·N + k`，坐标 `(x,y,z) = domain_lo + (i,j,k)·Δ`，`Δ = (hi−lo)/(N−1)`，默认域 `[−1.5,1.5]³`、N=64。从求解器网格插值时必须插值到同一组点、同一顺序：

```python
lin = np.linspace(-1.5, 1.5, 64)
xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)  # (G,3)，即 fields 行序
```

**语义约定（代码不校验，必须遵守）**：

1. **坐标系/归一化**：快照与 DeepSDF 预处理同一坐标约定（形状居中、尺度约 [−1,1]³）；真实尺度的 STL/CFD 结果须先按 preprocess 的同一变换归一化再采样；
2. `bc` 的**方向必须是单位向量**，`U` 为来流速度大小（m/s）；`load_snapshot` 只查长度、不查单位性；
3. **`p` 通道存无量纲 Cp**（= p/(½ρU²)），与 surface 分支标签约定一致；代码对 p 原样训练、不做转换，混入有量纲压力会破坏 Cp 体系；
4. **体内点**（SDF<0）速度置 0（no-slip 延拓），压力取近壁延拓值或有界值（无意义野值会浪费 POD 模态能量）；
5. 每形状案例数任意（不同 U / 不同方向），每案例一个文件。

**OpenFOAM 侧建议**：在 case 内配 `sampleDict`/`postProcess` 让求解器直接在上述探针点输出 (u, p)（按同一行序），再组装 npz，避免二次插值误差。

#### PIPOD-DeepONet 三阶段训练（§2.8）

推荐用 JSON 配置（`--config`，避免三阶段间抄错数据参数；顶层键 = 任意 CLI flag 的默认值，`stages` 列表放逐阶段覆盖，CLI 仍最优先；`--stage N>1` 且未给 `--init_from` 时自动链到 `PipodONet/stage<N-1>.pth`；解析后的完整参数存档到 `PipodONet/config_stage<N>.json`）：

```bash
# 合成场（examples/ellipsoids/pipod_config.json，复现 §6.7 合成线）
.venv/bin/python train_pipod_deeponet.py --config examples/ellipsoids/pipod_config.json --stage 1
# OpenFOAM 真实场（examples/ellipsoids_of/pipod_config.json，含 --snapshots +
# --latent_manifest、stage3 自动 lr 1e-4 + wall_bc noslip + re 150）
.venv/bin/python train_pipod_deeponet.py --config examples/ellipsoids_of/pipod_config.json --stage 3
```

等价的纯 CLI 形式（与旧行为一致，config 文件就是从它收敛来的）：

```bash
# 合成场（--synthetic，势流 + 尾迹；同 split/seed 下快照自动复用）
.venv/bin/python train_pipod_deeponet.py -e examples/ellipsoids -d data/ellipsoids \
    -s examples/ellipsoids/split.json --synthetic --stage 1 \
    --grid_resolution 64 --cases_per_shape 4 --u_range 10 20 --dir_cone_deg 180 \
    --iters 20000 --lr 1e-3 --val_fraction 0.2 --seed 0
# stage 2/3 依次加 --init_from <experiment>/PipodONet/stage1|2.pth（stage 3 lr 1e-4）

# OpenFOAM 真实数据（LHS 形状经 manifest 提供 latent，不经 split 重建）
.venv/bin/python train_pipod_deeponet.py -e examples/ellipsoids_of -d data/ellipsoids \
    -s examples/ellipsoids/split.json \
    --snapshots data/openfoam/ellipsoids/snapshots \
    --latent_manifest data/openfoam/ellipsoids/lhs_latents.npz \
    --stage 1 --grid_resolution 64 --iters 20000 --lr 1e-3 --val_fraction 0.2 --seed 0
# stage 2 同上 + --init_from .../stage1.pth；
# stage 3 同上 + --init_from .../stage2.pth --lr 1e-4 --wall_bc noslip --re 150
```

输出：`<experiment>/PipodONet/`：`pod_basis_{u,v,w,p}.pth`（cPOD 逐变量基 + 均值场 + 奇异值，仅 train 案例拟合）、`stage1/2/3.pth`（best-on-val；含 model_kwargs / rank / re / val 指标 / seed）、`metrics_stage<1|2|3>.jsonl`（每 50 iter 一条评估记录，含 val 系数 MSE、全场相对 L2（逐变量）、projection 下界、均值基线，stage 3 另有 val continuity/momentum/wall 残差与 λ_phys）、`train_state_stage<N>.pth`（每 50 iter 保存的完整训练状态：模型 + optimizer + iter + best，`--resume` 无损续训）。`--latent_manifest` 与 `--synthetic` 互斥。训练曲线可视化：`plot_pipod_metrics.py -e <experiment>`（出 `metrics.png`；`--overlay <exp2>` 叠加对比、`-w 30` 训练中每 30 s 实时刷新）。

#### PIPOD 训练参数说明（按阶段分组）

以下参数均可写在 `--config` JSON 顶层（`stages` 列表可放逐阶段覆盖），CLI 同名 flag 优先。标注 ✦ 的是实测对本项目结果有显著影响的参数。

**数据与网格（三阶段共用，阶段间必须一致）**

| 参数 | 默认 | 含义与影响 |
|---|---|---|
| `--snapshots` / `--synthetic` | 二选一 | 真实 npz 快照目录 / 现场生成合成势流场。合成场仅供流程验证（泛化差，见 §6.7） |
| `--latent_manifest` | 无 | LHS 形状的 z 来源（npz: names+latents），跳过 split 的 latent 重建 |
| `--grid_resolution` | 64 | 均匀参考网格 N³。与快照 G=N³ 强绑定，不一致直接报错 |
| `--grid_stretch` | off | ✦ 双分辨率拉伸网格（107³，近壁 h=0.027/远场 0.064），配 `--grid_dense_half 1.35 --grid_h_fine --grid_growth`。边界层 δ 内采样点 2.1→3.7；点数 ×4.7，训练/评估同步变慢；与均匀网格快照不可混用 |
| `--pod_energy` | 0.999 | POD 能量截断阈值。越大 rank 越大、投影下界越低但回归越难（实测 586 案例 r=351 时 stage-3 OOM 风险高） |
| `--pod_rank` | None | 固定公共秩（覆盖能量截断）。数据/内存受限时的手动阀 |
| `--cases_per_shape --u_range --dir_cone_deg --wake_amp/--wake_sigma` | 4 / [10,20] / 180 / 0.15 / 0.5 | 仅 `--synthetic`：每形状案例数、来流速度范围、方向锥角、尾迹幅度/宽度 |
| `--val_fraction` | 0.2 | 留出案例比例。val 集决定 best-on-val；太小评估噪声大，太大浪费训练数据 |
| `--seed` | 0 | 播种全部随机源（案例采样/val 划分/POD/训练循环），同配置逐位可复现 |

**Stage 1（branch 系数回归，L = L_POD）**

| 参数 | 默认 | 含义与影响 |
|---|---|---|
| `--hidden --num_layers --bc_hidden` | 256 / 4 / 64 | branch MLP 宽度/深度、bc 编码器宽度。stage-1 是结构上限（系数与空间无关），加大容量对 val 收益很小 |
| `--iters --lr` | 20000 / 1e-3 | stage-1 通常 2–4k iter 即达 best，余弦日程下长 iters 无害（后期 lr≈0） |

**Stage 2（branch+trunk 全场训练，L = L_POD + λ_f·L_field）**

| 参数 | 默认 | 含义与影响 |
|---|---|---|
| `--trunk_hidden` | [256, 512] | trunk MLP 结构（in_dim=8 → rank×4）。场拟合能力的主要来源 |
| `--lambda_pod --lambda_field` | 1.0 / 1.0 | 两项权重。λ_field 过小退化为 stage-1；过大则系数监督弱化 |
| `--n_field` | 16384 | 每 iter 场监督点数。G=1.2M 时占比 ~1.3%，噪声大但覆盖快 |
| `--field_near_frac --field_near_band` | 0 / 0.15 | ✦ 近壁重要性采样比例与 |sdf| 带宽。误差集中在边界层/尾迹，0.3–0.5 实测明显改善近壁拟合；拉伸网格下 0.3 即可（网格本身已聚近壁） |
| `--init_from` | None | stage-1 checkpoint（branch 初始化）；**也接受 stage-2 checkpoint 做 warm start**（保权重、重置 optimizer 动量，用于中断后续跑） |
| `--resume` | off | ✦ 从 `train_state_stage2.pth` 无损续训（含 optimizer 状态）；LR 日程按当前 `--iters` 重建并快进——调整总 iters 不浪费已跑进度 |

**Stage 3（物理微调，L += λ_phys(t)·(L_cont+L_mom+L_wall+L_ff)）**

| 参数 | 默认 | 含义与影响 |
|---|---|---|
| `--lr` | 1e-4（配置中） | 比 stage 2 低 10×，精修而非重学 |
| `--lambda_phys` | None（日程） | ✦ 固定物理权重；不设时按日程 0→0.01→0.05→0.1（20%/50%/80% iter 处升档）。过早给大 λ 会让场误差上升，日程式最稳 |
| `--re` | 1e4 | 动量方程特征雷诺数（ν=1/Re 进残差）。**必须与 CFD 工况一致**（U=10 ν=0.1 → Re=100），否则物理约束与数据打架 |
| `--n_collocation` | 4096 | 每 iter 物理点数的（近壁 30%/尾迹 30%/高梯度 20%/均匀 20% 分层采样） |
| `--phys_chunk` | 1024 | 二阶 autodiff 分块大小。**不改变结果**（梯度按计数归一，实测相对差 1.2e-7），只控显存：每 chunk 二阶图 ~3MB/点，256 是 8GB 卡的安全值 |
| `--margin` | 2.0 | 流体掩码边距（sdf > margin·h 才算流体点），固体内部不算物理残差 |
| `--wall_bc` | slip | 壁面约束类型：slip（u·n=0，无粘/势流）/ **noslip**（u=0，粘性数据必选，否则与 CFD 数据矛盾） |

**优化器日程（各阶段通用）**

| 参数 | 默认 | 含义与影响 |
|---|---|---|
| `--lr_schedule` | constant | ✦ `cosine` = 2% warmup + 余弦衰减到 `lr×lr_final_ratio`（0.01）。实测同数据下 stage-2 best 0.567→0.474（配合数据扩充），主要收益在末段低噪声精修；`constant` 下 best 全靠噪声低点、终值常回弹 |
| `--iters` | 20000 | 余弦日程绑定总 iter 数；调整后用 `--resume` 续训即可（日程自动重建快进） |

#### OpenFOAM 快照批量生成（§6.7 数据线）

```bash
# 2 案例端到端验证（解析椭球 + 物理判据）
.venv/bin/python generate_openfoam_snapshots.py -e examples/ellipsoids_of \
    --out data/openfoam/ellipsoids --cases_per_shape 4 --u_range 10 20 --seed 0
# LHS 形状批量化（--lhs 60：训练 latent 逐维 [min,max] 外扩 10% 采样，
# decoder→DeepMesh 提取→合法性检查→STL；4 路并行）
.venv/bin/python generate_openfoam_snapshots.py -e examples/ellipsoids_of \
    --out data/openfoam/ellipsoids --lhs 60 --cases_per_shape 4 --seed 0
```

案例设置：域 [−18,18]³（≥10× 最长轴）；blockMesh 36³；snappyHexMesh 表面 level 3 + 尾迹 refinementBox + **6 层边界层**（expansionRatio 1.25、首层 ~0.011）；simpleFoam 层流（ν=0.1，Re≈150）、SIMPLE 残差达标即止；外边界 freestream（任意来流方向无需旋转几何）；全部在 DeepSDF 归一化坐标系。产出 `batch_summary.json`（形状接受/拒绝、逐案例成败与耗时）。

### 5.6 Web 端实时 latent 探索

浏览器中拖动 latent code 滑条实时查看重建表面。架构：FastAPI 后端加载 decoder checkpoint，WebSocket 收 `{z, resolution}` 后调用 `extract_differentiable_mesh`（纯前向，GPU 锁串行），回传二进制帧 `[uint32 nv, uint32 nf | float32 verts | int32 faces | float32 normals]`；前端 three.js 增量更新 BufferGeometry。设计文档：`docs/superpowers/specs/2026-09-10-web-latent-explorer-design.md`。

```bash
.venv/bin/python server.py -e <experiment> --checkpoint latest --port 8000
# 打开 http://127.0.0.1:8000（局域网访问加 --host 0.0.0.0）
```

- 启动时读 `specs.json` 构建 decoder（兼容 weight_norm / norm_layers / latent_in / FieldType），加载 `ModelParameters/<ckpt>.pth`（容忍 DataParallel `module.` 前缀）；有 GPU 自动用 `mc_backend="torch"`，否则 skimage；
- 预设形状下拉来自 `LatentCodes/<ckpt>.pth`；z 滑条按 8 维一组折叠，范围取全部训练 latent 每维 ±3σ；另有随机 z、A→B 插值、分辨率 63/125/249；
- MC 空场返回空 mesh 提示而不崩溃；WebSocket 断连自动重连；
- 提取域：默认 [-1,1]³；米制数据（如 DrivAerNet）须在 specs.json 加 `"ExtractBounds": [xmin,ymin,zmin,xmax,ymax,zmax]`（如 drivaernet_f1 的 [-1.2,-1.2,-0.2,4.2,1.2,1.9]）或用 `--bounds` 覆盖，前端按 bounds 自动取景。

### checkpoint 格式

`ModelParameters/<ckpt>.pth` = `{"model_state_dict": ..., "epoch": ...}`；`LatentCodes/<ckpt>.pth` = `{"latent_codes": Embedding.weight 或 (N, latent_size) Tensor}`（见 `deep_sdf/workspace.py`）。

## 6. 验证结果汇总

所有验证脚本位于 /tmp（不入库），均实际运行通过。

### 6.1 数学正确性

- 解析球面场：IFT 梯度 vs 解析梯度相对误差 **0.00000**；有限差分复核 ~1%（MC 离散噪声内）；
- 提取精度：63³ 网格球面半径误差 2e-4；米制 bounds 下球心定位正确、半径误差 4e-5（skimage/torch 两后端、IFT 梯度通路均验证）。

### 6.2 拓扑可微（论文 Fig. 3 复现）

| 实验 | 初 → 末 | Chamfer |
|---|---|---|
| 合成：球 → 环面 | χ: 2 → 0 | 0.230 → 0.031 |
| 真实数据（MeshSDF 官方 bob/spot）：牛 → 鸭 | χ: 4 → 2 → 0 | 0.286 → 0.103 |
| occupancy 场：球 → 环面 | χ: 2 → 0 | 0.229 → 0.092 |
| 2D 剪影监督：球 → 环面 | χ: 2 → 0，IoU 0.38 → 1.00 | L1 0.020 → 0.0006 |

### 6.3 GPU marching cubes

| 网格 | 顶点/面 | torch GPU | skimage CPU | 加速比 |
|---|---|---|---|---|
| 128³ | 34,720 / 69,440 | 4.70 ms | 21.43 ms | 4.6× |
| 249³ | 130,696 / 261,392 | 26.23 ms | 129.43 ms | 4.9× |

与 skimage 输出对照：顶点数一致、Chamfer 2e-8、Euler 数相同、welded 无重复顶点；空场/边界 level 不崩溃。

### 6.4 CFD（27 椭球族 {0.5,0.7,0.9}³ 端到端）

单一场景版本（2026-09-09，固定 +x 来流）：

| 验证项 | 局部 MLP 基线 | 物理神经算子 |
|---|---|---|
| 留出泛化（22 训练/5 留出）压力 MSE | 3.6e-3 | **7.6e-4** |
| 留出集 drag 相对误差 | 3.3% | **1.1%** |
| 分辨率无关性（63³ vs 125³ drag 偏差） | — | **0.70%** |
| 端到端降阻（从 (0.9,0.9,0.9) 出发，150 轮） | −50% | **−56%**（x 向 +4.6%，y/z 向 −22%） |
| 单次前向（≈2.2 万面片） | 0.5 ms | **1.07 ms**（替代 OpenFOAM 分钟级求解） |
| 曲率 sanity | — | 解析球面 autograd 精确（max err 2.4e-7）；训练 decoder FD 路径 \|κ\|·r ≈ 1.03 |

案例一般化升级（2026-09-10，Cp 标签 + BC 条件化；同 27 椭球族，22 训练/5 留出形状，每形状 4 案例，U∈[10,20]，operator 配置 hidden=128、num_layers=3、--sdf_features --gnn_layers 2，分辨率 63，20k iter @ lr 3e-4）：

| 模型 | 留出形状 Cd 相对误差 |
|---|---|
| 锥内新方向（训练锥 = 绕 +x 60°，α≈15°/45°） | **6.3% / 4.0%**（Cp MSE 3.3e-3 / 2.2e-3，best-val MSE 4.4e-3） |
| 锥外新方向 α=75°/90°/120° | 22% / **6.1%** / 57%（优雅退化） |
| 全球面训练（dir_cone 180°）× 4 个随机未见方向 | **6.2%–8.2%** |
| bc_dim=0 基线（固定 +x 训练）@ α=45°/90°/120° | 25% / **100%** / 173%（换方向即失效，对照成立） |
| 降阻优化（--alpha 20，150 轮，(0.9,0.9,0.9) 出发） | Cd 1.096 → 0.640（**−41.6%**）；代理真值复核 Cd 1.205 → 0.639（**−47%**），x 向 +11%、y/z 向 −27%/−26%，网格非空（21,476 面） |
| 回归 | 旧格式 checkpoint（无 model_kwargs/buffer）strict=False 加载并正常优化；`--cases_per_shape 1 --val_fraction 0` 恢复旧训练行为；无新 flag 默认路径可运行 |

代码审核修正（2026-09-10 第二轮，单元测试 5/5 + 全量 29/29 通过）：BC 向量标准化（`bc_mean`/`bc_std` buffer，精确数值验证 max diff 0.0；修复前旧 checkpoint 以恒等统计量兼容加载）；`--seed` 播种全部随机源（同配置两次运行逐位一致：iter 0/200 mse 与 val 指标完全相同）；`optimize_drag.py` 的改进率计算加除零保护（初始 Cd≈0 的合法情况）；`--model local` 与 operator-only flag 组合时显式 warning 且不再被 val 留出削减训练数据；checkpoint 增加 `train_mse`/`seed` 字段。

### 6.5 体积场 POD-ROM（2026-09-10，27 椭球族 × 4 案例合成场端到端）

配置：`--synthetic --grid_resolution 64`（G=262,144，D=G·4=1,048,576）、`--pod_energy 0.999`、每形状 4 案例（U∈[10,20]、全球面方向）、22 训练/5 留出形状、regressor hidden=256 num_layers=4、20k iter @ lr 1e-3、seed 0。

- 单元测试（/tmp/test_volume_rom.py，27/27 通过）：已知低秩矩阵（rank 10 + 1e-6 噪声）randomized SVD 子空间恢复误差 **3.4e-7**（< 1e-4 要求）、基正交性 max\|VᵀV−I\| 2.4e-7、能量截断正确取 r=10；project/reconstruct 往返 6.1e-7；relative_error/projection_error 对投影内/外向量与解析值一致（~1e-7 量级）；标准化 setter 精确生效（max diff 0.0，含恒等默认/1e-8 clamp/尺寸校验）；快照 npz 读写一致且网格不匹配可被检测；合成场 sanity：体内 u≡0、远场速度 = U（偏差 1.3e-6）、尾迹区平均速度 0.63·U < 自由流、Cp 与 Bernoulli 式逐点一致。
- POD 能量谱（88 训练案例，精确 Gram 谱）：前 3 个模态累积能量 34.9% / 69.1% / **98.65%**，第 20 模态 99.74%；能量截断 **r=37**，捕获能量 99.904%；train 投影下界（不可约）**3.22%**。

| 留出形状 + 新方向 val（20 案例） | 数值 |
|---|---|
| 系数 MSE（标准化空间） | 6.80e-1 |
| 重构全场相对 L2 | **3.07e-1** |
| 真值 projection_error 下界 | 6.62e-2 |
| 均值场基线 | 1.012 |
| best-on-val checkpoint 重载复现 | 两次加载逐位一致，与训练日志 best val 完全相同 |

结论：合成场上重构误差（30.7%）显著优于均值场基线（101%，为其 30%），但仍高于投影下界（6.6%）——训练曲线显示 train 系数 MSE 降到 ~3e-2 而 val 停在 ~0.7，瓶颈在 **88 个训练案例对 37 维系数回归的泛化**（留出全新形状 + 全球面新方向），而非 POD 截断；接真实快照或扩充案例数后回归侧仍有明确余量。回归零影响：/tmp/test_cfd_upgrade.py 29/29、/tmp/test_cfd_review_fixes.py 5/5 通过。

### 6.6 Web latent 探索工具

手工构造的解析八面体场 decoder（输出 = tanh(|x|+|y|+|z|−0.5)，256 维 latent 不影响输出）写入假 checkpoint 端到端验证：`/api/info` 与静态页 200；WebSocket 往返返回合法二进制帧；63³/125³ 提取网格 L1 半径恰为 0.5000，面索引/法线合法，单帧 120–220ms；随机权重 decoder 的空场（无零交叉）路径返回空 mesh 不崩溃。验证后假 checkpoint 已删除，未污染实验目录。

### 6.7 PIPOD-DeepONet 三阶段（2026-09-11/12，合成场 + OpenFOAM 真实数据）

架构与流程见 §2.8 / §5.5。单元测试（/tmp/test_deeponet.py、test_physics.py、test_pipod_stage1/2/3.py 等全部通过）：**physics 真值基准**——解析球势流场（`sphere_potential_flow`）经 PDEInformer 的连续性残差 < 1e-8、动量残差 < 1e-5（float64），确认 autodiff 残差实现正确。

**OpenFOAM 数据线**（`data/openfoam/ellipsoids/`）：LHS 60 个 latent 采样形状中 **59 个接受**（1 个因空网格拒绝）+ 27 个解析椭球，共 86 形状 × 4 BC（U∈[10,20]、全球面方向）= **344 案例全部成功**（成功率 100%，无网格/收敛失败；另有 2 个管道验证案例，快照共 346 个 npz）；4 路并行总耗时 2429 s（逐案例 24.7–66.5 s、均值 35.2 s，RTX 3070 空闲时 CPU 20 核）。网格/求解参数见 §5.5。

| 留出 val（20% 案例，seed 0） | 合成场（27 形状 × 4 案例 = 88 train 案例） | OpenFOAM 真实场（344 案例，~275 train） |
|---|---|---|
| rank（cPOD 公共秩） | 79 | 220 |
| Stage 1 全场相对 L2 | —（best-on-val checkpoint 被后续运行覆盖） | 0.7428 |
| Stage 2 全场相对 L2 | 0.934 | **0.567**（u/v/w/p = 0.613/0.414/0.544/0.697） |
| Stage 3 全场相对 L2 | 0.933 | **0.423**（0.372/0.298/0.361/0.660） |
| Stage 3 val continuity / momentum / wall | 0.0735 / 0.357 / 0.423 | 0.194 / 0.218 / 0.314 |
| 真值 projection 下界 / 均值场基线 | 0.174 / 0.953 | 0.127 / 0.992 |
| 训练耗时（20k iter） | — | S1 240 s / S2 3037 s / S3 11860 s |

结论：

1. **数据量是泛化瓶颈，且已被证实可解**：合成场 88 训练案例下 stage-2 val rel L2 0.934 ≈ 均值基线 0.953（几乎无泛化）；换 344 个真实案例后同配置降到 0.567，stage-3 物理微调进一步到 0.423（基线的 43%）；
2. **Stage 3 物理微调在真实数据上双降**：场误差 0.567→0.423（−25%）的同时学到物理一致性（continuity 0.19 / momentum 0.22）；合成场线上 λ_phys 微调后场误差基本持平（0.934→0.933），因合成势流本身已近似满足 PDE、残差信号弱；
3. 真实数据投影下界 0.127 仍有明显余量（0.423 vs 0.127），瓶颈仍在 (z,μ)→系数回归的泛化，方向：更多 LHS 形状、更高 rank、trunk 容量；
4. **在线推理**：stage-3 模型全网格 64³（262,144 点）单次前向 **88.7 ms**（RTX 3070，20 次均值；CPU 2.09 s），对比单案例 simpleFoam ~35 s（CPU 并行摊销）加速约 **400×**，且推理为纯前向、无迭代求解。

**v2/v3 改进轮（2026-09-13/14）**：针对「泛化差距 0.423 vs 下界 0.127」做数据侧改进——(a) LHS 扩展 60 个新形状（seed 1，offset 60；240/240 案例成功，manifest 合并至 146 形状 / 586 案例）；(b) `L_field` 采样 50% 近壁带（|sdf|<0.15）；(c) 余弦 lr（2% warmup → 1e-2 末值）。v2（仅余弦）中止于 stage-2 中期（best 0.635 @4700，已优于旧 run 同期）；v3（三者叠加，`examples/ellipsoids_of3`）结果：

| v3（586 案例，~469 train；rank 351，投影下界 0.106） | best val rel L2 | 物理残差（cont/mom/wall） |
|---|---|---|
| Stage 1 | 0.6943 | — |
| Stage 2 | 0.4744（@19400） | — |
| Stage 3 | **0.3384**（@17450，iter 17900 手动终止） | **0.139 / 0.208 / 0.297** |

对照旧 run：stage-2 0.567→0.474（−16%）、stage-3 0.423→0.338（−20%），三项物理残差全部更优（旧 0.194/0.218/0.314）；POD 下界也随数据增加从 0.127 降到 0.106。**结论：留出形状的泛化瓶颈主要由形状多样性决定，扩充 LHS 形状 + 近壁采样 + 余弦衰减带来 ~20% 的终值改善；0.338 vs 0.106 的差距仍指向回归侧。** 工程排障：stage-3 在 586 案例/rank 351 下 OOM 的根因是 physics loss 的二阶 autodiff 图跨 chunk 累积（实测 ~3 MB/点/chunk），修复为逐 chunk 立即 backward（梯度相对差 1.2e-7），并将快照 fields 与 POD 基移到 CPU（POD 拟合后 empty_cache）。

## 7. 实现要点与坑

1. **训练采样分布**：隐式场训练必须混合「均匀采样 + 近表面采样（表面点加 σ=0.005/0.05 噪声）」；纯均匀采样 + SDF clamp 会塌缩为恒正常数场（实测复现）。DrivAerNet 预处理已按 94% 近表面（σ=5mm）+ 6% 均匀执行。
2. **ReLU 网络的解析曲率为零**：分段线性场的零等值面是分片平面，`compute_mean_curvature` 的 `method="auto"` 检测后回退到解析梯度的有限差分；若需精确曲率，可用光滑激活（softplus）训练 decoder。
3. **Chamfer 下界**：稀疏目标点云的最近邻间隙构成 Chamfer 损失下界（如 2k 点对应 ~0.03），验证收敛应对照几何量（半径/extent）而非损失绝对值。
4. **L1 剪影损失**梯度幅值不随误差缩小，需配学习率衰减（本实现每 20 轮 ×0.6）。
5. **法向符号约定**：压力标签用外法向（驻点压力 `max(0,−n·v̂)²` 在迎风面），drag 积分用体内法向使 `Σ p·(n·v̂)·A` 为正值物理压阻；两者不可混用同一约定。
6. **skimage LUT 版本兼容**：skimage ≥0.25 将 marching cubes LUT 序列化为 base64（`CASESCLASSIC`），`marching_cubes_torch.py` 含新旧两条导入路径。
7. **float32 cdist 精度伪影**：`torch.cdist` 默认 matmul 路径在 float32 下有 ~1e-3 误差，精度敏感对比用 `compute_mode="donot_use_mm_for_euclid_dist"`。
8. **GNN message passing 必须用残差形式**：按 `h ← ReLU(MLP([h, mean_agg]))` 直接替换特征会在初始化时破坏含符号的法向/质心信息（ReLU 半波整流），多案例训练中收敛慢且对初始化高方差（实测 val MSE 停留在 0.25 ≈ 仅预测均值）；改为 `h ← h + ReLU(MLP([h, mean_agg]))` 后与其他配置收敛速度一致（8k iter val MSE 5.8e-3）。
9. **多案例训练的迭代数需求**：单方向算子 2k iter 即收敛；引入方向/U 条件化后任务变难，2k iter 严重欠拟合（val MSE 0.25），12k–20k iter（lr 3e-4）达到 val MSE 4e-3 量级。
10. **Cp/Cd 归一化约定**：标签与预测均为无量纲 Cp（驻点 1）；Cd = Σ Cp·(n·v̂)·A / A_ref，A_ref 为迎风投影面积（`reference_area`），对闭曲面均匀 Cp=1 时 Cd=0（压力前后抵消），Cp=1 仅作用迎风面时 Cd=1——sanity check 须按此构造。
11. **BC 向量必须标准化**：`bc=[U, dir]` 中 U∈[10,20] 与方向分量 ∈[−1,1] 量级差 10 倍，直接进 `bc_encoder` 导致条件化病态；模型内 `bc_mean`/`bc_std` buffer 默认恒等（旧 checkpoint 行为不变），训练脚本用全部训练案例统计量设置。
12. **decoder 输出恒有界**：`networks/deep_sdf_decoder.py` 末层恒有 tanh（`self.th`），输出 ∈(−1,1)——occupancy 的 `occ=(pred+1)/2` 映射因此天然成立，无需 `use_tanh`（该选项是在末层**之前**再加一道 tanh，occupancy 下反而加重饱和）。
13. **occupancy 下游必须配对**：occupancy 训练的 checkpoint，所有提取命令要带 `--field_type occupancy`（按 iso=0 去切值域 ∈(0,1) 的场会得到错误/空网格）；`server.py` 从 specs.json 自动读取是唯一的例外。

## 8. 局限与遗留

- `run_openfoam` 按论文参数编写（simpleFoam + snappyHexMesh），本机实测环境为 OpenFOAM 12（见 §4.5）；snappyHexMesh 重划网格后 patch 面序与输入 STL 不一一对应，需按质心回映射（代码中已检测并提示）；多案例模式下 OpenFOAM 路径来流方向固定 +x（方向变化需旋转 STL，未实现），且不可压流的 Cp 对速度不变——同形状各案例标签几乎相同，BC 条件化在 OpenFOAM 路径下无实际收益（训练时会明确警告）。
- 椭球族 CFD 验证的标签是几何代理 Cp（非物理模拟）；DrivAerNet++ 分支已接入真实 OpenFOAM Cp（§4.4），但为单工况（u=30 m/s，+x），BC 条件化对真实标签同样无收益。
- BC 条件化算子对训练锥内/全球面未见方向泛化良好（Cd 误差 4–8%），但对锥外方向只是优雅退化（α=120° 时 57%），极端外推仍需全球面训练数据。
- 网格提取的 MC 仍非亚体素自适应；论文的 GPU MC + 32³ 粗网格起点已复现，分辨率链为 `N→2N−1`（如 128 实际到 249）。
- 体积场 POD-ROM：合成场非物理（非散度自由，仅流程验证）；POD 基绑定固定参考网格（64³，换分辨率需重建基），任意点查询需对网格场插值（分辨率无关性留给 GNO 路线）；体内延拓假设 u≡0（no-slip extension）；系数回归受训练案例数限制，留出全新形状上回归误差明显高于 POD 投影下界（30.7% vs 6.6%），需更多案例/数据增强；随机路径的能量截断 rank 依赖全局 RNG 播种（`--seed` 可复现）。
- PIPOD-DeepONet（§2.8/§6.7）：真实数据 val rel L2 0.423 仍高于投影下界 0.127，瓶颈在 (z,μ)→系数的泛化；OpenFOAM 线为层流（ν=0.1，Re≈150）椭球族，湍流/RANS、表面 Cp/Cf head 与 Cd/Cl 积分尚未接入；POD 基同样绑定 64³ 参考网格；stage-3 的 λ_phys 日程与 Re 依赖手工设定（合成场线 stage-1 的 best checkpoint 曾被真实数据运行误覆盖，合成 stage-1 数值缺失——实验目录分离后已不会发生）。
- 论文中未实现的下游应用：端到端训练未接入 DrivAerNet 流程（§4.5 用 Chamfer 反传 Θ，重连恒等式在技术上支持但需另写微调脚本）、三平面场景重建（§4.4 ConvOccNet 精修）、真实 ShapeNet/Pix3D 全量 benchmark。

## 9. 后续扩展

1. **加数据**：继续从 Dataverse 下载其余 zip 放入 `drivaernet/data/stl/`，重跑预处理与 split 即可（自动跳过已处理文件）。
2. **OpenFOAM 仿真**：网格已是 CFD 级质量，可直接 `surfaceCheck` 后 `snappyHexMesh` 外气动计算；注意物面尺寸是真实车长（~5m），参考速度设 40 m/s 时 Re≈1.4e7。
3. **重建**：`reconstruct.py` / `reconstruct_mesh.py` 直接用训练好的 latent code。

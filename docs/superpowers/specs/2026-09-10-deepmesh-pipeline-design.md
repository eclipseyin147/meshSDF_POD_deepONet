# DeepMesh 完整流水线扩展 — 设计文档

日期：2026-09-10
前置：`docs/superpowers/specs/2026-09-10-deepmesh-design.md`（可微等值面提取核心，已实现并验证）
目标：补全 DeepMesh 论文的四大剩余组件，搭建「训练 → 可微网格 → 渲染精修 / CFD 快速预测与优化」完整流程。

## 1. 子系统划分

| 子系统 | 论文依据 | 新增/修改文件 |
|---|---|---|
| A. Occupancy 场支持 | §3.1 Eq. 3（BCE）、§6.3（logit 放大） | 改 `deep_sdf/differentiable_mesh.py`、`train_deep_sdf.py`、`reconstruct_mesh.py` |
| B. GPU marching cubes | §3.2.2 Algorithm 1（GPU MC） | 新增 `deep_sdf/marching_cubes_torch.py` |
| C. 可微渲染精修 | §4.2 Eq. 14/15、图 13 | 新增 `deep_sdf/soft_renderer.py`、`refine_silhouette.py` |
| D. CFD 代理 + drag 优化 | §4.3 Eq. 16 | 新增 `deep_sdf/cfd/`（labels/surrogate）、`train_pressure_surrogate.py`、`optimize_drag.py` |

全局约束沿用前一设计：不破坏现有行为（新功能全部由可选参数开启）、代码风格一致、验证脚本放 /tmp、不做 git 提交。

## 2. Task A：Occupancy 场支持

- `train_deep_sdf.py`：specs 增加可选项 `"FieldType": "occupancy"`（默认 `"sdf"`，缺省行为不变）。occupancy 模式下：decoder 输出经 `occ = (pred + 1) / 2`（末层 tanh 输出 ∈(−1,1) → (0,1)）；标签 `occ_gt = (sdf_gt < 0)`；损失换为 BCE（论文 Eq. 3）；不做 clamp。
- `deep_sdf/differentiable_mesh.py`：
  - `sample_sdf_octree(..., field_type="sdf")`：occupancy 时细分判据改为「体素 8 角点不全部位于 0.5 同侧」（§3.2.2）；三线性填充不变（凸组合保持 0.5 同侧性，无伪交叉）。
  - `extract_differentiable_mesh(..., field_type="sdf")`：iso = 0.0（sdf）/ 0.5（occupancy）；occupancy 时先对网格做 logit 放大 `logit(clamp(g, 1e-6, 1-1e-6))` 再送 MC（§6.3，近似 SDF 线性区域、消除网格 artifacts）；法向仍对原始网络输出求 ∇f；重连恒等式中用 `(f(v) − iso)`。
- `reconstruct_mesh.py`：加 `--field_type` 参数透传。
- 验证（/tmp）：训练一个 2 形状 occupancy 小网络（球+环面，BCE），occupancy 模式提取网格 χ 正确（2/0），且网格级 Chamfer 优化同样能改拓扑。

## 3. Task B：GPU marching cubes

- `deep_sdf/marching_cubes_torch.py`：`marching_cubes_torch(sdf_grid, level=0.0, voxel_size=1.0) -> (verts (M,3) float32, faces (F,3) int64)`，输入 cuda 张量则全程 GPU。
- 实现：纯 PyTorch 向量化。逐边生成顶点（3 个方向的边网格，跨体素天然共享顶点 id，无需去重）：计算每条边的符号交叉掩码与线性插值位置，cumsum 压缩成顶点 id 表；逐体素查 case 表（256 种）生成三角形，引用 12 条边的全局顶点 id。case 表复用 skimage 的 LUT（`skimage.measure._marching_cubes_lewiner_luts`，导入失败时回退 skimage CPU 并 warning）。
- `extract_differentiable_mesh(..., mc_backend="skimage")`：新增参数，`"torch"` 时走 GPU 路径；IFT 重连不变（梯度公式与提取器无关）。
- 验证（/tmp）：平滑随机场（高斯叠加）上与 skimage 对比：Chamfer < 1e-4、χ 相同；128³/249³ 计时对比（GPU 应显著快于 CPU skimage）。

## 4. Task C：可微渲染精修

- `deep_sdf/soft_renderer.py`：纯 PyTorch SoftRas 风格剪影渲染器（无新依赖）：
  - `look_at(eye, center, up) -> R, t`；正交投影即可（剪影任务足够）。
  - `render_silhouette(verts, faces, image_size=64, sigma=1e-4, ...) -> (H,W) ∈ [0,1]`：逐面片计算像素到 2D 三角形的有向距离 `D`（重心坐标内部 + 边距离外部），软覆盖 `A = sigmoid(-D/σ)`，聚合 `S = 1 − Π_f (1 − A_f)`；按面片分块防显存爆炸；对 verts 可微。
- `refine_silhouette.py`（仿 `reconstruct_mesh.py`）：加载训练好的实验；目标剪影来自 (a) 渲染目标 latent 的网格（--target_latent）或 (b) 图片文件；损失 Ltask3 = ‖S(M(z)) − T‖₁（Eq. 14），可多视角；经 `extract_differentiable_mesh` 回传 latent。
- 验证（/tmp）：球/环面小 decoder；目标 = 环面正视图剪影；从球 latent 出发精修 → 剪影 IoU 显著提升（>0.9），且网格 χ 从 2 变 0（拓扑经 2D 监督改变，对应论文 Fig. 3b）。

## 5. Task D：CFD 代理 + drag 优化

论文 §4.3：`L_task(M) = ∫∫_M g_β·n_x dM + L_constraint + L_reg`，`g_β` 为预测压力场的代理网络（OpenFOAM 标签）。

**代理架构（应用户要求采用物理神经算子，非 CNN/局部 MLP）**：`PressureNeuralOperator`——DeepONet 风格 branch/trunk + FiLM 条件：branch 吃形状 latent z（全局形状上下文，表达阻塞等非局部效应）；trunk 吃逐面片物理特征 [质心, 外法向, 平均曲率 κ, log面积]，κ 由 DeepSDF 场解析/差分求得（注意：ReLU decoder 的场分段线性，解析 κ≡0，实现带 FD 回退）。性质：分辨率无关（任意网格采样上查询）、形状条件化、物理特征驱动。保留局部 MLP 作为基线对照。

- `deep_sdf/cfd/labels.py`：`export_stl`；`run_openfoam`（检测 simpleFoam，无则抛带说明异常）；`proxy_pressure`（几何代理标签 = 驻点压力 × 非局部阻塞因子 (1+β·D_yz/L_x) + 尾部吸力，明确标注非物理模拟）。
- `deep_sdf/cfd/surrogate.py`：`PressureNeuralOperator` / 基线 `PressureSurrogate`、`face_features`、`compute_mean_curvature`、`drag_from_pressure`（Σ_f p_f·(n_f·v̂)·A_f，对 verts 可微，经 DeepMesh 回传 latent；operator 时梯度同时经 branch 直连路径回传）。
- `train_pressure_surrogate.py`：`--model {local,operator}`（默认 operator），latent → 网格 → 压力标签 → MSE 训练 → 存 `<experiment>/CfdSurrogate/latest.pth`。
- `optimize_drag.py`：`L = drag + λ_reg‖z − z_0‖²`，Adam 优化 latent，每轮重新提取网格；输出优化前后 drag、相对改善、surrogate 单次前向耗时（毫秒 vs OpenFOAM 分钟级）。
- 验证（/tmp）：27 椭球族（{0.5,0.7,0.9}³）训练小 DeepSDF → 代理标签 → 训练 surrogate。四项验证：(1) 留出泛化 22/5 划分，operator vs local MLP 对比；(2) 分辨率无关性（63 vs 125 网格 drag 一致性）；(3) 端到端降阻（drag −15%+ 且沿来流拉长）；(4) κ sanity（球面 |κ| ≈ 1/r）。

## 6. 集成与端到端验证

- `extract_differentiable_mesh` 最终签名：
  `extract_differentiable_mesh(decoder, latent_vec, resolution=128, initial_resolution=32, max_batch=2**18, threshold_factor=2.0, eps=1e-8, field_type="sdf", mc_backend="skimage")`
- 端到端串联验证（/tmp）：occupancy 训练 → GPU MC 提取 → 剪影精修 → CFD 优化，全部复用 bob/spot 或合成族，确认接口互相兼容。
- 回归：`train_deep_sdf.py --help`、`reconstruct.py --help`、Task 1-3 既有 /tmp 验证脚本重跑通过。

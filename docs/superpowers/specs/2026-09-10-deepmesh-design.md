# DeepMesh：基于 DeepSDF 的可微等值面提取 — 设计文档

日期：2026-09-10
参考论文：*DeepMesh: Differentiable Iso-Surface Extraction*（Guillard et al., arXiv:2106.11795v2，仓库根目录 PDF）

## 1. 背景与差距分析

当前仓库是 DeepSDF：MLP 学习隐式场 `f_Θ(z, x) → SDF`。与 DeepMesh 论文的差距集中在「隐式场 → 显式网格」环节：

| 环节 | 当前 DeepSDF | DeepMesh 论文 |
|---|---|---|
| 网格提取 | `deep_sdf/mesh.py:create_mesh`：256³ 密集均匀采样 + skimage marching cubes | 分层 octree 采样（32³ 起步，体素任一角点 \|f\| < 体素对角线 2Δx 时细分；论文 Algorithm 1） |
| 可微性 | MC 顶点位置为线性插值 `x = si/(si−sj)`，`si = sj` 时不连续，拓扑变化无法反传（论文 §6.1） | 隐函数定理（Theorem 1）：`∂x/∂c = −(n/‖n‖²)·∂f/∂c`，`n = ∇f(x)`；`∂L/∂c = −Σ_v (∂L/∂v)·(∇f/‖∇f‖²)·∂f/∂c`（Eq. 5） |
| 反向目标 | 仅隐式域（`reconstruct.py` 对 SDF 采样值做 L1） | 网格顶点级损失（Chamfer、可微渲染）可回传到 `z` 或 `[z\|Θ]`（Algorithm 2、§4.5） |
| 顶点法向 | 无 | 用网络解析梯度 `∇f(v)` 作法向 |

网络结构（`networks/deep_sdf_decoder.py`）两者一致，直接复用。

## 2. 方案

**IFT 梯度重连**（不用自定义 C++/CUDA 算子）：

- 前向：octree 采样 + skimage marching cubes 提取 `(verts, faces)`；每次前向重新提取，拓扑天然可变。
- 用 autograd 在顶点处求解析法向 `n = ∇_x f(z, x)|_v`（detach 为常数）。
- 重连恒等式：`v_diff = v − (n/‖n‖²)·(f(z,v) − f(z,v).detach())`。
  - 数值上等于 `v`（括号内为 0）；
  - 梯度 `∂v_diff/∂c = −(n/‖n‖²)·∂f/∂c`，恰为 Theorem 1；
  - 普通 autograd 即可同时支持 `c = z` 与 `c = [z|Θ]`（端到端训练）。

被否决的替代方案：
- 纯 PyTorch 可微 MC 前向：线性插值在符号翻转处有奇异性（论文 §6.1），拓扑变化时梯度错误。
- 引入官方实现 / Kaolin GPU MC：重型 CUDA 依赖，与本仓库风格不符。

## 3. 模块设计

### 3.1 `deep_sdf/differentiable_mesh.py`（新增）

- `sample_sdf_octree(decoder, latent_vec, resolution, max_batch, initial_resolution=32, threshold_factor=2.0) -> (N,N,N) torch.FloatTensor`
  - 分层评估：当前分辨率下评估全部格点；对每个体素，若任一角点 `|f| < threshold_factor * voxel_size`（体素对角线准则）则标记细分；下一层只评估新增格点；直到目标分辨率。返回密集 SDF 网格。
  - `decoder` 调用复用 `deep_sdf.utils.decode_sdf`，在 `torch.no_grad()` 下分批进行。
- `compute_sdf_gradients(decoder, latent_vec, points, max_batch) -> (M,3) tensor`
  - 对查询点求 `∇_x f`：分批 `points.requires_grad_(True)` → `decode_sdf` → `torch.autograd.grad`。
- `extract_differentiable_mesh(decoder, latent_vec, resolution=128, initial_resolution=32, max_batch=2**18) -> (verts, faces, normals)`
  1. `sample_sdf_octree` 得到 SDF 网格；
  2. `skimage.measure.marching_cubes`（level=0）得 `verts, faces`（numpy → torch，cuda）；
  3. `compute_sdf_gradients` 得顶点法向 `n`（detach）；
  4. IFT 重连：`sdf = decode_sdf(decoder, latent_vec, verts)`；`verts = verts − (n/‖n‖²)·(sdf − sdf.detach())`；
  5. 返回 `(verts[可微], faces, normals)`。`faces` 仅为整数索引，不参与梯度。
  - 空网格（无零等值面）时返回空张量并给出警告。

### 3.2 `reconstruct_mesh.py`（新增脚本，接口仿照 `reconstruct.py`）

网格级 latent 优化（对应论文 §4.1/§4.2 的 refinement 流程）：

- 参数：`--experiment`、`--checkpoint`、`--data`、`--split`、`--iters`、`--resolution`、`--lr`、`--skip`，另加 `--init {sdf,rand}`：
  - `sdf`：先用隐式域 SDF L1 优化（复用 `reconstruct.reconstruct`）得到初始 latent，再做网格级 Chamfer 精修；
  - `rand`：从先验 `N(0, 0.01²)` 采样直接做网格级优化（演示拓扑变化）。
- 目标点云：对目标形状（split 中的 npz）先以其真值 SDF 样本用隐式域拟合 target latent，再 `create_mesh` 提取目标网格，按三角形面积均匀采样 10K 表面点（论文 §4.5 用 10K）。
- 优化循环：每轮 `extract_differentiable_mesh` → 网格顶点与目标点云的双边 Chamfer 损失（顶点数较多时随机子采样 10K）→ Adam 更新 latent。每 50 轮保存中间网格观察拓扑变化。
- 输出目录结构与 `reconstruct.py` 一致（`ReconstructionsMesh/<epoch>/Meshes|Codes`）。

### 3.3 现有文件改动

- `deep_sdf/__init__.py`：导出 `differentiable_mesh` 子模块。
- 不改动 `deep_sdf/mesh.py`、`reconstruct.py` 的既有行为。

## 4. 验证计划（项目无测试框架，验证脚本放 /tmp，不入库）

1. **梯度正确性**：随机初始化小 decoder（latent=8，4×64），构造目标点云，Chamfer 经 `extract_differentiable_mesh` 对 latent 求梯度，与中心有限差分对比（相对误差 < 1e-2 量级可接受，注意 MC 拓扑在微扰下不变才有意义——取小 ε）。
2. **端到端拓扑变化（复现论文 Fig. 3）**：小 auto-decoder 过拟合球（genus 0）与环面（genus 1）两形状的解析 SDF 样本；从球 latent 出发，用 `extract_differentiable_mesh` + Chamfer 优化至环面点云；逐轮记录 Euler 示性数 χ = V − E + F，期望从 2 变为 0（出现孔洞），证明拓扑可经梯度改变。
3. **回归**：`reconstruct.py` 与 `deep_sdf/mesh.py` 行为不变（接口未动，人工核对即可）。

## 5. 错误处理

- 网格为空（latent 处于无表面区域）：`extract_differentiable_mesh` 发出警告并返回空张量；`reconstruct_mesh.py` 检测到空网格时跳过该轮更新并计数，连续为空则提前终止并提示用 `--init sdf`。
- `‖n‖` 过小（梯度消失点）：重连时分母加 `eps=1e-8`。

## 6. 范围之外（YAGNI）

- occupancy 场的 inverse-sigmoid 放大（论文 §6.3）：本仓库为 SDF 模型，不需要。
- 可微渲染（§4.2 DR）、CFD 气动优化（§4.4）、三平面场景重建（§4.3）：均为下游应用，不在本次实现范围。
- GPU marching cubes：skimage 已够用，octree 采样已降低主要开销。

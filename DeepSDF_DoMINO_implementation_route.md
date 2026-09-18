# DeepSDF + NVIDIA PhysicsNeMo DoMINO：几何条件化三维外流场快速预测实现路线

> **面向对象**：已经具备 CFD 数值算法、GPU/HPC 和 PyTorch 基础，希望把自有 `octree Cartesian + cut-cell` CFD 数据接入 NVIDIA PhysicsNeMo，并建立“新几何 + 新工况 → 三维外流场 / 壁面压力 / 壁面剪切”的快速预测模型。
>
> **推荐主线**：`DeepSDF geometry latent → DoMINO GeometryRep 条件化 → DoMINO surface + volume decoder`
>
> **建议版本基线**：PhysicsNeMo 26.08 / 当前官方 main API。PhysicsNeMo 26.08 新增 AeroJEPA、xDeepONet 以及 batched radius-search 支持；DoMINO 仍然是当前外流场中最直接的 surface+volume operator 基线。正式实现时应固定你实际安装的 PhysicsNeMo commit，因为模型内部 API 在快速演进。

---

## 1. 最终目标

希望最终部署成一个连续坐标查询型 surrogate：

\[
(G,\; \mathrm{BC},\;\mathbf{x})
\;\longrightarrow\;
\hat{\mathbf{y}}(\mathbf{x})
\]

其中：

- `G`：CAD/STL/表面三角网格
- `BC`：\(U_\infty, Re, Ma, \alpha, \beta, \rho_\infty, T_\infty,\ldots\)
- \(\mathbf{x}=(x,y,z)\)：任意 volume query point 或 surface point
- \(\hat{\mathbf y}\)：
  - volume：\([u,v,w,p]\)，也可以进一步加入 \(k,\omega,T\)
  - surface：\([p,\tau_x,\tau_y,\tau_z]\)，或者采用 \([C_p,C_f,\mathbf n]\) 等无量纲目标

最终推理链希望达到：

```text
             CAD / STL
                 │
                 ▼
          DeepSDF Encoder
                 │
                 ▼
             z_geo
                 │
                 ├────────────────────────────┐
                 │                            │
                 ▼                            ▼
       DeepSDF SDF decoder              DoMINO geometry path
                 │                            │
                 │                         SDF + point cloud
                 │                            │
                 └──────────────┬─────────────┘
                                ▼
                    Geometry-conditioned DoMINO
                                │
        BC ───────────────► Parameter Encoder
                                │
              query point ─────┤
                                ▼
                         surface / volume
                          flow prediction
```

核心原则是：

> **DeepSDF 不应该简单替换 DoMINO 原有 SDF，也不建议一开始把 DeepSDF latent 直接粗暴 concat 到最终 MLP；最佳第一版应把 `z_geo` 作为“全局几何条件”，通过 FiLM/gating 调制 DoMINO 的 GeometryRep 输出。**

---

# 2. 为什么 DoMINO 是当前最适合的 PhysicsNeMo 基础

当前 DoMINO API 明确是用于同时预测 surface 和 volume quantities 的模型；模型内部包含 `GeometryRep`、FourierMLP/basis functions、global parameter encoding、local/global geometry processing 和 aggregation。官方描述也明确强调其多尺度 point-cloud geometry representation 与 SDF enrichment。  

当前源码中，DoMINO `forward()` 需要的主要几何输入包括：

- `geometry_coordinates`
- `grid`, `surf_grid`
- `sdf_grid`, `sdf_surf_grid`
- `sdf_nodes`
- `pos_volume_closest`
- `pos_volume_center_of_mass`
- `pos_surface_center_of_mass`
- `surface_mesh_centers`
- `surface_mesh_neighbors`
- `surface_normals`
- `surface_neighbors_normals`
- `surface_areas`
- `surface_neighbors_areas`
- `volume_mesh_centers`
- `global_params_values`
- `global_params_reference`

因此你的自有 cut-cell CFD 数据其实与 DoMINO 的 volume-point/surface-point 机制天然兼容。

当前源码的 forward 流程大致为：

```text
STL geometry points
       │
       ├── volume GeometryRep ──┐
       │                        │
       └── surface GeometryRep ─┤
                                ▼
                      local geometry encoding
                                │
          ┌─────────────────────┼──────────────────┐
          │                     │                  │
       volume x              volume SDF         BC encoding
          │                     │                  │
          └─────────────────────┴──────────────────┘
                                │
                         SolutionCalculator
                                │
                         volume / surface
```

PhysicsNeMo 当前源码还明确显示：global parameters 经 `FourierMLP` 编码，然后进入 surface / volume `SolutionCalculator`；aggregation model 的输入维度也会包含 global-parameter embedding。  

这意味着：

> **BC 已经有比较成熟的注入路径；真正需要新增的是 `z_geo` 的全局几何条件路径。**

---

# 3. DeepSDF 在整个模型中的正确职责

## 3.1 普通 SDF 和 DeepSDF latent 不要混为一谈

普通 SDF：

\[
d(\mathbf x)=\operatorname{SDF}(\mathbf x)
\]

主要告诉网络：

> 当前 query point 距离壁面多远、位于流体侧还是实体侧。

DeepSDF latent：

\[
z_g=E_G(G)
\]

主要告诉网络：

> 当前整个物体是什么形状。

所以真正希望模型看到的是：

\[
\boxed{z_g + SDF(\mathbf x)+\text{local geometry}}
\]

而不是：

\[
\boxed{z_g\;\text{取代 SDF}}
\]

### 例子

两个不同车身在某个空间位置都可能满足：

\[
SDF(\mathbf x)=0.02L_{ref}
\]

但是：

\[
z_{g,A}\neq z_{g,B}
\]

所以模型既需要局部距离信息，也需要全局形状信息。

---

# 4. 推荐的总体网络结构

## 4.1 V1：最推荐、改 DoMINO 最少的架构

```text
                          Geometry G
                              │
                              ▼
                       DeepSDF Encoder
                              │
                              ▼
                         z_geo [B,Z]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            Geo-FiLM Volume      Geo-FiLM Surface
                    │                   │
                    ▼                   ▼
             DoMINO GeometryRep Volume / Surface
                    │                   │
                    ▼                   ▼
              modulated global geometric encoding
                    │                   │
                    ▼                   ▼
            MultiGeometryEncoding / local features
                    │                   │
         BC ─────► ParameterModel      │
                    │                   │
                    └────────┬──────────┘
                             ▼
                       Aggregation
                             │
                 ┌───────────┴──────────┐
                 ▼                      ▼
            volume field             surface field
             u,v,w,p                  p,tau
```

这里使用：

\[
\gamma_g=MLP_\gamma(z_g),\qquad
\beta_g=MLP_\beta(z_g)
\]

然后对 DoMINO 的 geometry feature 做：

\[
H_g'=\gamma_g\odot H_g+\beta_g
\]

### 为什么首选 FiLM

因为它有四个优点：

1. 不需要让最终 aggregation MLP 直接处理 128/256/512 维 latent。
2. 不破坏 DoMINO 原来的 local geometry feature dimension。
3. 让整个全局几何编码受到 geometry latent 控制。
4. 可以在预训练 DoMINO 上增加少量参数做 fine-tuning。

---

# 5. V1 的精确插入位置

当前 DoMINO 源码中的关键步骤是：

```python
encoding_g_vol = self.geo_rep_volume(
    geo_centers_vol, p_grid, sdf_grid
)
```

和：

```python
encoding_g_surf = self.geo_rep_surface(
    geo_centers_surf, s_grid, sdf_surf_grid
)
```

随后进入：

```python
self.volume_local_geo_encodings(...)
self.surface_local_geo_encodings(...)
```

再进入：

```python
self.solution_calculator_vol(...)
self.solution_calculator_surf(...)
```

因此第一版最干净的插入点就是：

```text
GeoRep output
      ↓
DeepSDF FiLM
      ↓
local geometry encoding
      ↓
solution calculator
```

而不是直接修改 `geometry_coordinates`。

当前源码正是在 GeoRep 完成之后、local geometry encoding 之前进入 solution path，因此这一位置对自定义 fork 最合适。  

---

# 6. 建议的 DeepSDF Encoder 设计

## 6.1 不建议最终使用“每个 shape 一个可学习 latent”的纯 auto-decoder

如果模型最终需要：

```text
新 STL → 自动得到 z_geo → CFD inference
```

那么最好最终拥有：

\[
E_G:G\rightarrow z_g
\]

即显式 geometry encoder。

推荐训练成：

\[
G\rightarrow E_G(G)=z_g
\]

以及：

\[
(z_g,\mathbf x)\rightarrow \hat d(\mathbf x)
\]

形成真正可部署的 shape encoder + SDF decoder。

---

## 6.2 Geometry Encoder 输入

第一阶段推荐使用经过一致化处理后的 surface point cloud：

```text
surface_points : [B, Ngeo, 3]
normals        : [B, Ngeo, 3]
area           : [B, Ngeo, 1]
```

建议：

- 所有 geometry 统一到同一个参考坐标系
- 平移到几何中心
- 使用统一的 `L_ref`
- 保证训练/推理时旋转增强规则一致

可形成：

\[
X_G=[x,y,z,n_x,n_y,n_z,A]
\]

但 V1 也可以先只用：

\[
X_G=[x,y,z,n_x,n_y,n_z]
\]

---

# 7. DeepSDF latent dimension 的建议

第一版推荐：

```yaml
latent_dim: 256
```

实验范围：

```text
128  → 参数少、容易正则化
256  → 推荐起点
512  → 几何变化很复杂时再尝试
```

不建议第一版直接用 1024+。

你的最终目标不是让 latent 完美重构每一个几何细节，而是让：

\[
z_g
\]

成为对 CFD 最有用的 geometry coordinate。

因此过大的 latent 反而容易让模型记忆训练 geometry。

---

# 8. DeepSDF 训练数据如何生成

对每一个 geometry：

```text
CAD/STL
   │
   ├── surface point sampling
   │
   ├── near-surface random sampling
   │
   ├── uniform volume sampling
   │
   └── bounding-box sampling
          │
          ▼
      (x,y,z,SDF)
```

建议采样组成：

```text
near surface     50%
uniform volume   30%
far field        20%
```

对于 CFD surrogate，near-wall SDF 精度非常重要，因此不要完全照搬只适合 shape reconstruction 的均匀采样策略。

建议使用 normalized SDF：

\[
\tilde d=\frac{d}{L_{ref}}
\]

并采用 truncation：

\[
\tilde d_t=\operatorname{clip}(\tilde d,-d_{max},d_{max})
\]

初始可以测试：

```text
d_max = 0.1 ~ 0.2 L_ref
```

然后针对壁面 CFD surrogate 做单独消融。

---

# 9. DeepSDF loss

建议最初：

\[
L_{SDF}
=
L_{value}+\lambda_z L_{latent}
\]

其中：

\[
L_{value}
=
\operatorname{SmoothL1}
(\hat d,d)
\]

latent regularization：

\[
L_{latent}=\|z_g\|_2^2
\]

如果增加 Eikonal regularization：

\[
L_{eik}
=
\mathbb E_
\mathbf x
\left(
\|\nabla_\mathbf x d_\theta\|-1
\right)^2
\]

则：

\[
L_{DeepSDF}
=
L_{value}
+
\lambda_zL_{latent}
+
\lambda_eL_{eik}
\]

但第一版建议先确保 shape latent 稳定，再考虑 Eikonal；不要在一个实验里同时加入太多正则项。

---

# 10. DoMINO 输入应该如何从你的 cut-cell CFD 数据生成

这是整个项目最关键的数据工程部分。

## 10.1 Geometry point

DoMINO 当前接口需要：

```python
geometry_coordinates  # [B, Ngeo, 3]
```

对于你的 cut-cell solver，建议使用：

```text
STL triangle centroid
```

而不是 cut-cell centroid 作为 geometry_coordinates。

---

## 10.2 Volume prediction points

建议：

```python
volume_mesh_centers   # [B, Nvol, 3]
```

直接使用你的 fluid cell center。

特别是你使用：

```text
octree Cartesian + boundary cut-cell
```

因此 volume query point 应保留：

```text
x,y,z
cell size
fluid volume fraction
wall distance
nearest wall point
```

其中后三者可以作为你的 custom feature。

---

# 11. 推荐加入 cut-cell geometry features

这一点是你的 solver 相比标准 automotive CFD dataset 最有特色的部分。

建议增加：

\[
f_{cut}=
[\alpha_v,
\mathbf n_w,
A_w/V,
d_w,
\Delta x,
\Delta y,
\Delta z,
N_{cut},\ldots]
\]

其中：

- \(\alpha_v\)：fluid volume fraction
- \(\mathbf n_w\)：局部 wall normal
- \(A_w/V\)：wall area / cell volume
- \(d_w\)：wall distance
- \(\Delta x,\Delta y,\Delta z\)：局部 cell size
- `N_cut`：cell 与 boundary intersection/切割相关统计量

### 但 V1 不要一开始全部塞进去

推荐实验顺序：

```text
V1：x,y,z + SDF + z_geo
V2：V1 + wall distance + wall normal
V3：V2 + volume fraction + cut-face descriptors
```

这样才可以判断真正的增益来自哪里。

---

# 12. SDF 数据路径

PhysicsNeMo 当前 DoMINO 已经原生使用：

```text
sdf_grid
sdf_surf_grid
sdf_nodes
```

而且源码中明确会对 volume node SDF 做 scaling，再和 closest-surface position / center-of-mass positional information 一起形成 node encoding。  

因此推荐保留两套 SDF 信息：

### 全局 SDF

\[
SDF_{grid}(x,y,z)
\]

用于 GeometryRep 的全局几何处理。

### Query-point SDF

\[
SDF(x_i)
\]

用于具体 volume/surface query point 的局部表示。

### DeepSDF latent

\[
z_g
\]

用于整体 geometry conditioning。

最终：

\[
\boxed{
GeometryInfo=
\{SDF_{grid}, SDF(x), z_g, local\ geometry\}
}
\]

---

# 13. DeepSDF 和 DoMINO 的三层 Geometry Information

这是整个架构最重要的理解框架：

| 几何信息 | 数学对象 | 主要作用 |
|---|---|---|
| 全局几何 | \(z_g\) | 这个物体整体是什么形状 |
| 中尺度几何 | `GeometryRep` | 多尺度几何上下文 |
| 局部几何 | surface neighbors / cell descriptors | 当前点附近具体几何 |
| 距离几何 | SDF | query point 到壁面的距离 |
| 离散几何 | cut-cell features | CFD cell 离散方式 |

因此建议不要追求“让 DeepSDF 把所有信息全部压缩掉”。

真正合理的是：

\[
\boxed{
CFD\ Geometry
=
Global\ Shape
+
Multi-scale\ Shape
+
Local\ Boundary
+
Distance
+
Discretization
}
\]

---

# 14. DeepSDF latent 如何进入 DoMINO：三种方案

## 14.1 方案 A：直接 concat —— 适合 baseline

```python
h = torch.cat([h_domino, z_geo_broadcast], dim=-1)
```

优点：简单。

缺点：需要修改 downstream feature dimension，并且容易让网络把 latent 当成普通 scalar feature。

**只建议做 baseline。**

---

## 14.2 方案 B：FiLM —— V1 首选

定义：

```python
gamma = gamma_mlp(z_geo)
beta  = beta_mlp(z_geo)
```

然后：

```python
h = gamma[..., None, None, None] * h + beta[..., None, None, None]
```

对 volume/surface 可以各自使用：

```text
z_geo
 │
 ├── volume gamma/beta
 │
 └── surface gamma/beta
```

推荐：

```text
latent_dim = 256
condition_dim = 128~256
```

不要直接令 `gamma/beta` = 256，而应该让其输出匹配 GeometryRep 当前 channel 数量：

\[
\gamma,\beta\in\mathbb R^{C_g}
\]

---

## 14.3 方案 C：latent → query-point feature —— V2

对每个 query point：

\[
q_g=MLP(z_g)
\]

广播为：

\[
Q_g(\mathbf x_i)=q_g
\]

然后：

\[
H_i=
[H_i^{DoMINO},q_g,H_i^{SDF},H_i^{cut}]
\]

这种方案让 latent 直接参与最终 prediction。

推荐作为 V2，而不是第一版。

---

# 15. 最终推荐的 V1 融合公式

先得到：

\[
H^V_{geo}
=
GeometryRep_V(G,SDF_{grid})
\]

\[
H^S_{geo}
=
GeometryRep_S(G,SDF_{surf,grid})
\]

DeepSDF：

\[
z_g=E_G(G)
\]

两个 FiLM：

\[
(\gamma_V,\beta_V)=M_V(z_g)
\]

\[
(\gamma_S,\beta_S)=M_S(z_g)
\]

最终：

\[
H_V'=\gamma_V\odot H_V+\beta_V
\]

\[
H_S'=\gamma_S\odot H_S+\beta_S
\]

再进入：

\[
H_V''=LocalGeo_V(H_V')
\]

\[
H_S''=LocalGeo_S(H_S')
\]

最后：

\[
\hat y_V
=
DoMINO_V(H_V'',x,SDF,BC)
\]

\[
\hat y_S
=
DoMINO_S(H_S'',x,n,A,BC)
\]

---

# 16. 一个很重要的实现细节：combined volume + surface 模式

当前 DoMINO 在 `combine_volume_surface` 打开时，会先：

```python
encoding_g = torch.cat((encoding_g_vol, encoding_g_surf), axis=1)
encoding_g_surf = self.combined_unet_surf(encoding_g)
encoding_g_vol = self.combined_unet_vol(encoding_g)
```

所以推荐：

```text
DeepSDF z_geo
       │
       ▼
GeometryRep V ──┐
                ├── Combined UNet ── FiLM ── LocalGeo V ── Volume
GeometryRep S ──┘
                                └──── FiLM ── LocalGeo S ── Surface
```

或者做成：

```text
GeometryRep
    ↓
Combined UNet
    ↓
DeepSDF FiLM
    ↓
LocalGeo
```

我更推荐后者，因为它只对已经融合的 multi-scale geometry representation 做一次 global shape modulation。

---

# 17. PyTorch 代码骨架

## 17.1 DeepSDF encoder

```python
import torch
import torch.nn as nn


class ShapeLatentEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden=256, latent_dim=256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.latent_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, points, normals):
        # points  : [B, N, 3]
        # normals : [B, N, 3]
        x = torch.cat([points, normals], dim=-1)
        h = self.point_mlp(x)                 # [B, N, hidden]
        h = h.transpose(1, 2)                 # [B, hidden, N]
        h = self.pool(h).squeeze(-1)          # [B, hidden]
        z = self.latent_head(h)               # [B, latent_dim]
        return z
```

这只是最小骨架。

真正用于复杂车辆 geometry 时，可以进一步换成：

- PointNet++
- Point Transformer
- GeometryRep 本身的 global encoder
- Surface graph encoder

但第一版优先保证端到端训练稳定。

---

# 18. FiLM condition module

```python
class GeoFiLM(nn.Module):
    def __init__(self, latent_dim, channels, hidden=256):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )

        # Identity-like initialization is preferred.
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, h, z):
        # h: [B, C, ...]
        gamma_beta = self.to_gamma_beta(z)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # Start near identity: h' ~= h
        gamma = 1.0 + gamma

        while gamma.ndim < h.ndim:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return gamma * h + beta
```

> 如果你的实际 DoMINO tensor layout 不是 `[B,C,X,Y,Z]`，按当前实现调整 broadcast 维度即可；不要假设 future PhysicsNeMo 版本保持完全相同的 tensor layout。

---

# 19. DoMINO fork 的推荐修改方式

不建议第一版通过 monkey-patch 修改安装包。

推荐：

```text
physicsnemo/
  models/
    domino/
      model.py                 # upstream
      domino_deepsdf.py        # your fork
      geometry_latent.py       # DeepSDF encoder + FiLM
```

`domino_deepsdf.py`：

```python
from physicsnemo.models.domino.model import DoMINO


class DeepSDFDoMINO(DoMINO):
    def __init__(
        self,
        input_features,
        output_features_vol=None,
        output_features_surf=None,
        global_features=2,
        model_parameters=None,
        latent_dim=256,
        freeze_deepsdf=False,
    ):
        super().__init__(
            input_features=input_features,
            output_features_vol=output_features_vol,
            output_features_surf=output_features_surf,
            global_features=global_features,
            model_parameters=model_parameters,
        )

        self.latent_dim = latent_dim

        # Exact channel numbers should be obtained from the actual model config.
        # Do not hard-code them before checking the installed PhysicsNeMo version.
        self.shape_encoder = ShapeLatentEncoder(
            in_dim=6,
            hidden=256,
            latent_dim=latent_dim,
        )

        self.geo_film_vol = ...
        self.geo_film_surf = ...

        if freeze_deepsdf:
            for p in self.shape_encoder.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
        z_geo = self.shape_encoder(
            data_dict["geometry_points"],
            data_dict["geometry_normals"],
        )

        # Copy the upstream DoMINO forward and insert:
        # encoding_g_vol = self.geo_film_vol(encoding_g_vol, z_geo)
        # encoding_g_surf = self.geo_film_surf(encoding_g_surf, z_geo)
        # before MultiGeometryEncoding / SolutionCalculator.

        ...
```

### 为什么推荐 fork

当前 DoMINO 的官方 `forward(data_dict)` 并没有公开一个 `geometry_latent` hook；它直接在内部读取 SDF、geometry points 并执行 GeometryRep → local encoding → SolutionCalculator。  

所以真正干净的实现应该是：

> **fork DoMINO 的 model.py，而不是试图从外部 wrapper 截断/重构已有 forward。**

---

# 20. 更稳妥的工程实现：不要让 DeepSDF 立即参与 SDF ground truth 计算

训练 CFD surrogate 时推荐：

```text
真实 STL
   │
   ├── PhysicsNeMo / CUDA SDF query ──► SDF ground truth
   │
   └── DeepSDF Encoder ───────────────► z_geo
```

而不是：

```text
STL → DeepSDF → predicted SDF → DoMINO
```

原因：

如果第一版同时依赖 DeepSDF 生成 SDF，则：

\[
CFD\ error
\supset
DeepSDF\ reconstruction\ error
\]

你会很难判断：

- CFD predictor 不准
- geometry latent 不准
- DeepSDF SDF reconstruction 不准

到底是哪一个导致性能下降。

因此推荐：

> **DeepSDF 负责 latent；DoMINO 继续使用高质量数值 SDF。**

待模型稳定以后，再研究 learned SDF 是否可以替换部分 SDF grid。

PhysicsNeMo-Mesh 当前已经提供 signed-distance queries，可用来统一预处理/验证 SDF；其 API 返回 SDF、closest hit points 和 hit faces。  

---

# 21. BC 的正确接法

当前 DoMINO 已支持多个 global parameters，并在 `encode_parameters=True` 时创建 FourierMLP parameter model。  

你的 global condition 可以定义为：

```text
U_inf
alpha
beta
Re
Ma
rho_inf
T_inf
L_ref
nu_inf
```

但注意：

> 不要把所有物理量都机械地丢给网络。

对于定密度低速外流场，很多变量可以通过无量纲化吸收。

推荐优先使用：

\[
BC=[Re,Ma,\alpha,\beta]
\]

对于压缩性问题再加入：

\[
BC=[Re,Ma,\alpha,\beta,T_\infty,\ldots]
\]

---

# 22. 推荐使用无量纲 CFD target

建议把预测目标优先定义成：

### Volume

\[
\tilde u=\frac{u}{U_\infty},
\quad
\tilde v=\frac{v}{U_\infty},
\quad
\tilde w=\frac{w}{U_\infty}
\]

压力可以考虑：

\[
C_p=
\frac{p-p_\infty}
{\frac12\rho_\infty U_\infty^2}
\]

### Surface

\[
C_p
\]

以及：

\[
C_f=
\frac{\tau_w}
{\frac12\rho_\infty U_\infty^2}
\]

这样做会显著减少跨 geometry / BC 数据集之间的尺度差异。

---

# 23. Surface target 推荐

第一版建议：

```text
surface output = [Cp, Cfx, Cfy, Cfz]
```

而不是直接预测：

```text
[p, tau_x, tau_y, tau_z]
```

原因：

- 数据跨工况更容易统一
- force integration 更自然
- loss scale 更稳定

之后可以同时输出：

```text
Cp
Cf magnitude
Cf vector
```

---

# 24. Volume target 推荐

第一版：

```text
[u/Uinf, v/Uinf, w/Uinf, Cp]
```

第二版：

```text
[u*, v*, w*, Cp, k*, omega*]
```

如果你的目标是完整 RANS surrogate，则：

```text
velocity
pressure
k
omega
nut
```

都可以纳入，但不建议第一版直接一次预测所有 turbulence variables。

原因是：

> velocity/pressure 是首要宏观量；turbulence variable 的局部尺度、壁面行为、数据 noise 和 solver-model bias 通常更难学。

---

# 25. Dataset 组织方式

推荐每个 CFD case 是一个独立样本：

```text
case_0001/
├── geometry/
│   ├── surface_points.npy
│   ├── surface_normals.npy
│   └── surface_areas.npy
│
├── sdf/
│   ├── grid.npy
│   ├── sdf_grid.npy
│   └── sdf_surface_grid.npy
│
├── volume/
│   ├── centers.npy
│   ├── sdf_nodes.npy
│   ├── pos_closest.npy
│   ├── pos_com.npy
│   ├── volume_fraction.npy
│   └── wall_normal.npy
│
├── surface/
│   ├── centers.npy
│   ├── neighbors.npy
│   ├── normals.npy
│   └── areas.npy
│
├── conditions.json
│
└── fields/
    ├── volume_uvwp.npy
    └── surface_Cp_Cf.npy
```

实际 production 环境可以转成：

- `.pdmsh` / TensorDict
- Zarr
- HDF5
- Parquet + binary tensors

PhysicsNeMo 26.08 的 mesh/data tooling 已增加 Zarr backend 等能力；但对于你已有的 cut-cell solver，第一阶段完全可以先做一个轻量自定义 datapipe，等数据结构稳定后再与 PhysicsNeMo-Mesh/Curator 接轨。  

---

# 26. Batch 内 geometry 数量不同怎么办

不同 geometry 的：

```text
N_surface
N_volume
N_geometry
```

通常都不同。

因此需要：

```text
random sampling
padding + mask
或 point-wise packed representation
```

当前 PhysicsNeMo 已增加 batched radius search，以支持 DoMINO / GeoTransolver 使用 local features 的 batch size > 1；因此建议以当前版本能力为基础，而不要设计成“永远只能 batch=1”的代码。  

但实际显存仍然可能使 DoMINO external-aero dataset 选择 batch=1，这需要通过 sample size、gradient accumulation 和 distributed data parallel 平衡。

---

# 27. 训练数据划分必须按 geometry，而不是按 CFD case 随机划分

这是非常重要的实验设计。

错误：

```text
同一个 geometry
 ├── Re=1e6 → train
 └── Re=2e6 → test
```

如果你真正想验证“新几何泛化”，应采用：

```text
Geometry A/B/C/D → train
Geometry E        → validation
Geometry F        → test
```

然后在每一个 geometry 内随机采样多个工况。

推荐三个 benchmark：

### Test A：同 geometry，新 BC

测试参数插值能力。

### Test B：新 geometry，新 BC

测试真正的 geometry-conditioned generalization。

### Test C：OOD geometry

测试 latent manifold 外推能力。

---

# 28. 最推荐的三阶段训练策略

## Stage 0：几何标准化

先统一：

```text
center
scale
rotation convention
L_ref
SDF sign convention
normal orientation
```

所有后续工作都依赖这个标准化。

---

## Stage 1：单独训练 DeepSDF

目标：

\[
G\rightarrow z_g\rightarrow SDF
\]

只训练 geometry model。

输出并缓存：

```text
case_0001 → z_geo
case_0002 → z_geo
...
```

第一阶段可以完全不涉及 CFD。

### 推荐初始配置

```yaml
latent_dim: 256
hidden_dim: 256
num_layers: 6-8
activation: GELU
optimizer: AdamW
lr: 1e-4
weight_decay: 1e-6
```

这些是工程起始值，不是固定最优超参数。

---

## Stage 2：冻结 DeepSDF，训练 DoMINO surrogate

此时：

\[
z_g=\operatorname{stopgrad}(E_G(G))
\]

只优化：

```text
DoMINO
GeoFiLM
BC ParameterModel
```

目标：

\[
(z_g,BC,geometry,SDF)
\rightarrow CFD
\]

这是**强烈推荐的第一版可用模型**。

---

## Stage 3：联合 fine-tuning

解冻：

```text
DeepSDF encoder
GeoFiLM
DoMINO
```

但不要让 DeepSDF reconstruction objective 消失。

使用：

\[
L
=
L_{CFD}
+
\lambda_{SDF}L_{SDF}
+
\lambda_{phys}L_{phys}
\]

推荐开始时：

```text
lambda_SDF = 0.05 ~ 0.2
lambda_phys = 0.01 ~ 0.1
```

然后根据不同 loss 的实际 magnitude 调整。

---

# 29. CFD loss 的推荐组成

建议第一版：

\[
L_{CFD}
=
\lambda_V L_V
+
\lambda_S L_S
+
\lambda_F L_F
\]

其中：

### Volume loss

\[
L_V
=
\operatorname{MSE}(\hat y_V,y_V)
\]

或 relative MSE。

### Surface loss

\[
L_S
=
\operatorname{MSE}(\hat y_S,y_S)
\]

### Force consistency

由 surface prediction 得到：

\[
\hat C_D,
\hat C_L,
\hat C_M
\]

再和 CFD 真值比较：

\[
L_F=
|\hat C_D-C_D|^2+
|\hat C_L-C_L|^2+
|\hat C_M-C_M|^2
\]

这项对于你的最终工业应用很有价值。

---

# 30. 为什么强烈建议加入 force loss

因为很多模型可能出现：

```text
局部 Cp 看起来不错
```

但是：

```text
integrated drag 错很多
```

你最终真正关心的通常是：

```text
Cd
Cl
Cm
surface pressure distribution
wall shear
```

因此可以把 surface field prediction 和 aerodynamic integral prediction 绑在一起。

最终：

```text
                surface Cp / Cf
                      │
                      ▼
              force integration
                      │
                Cd / Cl / Cm
                      │
                      ▼
               global loss
```

这也是非常适合你后续做 shape optimization 的结构。

---

# 31. Physics loss：建议第二阶段再加入

如果采用 volume velocity + pressure prediction，则可以考虑：

\[
r_c=\nabla\cdot\mathbf u
\]

以及简化 momentum residual。

但你的 solver 是工业 CFD / turbulence model / cut-cell discretization，所以直接构造完整连续方程 + RANS momentum residual 会涉及：

- turbulence closure
- viscosity model
- pressure normalization
- boundary condition
- non-dimensionalization
- cell-centered discrete operator

因此推荐：

```text
V1：纯监督 CFD loss
V2：+ force consistency
V3：+ continuity / weak physics
V4：+ solver-aware discrete residual
```

不要 V1 就做 full PINN。

---

# 32. 最有价值的 V3：solver-aware physics loss

既然你自己的 CFD solver 已经拥有：

```text
cut-cell geometry
face area
face normal
cell volume
neighbor connectivity
```

那么你最终可以直接定义：

\[
L_{disc}
=
\|R_{continuity}^{cut-cell}\|^2
+
\lambda_m
\|R_{momentum}^{cut-cell}\|^2
\]

这里直接使用你 solver 的离散算子结构，而不是再用一个完全连续的 PDE residual。

这是以后 DeepSDF + DoMINO 模型最值得做的技术亮点之一。

---

# 33. 训练模式建议

## 第一版

```text
AMP/BF16
DDP
batch size = 尽可能大
gradient accumulation = 必要时开启
```

DoMINO 属于大规模 geometry / point-cloud operator，不要只盯着传统 MLP 的 batch-size 思路。

PhysicsNeMo 也在持续对 external aerodynamics 和大型 point cloud 训练做性能优化。  

---

# 34. 推荐 sampling strategy

不要把所有 CFD volume cells 全部送入网络。

推荐每次 sample：

```text
near-wall        40%
near wake        25%
outer flow       20%
far field        15%
```

如果 volume cell 数量极大，可以按以下权重采样：

\[
p_i\propto
w_{wall}
+w_{wake}
+w_{grad}
+w_{cut}
\]

其中：

- `wall`：wall distance 权重
- `wake`：尾迹区域
- `grad`：CFD gradient / vorticity 权重
- `cut`：cut-cell fraction/complexity 权重

这样比全空间均匀随机 sampling 更适合汽车/外流场。

---

# 35. Wall region 的特殊采样

建议至少做三层：

```text
Layer 1:
0 < y+ <= 5

Layer 2:
5 < y+ <= 50

Layer 3:
y+ > 50
```

或者在 geometry-normalized surrogate 中直接按：

```text
wall distance / L_ref
```

分桶。

因为 surface pressure / shear 对 near-wall representation 极其敏感。

---

# 36. DeepSDF latent 是否应该对每个 query point 重新计算？

**绝对不要。**

正确流程：

```text
Geometry G
   │
   ▼
DeepSDF Encoder
   │
   ▼
 z_geo [B,Z]
   │
   ├── query 1
   ├── query 2
   ├── query 3
   └── ...
```

即：

\[
z_g
\]

对一个 geometry 全局共享。

否则会失去它作为 global shape coordinate 的意义，并产生巨大冗余计算。

---

# 37. z_geo 应该 cache

对于一个 geometry 有很多 BC：

```text
geometry A
 ├── BC1
 ├── BC2
 ├── BC3
 ├── BC4
 └── BC5
```

只需要：

```text
DeepSDF Encoder(A) → z_A
```

一次。

因此训练 dataset 可以提前缓存：

```text
geometry_id → z_geo
```

联合 fine-tuning 阶段再关闭 cache。

这样能够显著降低训练阶段 DeepSDF encoder 的重复计算。

---

# 38. 两种 training mode 都要保留

## Mode A：latent cache

```text
STL
 ↓
pretrained DeepSDF
 ↓
z_geo cache
 ↓
DoMINO
```

用于：

- 大规模 CFD dataset
- 快速实验
- hyperparameter search

## Mode B：end-to-end

```text
STL
 ↓
DeepSDF encoder
 ↓
z_geo
 ↓
DoMINO
```

用于：

- 最终模型
- geometry adaptation
- active learning
- shape optimization

---

# 39. 推荐配置文件结构

```text
project/
├── conf/
│   ├── train.yaml
│   ├── model/
│   │   └── domino_deepsdf.yaml
│   └── dataset/
│       └── cutcell_external_aero.yaml
│
├── src/
│   ├── train.py
│   ├── infer.py
│   ├── models/
│   │   ├── deepsdf_encoder.py
│   │   ├── geo_film.py
│   │   └── domino_deepsdf.py
│   ├── datapipe/
│   │   └── cutcell_domino_datapipe.py
│   ├── losses/
│   │   ├── field_loss.py
│   │   ├── force_loss.py
│   │   └── physics_loss.py
│   └── utils/
│       ├── normalize.py
│       ├── sdf.py
│       └── geometry.py
│
└── checkpoints/
```

---

# 40. `domino_deepsdf.yaml` 初始模板

```yaml
model:
  name: domino_deepsdf

  input_features: 3
  output_features_vol: 4
  output_features_surf: 4
  global_features: 4

  latent_dim: 256
  freeze_deepsdf: true

  encode_parameters: true
  use_surface_normals: true
  use_surface_area: true

  geometry_rep:
    geo_processor:
      cross_attention: true
      self_attention: true

  geometry_latent:
    encoder:
      hidden_dim: 256
      num_layers: 4
    conditioning: film
    condition_dim: 256

training:
  optimizer:
    name: adamw
    lr: 1e-4
    weight_decay: 1e-6

  amp: true

  loss:
    volume: 1.0
    surface: 1.0
    force: 0.1
    sdf: 0.05
    physics: 0.0
```

> 上面是项目级建议模板，不是 NVIDIA 官方 DoMINO YAML 的逐字段复制。实际参数名应以你 checkout 的 PhysicsNeMo 版本为准。

---

# 41. DoMINO 当前官方 recipe 的现实情况

当前 PhysicsNeMo 官方 unified external-aerodynamics recipe 已提供：

```text
Domino
GeoTransolver
Transolver
FLARE
GLOBE
```

以及对应的 surface / volume dataset 配置；但截至目前公开页面仍注明 DoMINO 部分为 draft / 不保证 unified recipe 直接 end-to-end 运行。因此开发 DeepSDF + DoMINO 时，建议以 PhysicsNeMo 的独立 DoMINO example / model implementation 为代码基础，而不是把 unified recipe 当作稳定 API。  

---

# 42. 推荐的实际开发顺序

不要直接开始联合训练。严格按照以下顺序：

```text
Phase 1
  ↓
STL → SDF
  ↓
验证几何一致性

Phase 2
  ↓
STL → DeepSDF z_geo
  ↓
验证 shape reconstruction

Phase 3
  ↓
z_geo + BC + query point
  ↓
简单 MLP baseline

Phase 4
  ↓
原始 DoMINO
  ↓
先不加 DeepSDF
  ↓
验证自有 cut-cell 数据能正确跑通

Phase 5
  ↓
DeepSDF z_geo + DoMINO FiLM
  ↓
冻结 DeepSDF

Phase 6
  ↓
force loss

Phase 7
  ↓
解冻 DeepSDF encoder
  ↓
joint fine-tuning

Phase 8
  ↓
cut-cell local features

Phase 9
  ↓
solver-aware discrete physics loss
```

这是最稳妥的研发路线。

---

# 43. 四组必须做的 baseline

为了证明 DeepSDF 真的有用，至少做：

### Baseline A

```text
DoMINO + SDF
```

### Baseline B

```text
DoMINO + SDF + geometry global embedding
```

### Baseline C

```text
DoMINO + SDF + DeepSDF latent concat
```

### Proposed

```text
DoMINO + SDF + DeepSDF latent FiLM
```

然后再加：

```text
+ cut-cell features
```

否则你很难说明 DeepSDF latent 的真正贡献。

---

# 44. 必须做的 geometry ablation

建议至少：

| 模型 | z_geo | SDF | local geometry | cut-cell |
|---|---:|---:|---:|---:|
| A | ✗ | ✓ | ✓ | ✗ |
| B | ✓ | ✗ | ✓ | ✗ |
| C | ✓ | ✓ | ✗ | ✗ |
| D | ✓ | ✓ | ✓ | ✗ |
| E | ✓ | ✓ | ✓ | ✓ |

关键观察：

- A → D：DeepSDF global geometry value
- D → E：cut-cell representation value
- C → D：DoMINO local geometry value

---

# 45. 必须做的泛化实验

## Geometry interpolation

训练：

```text
SUV A
SUV B
SUV C
```

测试：

```text
A/B 中间形状
```

观察：

\[
z_g
\]

是否表现出连续的 geometry manifold。

---

## Geometry extrapolation

测试明显不同于训练 geometry distribution 的外形。

观察：

```text
field error
Cd error
Cp distribution error
wall shear error
```

并配合 latent-distance / OOD score。

PhysicsNeMo 26.08 已在 external-aero active learning workflow 中加入基于 latent novelty 的 acquisition 思路，可以作为后续主动学习设计的参考。  

---

# 46. 新 geometry 推理流程

最终推理应该是：

```text
             New STL
                │
                ▼
       geometry normalization
                │
                ├───────────────┐
                ▼               ▼
          DeepSDF encoder    SDF query
                │               │
                ▼               │
             z_geo              │
                │               │
                └───────┬───────┘
                        ▼
                 DoMINO model
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      arbitrary volume x      surface points
             │                     │
             ▼                     ▼
        u,v,w,Cp               Cp,Cf
```

之后可以任意重新采样：

```text
Cartesian grid
Octree grid
cut-cell mesh
surface mesh
VTK sampling points
```

而无需重新求解 CFD。

---

# 47. 与你的 cut-cell solver 最终应该如何闭环

理想的数据闭环：

```text
                 ┌───────────────────────┐
                 │   CAD / geometry DB   │
                 └───────────┬───────────┘
                             │
                       DeepSDF Encoder
                             │
                           z_geo
                             │
                             ▼
┌─────────────────────────────────────────────────────┐
│                  DeepSDF + DoMINO                    │
│                                                     │
│ z_geo + SDF + local geometry + BC                   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
                 predicted CFD field
                        │
           ┌────────────┼─────────────┐
           ▼            ▼             ▼
         flow          Cp/Cf        Cd/Cl/Cm
           │            │             │
           └────────────┴─────────────┘
                        │
                        ▼
                 design / optimization
                        │
                        ▼
                       CAD'
                        │
                        ▼
                 DeepSDF latent z'
                        │
                        ▼
                    CFD surrogate
```

这时模型就不再只是：

> “一个 CFD field regression network”

而变成：

> **Geometry-conditioned physical surrogate / AI simulator。**

---

# 48. 与 shape optimization 结合

一旦有：

\[
G\rightarrow z_g
\]

并且：

\[
(z_g,BC)\rightarrow C_D
\]

就可以形成：

\[
G
\rightarrow z_g
\rightarrow \hat C_D
\]

然后求：

\[
\frac{\partial \hat C_D}{\partial z_g}
\]

以及：

\[
\frac{\partial z_g}{\partial G}
\]

最终：

\[
\frac{\partial C_D}{\partial G}
=
\frac{\partial C_D}{\partial z_g}
\frac{\partial z_g}{\partial G}
\]

这为下一步做：

```text
DeepSDF / DeepMesh geometry optimization
+
DoMINO aerodynamic surrogate
```

提供了非常自然的 differentiable path。

---

# 49. 为什么这里暂时不优先选 FNO

对于你的问题：

```text
different geometry
+ irregular cut-cell
+ arbitrary query points
+ surface + volume
```

FNO 的固定 Cartesian tensor grid 约束比较明显。

而 DoMINO 本身就是针对：

- local multi-scale point-cloud geometry
- surface + volume
- SDF
- large-scale external aerodynamics

设计的。

因此当前阶段推荐：

```text
DeepSDF + DoMINO
```

而不是：

```text
DeepSDF + FNO
```

---

# 50. 为什么 DeepONet 可以作为第二个 baseline

如果做一个更简洁的 operator baseline，可以使用：

\[
Branch(z_g,BC)
\]

和：

\[
Trunk(x,y,z,SDF)
\]

最终：

\[
\hat y(x)
=
Branch(z_g,BC)\cdot Trunk(x,SDF)
\]

这个模型非常适合回答一个问题：

> **DeepSDF latent 本身是不是足够成为一个 geometry coordinate？**

但是对于当前 industrial 3D external aero，DoMINO 的 local multi-scale geometry path 更接近你的 cut-cell geometry structure，所以建议 DeepONet 作为对照，而非第一主模型。

---

# 51. 下一代：DeepSDF + AeroJEPA

PhysicsNeMo 26.08 已加入实验性的 AeroJEPA，用于 3D aerodynamic field joint-embedding / field prediction；其思路更接近：

\[
geometry + operating\ conditions
\rightarrow
flow\ latent
\rightarrow
continuous\ field
\]

所以后续可以把你的系统升级成：

```text
                Geometry
                   │
                DeepSDF
                   │
                 z_geo
                   │
                   ▼
              AeroJEPA
                   │
                z_flow
                   │
                   ▼
          continuous implicit decoder
                   │
                   ▼
               y(x)
```

这一步比 DoMINO + DeepSDF 更进一步：

```text
geometry latent
       ↓
physics latent
       ↓
continuous field
```

非常适合作为未来 physical-world AI simulation platform 的架构。

但推荐先把 DoMINO 版本做成稳定基线，再转 AeroJEPA。

---

# 52. 推荐的最终研发里程碑

## M1

```text
DeepSDF geometry reconstruction
```

验收：

- SDF reconstruction error
- surface reconstruction
- latent interpolation

## M2

```text
Vanilla DoMINO + custom cut-cell data
```

验收：

- volume field
- surface Cp
- Cd/Cl/Cm

## M3

```text
DeepSDF + DoMINO FiLM
```

验收：

- new BC
- new geometry
- OOD geometry

## M4

```text
+ cut-cell local features
```

验收：

- near-wall error
- wake error
- force error

## M5

```text
joint fine-tuning
```

验收：

- lower geometry-conditioned error
- latent consistency
- differentiability

## M6

```text
shape optimization
```

验收：

```text
CAD → latent → CFD → objective → gradient → new CAD
```

## M7

```text
DeepSDF + AeroJEPA / continuous implicit simulator
```

最终形成 geometry-conditioned physical AI simulator。

---

# 53. 我最推荐的最终网络

如果现在开始实现，我会采用下面这一版作为正式主线：

```text
                         ┌────────────────┐
                         │   CAD / STL     │
                         └───────┬────────┘
                                 │
                ┌────────────────┴─────────────────┐
                │                                  │
                ▼                                  ▼
        DeepSDF Shape Encoder              Numerical SDF pipeline
                │                                  │
                ▼                                  ▼
             z_geo                       sdf_grid / sdf(x)
                │                                  │
                └──────────────┬───────────────────┘
                               ▼
                    DoMINO GeometryRep
                               │
                               ▼
                         combined geometry
                               │
                        Geo-FiLM(z_geo)
                               │
                               ▼
                   Multi-scale local encoding
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
        query x             SDF(x)              cut-cell
                                                   features
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
               BC → ParameterModel/FourierMLP
                               │
                               ▼
                    DoMINO SolutionCalculator
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
                volume field          surface field
              u,v,w,Cp,...             Cp,Cf,...
                    │                     │
                    └──────────┬──────────┘
                               ▼
                        force integration
                               │
                         Cd / Cl / Cm
```

---

# 54. 最关键的设计决策总结

### 决策 1：DeepSDF 不替代 DoMINO SDF

保持：

```text
Numerical SDF → local distance / geometry
DeepSDF latent → global shape
```

### 决策 2：z_geo 第一版使用 FiLM

而不是直接 concat 到最终 MLP。

### 决策 3：DeepSDF 先冻结

先验证：

```text
latent quality
```

再联合优化。

### 决策 4：volume + surface 联合训练

因为：

```text
surface Cp/Cf
      ↓
force
      ↓
Cd/Cl/Cm
```

能形成很强的 global supervision。

### 决策 5：保留你的 cut-cell metadata

不要把这些信息扔掉。

你真正有价值的特色不是简单复制 DrivAerML，而是：

```text
DeepSDF global CAD geometry
+
DoMINO multi-scale geometry
+
cut-cell discrete geometry
```

### 决策 6：先做监督 operator，再加 physics

推荐：

```text
supervised CFD
→ force consistency
→ weak physics
→ solver-aware discrete residual
```

---

# 55. 最终推荐的第一版实验配置

如果要直接开始 coding，我建议第一版固定成：

```text
Geometry latent:
    DeepSDF encoder
    latent_dim = 256

Geometry representation:
    DoMINO GeometryRep
    numerical SDF retained

Latent conditioning:
    FiLM
    separate surface / volume heads

Operating condition:
    [Re, Ma, alpha, beta]
    DoMINO ParameterModel

Volume target:
    [u/Uinf, v/Uinf, w/Uinf, Cp]

Surface target:
    [Cp, Cfx, Cfy, Cfz]

Additional cut-cell feature:
    Vfluid/Vcell
    wall distance
    wall normal

Training:
    Stage 1: DeepSDF pretrain
    Stage 2: frozen DeepSDF + DoMINO
    Stage 3: joint fine-tuning

Loss:
    volume field
    + surface field
    + Cd/Cl/Cm
    + small SDF regularization

Evaluation:
    same-geometry / new-BC
    new-geometry / seen-BC
    new-geometry / new-BC
    OOD geometry
```

这套配置已经足够作为第一个真正可执行的 research prototype。

---

# 56. 后续代码实现建议

实际开始编程时，建议按以下三个核心文件依次实现：

```text
01  src/models/deepsdf_encoder.py
    └── STL/surface point cloud → z_geo

02  src/models/domino_deepsdf.py
    └── NVIDIA DoMINO fork
    └── GeoFiLM(z_geo)

03  src/datapipe/cutcell_domino_datapipe.py
    └── cut-cell solver output
        → DoMINO data_dict
```

然后再实现：

```text
04  force_loss.py
05  physics_loss.py
06  inference.py
07  visualization.py
```

尤其建议先把 `data_dict` 完全对齐 DoMINO 当前接口，再加入 DeepSDF；不要同时修改 datapipe、模型和 loss 三个方向，否则出现误差时很难定位。

---

# 57. NVIDIA 官方参考资料

1. **PhysicsNeMo DoMINO API / source**  
   当前 DoMINO 的模型定义、输入 tensor keys、GeometryRep、ParameterModel 和 forward 数据流。
   
2. **PhysicsNeMo DoMINO External Aerodynamics example**  
   DoMINO 针对外流场 surface + volume 预测的官方 recipe。
   
3. **PhysicsNeMo Unified External Aerodynamics Recipe**  
   当前统一外流场训练框架、dataset/model YAML、TensorDict/mesh 数据接口。
   
4. **PhysicsNeMo 26.08 Release Notes**  
   当前版本新增 AeroJEPA、xDeepONet、batched radius search、Mesh SDF query 等能力。
   
5. **PhysicsNeMo-Mesh Spatial Queries**  
   当前 signed-distance query API，可用于统一 SDF 预处理与校验。

---

## 参考链接

- https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/models/operators.html
- https://docs.nvidia.com/physicsnemo/latest/_modules/physicsnemo/models/domino/model.html
- https://docs.nvidia.com/physicsnemo/26.08/physicsnemo/examples/cfd/external_aerodynamics/domino/README.html
- https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/unified_external_aero_recipe/README.html
- https://docs.nvidia.com/physicsnemo/26.08/physicsnemo/api/mesh/spatial.html
- https://docs.nvidia.com/physicsnemo/26.08/release-notes/index.html
- https://github.com/NVIDIA/physicsnemo

---

# 58. 一句话结论

\[
\boxed{
\textbf{STL}
\rightarrow
\textbf{DeepSDF }z_{geo}
\rightarrow
\textbf{DoMINO GeometryRep + Geo-FiLM}
\rightarrow
\textbf{local geometry + SDF + cut-cell features + BC}
\rightarrow
\textbf{surface/volume CFD}
}
\]

这是目前最适合你已有 `octree Cartesian + cut-cell CFD solver` 的第一版实现路线。

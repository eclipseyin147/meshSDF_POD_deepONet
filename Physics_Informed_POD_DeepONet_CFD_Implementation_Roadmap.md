# Physics-Informed POD + DeepONet for 3D CFD：基于 DeepSDF + POD + Octree/Cut-Cell 的具体实现路线

## 1. 项目目标

当前已经具备：

- DeepSDF 几何表示模块
- POD 分解/重构模块
- 3D CFD 求解器，网格具有 **octree-based Cartesian + boundary cut-cell** 特征

目标是构建一个面向三维外流 CFD 的实时/准实时 surrogate：

\[
\boxed{
\text{Geometry} + \text{Boundary Conditions}
\rightarrow
\text{CFD Flow Field}
}
\]

同时希望输出：

- 体场：\(u,v,w,p\)，后续可扩展到 \(T,k,\omega,\nu_t\)
- 壁面压力：\(C_p\)
- 壁面剪切：\(C_f,\tau_w\)
- 气动力积分：\(C_D,C_L,C_M\)

推荐总体方案：

\[
\boxed{
\text{DeepSDF}
+
\text{POD}
+
\text{DeepONet}
+
\text{Physics-informed loss}
}
\]

更具体地说：

> **DeepSDF-conditioned Physics-informed POD-DeepONet**

---

# 2. 最终总体架构

```text
                         Geometry G
                              │
                              ▼
                         Your DeepSDF
                              │
                              ▼
                           latent z_g
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             │ BC / operating conditions       │
             │ U∞, AoA, Re, Ma, T∞ ...        │
             │                                 │
             ▼                                 │
                    Branch Network             │
             B(z_g, μ_BC) → a₁...a_r         │
                              │                │
                              ▼                │
                       POD coefficients        │
                              │                │
             ┌────────────────┘                │
             │                                 │
       query x, SDF d(x), normal ∇d, h        │
             │                                 │
             ▼                                 │
         Trunk Network                         │
             │                                 │
             ▼                                 │
     POD / DeepONet operator                   │
             │                                 │
             ▼                                 │
          q(x) = u,v,w,p,T,...                 │
             │                                 │
       ┌─────┴─────────────────┐               │
       │                       │               │
       ▼                       ▼               ▼
 CFD supervised loss     Physics loss     Surface Head
                         continuity       Cp/Cf/τw
                         momentum
                         BC
                         wall
```

核心表达式：

\[
q(x;G,\mu)
=
\bar q(x)+
\sum_{i=1}^{r}
a_i(z_g,\mu)\,
\phi_i(x,d,\nabla d,h)
\]

其中：

\[
z_g = DeepSDF(G)
\]

\[
a=B_\theta(z_g,\mu)
\]

这里：

- \(G\)：几何
- \(z_g\)：DeepSDF geometry latent code
- \(\mu\)：边界条件/工况参数
- \(a_i\)：POD coefficients
- \(x\)：空间位置
- \(d(x)\)：SDF value
- \(\nabla d(x)\)：SDF gradient
- \(h(x)\)：局部 octree cell size
- \(\phi_i\)：POD basis 或 geometry-aware DeepONet trunk basis

---

# 3. GitHub 开源项目如何组合

不要试图找一个仓库把全部功能一次解决。最合理的是把已有项目按功能拆开使用。

## 3.1 mPOD-DeepONet

GitHub：

**Chieh997/mPOD-DeepONet**

地址：

https://github.com/Chieh997/mPOD-DeepONet

主要用途：

- POD-DeepONet
- cPOD-DeepONet
- mPOD-DeepONet
- 多输出
- 多物理场
- DeepONet Branch/Trunk 结构

对当前项目价值：

> **最高**

尤其适合作为 POD + DeepONet 部分的代码参考。

该项目中值得重点关注：

- `ChannelPodONet`
- `MPodONet`
- `MultiTrunkDeepONet`
- `ShareTrunkDeepONet`

如果以后预测：

\[
u,v,w,p,T,k,\omega
\]

则可以参考它对多输出/多物理变量的组织方式。

### 推荐用法

第一版先采用类似 cPOD 的结构：

\[
u=\sum_i a_i^u\phi_i^u
\]

\[
v=\sum_i a_i^v\phi_i^v
\]

\[
w=\sum_i a_i^w\phi_i^w
\]

\[
p=\sum_i a_i^p\phi_i^p
\]

不要一开始就强行 shared POD。

---

## 3.2 NVIDIA PhysicsNeMo

GitHub：

https://github.com/NVIDIA/physicsnemo

重点参考：

`examples/cfd/darcy_physics_informed/darcy_physics_informed_deeponet.py`

它的价值主要不是 POD，而是：

- Physics-informed DeepONet
- autodiff
- PDE residual
- data loss + physics loss
- physics weighting
- coordinate `requires_grad=True`
- 物理约束训练框架

其思想可以直接迁移到：

\[
L=
L_{data}
+
\lambda_{physics}L_{PDE}
\]

### 对当前工程的直接借鉴

```text
DeepONet forward
      ↓
coordinates.requires_grad_(True)
      ↓
autodiff
      ↓
PDE residual
      ↓
physics loss
      ↓
data loss + λ * physics loss
```

---

## 3.3 PIOperator

GitHub：

https://github.com/pidneuralode/PIOperator

它的优势是可以直接比较：

- DeepONet
- POD-DeepONet
- PI-DeepONet
- FNO
- PI-FNO

适合：

> 快速验证不同 operator architecture 的优劣。

第一版不一定直接依赖它作为最终框架，但建议阅读其：

- POD-DeepONet
- PI-DeepONet
- training loop
- physics loss

实现。

---

## 3.4 NeuralOperator

GitHub：

https://github.com/NeuralOperator/neuraloperator

主要用来做：

- FNO baseline
- GNO 等 neural operator 对照实验

当前项目中：

> **建议把 FNO 作为 benchmark，而不是第一主线实现。**

原因：

- FNO 更喜欢固定规则网格
- 你的几何会变化
- octree/cut-cell topology 不固定
- arbitrary geometry 不天然适合标准 FFT-based FNO

---

# 4. 为什么你的项目优先选择 POD-DeepONet，而不是传统 PINN

你的已有条件是：

```text
已有：
DeepSDF
POD
高保真 CFD solver
大量 CFD snapshots
```

因此不适合让传统 PINN 直接承担完整 CFD 求解。

传统 PINN：

\[
BC + Geometry
\rightarrow
NN
\rightarrow
PDE residual optimization
\]

通常意味着在线仍需要大量优化迭代。

而目标是实时预测，所以应改成：

\[
Geometry+BC
\rightarrow
latent/operator
\rightarrow
flow field
\]

即：

```text
Offline training:
CFD data + Physics

Online inference:
仅前向传播
```

在线不再包含：

- Newton
- CG
- AMG
- multigrid
- MPI CFD iteration

这才真正有机会实现实时/准实时预测。

---

# 5. 第一阶段：直接利用现有 POD

假设已有：

```python
mean_field
pod_basis
pod_coeff
```

可以组织成：

```text
Case i
├── geometry
│    └── DeepSDF latent z_i
├── BC
│    ├── U∞
│    ├── AoA
│    ├── Re
│    ├── Ma
│    └── ...
├── POD coefficients
│    └── a_i
└── CFD snapshots
```

第一步只做：

\[
(z_g,\mu)\rightarrow a
\]

即：

```python
a_pred = branch_net(
    torch.cat([z_g, bc], dim=-1)
)
```

Loss：

\[
L_{POD}
=
\frac1r
\sum_i
\left(a_i^{pred}-a_i^{CFD}\right)^2
\]

### 第一阶段目标

验证：

> DeepSDF latent 是否已经携带足够的 geometry information，使得 POD coefficients 可以由 geometry + BC 预测出来。

推荐 POD rank：

- 64
- 128
- 再视重构误差决定是否提高到 256

第一版不要直接用上千个 mode。

---

# 6. Branch Network 设计

推荐：

```text
z_g
 │
 ├── FC
 ├── SiLU
 ├── FC
 └── latent feature
              │
BC ───────────┤
              ▼
          concatenate
              │
            MLP
              │
              ▼
       a₁ ... a_r
```

输入：

\[
[z_g,\mu]
\]

例如：

```text
z_g = 128
U∞  = 1
AoA = 1
Re  = 1
Ma  = 1
T∞  = 1
```

总输入维数：

\[
128+5=133
\]

输出：

\[
r=64/128
\]

---

# 7. 第二阶段：DeepONet

标准 DeepONet：

\[
G(u)(x)
=
\sum_{i=1}^{r}
B_i(u)T_i(x)
\]

对于 CFD：

\[
q(x;G,\mu)
=
\sum_{i=1}^{r}
B_i(z_g,\mu)
T_i(x)
\]

其中：

### Branch

\[
B(z_g,\mu)
\]

代表：

> geometry + operating condition

### Trunk

\[
T(x)
\]

代表：

> spatial coordinate

---

# 8. 你的 Trunk 不应该只有 xyz

对于普通问题：

\[
T=T(x,y,z)
\]

但对于你的：

> octree-based Cartesian + cut-cell CFD

建议：

\[
T=
T(x,y,z,d,\nabla d,h)
\]

即：

```text
x
y
z
SDF
∂SDF/∂x
∂SDF/∂y
∂SDF/∂z
cell size h
```

这一步非常关键。

---

# 9. DeepSDF 在 Physics-informed CFD 中真正的价值

DeepSDF 不仅可以告诉模型 geometry latent：

\[
z_g
\]

它还可以在任意空间点计算：

\[
d(x)
\]

以及：

\[
n(x)
=
\frac{\nabla d(x)}
{|\nabla d(x)|}
\]

于是网络实际上学习：

\[
q=
F(z_g,\mu,x,d,\nabla d,h)
\]

因此模型可以识别：

- 是否接近壁面
- 到壁面的距离
- 壁面法向方向
- 局部几何变化
- 当前 octree cell 的空间尺度

相比：

\[
q=F(z_g,\mu,x)
\]

信息量明显更丰富。

---

# 10. POD 和 DeepONet 的两种组合方式

## 10.1 方案 A：POD coefficient prediction

最推荐作为第一版：

\[
(z_g,\mu)
\rightarrow
a
\]

然后：

\[
q(x)=\bar q+\Phi(x)a
\]

即：

```text
DeepSDF
   ↓
z_g + BC
   ↓
Branch
   ↓
POD coefficients
   ↓
existing POD reconstruction
   ↓
CFD field
```

优点：

- 极快
- 稳定
- 充分复用现有 POD 模块
- 训练简单
- 适合快速做 baseline

---

## 10.2 方案 B：Geometry-conditioned DeepONet

进一步升级为：

\[
q(x)
=
\sum_i
B_i(z_g,\mu)
T_i(x,d,\nabla d,h)
\]

此时 trunk 不再只是固定坐标映射，而是：

> geometry-aware basis

可以看成：

\[
\phi_i(x,G)
=
T_i(x,d_G(x),\nabla d_G(x),h)
\]

这更适合：

- 不同几何
- 不同拓扑
- arbitrary geometry
- cut-cell

---

# 11. 为什么不要一开始就直接做复杂的 Geometry-conditioned POD

第一阶段建议：

\[
(z_g,\mu)\rightarrow a
\]

只验证 POD coefficient prediction。

第二阶段：

\[
(z_g,\mu,x,d,\nabla d,h)\rightarrow q(x)
\]

第三阶段才：

\[
L_{data}+L_{physics}
\]

原因：

如果三者同时优化：

```text
DeepSDF latent
+
POD representation
+
DeepONet
+
physics residual
```

训练失败时很难判断问题来自哪里。

---

# 12. Physics-informed Loss

对于 incompressible steady flow：

## 12.1 Continuity

\[
\nabla\cdot\mathbf u=0
\]

即：

\[
R_c=
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
+
\frac{\partial w}{\partial z}
\]

loss：

\[
L_{continuity}
=
\left|R_c\right|^2
\]

---

## 12.2 Momentum

稳态 incompressible NS：

\[
\rho(\mathbf u\cdot\nabla)\mathbf u
=
-\nabla p
+
\mu\nabla^2\mathbf u
\]

定义：

\[
R_m=
\rho(\mathbf u\cdot\nabla)\mathbf u
+
\nabla p
-
\mu\nabla^2\mathbf u
\]

然后：

\[
L_{momentum}
=
\|R_m\|^2
\]

---

## 12.3 Physics Loss

第一版：

\[
L_{phys}
=
L_{continuity}
+
\lambda_mL_{momentum}
\]

最终：

\[
L=
L_{POD}
+
\lambda_fL_{field}
+
\lambda_pL_{phys}
+
\lambda_bL_{BC}
\]

---

# 13. 为什么第一版不要马上加入 RANS k-ω SST

如果你的实际 CFD 是 RANS + k-ω SST，不建议第一天就同时加入：

- continuity
- momentum
- k equation
- omega equation
- turbulent viscosity
- wall function

因为 PDE system 会变得很 stiff。

推荐分阶段：

### V1

```text
u v w p
+
continuity
+
momentum
```

### V2

```text
u v w p k omega
+
continuity
+
momentum
+
k equation
+
omega equation
```

### V3

再逐步加入：

- \(\nu_t\)
- wall function
- turbulence boundary condition
- near-wall treatment

---

# 14. Wall boundary condition

壁面：

\[
u|_\Gamma=0
\]

因此：

\[
L_{wall}
=
\|u(x_w)\|^2
\]

由于已有 DeepSDF，可以很方便找到：

\[
d(x_w)\approx0
\]

的 surface points。

于是：

```text
DeepSDF
   ↓
d(x)=0 surface points
   ↓
surface BC loss
```

---

# 15. Solid region 的处理

对于你的 cut-cell CFD，query points 可能处于：

```text
solid
fluid
cut-cell
```

必须避免在 solid interior 上计算流体 PDE residual。

可以引入：

\[
\chi(x)
=
\begin{cases}
1,& fluid\\
0,& solid
\end{cases}
\]

训练时：

```python
physics_points = points[fluid_mask]
```

只对流体区域计算：

\[
L_{physics}
\]

---

# 16. Local cell size h 必须加入

由于你的网格是 octree：

\[
h=h(x)
\]

可能发生：

```text
far field: coarse
near body: fine
wake: locally refined
cut-cell: special geometry
```

所以建议 Trunk：

\[
(x,d,\nabla d,h)
\]

其中 \(h\) 可做归一化：

\[
h^*=\frac{h}{L_{ref}}
\]

这样网络可以感知 local resolution / spatial scale。

---

# 17. Surface pressure / shear 建议独立 Head

不要强行让一个 volume network 同时负责所有壁面量。

推荐：

```text
                     shared latent
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       Volume DeepONet          Surface Head
              │                       │
          u,v,w,p                    Cp,Cf
```

Surface head：

\[
[C_p,C_f,\tau_w]
=
H(z_g,\mu,x_s,n_s)
\]

这里使用：

- \(z_g\)
- BC
- surface point \(x_s\)
- wall normal \(n_s\)

这样更适合壁面物理量。

---

# 18. Drag / Lift / Moment

有 surface \(C_p\) 和 \(C_f\) 后，可以积分得到：

\[
C_D
=
\int_S
(C_p n_x+C_f)_dS
\]

概念上：

```text
surface network
      ↓
Cp / Cf
      ↓
surface integration
      ↓
Cd / Cl / Cm
```

可以增加一个 global aerodynamic loss：

\[
L_{aero}
=
\|C_D^{pred}-C_D^{CFD}\|^2
+
\|C_L^{pred}-C_L^{CFD}\|^2
\]

这是非常值得加入的一项，因为最终工业 CFD surrogate 往往最关心：

- Drag
- Lift
- Moment

而不只是单元级 MSE。

---

# 19. CFD Data Loss

预测：

\[
q^{pred}(x)
\]

CFD：

\[
q^{CFD}(x)
\]

不要直接混合原始量纲做 MSE：

```text
MSE(u,v,w,p,T)
```

建议 nondimensionalization。

速度：

\[
u^*=\frac{u}{U_\infty}
\]

压力：

\[
p^*
=
\frac{p-p_\infty}
{\frac12\rho U_\infty^2}
\]

这样可以减少不同变量量纲造成的 loss imbalance。

---

# 20. Collocation Point Sampling

physics points 不需要等于全部 CFD cells。

推荐：

```text
CFD points
   └── supervised data

collocation points
   └── PDE physics
```

### 推荐初始比例

```text
30% near-wall
30% wake
20% high-gradient
20% far-field
```

其中 near-wall 可按：

\[
|d(x)| < C h(x)
\]

构造，例如：

\[
|d|<10h
\]

具体常数后续通过实验调整。

---

# 21. High-gradient sampling

重点关注：

\[
|\nabla p|
\]

\[
|\nabla u|
\]

较大的区域。

因为外流 CFD 中这些位置往往对应：

- boundary layer
- separation
- shock
- wake
- shear layer

如果只均匀随机 sampling，网络很容易出现：

> far-field 很准，但 near-wall / wake 很差。

---

# 22. 推荐四阶段训练流程

## Stage 1：DeepSDF → POD coefficients

固定 DeepSDF：

```python
for p in deep_sdf.parameters():
    p.requires_grad = False
```

训练：

\[
[z_g,\mu]
\rightarrow
a
\]

Loss：

\[
L=L_{POD}
\]

目标：

检查：

- POD coefficient MSE
- coefficient \(R^2\)
- POD reconstruction error

---

## Stage 2：Field DeepONet

加入：

\[
x,d,\nabla d,h
\]

训练：

\[
(z_g,\mu,x,d,\nabla d,h)
\rightarrow
q(x)
\]

Loss：

\[
L=L_{POD}+\lambda_fL_{field}
\]

这一步仍然暂时不加入 PDE loss。

---

## Stage 3：Physics-informed fine-tuning

加入：

- continuity
- momentum
- wall BC
- far-field BC

最终：

\[
L=
L_{POD}
+
\lambda_fL_{field}
+
\lambda_pL_{phys}
+
\lambda_bL_{BC}
\]

---

## Stage 4：Surface prediction

加入：

\[
C_p,\quad C_f,\quad \tau_w
\]

并进一步加入：

\[
C_D,C_L,C_M
\]

最终形成：

```text
Volume:
u,v,w,p,T,k,omega

Surface:
Cp,Cf,tau_w

Integrated:
Cd,Cl,Cm
```

---

# 23. Physics loss 权重

不要一开始把 physics loss 权重设得很高。

推荐：

\[
L=
L_{data}
+
\lambda_{phys}L_{phys}
\]

训练初期：

\[
\lambda_{phys}\approx0
\]

之后逐渐提高。

一个可作为实验起点的 schedule：

```text
0–20% epochs
λphys = 0

20–50%
λphys = 0.01

50–80%
λphys = 0.05

80–100%
λphys = 0.1
```

注意：

> 这些数值只是起始实验范围，实际应根据 nondimensionalized residual 的量级进行动态调整。

更进一步可以采用：

- gradient balancing
- uncertainty weighting
- adaptive residual weighting

---

# 24. 一个关键数学问题：POD 本身不一定适合作为完整 physics operator

如果简单使用：

\[
q=\bar q+\Phi a
\]

并且 \(a\) 只依赖：

\[
(z_g,\mu)
\]

那么：

\[
a
\]

与空间 \(x\) 无关。

这对于构造完整 spatial PDE residual 时并不天然。

因此必须区分：

### 第一阶段

POD 是：

> latent compression / state reduction

即：

\[
(z_g,\mu)\rightarrow a
\]

### 第二阶段

DeepONet 是：

> spatial operator

即：

\[
(z_g,\mu,x,d,\nabla d,h)\rightarrow q(x)
\]

所以：

> **POD 主要负责降低状态维数，DeepONet 负责 spatial field representation。**

Physics loss 最适合放在第二阶段。

---

# 25. 推荐的最终数学形式

最终建议使用：

\[
\boxed{
q(x;G,\mu)
=
\bar q(x)
+
\sum_{k=1}^{r}
a_k(z_G,\mu)
\,
\phi_k(x,d_G(x),\nabla d_G(x),h)
}
\]

其中：

\[
\boxed{
z_G=DeepSDF(G)
}
\]

\[
\boxed{
a=B_\theta(z_G,\mu)
}
\]

\[
\boxed{
\phi_k=
T_{\theta,k}(x,d,\nabla d,h)
}
\]

最终 Loss：

\[
\boxed{
L=
L_{POD}
+\lambda_fL_{field}
+\lambda_cL_{continuity}
+\lambda_mL_{momentum}
+\lambda_bL_{BC}
+\lambda_wL_{wall}
}
\]

---

# 26. 推荐的工程目录

```text
cfd_ai/
│
├── geometry/
│   └── deepsdf.py
│
├── reduction/
│   ├── pod.py
│   ├── basis.py
│   └── normalization.py
│
├── operator/
│   ├── branch.py
│   ├── trunk.py
│   ├── pod_deeponet.py
│   └── surface_head.py
│
├── physics/
│   ├── continuity.py
│   ├── navier_stokes.py
│   ├── boundary.py
│   └── turbulence.py
│
├── dataset/
│   ├── cfd_dataset.py
│   ├── sampler.py
│   └── collocation.py
│
├── losses/
│   ├── pod_loss.py
│   ├── field_loss.py
│   ├── physics_loss.py
│   └── surface_loss.py
│
├── train/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── train_stage3.py
│   └── train_stage4.py
│
└── inference/
    └── realtime.py
```

---

# 27. 推荐的 Tensor / Dataset 组织

每个 CFD case：

```text
case_i
├── geometry
├── z_g                  [latent_dim]
├── bc                   [bc_dim]
├── pod_coeff            [n_outputs, r]
├── surface
│   ├── x_s              [Ns, 3]
│   ├── normal            [Ns, 3]
│   ├── Cp                [Ns, 1]
│   └── Cf                [Ns, 1]
└── volume
    ├── x                 [Nv, 3]
    ├── sdf               [Nv, 1]
    ├── grad_sdf          [Nv, 3]
    ├── h                 [Nv, 1]
    ├── fluid_mask        [Nv, 1]
    └── q                 [Nv, n_outputs]
```

推荐的网络输入：

```text
Branch:
[z_g, bc]

Trunk:
[x, sdf, grad_sdf, h]

Surface Head:
[z_g, bc, x_s, normal]
```

---

# 28. DeepSDF 模块的冻结策略

推荐：

### Stage 1–3 前期

```text
DeepSDF = frozen
```

原因：

它已经负责：

> geometry representation

先避免 CFD surrogate 破坏已经训练好的 geometry latent space。

### 后期

可以进行：

```text
end-to-end fine tuning
```

即：

\[
G
\rightarrow
DeepSDF
\rightarrow
z_g
\rightarrow
CFD surrogate
\]

同时：

\[
\frac{\partial L_{CFD}}{\partial \Theta_{DeepSDF}}
\]

允许 CFD loss 反传回 geometry representation。

这会形成更强的：

> physics-aware geometry latent space

---

# 29. 在线推理阶段

训练结束后：

```text
Geometry
   │
   ▼
DeepSDF
   │
   ▼
z_g
   │
   ├──────── BC
   │
   ▼
Branch
   │
   ▼
POD / latent coefficients
   │
   ▼
DeepONet
   │
   ├── volume field
   │
   └── surface quantities
```

在线过程只有：

- encoder
- MLP
- matrix multiplication
- query evaluation

不会再调用：

- CFD nonlinear iterations
- linear solver
- AMG
- CG/GMRES
- MPI

因此特别适合：

- design space exploration
- optimization
- uncertainty quantification
- digital twin
- real-time aerodynamic estimation
- interactive geometry design

---

# 30. 推荐的具体开发顺序

按照你当前已经具备的模块，建议按以下顺序实施。

## Step 1

保持现有 DeepSDF 不动。

确认：

\[
G\rightarrow z_g
\]

稳定。

---

## Step 2

保持现有 POD 不动。

确认：

\[
snapshot
\rightarrow
a
\rightarrow
reconstruction
\]

误差已经达到要求。

---

## Step 3

实现 BranchNet：

\[
[z_g,\mu]
\rightarrow
a
\]

先只做 \(L_{POD}\)。

---

## Step 4

验证：

```text
POD coefficient error
POD reconstruction error
```

---

## Step 5

实现基础 DeepONet：

\[
[z_g,\mu,x]
\rightarrow
q(x)
\]

---

## Step 6

加入：

\[
d,\nabla d,h
\]

变成：

\[
[z_g,\mu,x,d,\nabla d,h]
\rightarrow
q(x)
\]

---

## Step 7

加入：

\[
L_{field}
\]

完成监督式 DeepONet。

---

## Step 8

加入：

\[
L_{continuity}
\]

---

## Step 9

加入：

\[
L_{momentum}
\]

---

## Step 10

加入：

\[
L_{wall}
+
L_{farfield}
\]

---

## Step 11

增加 surface head：

\[
Cp,Cf,\tau_w
\]

---

## Step 12

增加 global aerodynamic loss：

\[
C_D,C_L,C_M
\]

---

## Step 13

最后才尝试：

- RANS \(k-\omega\)
- end-to-end DeepSDF fine-tuning
- active learning
- uncertainty estimation
- multi-fidelity CFD

---

# 31. 推荐的第一版最小模型

不要一上来实现完整工业 CFD surrogate。

建议 MVP：

```text
Geometry:
DeepSDF latent

BC:
U∞
AoA
Re

Volume:
u,v,w,p

Reduction:
POD r = 64 or 128

Branch:
[z_g, U∞, AoA, Re]
       ↓
POD coefficients

Trunk:
[x,y,z,SDF,nx,ny,nz,h]

Physics:
continuity
momentum

Surface:
Cp
```

最终目标：

\[
\boxed{
(G,U_\infty,\alpha,Re)
\rightarrow
(u,v,w,p,C_p)
}
\]

这是最容易验证整条技术链路是否正确的版本。

---

# 32. 建议的 Benchmark

至少比较下面几种：

### Baseline 1

\[
BC\rightarrow MLP\rightarrow POD
\]

### Baseline 2

\[
DeepSDF+BC\rightarrow MLP\rightarrow POD
\]

### Baseline 3

\[
DeepSDF+BC\rightarrow DeepONet
\]

### Baseline 4

\[
DeepSDF+BC\rightarrow POD\text{-}DeepONet
\]

### Baseline 5

\[
DeepSDF+BC\rightarrow PI\text{-}POD\text{-}DeepONet
\]

### Baseline 6

FNO / PI-FNO

这样最终可以非常清楚地回答：

> DeepSDF 是否有效？

> POD 是否有效？

> DeepONet 是否比普通 MLP 好？

> physics-informed 是否真的降低 generalization error？

---

# 33. 推荐评价指标

不要只看 global MSE。

至少记录：

## Volume

\[
L_2(u)
\]

\[
L_2(v)
\]

\[
L_2(w)
\]

\[
L_2(p)
\]

---

## Surface

\[
L_2(C_p)
\]

\[
L_2(C_f)
\]

---

## Integral quantities

\[
|C_D^{pred}-C_D^{CFD}|
\]

\[
|C_L^{pred}-C_L^{CFD}|
\]

\[
|C_M^{pred}-C_M^{CFD}|
\]

---

## Physics

\[
\|\nabla\cdot u\|
\]

\[
\|R_{momentum}\|
\]

---

## Inference speed

最终必须记录：

\[
T_{CFD}
\]

vs.

\[
T_{AI}
\]

以及：

\[
Speedup=
\frac{T_{CFD}}
{T_{AI}}
\]

这对于你的实时 CFD 平台最终会是非常重要的指标。

---

# 34. 最终技术路线总结

完整路线可以浓缩为：

```text
                     Geometry
                        │
                        ▼
                    DeepSDF
                        │
                        ▼
                       z_g
                        │
                  ┌─────┴─────┐
                  │           │
                 BC          SDF
                  │           │
                  ▼           │
               Branch         │
                  │            │
                  ▼            │
            POD coefficients   │
                  │            │
                  └─────┬──────┘
                        ▼
                  DeepONet Trunk
                        ▲
                        │
                 x,d,∇d,h
                        │
                        ▼
                   CFD field
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
        Data loss             Physics loss
                              │
                     ┌────────┼─────────┐
                     ▼        ▼         ▼
                 continuity momentum   BC
                        │
                        ▼
                    total loss
```

最终数学形式：

\[
\boxed{
q(x;G,\mu)
=
\bar q
+
\sum_k
B_k(z_G,\mu)
T_k(x,d,\nabla d,h)
}
\]

以及：

\[
\boxed{
L=
L_{POD}
+
\lambda_fL_{field}
+
\lambda_cL_{continuity}
+
\lambda_mL_{momentum}
+
\lambda_bL_{BC}
+
\lambda_wL_{wall}
}
\]

最终工程定位：

> **DeepSDF = geometry representation**

> **POD = low-dimensional state representation**

> **DeepONet = geometry/BC → field operator**

> **Physics-informed loss = CFD physics regularization**

> **Surface head = wall pressure/shear operator**

> **Online inference = pure neural-network forward pass**

---

# 35. 当前最推荐的实施路线

如果只保留最关键的开发优先级：

```text
① 你的 DeepSDF
       ↓
② 你的 POD
       ↓
③ mPOD-DeepONet 的 Branch/POD 思路
       ↓
④ DeepSDF + BC → POD coefficients
       ↓
⑤ DeepSDF + BC + x + SDF + ∇SDF + h → DeepONet
       ↓
⑥ supervised CFD loss
       ↓
⑦ PhysicsNeMo 风格 autodiff
       ↓
⑧ continuity + momentum + BC physics loss
       ↓
⑨ surface Cp/Cf/tau_w head
       ↓
⑩ Cd/Cl/Cm
       ↓
⑪ RANS/turbulence
       ↓
⑫ end-to-end DeepSDF fine-tuning
```

## 最重要的工程原则

**不要一次性把 DeepSDF、POD、DeepONet、PINN、RANS 全耦合。**

建议严格分解：

\[
\boxed{
DeepSDF
\rightarrow
POD
\rightarrow
DeepONet
\rightarrow
Physics-informed
\rightarrow
Surface
\rightarrow
RANS
}
\]

这样每一步都可以独立验证，出现误差时也容易定位。

---

# 36. GitHub 参考项目清单

| 项目 | GitHub | 建议用途 |
|---|---|---|
| mPOD-DeepONet | https://github.com/Chieh997/mPOD-DeepONet | **POD-DeepONet 核心参考** |
| PhysicsNeMo | https://github.com/NVIDIA/physicsnemo | **Physics-informed DeepONet / PDE loss** |
| PIOperator | https://github.com/pidneuralode/PIOperator | **PI-DeepONet / POD-DeepONet 对比参考** |
| NeuralOperator | https://github.com/neuraloperator/neuraloperator | **FNO/GNO benchmark** |

---

# 37. 最值得优先实现的代码接口

第一版可以把 API 控制在下面几个核心类：

```python
class DeepSDFEncoder:
    def encode_geometry(self, geometry):
        ...

class PODReducer:
    def project(self, field):
        ...

    def reconstruct(self, coeff):
        ...

class BranchNet(nn.Module):
    def forward(self, z_g, bc):
        ...

class TrunkNet(nn.Module):
    def forward(self, x, sdf, grad_sdf, h):
        ...

class PODDeepONet(nn.Module):
    def forward(self, z_g, bc, query):
        ...

class SurfaceHead(nn.Module):
    def forward(self, z_g, bc, x_s, normal):
        ...

class PhysicsLoss:
    def continuity(self, q, x):
        ...

    def momentum(self, q, x):
        ...

    def boundary(self, q, x):
        ...
```

训练层：

```python
loss = (
    loss_pod
    + lambda_field * loss_field
    + lambda_phys * loss_physics
    + lambda_bc * loss_bc
    + lambda_surface * loss_surface
)
```

最终 inference：

```python
z_g = deepsdf.encode_geometry(geometry)

a = branch(z_g, bc)

field = operator(
    z_g=z_g,
    bc=bc,
    query=points
)

surface = surface_head(
    z_g,
    bc,
    surface_points,
    normals
)
```

这就是整个项目最适合的第一版软件接口。

# DeepSDF、DeepMesh 与 Neural Operator 结合的 CFD 预测网络设计

结合 **DeepSDF (CVPR 2019)** 和 **DeepMesh**，这个方向非常适合构建一个：

> **Geometry Latent Space + Physics Neural Operator**

架构。

目标可以定义为：

- **输入：** 几何外形 + 边界条件
- **输出：** 3D 外流场 + 壁面压力 + wall shear / $C_f$，以及后续的 drag/lift

不建议简单地把 DeepSDF 的 `z-code` 直接输入普通 CNN。更合理的方案是：

> **DeepSDF/DeepMesh 负责“可生成、可微、低维”的几何表示；Geo-FNO / FNO 类 Neural Operator 负责学习 Navier–Stokes solution operator；DeepMesh 的显式 surface 则作为高精度 wall boundary branch。**

推荐主干：

**DeepSDF + DeepMesh + Geometry-conditioned Geo-FNO**

对于壁面压力/shear，再增加：

**Surface Mesh GNN / MeshGraphNet branch**

形成一个双分支 surrogate。

---

# 1. 先把 DeepSDF 和 DeepMesh 在问题中的角色分清楚

DeepSDF 的核心是：

$$
f_\theta(z,\mathbf{x})\rightarrow d
$$

其中：

- $z$：低维 shape latent code
- $\mathbf{x}=(x,y,z)$：空间坐标
- $d$：SDF
- $f_\theta(z,\mathbf{x})=0$：物体表面

也就是说，一个 256 维左右的 `z-code` 可以表达整个复杂几何，而不是直接保存 10 万甚至百万个 surface vertices。

DeepSDF 本身的核心思想就是用 latent-code-conditioned feed-forward decoder 表示一个 shape class 的连续 SDF。

DeepMesh 又进一步解决：

$$
z
\rightarrow
f_\theta(z,\mathbf{x})
\rightarrow
SDF
\rightarrow
Mesh(V,F)
$$

这个过程中 **surface mesh 对 $z$ 是可微的**。

DeepMesh 给出的关键关系是：

$$
\frac{\partial \mathbf{x}}{\partial c}
=
-\frac{\nabla f_\theta}
{\|\nabla f_\theta\|^2}
\frac{\partial f_\theta}{\partial c}
$$

其中 $c$ 可以是 $z$，也可以是 $[z,\theta]$。

因此可以把 mesh 上的 loss 梯度反传回 latent space。

这对 CFD 很重要，因为：

$$
z
\rightarrow
Geometry
\rightarrow
CFD\ response
$$

这一整个 pipeline 可以理论上保持 differentiable。

---

# 2. DeepMesh 论文其实已经给出了这个想法的雏形

DeepMesh 自己就做了一个：

$$
Mesh \rightarrow Pressure
$$

的 CFD surrogate。

他们选了约 1400 个 car shape，每一个用 OpenFOAM 进行 CFD，得到表面压力，然后训练：

$$
p=g_\beta(M)
$$

也就是 **Mesh Convolutional Neural Network**，输入 surface mesh，预测 pressure field。

然后：

$$
p=g_\beta(M(z))
$$

再把预测的 pressure 和表面法向积分，就得到 drag，并可以直接：

$$
\frac{\partial L_{drag}}{\partial z}
$$

然后优化 DeepMesh 的 latent code。

论文的实验发现 DeepMesh-SDF 的优化明显优于直接 vertex optimization、Scaling、FreeForm 和 PolyCube。

更加关键的是，论文明确发现直接优化每个 vertex 容易产生类似“adversarial geometry”的不真实形状，甚至导致最终 CFD 不收敛，因此需要低维 geometry parameterization 来 regularize shape space。

所以这里的思路实际上是对 DeepMesh CFD 部分进行一次自然升级：

### DeepMesh 原始思路

$$
z \rightarrow Mesh \rightarrow MeshCNN \rightarrow p
$$

### 更完整的方案

$$
z,\ BC
\rightarrow
\boxed{\text{Neural Operator}}
\rightarrow
\boxed{3D\ Flow\ Field}
$$

同时：

$$
z,\ BC
\rightarrow
\boxed{\text{Surface Network}}
\rightarrow
\boxed{p_w,\tau_w}
$$

这样能力会比原论文大很多。

---

# 3. 建议的整体架构

```text
                    Geometry
                       │
                 DeepSDF encoder
                       │
                       ▼
                latent code z
                  (e.g. 128/256)
                       │
             ┌─────────┴──────────┐
             │                    │
             ▼                    ▼
      DeepSDF decoder        DeepMesh
     f(z,x) → SDF             z → Mesh
             │                    │
             │                    │
             ▼                    ▼
        SDF(x), ∇SDF          V, F, normal
             │                    │
             │                    │
             └────────┬───────────┘
                      │
                      ▼
              Geometry features
                      │
         ┌────────────┴─────────────┐
         │                          │
         │ Boundary Conditions      │
         │ U∞, α, Re, Ma, TI...     │
         ▼                          ▼
      BC Encoder              Geometry Encoder
         │                          │
         └────────────┬─────────────┘
                      │
                      ▼
             Geometry-conditioned
                  Geo-FNO
                      │
                      ▼
             3D External Flow
       ┌────────┬────────┬─────────┐
       │        │        │         │
       ▼        ▼        ▼         ▼
       u        p        k/ω       ...
       │
       ▼
    wall region
       │
       ├──────────────► surface pressure p_w
       │
       └──────────────► wall shear τ_w
```

---

# 4. 为什么更推荐 Geo-FNO，而不是直接 Mesh CNN

真正要预测的是：

$$
\mathcal{G}:
(\text{geometry},BC)
\rightarrow
(u,p,k,\omega,\ldots)
$$

这本质上不是普通的 regression，而是：

> **PDE solution operator learning**

即：

$$
\mathcal{G}_{NS}
:
\{Geometry, BC\}
\rightarrow
\{Flow\ Field\}
$$

这正是 Neural Operator 比普通 CNN / MLP 更合适的地方。

FNO 通过 Fourier representation 学习整个 operator；普通 CNN 更接近：

$$
\mathbf{x}\rightarrow y
$$

的局部映射。

Geo-FNO 则进一步针对 arbitrary geometry，将 irregular physical domain 映射到 latent uniform domain，再在 latent domain 使用 FNO。

这和：

$$
z\rightarrow geometry
$$

非常契合。

---

# 5. 不要把 DeepSDF 的 z 直接当成唯一 geometry input

例如：

$$
z\in\mathbb R^{256}
$$

本质上是一个**隐式 geometry coordinate**。

它告诉网络：

> “这是 latent shape space 中的某种形状。”

但是它没有明确告诉 neural operator：

> “在当前 query point $\mathbf{x}$ 上，距离 wall 是多少。”

而对于 CFD，这个信息非常重要。

所以建议 physical network 同时使用：

$$
\boxed{
z
+
SDF(\mathbf{x})
+
\nabla SDF(\mathbf{x})
+
\mathbf{x}
+
BC
}
$$

也就是说：

$$
h_{geo}(\mathbf{x})
=
Encoder
\left[
z,\,
x,y,z,\,
d(\mathbf{x}),\,
\nabla d(\mathbf{x})
\right]
$$

其中：

$$
d(\mathbf{x})=f_\theta(z,\mathbf{x})
$$

而：

$$
\nabla d(\mathbf{x})
$$

直接给出 wall normal 的信息。

对于 CFD 来说，这比只给 `z` 强很多。

---

# 6. 为什么 SDF 在 CFD surrogate 中尤其有价值

例如一个空间点：

$$
\mathbf{x}=(x,y,z)
$$

仅仅知道：

$$
(x,y,z)
$$

是不够的。

还需要知道：

$$
d(\mathbf{x})=\text{distance to wall}
$$

因为外流问题存在非常强烈的 wall-distance dependence。

例如：

$$
u\rightarrow 0
$$

发生在：

$$
d\rightarrow0
$$

附近。

而 pressure gradient、velocity gradient、turbulent viscosity、wall shear 同样都强烈依赖 wall distance。

因此 SDF 实际上可以作为一种：

> **Geometry-aware spatial coordinate**

这对于 neural operator 非常有价值。

建议输入不仅采用：

$$
[d]
$$

而是：

$$
\boxed{
[d,\; \nabla d,\; |\nabla d|,\; x,\;y,\;z]
}
$$

其中对于理想 SDF：

$$
|\nabla d|\approx1
$$

因此：

$$
\mathbf n=\nabla d
$$

天然就是 wall normal。

---

# 7. Geometry branch 应该是什么样

建议：

$$
z\rightarrow Geometry\ Encoder
$$

同时：

$$
SDF(\mathbf{x})
$$

进入 local geometry branch。

最终形成：

$$
h_{geo}(\mathbf{x})
=
\phi
(
z,\mathbf{x},d,\nabla d
)
$$

例如：

```text
z ───────────────┐
                 │
(x,y,z) ─────────┤
                 ├── MLP ── geometry feature
SDF(x) ──────────┤
                 │
∇SDF(x) ─────────┘
```

这相当于把：

> **global geometry information**

和：

> **local geometry information**

同时提供给网络。

---

# 8. Boundary condition 也应该独立编码

例如 external aerodynamics：

$$
BC=
[
U_\infty,\alpha,
Re,Ma,
T_\infty,
\rho_\infty,
I_\infty,
\omega_\infty,k_\infty,\ldots
]
$$

经过：

$$
h_{BC}=MLP(BC)
$$

之后与 geometry latent fusion：

$$
h_0=Fusion(z,h_{BC})
$$

然后：

$$
h(\mathbf{x})
=
FNO
\left[
x,y,z,
SDF,
\nabla SDF,
h_0
\right]
$$

---

# 9. 一个非常重要的设计：不要让 z 只在输入层出现一次

非常建议使用：

$$
\boxed{\text{Latent conditioning at multiple FNO blocks}}
$$

例如：

$$
h_{l+1}
=
FNO_l(h_l)
+
A_l(z)
+
B_l(BC)
$$

或者 FiLM：

$$
h_l'
=
\gamma_l(z,BC)\odot h_l
+
\beta_l(z,BC)
$$

因为：

$$
z
$$

包含的是全局 geometry，而 FNO 在不同 Fourier scales 上处理不同 spatial structures。

如果只在第一层 concatenate 一次：

```text
[z, SDF, BC] → FNO → FNO → FNO → output
```

geometry 信息很容易在深层被稀释。

更好的方式：

```text
               z
               │
         ┌─────┼─────┐
         ▼     ▼     ▼
        FNO1  FNO2  FNO3
         ▲     ▲     ▲
         │     │     │
       BC/SDF BC/SDF BC/SDF
```

---

# 10. 外流场输出最好不要只预测 velocity

第一版建议预测：

$$
\boxed{
[u,v,w,p,k,\omega]
}
$$

如果 CFD 是 RANS。

或者如果是更简单的 laminar / LES：

$$
[u,v,w,p]
$$

但是建议把 turbulence quantities 也加入：

$$
k,\omega
$$

因为对于 RANS：

$$
\nu_t=f(k,\omega)
$$

或者 SST：

$$
\nu_t=\nu_t(k,\omega,S,F_1,\ldots)
$$

这样网络学习的不是简单的：

$$
Geometry\rightarrow p
$$

而是：

$$
Geometry+BC
\rightarrow
(U,p,k,\omega)
$$

这个 surrogate 的物理表达能力会强很多。

---

# 11. Wall pressure 与 wall shear 建议不要完全依赖 volume interpolation

对于 pressure：

$$
p_w=p|_{\Gamma}
$$

问题还不算特别严重。

但是 wall shear：

$$
\tau_w
=
\mu
\left.
\frac{\partial u_t}{\partial n}
\right|_w
$$

难度明显大很多。

原因是：

> shear 是一个导数型 quantities。

即使：

$$
u_{pred}
$$

看起来误差很小：

$$
\|u_{pred}-u\|
$$

也可能导致：

$$
\left|
\frac{\partial u_{pred}}{\partial n}
-
\frac{\partial u}{\partial n}
\right|
$$

非常大。

所以建议做：

### Branch A：Volume Geo-FNO

$$
(z,BC,SDF,\nabla SDF)
\rightarrow
(u,p,k,\omega)
$$

### Branch B：Surface Network

$$
(z,BC,M)
\rightarrow
(p_w,\tau_w,C_f)
$$

而不是强迫同一个 volume network 同时解决所有问题。

---

# 12. Surface Branch 推荐 MeshGraphNet，而不是原论文的 Mesh CNN

DeepMesh 原论文使用 Mesh Convolutional Neural Network 来预测 pressure。

但是对于这里的情况，更倾向：

$$
\boxed{MeshGraphNet}
$$

或者一个现代 message-passing GNN。

原因是 surface mesh：

$$
M=(V,F)
$$

本身就是 graph：

$$
G=(V,E)
$$

每个 vertex 可以拥有：

$$
h_i=
[
x_i,
n_i,
z,
BC,
curvature,
SDF\_features
]
$$

edge 可以拥有：

$$
e_{ij}=
[
x_i-x_j,
|x_i-x_j|,
n_i\cdot n_j
]
$$

然后做：

$$
h_i^{l+1}
=
\Phi
(
h_i^l,
\sum_{j\in N(i)}
\Psi(h_i^l,h_j^l,e_{ij})
)
$$

最终：

$$
h_i\rightarrow
[p_i,\tau_{w,i}]
$$

MeshGraphNet 本身就是用 mesh graph 上的 message passing 来学习物理 simulation，并且实验覆盖 aerodynamics。

---

# 13. 推荐的最终模型其实是“双 Neural Operator”

最终：

## Volume branch

$$
\boxed{
(z,BC,SDF,\nabla SDF)
\xrightarrow{\text{Geo-FNO}}
(u,p,k,\omega)
}
$$

## Surface branch

$$
\boxed{
(z,BC,V,F,n)
\xrightarrow{\text{MeshGraphNet}}
(p_w,\tau_w,C_f)
}
$$

然后两者通过 wall consistency loss 耦合：

$$
L_{wall}
=
\|p_{volume}|_\Gamma-p_{surface}\|^2
$$

以及：

$$
L_{\tau}
=
\left\|
\tau_{surface}
-
\mu
\frac{\partial u_{volume,t}}{\partial n}
\right\|^2
$$

这样会比“一个网络全部预测”稳定很多。

---

# 14. DeepONet 其实也非常适合，但放在第二选择

DeepONet 的结构：

$$
Branch(BC,Geometry)
$$

产生：

$$
b_1,\ldots,b_N
$$

然后：

$$
Trunk(x,y,z)
$$

产生：

$$
t_1(x),\ldots,t_N(x)
$$

最后：

$$
u(x)
=
\sum_{k=1}^{N}
b_k t_k(x)
$$

它天然就是：

$$
\boxed{
\{Geometry,BC\}
\rightarrow
Field(x)
}
$$

所以理论上甚至可以做：

$$
Branch(z,BC)
$$

和：

$$
Trunk(x,SDF,\nabla SDF)
$$

得到：

$$
\boxed{
q(x)=DeepONet(z,BC,x,SDF,\nabla SDF)
}
$$

这个结构非常干净。

---

# 15. 但为什么仍然优先选择 Geo-FNO？

因为目标是：

$$
\boxed{3D\ external\ flow\ field}
$$

而不是只查询几个点。

比如：

$$
512^3
$$

或者几百万个 CFD cells。

DeepONet 如果直接对每个 query point 单独 evaluation：

$$
x_1,x_2,\ldots,x_N
$$

计算量会比较大。

FNO 则更适合：

$$
\text{整个 field}
$$

进行 global operator transformation。

所以：

| 模型 | Geometry | 3D volume | Surface pressure | Arbitrary mesh | 适合程度 |
|---|---|---|---|---|---|
| MLP | 差 | 差 | 差 | 差 | ★ |
| CNN | 一般 | 好 | 差 | 差 | ★★ |
| DeepONet | 很好 | 好 | 好 | 好 | ★★★★ |
| FNO | 好 | **非常好** | 一般 | 差 | ★★★★ |
| Geo-FNO | **非常好** | **非常好** | 一般 | **好** | **★★★★★** |
| MeshGraphNet | 好 | 好 | **非常好** | **非常好** | ★★★★½ |

所以第一选择是：

$$
\boxed{\text{DeepSDF + DeepMesh + Geo-FNO + Surface GNN}}
$$

而不是：

$$
DeepSDF+DeepONet
$$

---

# 16. 更进一步：可以做成 DeepONet + Geo-FNO Hybrid

如果后面真的要做比较先进的研究，可以考虑：

$$
\boxed{
DeepSDF/DeepMesh
+
Geometry\ Encoder
+
DeepONet
+
Geo-FNO
}
$$

结构：

```text
                  z
                  │
           ┌──────┴──────┐
           │             │
           ▼             ▼
       DeepSDF        DeepMesh
           │             │
           ▼             ▼
          SDF          Mesh
           │             │
           └──────┬──────┘
                  │
                  ▼
          Geometry Encoder
                  │
             ┌────┴────┐
             │         │
             ▼         ▼
          Branch      Local
         network      SDF
             │         │
BC ──────────┘         │
             │         │
             ▼         ▼
          Global latent field
                  │
                  ▼
              Geo-FNO
                  │
                  ▼
       ┌──────────┼─────────┐
       ▼          ▼         ▼
       U          p       turbulence
       │
       ▼
        Surface features
             │
             ▼
        MeshGraphNet
             │
       ┌─────┴─────┐
       ▼           ▼
      p_w         τ_w
```

这里：

### DeepONet 的作用

负责：

> **global geometry + BC → global physical latent representation**

### Geo-FNO 的作用

负责：

> **spatial propagation / long-range correlation / entire field**

两者实际上互补。

---

# 17. DeepSDF z-code 可以进一步变成“CFD-aware latent code”

传统 DeepSDF 的训练：

$$
L_{SDF}
=
\left|
f_\theta(z,x)-d(x)
\right|
+\lambda\|z\|^2
$$

DeepMesh 又加入：

$$
L_{Chamfer}
$$

进行 geometry refinement。

可以进一步：

$$
\boxed{
L=
L_{SDF}
+
\lambda_1L_{Chamfer}
+
\lambda_2L_{CFD}
}
$$

其中：

$$
L_{CFD}
=
\|p_{pred}-p_{CFD}\|^2
+
\lambda_\tau
\|\tau_{pred}-\tau_{CFD}\|^2
$$

甚至：

$$
+
\lambda_D
\|C_D^{pred}-C_D^{CFD}\|^2
$$

于是 latent space 不再只是：

> “shape similarity latent space”

而会逐渐变成：

> **CFD-relevant shape latent space**

---

# 18. 这意味着 z-space 中可能出现非常有价值的 aerodynamic semantics

例如理想情况下：

$$
z_1
$$

可能主要控制：

> vehicle frontal area

$$
z_2
$$

可能主要控制：

> roof curvature

$$
z_3
$$

可能主要控制：

> rear taper

虽然未经约束的 DeepSDF **不会自动保证这种 disentanglement**，但可以利用 CFD loss + auxiliary supervision 去推动这种 behavior。

例如：

$$
z
\rightarrow
C_D,C_L,C_M
$$

同时学习：

$$
z
\rightarrow
geometry
$$

于是得到：

$$
\boxed{
z
\rightarrow
\text{geometry}
\rightarrow
\text{flow}
\rightarrow
\text{aerodynamics}
}
$$

这就开始接近真正意义上的：

> **Physics-aware generative design space**

---

# 19. 对这个 CFD 项目，不建议直接把 DeepMesh mesh 喂给 FNO

因为 FNO 最喜欢：

$$
\text{regular grid}
$$

而 DeepMesh 输出：

$$
(V,F)
$$

是 irregular mesh。

因此：

### 不推荐路线

$$
DeepMesh
\rightarrow
Mesh
\rightarrow
FNO
$$

中间强行转换到规则网格。

### 更好的路线

$$
DeepSDF
\rightarrow
SDF(x)
\rightarrow
Geo-FNO
$$

而：

$$
DeepSDF
\rightarrow
DeepMesh
\rightarrow
MeshGNN
$$

作为 surface branch。

即：

$$
\boxed{
SDF\text{ 给 volume network}
}
$$

$$
\boxed{
Mesh\text{ 给 surface network}
}
$$

这是非常自然的 representation split。

---

# 20. Octree / cut-cell CFD mesh 可以利用起来

这和 octree-based Cartesian cut-cell CFD solver 架构非常契合。

solver 是：

$$
Octree
\rightarrow
Cartesian\ cells
\rightarrow
cut-cells
$$

这里其实存在一个很好的 surrogate representation：

$$
cell_i:
[
x_i,y_i,z_i,
\Delta x_i,\Delta y_i,\Delta z_i,
SDF_i,
V_i,
A_{cut},
n_{wall},
BC
]
$$

然后：

$$
G_{CFD}
=
(V,E)
$$

就可以直接交给：

$$
MeshGraphNet
$$

这样甚至不必把 CFD mesh 转换成 standard structured grid。

因此可以做两个 surrogate baseline：

### Model A

$$
SDF+Geo-FNO
$$

### Model B

$$
Octree/CutCell+MeshGraphNet
$$

然后比较：

$$
accuracy
$$

$$
memory
$$

$$
inference\ speed
$$

$$
generalization\ to\ unseen\ geometry
$$

这对 octree/cut-cell CFD solver 的数据结构具有直接研究价值。

---

# 21. 对当前研究目标，建议这样定第一版

不要一开始就：

$$
u,v,w,p,k,\omega,\tau_w
$$

全部一起做。

第一阶段建议：

$$
\boxed{
(z, U_\infty,\alpha,Re)
\rightarrow
p(\mathbf{x}),C_p(\mathbf{x})
}
$$

先预测 pressure。

具体：

$$
f_\theta(z,x,y,z)
\rightarrow SDF
$$

然后：

$$
\boxed{
(z,BC,x,y,z,SDF,\nabla SDF)
\rightarrow C_p(x,y,z)
}
$$

使用 Geo-FNO。

同时：

$$
\boxed{
(z,BC,M)
\rightarrow C_p|_{\Gamma}
}
$$

使用 MeshGraphNet。

然后验证：

$$
C_D
=
\int_\Gamma
-C_p
(\mathbf n\cdot\mathbf e_x)
\,dA
$$

网络预测：

$$
\hat C_D
$$

与真实 CFD：

$$
C_D^{CFD}
$$

比较。

---

# 22. 第二阶段再增加 velocity

然后：

$$
\boxed{
(z,BC,SDF)
\rightarrow
[u,v,w,p]
}
$$

此时可以增加：

$$
L_{continuity}
=
\|\nabla\cdot\mathbf u\|^2
$$

以及 momentum residual：

$$
L_{NS}
=
\left\|
\rho(\mathbf u\cdot\nabla)\mathbf u
+
\nabla p
-
\nabla\cdot\tau
\right\|^2
$$

形成：

$$
\boxed{
L
=
L_{data}
+
\lambda_{NS}L_{NS}
+
\lambda_{wall}L_{wall}
}
$$

这样就开始进入真正的：

> **physics-aware neural operator**

---

# 23. 第三阶段才处理 wall shear

最后：

$$
\boxed{
\tau_w
=
\mu
\left.
\frac{\partial u_t}{\partial n}
\right|_\Gamma
}
$$

这时建议同时预测：

$$
\tau_w
$$

和：

$$
\frac{\partial u_t}{\partial n}
$$

或者增加专门 surface head：

$$
(z,BC,M,n)
\rightarrow
\tau_w
$$

然后：

$$
L_{\tau}
=
\|\tau_{pred}-\tau_{CFD}\|^2
$$

这样比单纯从 volume field numerical differentiation 得到 shear 更稳定。

---

# 24. 一个很重要的现实问题：DeepMesh 的 differentiability 对 surrogate training 其实不是刚需

如果目标单纯是：

$$
Geometry+BC\rightarrow CFD
$$

那么：

> **不需要 DeepMesh 的 differentiable Marching Cubes。**

因为完全可以：

$$
z
\rightarrow SDF
$$

直接生成训练所需 geometry representation。

DeepMesh 的真正价值在于：

$$
L_{CFD}
\rightarrow
\frac{\partial L_{CFD}}{\partial Mesh}
\rightarrow
\frac{\partial L_{CFD}}{\partial z}
$$

也就是：

> **CFD surrogate + gradient-based geometry optimization**

例如：

$$
z^*
=
\arg\min_z C_D(z)
$$

这时候 DeepMesh 就变得非常关键。

所以：

### Prediction

$$
DeepSDF + GeoFNO
$$

就够了。

### CFD design optimization

$$
\boxed{
DeepSDF+DeepMesh+GeoFNO
}
$$

才体现出整个体系真正的优势。

---

# 25. 最终会得到一个漂亮的闭环

训练阶段：

$$
\boxed{
Geometry
\rightarrow
DeepSDF
\rightarrow
z
}
$$

$$
\boxed{
z+BC
\rightarrow
GeoFNO
\rightarrow
CFD\ Field
}
$$

$$
\boxed{
z+BC+Mesh
\rightarrow
MeshGNN
\rightarrow
Wall\ quantities
}
$$

然后：

$$
CFD\ loss
\rightarrow
z
$$

最终：

$$
\boxed{
z
\overset{DeepSDF}{\longrightarrow}
Geometry
\overset{GeoFNO}{\longrightarrow}
Flow
\overset{integration}{\longrightarrow}
C_D
}
$$

再：

$$
\boxed{
C_D
\rightarrow
\nabla_z C_D
\rightarrow
z_{new}
\rightarrow
new\ geometry
}
$$

这就形成：

> **Generative Geometry → Physics Neural Operator → Differentiable Aerodynamic Design**

这比单纯的 DeepSDF + pressure CNN 要高一个层级。

---

# 26. 几种组合的最终推荐

| 方案 | Geometry | Physics | Surface | 评价 |
|---|---|---|---|---|
| DeepSDF + MLP | z | MLP | 无 | ★★ |
| DeepSDF + DeepONet | z+SDF | DeepONet | 无 | ★★★★ |
| DeepSDF + FNO | z+SDF | FNO | 无 | ★★★★ |
| **DeepSDF + Geo-FNO** | **z+SDF** | **Geo-FNO** | 无 | **★★★★★** |
| DeepMesh + MeshCNN | mesh | MeshCNN | pressure | ★★★★ |
| DeepMesh + MeshGraphNet | mesh | MGN | pressure/shear | **★★★★★** |
| **DeepSDF + DeepMesh + Geo-FNO + MeshGNN** | **z+SDF+mesh** | **Geo-FNO** | **MeshGNN** | **★★★★★+** |

---

## 最值得实际实现的版本

如果站在 **CFD solver + octree/cut-cell + CUDA** 的背景上，不建议照搬 DeepMesh 论文，而建议直接设计：

$$
\boxed{
\textbf{DeepSDF latent geometry}
+
\textbf{SDF-conditioned Geo-FNO}
+
\textbf{DeepMesh surface branch}
+
\textbf{MeshGraphNet}
}
$$

其中：

$$
\boxed{
z+BC+SDF+\nabla SDF
\rightarrow
Geo\text{-}FNO
\rightarrow
[u,v,w,p,k,\omega]
}
$$

以及：

$$
\boxed{
z+BC+\{V,F,n\}
\rightarrow
MeshGraphNet
\rightarrow
[p_w,\tau_w]
}
$$

最后通过：

$$
\boxed{
L=
L_{volume}
+\lambda_pL_{wall-p}
+\lambda_\tau L_{wall-\tau}
+\lambda_{phys}L_{NS}
+\lambda_{geo}L_{SDF/Chamfer}
}
$$

训练。

其中：

> **Geo-FNO 是整个体系里最推荐的 physics backbone；DeepMesh 不是替代 Geo-FNO，而是负责把 latent geometry 与高质量显式壁面连接起来。**

---

## 最关键的下一步

这个架构可以进一步具体到：

```text
DeepSDF
   │
   ├── z ∈ R^256
   │
   └── f(z,x) → SDF(x)
                  │
                  ├── volume grid
                  │      └── Geo-FNO
                  │
                  └── zero level set
                         └── DeepMesh
                               └── MeshGraphNet
```

下一步可以直接把这个架构落实成一份**可训练的 PyTorch 网络结构设计**，包括：

- `DeepSDF decoder`
- `SDF feature construction`
- `BC encoder`
- `Geo-FNO blocks`
- `surface MeshGraphNet`
- `multi-loss`
- 按照 octree-based Cartesian cut-cell CFD 数据结构设计 input/output tensor

而不是采用普通 CFD/Computer Vision 教科书式的数据格式。

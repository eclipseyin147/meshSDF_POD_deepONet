# 体积流场 POD + NN 系数回归 ROM 设计

日期：2026-09-10
关联：`docs/superpowers/specs/2026-09-10-cfd-case-generalization-design.md`（surface 分支）、`DeepSDF_DeepMesh_GeoFNO_CFD_Architecture.md` §10/§19/§22（volume 分支）

## 1. 目标

在现有 (z, BC) → 表面 Cp 算子之外，新增 **(z, BC) → 体积流场 (u, p)** 的 reduced-order 预测：

- POD 在 GPU 上完成（torch randomized SVD）；
- NN 只做低维系数回归：(z, BC) → a ∈ R^r；
- 数据接口先定义成与真实 CFD 快照兼容的格式，验证阶段用合成场跑通全链条。

明确不做：Geo-FNO/GINO 式整场算子（对照路线，后续）；PDE 残差物理损失（体积场可预测后再加，见 §7 展望）。

## 2. 数据表示

- **参考网格**：归一化坐标系下的规则网格，`--grid_resolution N`（默认 64），域 `[−1.5, 1.5]³`（DeepSDF 形状约 [−1,1]³，四周留出来流/尾迹余量）。所有案例共享同一网格 → POD 快照矩阵对齐。
- **快照格式**（每个 (shape, case) 一个文件，真实与合成数据共用同一格式）：
  `snapshots/<case_id>.npz`：`fields` (G, 4) float32（u,v,w,p，按参考网格 C-order 展平）、`bc` (4,)（BC_FIELDS 约定 [U, dir_x, dir_y, dir_z]）、`shape`（对应 SDF 样本 npz 名，用于取 latent）。
- **体内延拓**：SDF(x)<0 的点（固体内部）u≡0（no-slip 延拓，与 GINO 的 extension operator 一致）；p 取近壁延拓或验证场的定义值。SDF 由 decoder 现场评估，mask 不存盘。

## 3. POD（GPU，`deep_sdf/cfd/volume.py`）

- 快照矩阵 `S`：(N_cases, G·4) float32，列减均值 ȳ。
- `pod_fit(S, energy=0.999 或 rank=r, randomized=True, oversampling=10, niter=2, device="cuda")`：randomized range finder（Ω∼N(0,1) → Y=SΩ → QR → B=QᵀS → SVD(B)）→ 返回 `PODBasis(mean, basis (G·4, r), singular_values, energy)`；rank 按能量截断时报告实际 r。
- `PODBasis.project(Y) -> A (N, r)`、`reconstruct(A) -> Ŷ`、`relative_error(Y)`（全场相对 L2）、`projection_error(Y)`（用真值投影得到的不可约下界，评估时必须先报）、`save/load`（torch.save，basis 可存 float32）。
- 单元验证：已知低秩合成矩阵上 randomized SVD 与 exact SVD 的子空间投影误差 < 1e-6；reconstruct∘project 往返一致。

## 4. 系数回归网络（`deep_sdf/cfd/volume.py`）

`VolumeCoefficientRegressor(latent_size, rank, hidden=256, num_layers=4, bc_dim=4)`：

- 结构沿用 surface 算子惯例：`h_bc = bc_encoder(bc)`（MLP），`cat([z_norm, h_bc])` → MLP → r 维线性输出；
- 三组标准化 buffer（全部恒等默认 + setter）：`z_mean/z_std`（训练集 latent 统计）、`bc_mean/bc_std`（训练案例统计）、`coef_mean/coef_std`（**关键**：POD 系数幅值随奇异值衰减数个量级，不标准化会导致高阶模态欠拟合；用训练集投影系数的逐模态 std）；
- 预测重构：`ŷ = basis.reconstruct(a_pred)`，任意点查询通过对参考网格场三线性插值（`grid_sample` 风格，体积场分辨率无关性留给 GNO 路线）。

## 5. 训练脚本（`train_volume_rom.py`）

```
.venv/bin/python train_volume_rom.py -e <experiment> -d <data> -s <split> \
    --snapshots <dir>            # 真实/合成快照目录（上述 npz 格式）
    --synthetic                  # 无快照目录时：由 decoder SDF 现场生成合成场（仅流程验证）
    --grid_resolution 64 --pod_energy 0.999 [--pod_rank r] \
    --cases_per_shape 4 --u_range 10 20 --dir_cone_deg 180 \
    --iters 20000 --lr 1e-3 --val_fraction 0.2 --seed 0
```

- latent 获取复用 `train_pressure_surrogate.load_or_fit_latent`；方向/速度采样复用 `sample_flow_direction`/`make_bc`；
- 按形状切 train/val（沿用 surface 脚本逻辑）；POD 只用 **train** 案例的快照拟合（val 形状的场不得参与建基）；
- 损失：系数 MSE（标准化空间）；每 200 iter 报 val：系数 MSE、**重构全场相对 L2 误差**、以及 val 真值的 **projection_error**（不可约下界）并列对照；保存 best-on-val；
- checkpoint：model_kwargs、全部标准化统计量、pod 基文件路径（基单独存 `pod_basis.pth`）、u_range/dir_cone_deg/bc_fields、val 指标、seed；
- `--synthetic` 合成场（仅流程验证，写入相同快照格式后走统一路径）：自由流 `u∞=U·v̂`；近壁衰减 `u = u∞·σ(d/δ)`（d=decoder SDF，δ≈0.05）；尾迹亏损：下游坐标 ξ=(x−c)·v̂>0 且横向距离 ρ 在管内时 `u *= 1 − A·exp(−ρ²/w²)/(1+ξ)`，A ∝ 阻塞度（复用 proxy 的 D_yz/L_x）；p 由 Bernoulli 形式 `Cp = 1 − ‖u‖²/U²` 给出；体内 u≡0。docstring 明确标注非物理模拟。

## 6. 验证（全部实跑，/tmp 脚本）

1. 单元：POD 低秩恢复、往返一致、标准化 setter、快照格式读写；`--synthetic` 场的发散 sanity（尾迹区速度低于自由流）。
2. 端到端：沿用 27 椭球族（/tmp/cfd_data + /tmp/cfd_exp 的 decoder），每形状 4 案例（U∈[10,20]，全球面方向），grid 64³：
   - 报 POD 能量谱与截断 r、train 投影误差（不可约下界）；
   - 22 训练/5 留出形状 + 留出方向的 val 全场相对 L2 误差（目标：合成场上明显优于"只预测均值场"基线，且投影下界与回归误差分列）；
   - best-on-val checkpoint 重新加载后重构误差复现。

## 7. 展望（不在本次实现）

- 与 surface 分支联合损失：`L = L_vol + λ_wall·L_wall + λ_mom·L_mom`（两边均可微可算）；
- 残差细化网络 R_ψ(x, SDF, ∇SDF; a) 补 POD 截断高频；
- 真实 DrivAero/OpenFOAM 快照接入：写体积场采样器（OpenFOAM sampleDict → 上述 npz 格式）；
- 连续性约束：若快照散度自由则 POD 模态散度自由（by construction）；合成验证场不满足，属预期。

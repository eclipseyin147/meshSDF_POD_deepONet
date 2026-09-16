# DeepSDF + PhysicsNeMo DeepONet 第二轮增强设计：动态采样加权 / Surface 头 + Cd/Cl / PDE 微调

日期：2026-09-16
状态：设计已确认（brainstorming 流程产出）
上游：`DeepSDF_PhysicsNeMo_DeepONet_CFD_Roadmap.md`（§16 双头、§27 表面输入、§28 双路力、§34 重要性采样、Stage C）；第一轮 `docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md`
代码基线：分支 `roadmap-deeponet-mvp`（`deep_sdf/cfd/roadmap_deeponet.py`、`train_roadmap_deeponet.py`；MVP 结果 val 0.2513 / test 0.2640 / 189×）

## 1. 目标与范围

在第一轮 MVP（field stage：z+bc → xDeepONet → [u,v,w,Cp] 体积场）基础上增加三条工作流：

- **W1 近壁/离群形状分析 + 形状级动态采样加权**（field stage 重训增强）
- **W2 surface Cp/Cf 头 + Cd/Cl 双路预测**（积分 + 直接头 + consistency）
- **W3 PDE 物理微调**（复用 `deep_sdf/cfd/physics.py` 的 PDEInformer/IncompressibleNS/CollocationSampler）

W2 依赖新数据：OpenFOAM 壁面场重导出（全部 281 形状，用户已确认）。

显式排除：Cm（需力矩中心，后续）、湍流量（数据为层流）、多工况 BC、active learning 闭环、DoMINO benchmark。

## 2. 现状与事实

- MVP 产物：`examples/ellipsoids/RoadmapONet/`（best.mdlus、metrics.jsonl、eval.json、split.json、stats.pth、config.json）；eval.json 已有 per_case rel L2。
- 体积快照/SDF/法向缓存齐备（`data/openfoam/ellipsoids_u10/{snapshots,sdf_cache}`，113³ 网格）。
- **OpenFOAM case 目录已删除**（上次批跑成功后清理）→ 表面场必须连网格带求解全量重跑；相同模板与参数重跑得到相同解（上次 254/254 成功，均 101s/case，12 并发约 1.5~2h）。
- `openfoam_runner.py` 现有 FO 模式：probesDict + `foamPostProcess -latestTime -dict`；壁面 patch 正则 `body.*`，noSlip；模板 `data/openfoam/ellipsoids_u10/template_case/`；OpenFOAM 14 在 `/opt/openfoam14`（bashrc source 后用 `foamRun`）。
- `deep_sdf/cfd/labels.py:365 reference_area(verts, faces, flow_dir)`：`A_ref = Σ max(0, −n·v̂)·A_f`（投影迎风面积约定，保持沿用）。
- PiPOD stage-3（`train_pipod_deeponet.py:599-676`）物理微调模式：collocation 点 SDF 经冻结 decoder **带图**求（梯度贡献导数链）、SDF 梯度 detach、逐 chunk backward（phys_chunk 256）、physics AMP 关、`physics_weight_schedule`（0→0.01→0.05→0.1 @ 20/50/80%）、wall noslip |u|² + far-field u→dir。
- OpenFOAM patch 面法向（Sf）指向流体外（进入物体），符号易错——积分公式统一改用 decoder SDF 梯度法向（物体外向），并用 "所有 GT Cd > 0 且量级 0.01~2" 作导出校验。

## 3. W2a：表面场重导出（数据管线）

### 3.1 openfoam_runner 扩展

- `make_case` 增写 `system/surfaceDict`（surfaces FO）：`type surfaces; surfaceFormat vtk; interpolationScheme cell; fields (p wallShearStress); surfaces { wall { type patch; patches (".*body.*"); } }`（patch 名以 snappy 实际产出为准，导出时按正则匹配校验非空）。
- `controlDict` 的 functions 增加 `wallShearStress` FO（`libs "libfieldFunctionObjects.so"`，patches 同壁面，`writeControl endTime`）使 latestTime 存在壁面 `wallShearStress` 场。
- 新 `export_surface(case_dir, out_npz, U)`：
  1. `foamPostProcess -latestTime -dict system/surfaceDict`；
  2. pyvista 读 `postProcessing/surfaces/<time>/wall.vtp`（pip 安装 `pyvista`，新增依赖入报告）；
  3. 三角形 → 面心 `centers (F,3)`、面积 `areas (F,)`（顶点叉积；patch 自带法向弃用，统一改用 §3.2 的 decoder 法向）；
  4. cell 数据 p、wallShearStress → `Cp = p/(½U²)`、`cf = τw/(½U²)`（与体积线同约定）；
  5. 按 §3.2 公式积分出 `cd_gt`、`cl_gt`；
  6. 存 `data/openfoam/ellipsoids_u10/surface/<cache_key>.npz`：`centers (F,3)`、`normals (F,3)`（decoder 法向）、`areas (F,)`、`cp (F,)`、`cf (F,3)` 全 f32 + `cd_gt`、`cl_gt` 标量。

### 3.2 法向与力积分约定（GT 与预测共用同一公式）

面心处用冻结 decoder 求 SDF 梯度归一化得物体外向单位法向 `n`（patch 面心在体内侧附近，decoder 查询稳健）：

```
F_p = −Σ_f Cp_f · n_f · A_f          # 压力（法向向外，压力推向体内）
F_f =  Σ_f cf_f · A_f                # 摩擦（τw 方向即壁面力方向）
Cd  = (F_p + F_f) · dir / A_ref      # dir = bc[1:4] 归一化（本数据 = +x）
Cl  = (F_p + F_f) · ẑ   / A_ref
A_ref = Σ_f max(0, −n_f·dir) · A_f   # reference_area 约定
```

### 3.3 批量重跑

`generate_openfoam_snapshots.py` 加 `--surface_only`：make_case → run → export_surface → 校验 → 删 case；体积 probes 跳过（已有快照）。全 281 形状、12 并发、后台运行；复用 manifest 顺序与 STL。每形状校验：F > 1000、`cd_gt` 有限且 ∈ (0.001, 2)、法向模长 1±1e-3；写 `surface_summary.json`。

## 4. W1：分析与动态采样加权（field stage 增强）

### 4.1 `--analyze` 模式

加载 field checkpoint（best.mdlus），对 val+test 每 case 固定 65536 流体点：
- 分掩码 rel L2：近壁（|sdf|<0.15）/ 远域（其余），分变量；
- worst-10 清单 + 每形状的 latent 簇号、到 train 簇心距离；
- 写 `<out>/analysis.json` + 控制台表格。

### 4.2 动态误差自适应采样（field stage 重训时启用，`--adaptive_sampling`）

- 维护 train 形状权重 `w_s`，初始均匀；
- 每 `eval_every` 周期后：对全部 train 形状各采 4096 流体点快速前向（fp16，no autograd），估计 err_s（4 通道平均 rel L2）；
- `P(s) ∝ (err_s / mean_err + 0.1)`（ε=0.1、α=1），batch case 选取改 `torch.multinomial`；
- metrics.jsonl 增记当轮被采样最多的 top-5 形状名与 mean_err；形状误差不进 val 指标（val 保持均匀口径，与 MVP baseline 可比）。

## 5. W2b：surface stage（模型与训练）

### 5.1 架构

```python
surface_trunk = FullyConnected(42, 512×4, 256, silu)   # [x̃(3), n(3)] + γ(x̃,6)=36
surface_model = DeepONet(field_branch, trunk=surface_trunk, width=256,
                         out_channels=4, decoder mlp 128×2 silu)  # [Cp, Cfx, Cfy, Cfz]
force_head    = MLP(20 → 256×2 → 2, silu)              # (Cd, Cl)
```

- **共享 field stage 的 branch**（同一 nn.Module 传入两个 DeepONet 容器）。
- 冻结策略：volume trunk+decoder 冻结；branch 以 0.1× lr 联合微调；surface trunk/decoder/force head 全 lr。
- surface 标签 z-score（train 形状表面点统计，存 stats.pth 新增键 `surf_mean/surf_std (4,)`）；Cd/Cl 标签按 train 集分量 std 缩放，分量 std 过小时（如近对称形状的 Cl）以 `max(std_c, 0.1·std_Cd)` 为尺度，防止归一化爆炸。

### 5.2 surface 训练

- 采样：每 case 按**面积加权**抽 4096 表面点（`P(f) ∝ A_f`）；场真值 = surface npz 的 cp/cf。
- loss：`L_surf`（4 通道标准化 MSE 求和）+ `λ_f·(|Cd_int−Cd_gt| + |Cl_int−Cl_gt| + |Cd_head−Cd_gt| + |Cl_head−Cl_gt|)` + `λ_c·((Cd_head−Cd_int)² + (Cl_head−Cl_int)²)`；λ_f=1（力按上述尺度归一）、λ_c=0.1。
- Cd_int/Cl_int：对**预测**表面场（反标准化后）按 §3.2 公式积分；训练时用面积加权 MC 估计 `(A_total/n_sample)·Σ_i f_i`（无偏），评估时用全表面精确积分。

### 5.3 surface 评估

val/test 每形状：Cp、Cf（逐分量）rel L2；Cd/Cl 相对误差（head 与 integral 两路）中位数 + worst-5；写 eval_surface.json。

## 6. W3：physics stage（PDE 微调）

- `init_from` field best；branch+volume trunk/decoder 微调（lr 降 10×），surface 头冻结。
- loss：`L_field + λ_phys(t)·(L_cont + L_mom + L_wall + L_ff)`；λ 日程复用 `physics_weight_schedule`（0→0.01→0.05→0.1 @ 20/50/80%），可 `--lambda_phys` 固定。
- collocation 4096/iter（CollocationSampler 30/30/20/20），phys_chunk 256，physics 部分 AMP 关闭 fp32；预测反标准化后进 `PDEInformer(IncompressibleNS(re=100))`（ν=0.1、U=10、L=1 → Re=100）。
- collocation 特征：`[p, d, n_detached]` + Fourier——d 经冻结 decoder **带图**求（其梯度参与导数链），n = detach(grad d) 归一化（ReLU decoder 二阶导几乎处处为零）；照抄 PiPOD `physics_losses` 逐 chunk backward 模式。
- wall（noslip |u|²，近壁带采样点）/ far-field（u→dir，域边缘带）loss 照抄 `boundary_losses` 模式，特征用缓存值。

## 7. stage 编排与 CLI

`train_roadmap_deeponet.py` 增 `--stage field|surface|physics`（默认 field 保持 MVP 行为 + 新 `--adaptive_sampling`/`--analyze`），链式 init_from：surface←field best、physics←field best。输出统一 `<experiment>/RoadmapONet/`：

```text
field:    best.mdlus / train_state.pth / metrics.jsonl          （MVP 已有 + adaptive 增列）
surface:  best_surface.mdlus / train_state_surface.pth / metrics_surface.jsonl / eval_surface.json
physics:  best_physics.mdlus / train_state_physics.pth / metrics_physics.jsonl / eval_physics.json
analyze:  analysis.json
```

`--smoke` 对每 stage 生效（小点数小步数）。所有 stage 复用同一 split.json / stats.pth（surface/physics 不再改划分）。

## 8. 验收标准

1. W1：`analysis.json` 产出且含近壁/远域分解与 worst-10；field+adaptive run 的 val rel_l2 ≤ 0.27（MVP 0.2513 基准，允许波动）且 worst-10 平均误差较 MVP run 下降。
2. W2a：281 个 surface npz 全部通过 §3.3 校验；surface_summary.json 记录成功率。
3. W2b：eval_surface.json 含 Cp/Cf 分变量 rel L2 与 test 集 Cd/Cl 相对误差中位数（head/integral 两路）；Cd 中位数误差目标 < 10%。
4. W3：eval_physics.json 的 continuity/momentum residual 较 field-only 模型下降；val rel_l2 变化 ≤ +0.03。
5. 全部 stage 的 smoke 通过；产物不入 git；torch 2.5.1 不动；vendored physicsnemo 不改。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| OpenFOAM 重跑 ~2h 占用 | 后台运行，与代码任务并行；逐 case 校验防坏数据 |
| patch 名/FO 在 OF14 行为差异 | 先 1 个 case 冒烟导出（断言 F>1000、Cd_gt∈(0.001,2)、Cp 驻点≈1）再批量 |
| 符号约定错误（法向/Cd 符号） | §3.2 统一公式 + "全部形状 Cd_gt>0" 硬校验 + 椭球 Cd 量级先验 |
| 动态加权过拟合难形状牺牲一般形状 | ε=0.1 保底权重；val 口径保持均匀；与 MVP baseline 并排对比 |
| PDE 二阶导显存（8GB） | 逐 chunk backward（phys_chunk 256），PiPOD 已验证该规模可行 |
| pyvista 新依赖 | pip 安装记录于报告；仅导出脚本用，训练侧不依赖 |

## 10. 与第一轮的关系

不改 MVP 已提交代码的语义：`--stage field` 默认行为与第一轮相同（adaptive 默认关）；新数据（surface/）与旧数据（snapshots/、sdf_cache/）并存。PiPOD 线照旧不动。physicsnemo 仅用到 MVP 已验证的 import 面（FullyConnected/xDeepONet/Module）。

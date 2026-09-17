# DeepONet v3 精度改进设计：loss/日程 + trunk 特征 v2（physicsnemo 案例迁移）

日期：2026-09-17
状态：设计已确认（brainstorming 流程产出）
上游：`DeepSDF_PhysicsNeMo_DeepONet_CFD_Roadmap.md`（§21 Fourier、§22 日程、§64 ablation）；physicsnemo 案例挖掘报告（DoMINO/unified recipe/GeoTransolver）
基线：分支 `roadmap-deeponet-mvp`；MVP field（val 0.2513 / test 0.2640 / worst10-mean 0.8994）、adaptive（val 0.2491 / test 0.2729）、surface（test cd_int 0.0526）、physics（cont −34% / mom −44%）

## 1. 目标与范围

主攻**整体泛化**（test rel L2，未见几何），worst-shape 与 Cd 指标监控不劣化（用户裁定）。
允许手段（用户裁定）：仅 ① loss/日程/正则 与 ② trunk 特征增强。**不动**：数据集、DeepSDF（不重训/不换 latent）、真实 SDF 替换、surface/physics stage 语义。

已知误差结构（改进靶点）：v/w/p 通道远差于 u（0.34-0.47 vs 0.06）；尾迹区主导 worst case；过拟合早于 ~7500 iter；训练 loss（z-score MSE）与评测（rel L2）口径不一致。

## 2. 两个杠杆

### 2.1 E1：loss/日程/正则

- **per-channel relative-MSE** 替换 z-score MSE：逐 case 逐通道 `L_c = ‖pred_c−y_c‖²/(‖y_c‖²+ε)`，四通道求和；预测仍走 z-score 标准化输出再反变换（模型侧不变），loss 在物理（无量纲）空间计算。ε=1e-12。备选 `huber`（z-score 空间 SmoothL1 δ=1）作对照。
- **bf16 替换 fp16、去掉 GradScaler**（bf16 无需 loss scaling；与已验证的 physics stage 无 scaler 模式一致）。
- **early stop**：val 连续 `early_stop_patience=8` 个评估周期（默认 8×500=4000 iters）无 best 更新即终止；best-on-val checkpoint 机制不变。

### 2.2 E2：trunk 特征 v2（72 维）

v1（43 维，保留不动）：`[x̃(3), sdf(1), n(3)] + γ(x̃)(36)`。

v2（72 维）：

| 组 | 内容 | 维数 |
|---|---|---|
| raw | `x̃(3), sdf(1), scaled_sdf(1), inside(1), n(3), sdf·n(3)` | 12 |
| Fourier | `γ([x̃, sdf, scaled_sdf])`，5 通道 × 6 bands × 2 = 60 | 60 |

- `scaled_sdf = sdf/(0.04+|sdf|)`（DoMINO `scale_sdf`，近壁聚焦压缩到 (-1,1)）
- `inside = (sdf<0)` 显式体内/外标志
- `sdf·n` = 伪最近点向量（方向+幅值；DoMINO `x−x_closest` 的零成本近似）
- Fourier 联合编码坐标与 SDF 标量（DoMINO FourierMLP 对 SDF 特征联合编码的迁移）

## 3. 实现（配置开关，默认行为不变）

`deep_sdf/cfd/roadmap_deeponet.py`：
- `DEFAULT_CFG` 增：`loss_type: "zmse"`（|relmse|huber）、`feature_set: "v1"`（|v2）、`amp_dtype: "fp16"`（|bf16）、`early_stop_patience: 0`（0=关）。
- `trunk_features(..., feature_set="v1")`：v2 分支按 §2.2；v1 路径逐字不变。
- `build_model` 按 `feature_set` 算 trunk 输入维（43 或 72）。
- `data_loss(pred_norm, y, stats, cfg)` 新函数：`zmse` = 现行为；`relmse`/`huber` 按 §2.1。

`train_roadmap_deeponet.py`：
- AMP 按 `amp_dtype`：bf16 → 不建 GradScaler、直接 `loss.backward()/opt.step()`（physics stage 同款已验证模式）；fp16 → 现行为不变。
- 训练循环 data loss 改调 `rd.data_loss`；eval 块加 early stop 判定（`best` 计数器，patience 到即 break 并落 final 状态）。
- 所有 stage 共用这些开关；surface/physics stage 的既有 loss 语义不变（physics stage 已无 scaler，仅 bf16/fp16 影响其数据前向 dtype）。

## 4. 消融实验与验收

同一 split（RoadmapONet/split.json 复用机制）、关 adaptive、20000 iters 上限、其余超参同 MVP：

| run | 输出目录 | cfg 覆盖 |
|---|---|---|
| v3a | `RoadmapONet_v3a/` | `loss_type=relmse, amp_dtype=bf16, early_stop_patience=8` |
| v3b | `RoadmapONet_v3b/` | `feature_set=v2` |
| v3c | `RoadmapONet_v3c/` | 两者都开 |

每 run：`--config <json>` 训练 + `--eval_only --resume` 终评 + `--analyze` 产 analysis.json。
对照基线：MVP（val 0.2513 / test 0.2640 / worst10-mean 0.8994）。

验收：
1. 三 run 全通（smoke 先验证两新特征/loss 路径）且 val/test 可复现（同 seed 两次 eval 数字一致）。
2. 至少一 run test rel_l2 ≤ 0.238（较 0.2640 降 ≥10%）且 val ≤ 0.2613（0.2513+0.01）。
3. 各 run worst-10 均值与 surface 侧 Cd 相关性指标不显著劣化（--analyze 对照报告）。
4. 全部未达标也如实报告——消融负结果同样是结论；不在本轮追加新杠杆。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| relmse 对近零能量通道（固定 BC 下 v/w 均值≈0）分母小 | 分母为通道能量 Σy² 非均值，v/w 场能量非零；ε 兜底；探针先验证 |
| bf16 无 scaler 下溢 | physics stage 已验证同模式；smoke 检查 loss 有限 |
| 特征维数 43→72 改 trunk 第一层 | build_model 按 cfg 算 in_features；checkpoint 不跨 feature_set 复用（新 run 从头训） |
| early stop 误停（val 噪声） | patience 8 个周期（4000 iters）足够宽；best checkpoint 保底 |
| 三 run 抢 GPU | 串行跑（每个 ~10 min）；physics/surface 不重跑 |

## 6. 后续（本轮不做）

胜出配置可作为新 field best 重跑 surface/physics；真实 SDF 特征与 DeepSDF 重训（本轮被排除的手段）；DoMINO 式局部几何编码（粗网格 SDF 场 + 3D CNN）；surface 面积加权逐点 loss。

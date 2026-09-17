# DeepONet v3 精度改进（loss/日程 + 特征 v2，三 run 消融）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 经配置开关实现 E1（per-channel relative-MSE + bf16 + early stop）与 E2（trunk 特征 v2，72 维），跑 v3a/v3b/v3c 三个消融 run，对照 MVP 基线验证 test rel L2 下降 ≥10%。

**Architecture:** 全部改动经 `DEFAULT_CFG` 新键分流，默认行为与 MVP 逐字一致；v3 run 输出到独立目录 `examples/ellipsoids/RoadmapONet_v3{a,b,c}/`。

**Tech Stack:** torch 2.5.1、vendored PhysicsNeMo（经 compat shim）。规格：`docs/superpowers/specs/2026-09-17-deeponet-v3-accuracy-design.md`。

## Global Constraints

- 不升级 torch（2.5.1+cu121）；不修改 third-party/physicsnemo/；import physicsnemo 前先 import deep_sdf.cfd.physicsnemo_compat。
- 无 pytest；验证用可执行探针与 --smoke。
- Python 一律 .venv/bin/python；工作目录 /home/siqi/CLionProjects/DeepSDF。
- 产物不入 git；**不得覆盖** `examples/ellipsoids/RoadmapONet/`（MVP）与 `RoadmapONet_adaptive/`（R2）——v3 一律用新目录名。
- 向后兼容：`DEFAULT_CFG` 默认下（zmse/v1/fp16/patience 0）field stage 行为与 MVP 逐字节一致；surface/physics stage 语义不变。
- 关键事实：`trunk_features(xyz, sdf, normal, n_bands, domain_half)` 现为 v1（43 维）；`predict_normalized` 内 autocast 硬编码 fp16；`build_model` 按 `3+1+3+3*2*bands` 算 trunk_in；field 训练循环在 train_roadmap_deeponet.py（scaler 创建 :134、loss 组装 :261-268 一带）；MVP 基线 val 0.2513 / test 0.2640 / worst10-mean 0.8994。
- roadmap_surface.py / roadmap_physics.py 也调用 trunk_features/predict_normalized——改动必须向后兼容（默认参数），两 stage 本轮不重训。

---

### Task 1: 特征 v2 + loss 函数 + 配置键（roadmap_deeponet.py）

**Files:**
- Modify: `deep_sdf/cfd/roadmap_deeponet.py`

**Interfaces:**
- Consumes: 现有 fourier_encode/trunk_features/build_model/predict_normalized。
- Produces:
  - DEFAULT_CFG 新键：`"loss_type": "zmse"`、`"feature_set": "v1"`、`"amp_dtype": "fp16"`、`"early_stop_patience": 0`
  - `trunk_features(xyz, sdf, normal, n_bands, domain_half, feature_set="v1")`（v2 → (N,72)）
  - `build_model(cfg)` 按 feature_set 算 trunk_in（v1=43 / v2=72）
  - `data_loss(pred_norm (N,4), y (N,4), stats, cfg) -> scalar`
  - `predict_normalized` 的 autocast dtype 按 `cfg["amp_dtype"]` 分流

- [ ] **Step 1: 修改 roadmap_deeponet.py**

a) 顶部 import 区补 `import torch.nn.functional as F`。

b) DEFAULT_CFG 在 `"early_stop"` 无现有键——在 `"physics stage"` 段后追加一段：

```python
    # v3 accuracy levers (spec 2026-09-17 v3)
    "loss_type": "zmse",          # zmse | relmse | huber
    "feature_set": "v1",          # v1 | v2
    "amp_dtype": "fp16",          # fp16 | bf16 (bf16: no GradScaler)
    "early_stop_patience": 0,     # eval cycles without best update; 0=off
```

c) `trunk_features` 整个替换为：

```python
def trunk_features(xyz, sdf, normal, n_bands, domain_half,
                   feature_set="v1"):
    """v1 (3+1+3+3*2*bands): [x̃, sdf, n] + γ(x̃).
    v2 (12+5*2*bands = 72 @ bands=6), DoMINO-inspired (v3 spec 2.2):
    raw [x̃(3), sdf(1), scaled_sdf(1), inside(1), n(3), sdf·n(3)]
    + γ([x̃, sdf, scaled_sdf]) over 5 channels."""
    xn = xyz / domain_half
    if feature_set == "v1":
        return torch.cat([xn, sdf, normal, fourier_encode(xn, n_bands)],
                         dim=-1)
    if feature_set != "v2":
        raise ValueError("unknown feature_set: {}".format(feature_set))
    scaled = sdf / (0.04 + sdf.abs())
    inside = (sdf < 0).to(sdf.dtype)
    pseudo = sdf * normal
    raw = torch.cat([xn, sdf, scaled, inside, normal, pseudo], dim=-1)
    enc_in = torch.cat([xn, sdf, scaled], dim=-1)
    return torch.cat([raw, fourier_encode(enc_in, n_bands)], dim=-1)
```

d) `build_model` 的 trunk_in 一行替换为：

```python
    feature_set = cfg.get("feature_set", "v1")
    if feature_set == "v1":
        trunk_in = 3 + 1 + 3 + 3 * 2 * cfg["fourier_bands"]
    elif feature_set == "v2":
        trunk_in = 12 + 5 * 2 * cfg["fourier_bands"]
    else:
        raise ValueError("unknown feature_set: {}".format(feature_set))
```

e) `predict_normalized` 中 `with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):` 替换为：

```python
    amp_dtype = (torch.bfloat16 if cfg.get("amp_dtype") == "bf16"
                 else torch.float16)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
```
（其后 `xt = trunk_features(...)` 调用追加 `cfg.get("feature_set", "v1")` 参数。）

f) 文件末尾追加：

```python
def data_loss(pred_norm, y, stats, cfg):
    """Per-case field loss. pred_norm (N,4) normalized-space prediction,
    y (N,4) physical (nondimensional) targets. zmse: channelwise MSE in
    z-score space (MVP behavior); huber: SmoothL1(beta=1) per channel in
    z-score space; relmse: per-channel relative MSE in physical space
    sum_c ||p-y||^2/(||y||^2+eps) (v3 spec 2.1 - same shape as the rel-L2
    eval metric)."""
    yn = (y - stats["y_mean"]) / stats["y_std"]
    lt = cfg.get("loss_type", "zmse")
    if lt == "zmse":
        return sum(F.mse_loss(pred_norm[:, v], yn[:, v])
                   for v in range(4))
    if lt == "huber":
        return sum(F.smooth_l1_loss(pred_norm[:, v], yn[:, v], beta=1.0)
                   for v in range(4))
    if lt == "relmse":
        pred = pred_norm * stats["y_std"] + stats["y_mean"]
        return sum(((pred[:, v] - y[:, v]) ** 2).sum()
                   / ((y[:, v] ** 2).sum() + 1e-12) for v in range(4))
    raise ValueError("unknown loss_type: {}".format(lt))
```

- [ ] **Step 2: 探针验证**

```bash
.venv/bin/python - <<'EOF'
import torch
from deep_sdf.cfd import roadmap_deeponet as rd

# 向后兼容：v1 默认逐字不变
x = torch.randn(11, 3); s = torch.randn(11, 1); n = torch.randn(11, 3)
v1 = rd.trunk_features(x, s, n, 6, 1.5)
assert v1.shape == (11, 43), v1.shape
v2 = rd.trunk_features(x, s, n, 6, 1.5, feature_set="v2")
assert v2.shape == (11, 72), v2.shape
assert torch.equal(v1[:, :7], torch.cat([x / 1.5, s, n], dim=-1))  # v1 raw 段
scaled = s / (0.04 + s.abs())
assert torch.allclose(v2[:, 4], scaled[:, 0])                       # scaled_sdf
assert set(torch.unique(v2[:, 5]).tolist()) <= {0.0, 1.0}           # inside 标志
assert torch.allclose(v2[:, 9:12], (s * n))                         # sdf·n
try:
    rd.trunk_features(x, s, n, 6, 1.5, feature_set="v3")
    raise SystemExit("should have raised")
except ValueError:
    pass

cfg = dict(rd.DEFAULT_CFG)
m1 = rd.build_model(cfg)
assert sum(p.numel() for p in m1.parameters()) == 2447108
cfg2 = dict(cfg); cfg2["feature_set"] = "v2"
m2 = rd.build_model(cfg2)
n2 = sum(p.numel() for p in m2.parameters())
assert n2 == 2447108 + (72 - 43) * 512, n2   # 仅 trunk 首层变宽

stats = {"z_mean": torch.zeros(16), "z_std": torch.ones(16),
         "bc_mean": torch.zeros(4), "bc_std": torch.ones(4),
         "y_mean": torch.zeros(4), "y_std": torch.ones(4)}
pred = torch.randn(100, 4, requires_grad=True); y = torch.randn(100, 4)
for lt, ref in (("zmse", None), ("huber", None), ("relmse", None)):
    c = dict(cfg); c["loss_type"] = lt
    l = rd.data_loss(pred, y, stats, c)
    assert torch.isfinite(l) and l.requires_grad
# relmse 数值正确性：恒等 stats 下应等于 sum ||p-y||^2/||y||^2
c = dict(cfg); c["loss_type"] = "relmse"
l = rd.data_loss(pred, y, stats, c)
ref = sum(((pred[:, v] - y[:, v]) ** 2).sum()
          / ((y[:, v] ** 2).sum() + 1e-12) for v in range(4))
assert abs(l.item() - ref.item()) < 1e-5
# relmse 在物理空间：y_std 缩放应改变 loss（证明反标准化生效）
stats2 = dict(stats); stats2["y_std"] = torch.full((4,), 2.0)
l2 = rd.data_loss(pred, y, stats2, c)
assert abs(l2.item() - l.item()) > 1e-3
print("V3-LIB-OK v2_params=%d" % n2)
EOF
```
Expected: `V3-LIB-OK v2_params=2461956`（2447108+29×512）。

- [ ] **Step 3: Commit**

```bash
git add deep_sdf/cfd/roadmap_deeponet.py
git commit -m "v3 levers: trunk feature_set v2 (72d), data_loss zmse/relmse/huber, amp_dtype/bf16 plumbing"
```

---

### Task 2: 训练器接线（bf16 无 scaler + loss 分流 + early stop）

**Files:**
- Modify: `train_roadmap_deeponet.py`

**Interfaces:**
- Consumes: Task 1 的 `rd.data_loss`、cfg 新键、`predict_normalized` 的 amp_dtype 分流。
- Produces: field stage 支持 `loss_type`/`amp_dtype`/`early_stop_patience`；默认配置行为不变。

- [ ] **Step 1: 修改 train_roadmap_deeponet.py**

a) `scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])`（:134 一带）替换为：

```python
    use_scaler = cfg["amp"] and cfg["amp_dtype"] == "fp16"
    scaler = (torch.amp.GradScaler("cuda") if use_scaler else None)
```

b) field 训练循环的 loss 组装（`pred = rd.predict_normalized(...)` 后的 `yn = ...; loss = loss + sum(F.mse_loss...)` 两行）替换为：

```python
            pred = rd.predict_normalized(model, b["latent"], b["bc"],
                                         b["xyz"], b["sdf"], b["normal"],
                                         stats_g, cfg, amp=cfg["amp"])
            loss = loss + rd.data_loss(pred, b["y"], stats_g, cfg)
```
（注：`yn` 行删除；`F.mse_loss` 若在文件其他地方不再使用则保留 import 无碍。）

c) backward/step 三行（`scaler.scale(loss).backward()` / `scaler.step(opt)` / `scaler.update()`）替换为：

```python
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
```

d) early stop：`start_iter, best = 0, float("inf")` 行后加 `since_best = 0`；评估块中 `if res["rel_l2"] < best:` 分支内（保存 best.mdlus 之后）加 `since_best = 0`，并加 else 分支与终止：

```python
            if res["rel_l2"] < best:
                best = res["rel_l2"]
                model.save(os.path.join(out_dir, "best.mdlus"))
                since_best = 0
            else:
                since_best += 1
            torch.save({"model_state_dict": ...}, state_path)  # 既有行不动
            if (cfg["early_stop_patience"] and
                    since_best >= cfg["early_stop_patience"]):
                logging.info("early stop at iter %d (best %.4f)",
                             it + 1, best)
                break
```
（以 main() 实际代码为准对齐缩进与上下文；保持 train_state.pth 每次评估都保存的既有行为。）

e) `if args.eval_only:` 分支与 analyze 分支**不动**（它们不涉及 scaler/loss）。

- [ ] **Step 2: smoke 双配置验证**

```bash
printf '{"loss_type":"relmse","amp_dtype":"bf16","early_stop_patience":8}' > /tmp/cfg_v3a.json
printf '{"feature_set":"v2"}' > /tmp/cfg_v3b.json
.venv/bin/python train_roadmap_deeponet.py --smoke --config /tmp/cfg_v3a.json
.venv/bin/python train_roadmap_deeponet.py --smoke --config /tmp/cfg_v3b.json
.venv/bin/python train_roadmap_deeponet.py --smoke   # 默认配置回归
```
Expected：三个 smoke 都无异常结束（每个 ~30s-1min；缓存全命中）；`RoadmapONet_smoke/metrics.jsonl` 有 val 行；bf16 run 无 GradScaler 相关报错；默认配置行为与 MVP 一致（train_loss 量级与此前 smoke 同数量级 ~10→1）。**每个 smoke 跑完删除 smoke 目录**（`rm -rf examples/ellipsoids/RoadmapONet_smoke`）避免混淆。

- [ ] **Step 3: Commit**

```bash
git add train_roadmap_deeponet.py
git commit -m "Wire v3 levers into field stage (bf16 no-scaler, data_loss dispatch, early stop)"
```

---

### Task 3: 三 run 消融 + 对照报告

**Files:**
- 无新代码；产物 `examples/ellipsoids/RoadmapONet_v3{a,b,c}/`（不入 git）

**Interfaces:**
- Consumes: Task 1/2 的开关；MVP 基线数字（val 0.2513 / test 0.2640 / worst10 0.8994，analysis.json）。
- Produces: 三个 run 目录（各自 config.json/metrics.jsonl/best.mdlus/eval.json/analysis.json）+ 对照结论。

- [ ] **Step 1: 依次跑三个 run（串行，GPU 单卡；每个训练 ~10-35 min + 评估 ~1 min）**

```bash
printf '{"loss_type":"relmse","amp_dtype":"bf16","early_stop_patience":8}' > /tmp/cfg_v3a.json
printf '{"feature_set":"v2"}' > /tmp/cfg_v3b.json
printf '{"loss_type":"relmse","amp_dtype":"bf16","early_stop_patience":8,"feature_set":"v2"}' > /tmp/cfg_v3c.json
.venv/bin/python train_roadmap_deeponet.py --config /tmp/cfg_v3a.json --out_name RoadmapONet_v3a
.venv/bin/python train_roadmap_deeponet.py --eval_only --resume --out_name RoadmapONet_v3a
.venv/bin/python train_roadmap_deeponet.py --analyze --out_name RoadmapONet_v3a
```
（v3b/v3c 同理换 --config 与 --out_name。用 run_in_background 后台逐个跑，等一个完成再启动下一个。early stop 触发属正常，记录实际终止 iter。）

- [ ] **Step 2: 对照报告**

```bash
.venv/bin/python - <<'EOF'
import json
def load(d):
    e = json.load(open("examples/ellipsoids/%s/eval.json" % d))
    return e["val"]["rel_l2"], e["test"]["rel_l2"]
rows = [("mvp",) + load("RoadmapONet")]
for tag in ("v3a", "v3b", "v3c"):
    rows.append((tag,) + load("RoadmapONet_" + tag))
print("%-6s %8s %8s" % ("run", "val", "test"))
for r in rows:
    print("%-6s %8.4f %8.4f" % r)
# worst10 对照（MVP analysis 的 worst10 名单）
ana0 = json.load(open("examples/ellipsoids/RoadmapONet/analysis.json"))
w10 = ana0["worst10"]
mvp_pc = json.load(open("examples/ellipsoids/RoadmapONet/eval.json"))["val"]["per_case"]
for tag in ("v3a", "v3b", "v3c"):
    pc = json.load(open("examples/ellipsoids/RoadmapONet_%s/eval.json" % tag))["val"]["per_case"]
    both = [n for n in w10 if n in pc and n in mvp_pc]
    m = sum(pc[n] for n in both) / len(both)
    m0 = sum(mvp_pc[n] for n in both) / len(both)
    print("%s worst10-mean %.4f (mvp %.4f, n=%d)" % (tag, m, m0, len(both)))
EOF
```
Expected: 打印对照表。验收判定（规格 §4）：至少一 run test ≤ 0.238 且 val ≤ 0.2613 → 达标；记录各 run 的 early stop iter、worst10 均值变化。若全部未达标：如实报告（负结果也是结论），不追加新杠杆。

- [ ] **Step 3: 绘图留档**

```bash
.venv/bin/python plot_roadmap_metrics.py -d examples/ellipsoids/RoadmapONet_v3c -o /tmp/v3c.png
.venv/bin/python plot_roadmap_metrics.py -d examples/ellipsoids/RoadmapONet \
    --overlay examples/ellipsoids/RoadmapONet_v3c -o /tmp/mvp_vs_v3c.png
```
（v3c 换成实际最佳 run 的目录名。）

---

## Self-Review 记录

- 规格覆盖：§2.1→T1(f)/T2；§2.2→T1(c,d,e)；§3 开关与兼容→T1(b)/T2(a-d)；§4 消融与验收→T3。✅
- 向后兼容核验：trunk_features/build_model/predict_normalized 的默认路径逐字不变；roadmap_surface/roadmap_physics 以 5 参数位置调用 trunk_features 不受影响（新参数有默认值）；两 stage 的 predict_* 内部 autocast 仍 fp16 硬编码——本轮不重训它们，spec §6 已注明。
- 类型一致性：data_loss(pred_norm, y, stats, cfg) 在 T1 定义、T2 调用一致；v2 维数 72 = 12+5×2×6；参数量断言 2447108+29×512=2461956。
- early stop 与 best-on-val 交互：patience 只终止循环，best.mdlus 始终为 val 最优点；--eval_only 从 best.mdlus 评估，语义不变。

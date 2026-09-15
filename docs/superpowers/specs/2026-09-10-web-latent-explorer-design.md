# Web 端 SDF Latent 实时探索工具 设计文档

2026-09-10

## 目标

加载训练好的 DeepSDF decoder checkpoint（`<experiment>/ModelParameters/<ckpt>.pth`），在浏览器中实时拖动 latent code（z）等参数，即时显示重建出的物体表面网格。

## 架构：FastAPI + WebSocket + three.js（方案 A）

```
server.py                    # FastAPI 服务
web/index.html               # three.js 页面（CDN 引入，无构建步骤）
web/app.js                   # WebSocket 客户端 + 滑条 UI + OrbitControls
```

### 模型加载（启动时）

- CLI：`python server.py -e <experiment_dir> [--checkpoint latest] [--port 8000]`
- 读 `specs.json` 构建 decoder（复用 `deep_sdf.workspace.build_decoder` 的逻辑，兼容 weight_norm / norm_layers / latent_in / xyz_in_all）；
- 加载 `ModelParameters/<ckpt>.pth` 的 `model_state_dict`，strict=False 兼容 DataParallel `module.` 前缀；
- 加载 `LatentCodes/<ckpt>.pth` 得到训练形状 latent 库（预设下拉菜单的数据来源）；
- field_type 从 specs 的 `FieldType` 读（默认 sdf）；有 GPU 用 `mc_backend="torch"`，无 GPU 回退 skimage；
- 实验目录 / checkpoint 不存在：启动时报错退出。

### 数据流

拖动滑条 → 前端节流（~15fps）经 WebSocket 发 `{z: [...], resolution}` → 后端 `torch.no_grad()` 下 `sample_sdf_octree` + marching cubes 提取网格 → 二进制帧 `[uint32 nv, uint32 nf | float32 verts | int32 faces | float32 normals]` → 前端增量替换 BufferGeometry 属性。

### UI

- 预设形状下拉（来自 LatentCodes，名字取 split json，缺省用序号）；随机 z 按钮；两形状插值（A→B 下拉 + α 滑条）；
- z 滑条：按 8 维一组折叠（`<details>`），范围为全部训练 latent 每维 ±3σ；
- 分辨率滑条（63/125/249）：拖动时用低分辨率、松手后自动高分辨率精修一次；
- 状态栏：场类型、单帧提取耗时、网格顶点/面数。

### 错误处理

- MC 空场（无零交叉）：返回空 mesh，前端提示而不崩溃；
- WebSocket 断连自动重连；
- 后端串行处理请求（GPU 单流，拖动时后到请求覆盖式处理即可）。

### 测试

无真实训练 checkpoint（drivaernet_f1 尚未产出 ModelParameters），用随机初始化 decoder 按 specs 结构生成一个假 checkpoint 验证端到端：HTTP 静态页可达、WebSocket 往返、返回合法 mesh 帧。

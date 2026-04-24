# Complex RGB+Event Fusion Integration Pipeline

本文档描述把 `ComplexBiGRU.py` + `CompEvent_arch.py` 融合进 **当前 DynEventNeRF 工程** 后，端到端 pipeline 应该如何跑。

## 0. 现有工程中的数据入口

1. `data_loader_split.py::load_event_data_split` 读取每个视角的：
   - RGB 多帧路径（`rgb_paths`）
   - 事件流（`x, y, t, p`）
   - 相机内外参
2. 每个视角会构建一个 `RaySamplerSingleEventStream`。该对象里已经有：
   - 事件累积器 `EventStorage`
   - 每帧 RGB（`self.sRGB`）
   - `random_sample` 采样出的 ray + 监督信号

> 这意味着“多视角 + 时间”的原始输入已经在工程里，只需要在 sampler/train loop 中增加特征分支。

## 1. 每个时间窗构建 event voxel

新增模块 `EventVoxelizer`（`network/CompEvent_arch.py`）用于把每个时间窗内事件投影到 `B x bins x H x W`。

**建议接入点**：
- 在 `RaySamplerSingleEventStream` 中，基于每个训练 step 选定的时间窗 `[t_i, t_{i+1}]` 从事件流切片并构建 voxel。
- 或者在训练 loop（`ddp_train_nerf.py`）中统一构建（多视角 batch 方便并行）。

## 2. RGB feature 当实部，event feature 当虚部

`CompEventFusion.forward(rgb_feats, event_voxels)` 已按此规则实现：
- `event_voxels -> event_encoder -> event feature`
- `torch.complex(rgb_feats, event_feature)` 形成复特征。

**建议特征来源**：
- RGB feature 可直接来自图像 backbone（已有网络或轻量 CNN）。
- 早期验证时也可先使用“线性 RGB / 去马赛克 RGB”下采样后的 feature。

## 3. 用 ComplexBiGRU 做时间聚合

`ComplexBiGRU` 输入 `[B, T, N, C]` 复特征（`N=H*W`），输出同结构但通道会变成双向 hidden（`2*hidden`）。

`CompEventFusion` 内部已完成：
1. `z` reshape 为 `[B, T, H*W, C]`
2. 调用 `self.temporal(z_seq)`
3. 取最后时刻 `z_last` 作为当前时间窗融合表示。

## 4. 用 ComplexMixer 做局部 + 频域融合

`ComplexMixer` 内部两段：
1. 深度可分离局部卷积（real/imag 分支）
2. `FFT -> 频域门控 -> iFFT`

输出仍为复特征，`CompEventFusion` 当前默认返回实部作为 fused map。

## 5. 输出每个视角的 fused feature map

`CompEventFusion` 输出：`[B, C, H, W]`，可视作“每个视角当前时间窗”的 fused feature map。

多视角训练时，`B` 维即不同视角（或视角-时间组合 batch）。

## 6. NeRF 在投影采样时用 fused feature

在每条 ray、每个 sample 点投影到图像平面得到 `uv` 后，使用：
- `sample_fused_feature_for_nerf(fused_feature, uv)`

得到 `[B, N_samples, C]` 的每点条件特征，再拼接/调制到 NeRF 主干（`ddp_model.py::NerfNet.forward` 中 `fg_pts`、`fg_viewdirs` 的分支）。

---

## 推荐的工程落地顺序（最小风险）

1. **阶段 A：旁路验证（不改 NeRF）**
   - 先只跑 `CompEventFusion`，保存/可视化 fused map，检查时序稳定性。
2. **阶段 B：点采样接入**
   - 在 ray marching 时采样 fused feature，但只作为额外 loss 分支或辅助头输入。
3. **阶段 C：主干融合**
   - 将 fused feature 正式并入 `fg_net` 输入（concat 或 FiLM 调制）。
4. **阶段 D：联合训练**
   - 联合优化 NeRF + fusion 分支，必要时冻结部分模块稳定收敛。

## 一句话总览

**最终端到端链路**：

多视角 RGB/Events -> 时间窗事件 voxel -> 复特征构建(RGB实部+Event虚部) -> ComplexBiGRU 时序聚合 -> ComplexMixer 空间/频域融合 -> 每视角 fused feature map -> NeRF 投影采样 fused feature -> 体渲染输出。



## 原来的投影采样 vs 现在的投影采样

### 原来（项目当前默认）

- `NerfNet.forward` 在每条 ray 上做体采样，网络输入主要是：
  - 采样点坐标+时间 `fg_pts`（含时间编码）
  - 方向 `fg_viewdirs`
- 颜色/密度通过 MLP/Tensorf 后直接体渲染得到，不会在每个 3D sample 点上从“2D 特征图”再采一遍条件特征。
- 训练时 `RaySamplerSingleEventStream.random_sample` 负责从像素位置采样 ray，并提供对应监督（RGB/event）。

> 简单说：**原来是“像素采 ray + 3D 体采样”，但没有“3D 点反投影到2D特征图再取 feature”的步骤。**

### 现在（引入 complex fusion 后）

- 先得到每视角 fused feature map：`[B, C, H, W]`。
- 对每个 3D sample 点，利用相机模型投影得到 `uv`（归一化到 `[-1, 1]`）。
- 调用 `sample_fused_feature_for_nerf(fused_feature, uv)`（`grid_sample`）得到 `[B, N_samples, C]` 条件特征。
- 这些条件特征再与 NeRF 主干输入融合（concat/调制）。

> 简单说：新增了“**按 3D sample 点去2D fused feature map 做可微分采样**”这一步，因此 NeRF 每个采样点都能拿到对应视角的时空融合先验。

### 直观差异总结

1. **条件信息来源**
   - 原来：主要来自坐标编码与视角方向。
   - 现在：额外引入了图像平面上的 fused 语义/事件时序特征。

2. **采样空间**
   - 原来：只在 3D 体空间沿 ray 采样。
   - 现在：3D 采样点还会映射到 2D 特征平面进行二次采样。

3. **可微链路**
   - 原来：ray marching -> NeRF。
   - 现在：ray marching -> 投影 -> `grid_sample` -> NeRF（全链路可微）。

4. **计算代价**
   - 现在相比原来会增加：特征提取/融合开销 + 每点特征采样开销。

5. **潜在收益**
   - 对动态边缘、快速运动和事件稀疏区域，通常更容易提供稳定约束。



## 如何验证（回答：不能只跑 `run.sh`）

`run.sh` 当前是原始训练脚本入口，**默认不会自动验证 complex fusion 分支是否真正生效**（因为它没有显式检查 fused feature 的构建/采样链路）。

建议按下面顺序验证：

1. **模块级冒烟测试（必做）**
   - `python tools/test_complex_fusion_smoke.py`
   - 目的：确认 `CompEventFusion` 前向和 `sample_fused_feature_for_nerf` 维度/数值稳定。

2. **小规模训练验证（推荐）**
   - 用很小的 `N_iters` 跑一次训练（例如 100~500 iter），并在日志里确认你新增的 fusion 分支被调用（shape、loss、耗时）。
   - 这一步可以沿用 `run.sh` 的命令模板，但需确保你的训练代码里已经把 fused feature 接入 NeRF 前向。

3. **对照实验（最终）**
   - baseline（不启用 fusion） vs fusion（启用）同配置对比 PSNR/LPIPS/时序一致性。

一句话：`run.sh` 可以作为训练入口，但它本身**不是** complex fusion 的有效性验证脚本。

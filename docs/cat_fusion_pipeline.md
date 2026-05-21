# CTA-GRU Fusion Pipeline (non-complex)

按你的新需求，推荐的实现链路如下（对应 `network/cta_fusion_arch.py`）：

1. **数据准备（三帧）**
   - RGB: `I_{t-1}, I_t, I_{t+1}`
   - Event voxel: `E_{t-1→t}, E_{t→t+1}`（由 `EventVoxelizer` 构建）

2. **时序融合（非复数）**
   - `F_rgb = CNN(I_t)`（`FrameCNN`）
   - `F_event = CNN(E_t)`（`FrameCNN`）
   - `Z_t = concat(F_rgb, F_event)`
   - `F_t = CTA_GRU(Z_{t-1}, Z_t, Z_{t+1})`（`CTAGRU`）

3. **射线采样特征**
   - 对每条 ray 的投影坐标 `uv`，从 `F_t` 取 `f_t = F_t[u,v]`
   - 通过 `sample_feature_by_uv`（`grid_sample`）完成

4. **NeRF FiLM 调制**
   - `h = MLP_base(x,y,z,t)`
   - `gamma = MLP_gamma(f_t), beta = MLP_beta(f_t)`
   - `h' = gamma * h + beta`
   - 再预测 `sigma/rgb`
   - 对应 `NeRFFiLMModulator`

5. **Loss 设计**
   - 高频约束：`L_high = ||grad(C_hat) - Event_frame||`
   - 低频约束：`L_low = ||blur(C_hat) - RGB||`
   - 对应 `FusionLosses`

## 最小接入建议

- 先在训练代码中旁路计算 `F_t` 并打日志；
- 再把 `f_t` 接到 NeRF hidden layer 上做 FiLM；
- 最后加 `L_high + L_low`，并控制权重逐步增大
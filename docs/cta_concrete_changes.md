# CTA 融合：具体要改哪里（可直接照着做）

你问“具体呢”，这里给出**按文件粒度**的改动清单。

## A. 已完成（本次提交）

- 新增配置开关（`ddp_config.py`）：
  - `--use_cta_fusion`
  - `--cta_event_bins`
  - `--cta_feat_ch`
  - `--cta_loss_high_w`
  - `--cta_loss_low_w`
  - `--cta_warmup_iters`

这一步的意义：你现在可以在命令行/配置文件里显式开关 CTA 分支，而不是硬编码。

## B. 你下一步要改（建议顺序）

### 1) `create_nerf.py`

- 当 `args.use_cta_fusion=True` 时：
  - 创建 `ThreeFrameCTAFusion`、`NeRFFiLMModulator`、`FusionLosses`
  - 放到 `models` 字典（例如 `models['cta_fusion']`）
  - 把它们的参数并入优化器 param groups

### 2) `ddp_train_nerf.py`

- 在训练 step 中新增：
  1. 取三帧 RGB：`I_{t-1}, I_t, I_{t+1}`
  2. 构建两段 voxel：`E_{t-1->t}, E_{t->t+1}`
  3. `F_t = ThreeFrameCTAFusion(...)`
  4. 根据当前 ray 的投影 `uv`：`f_t = sample_feature_by_uv(F_t, uv)`
  5. 将 `f_t` 传入 NeRF 前向（建议新参数名 `ray_feat`）

- loss 上：
  - `l_high, l_low = FusionLosses(c_hat, event_frame, rgb)`
  - warmup 权重：
    - `w_high = args.cta_loss_high_w * min(1, it/args.cta_warmup_iters)`
    - `w_low  = args.cta_loss_low_w  * min(1, it/args.cta_warmup_iters)`
  - `loss += w_high * l_high + w_low * l_low`

### 3) `ddp_model.py` / `nerf_network.py`

- 给 NeRF 前向增加可选输入 `ray_feat`。
- 在 base hidden 后做 FiLM：
  - `h' = gamma(ray_feat) * h + beta(ray_feat)`
- 如果你暂时不想侵入太深：
  - 先在颜色头前做一次轻量调制（过渡方案）。

## C. 一组可直接尝试的参数

```bash
--use_cta_fusion True \
--cta_event_bins 8 \
--cta_feat_ch 32 \
--cta_loss_high_w 0.05 \
--cta_loss_low_w 0.1 \
--cta_warmup_iters 5000
```

## D. 成功标准（最小）

- 日志里出现 CTA 分支 forward 的 shape；
- loss 中出现 `l_high/l_low` 且数值稳定；
- baseline 与 CTA 对照渲染可比。

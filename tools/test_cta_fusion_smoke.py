#!/usr/bin/env python3

import torch

from network.cta_fusion_arch import ThreeFrameCTAFusion, sample_feature_by_uv, NeRFFiLMModulator, FusionLosses


def main() -> None:
    torch.manual_seed(0)
    b, h, w = 2, 24, 24
    cfeat = 32

    model = ThreeFrameCTAFusion(rgb_ch=3, event_bins=8, feat_ch=cfeat)

    rgb_prev = torch.randn(b, 3, h, w)
    rgb_t = torch.randn(b, 3, h, w)
    rgb_next = torch.randn(b, 3, h, w)
    evt_prev = torch.randn(b, 8, h, w)
    evt_next = torch.randn(b, 8, h, w)

    f_t = model(rgb_prev, rgb_t, rgb_next, evt_prev, evt_next)
    assert f_t.shape == (b, cfeat * 2, h, w)

    uv = torch.rand(b, 64, 2) * 2 - 1
    f_ray = sample_feature_by_uv(f_t, uv)
    assert f_ray.shape == (b, 64, cfeat * 2)

    hidden = torch.randn(b, 64, 128)
    film = NeRFFiLMModulator(feat_dim=cfeat * 2, hidden_dim=128)
    h_mod = film(hidden, f_ray)
    assert h_mod.shape == hidden.shape

    losses = FusionLosses(blur_kernel=5)
    c_hat = torch.randn(b, 3, h, w)
    event_frame = torch.randn(b, 1, h, w)
    rgb = torch.randn(b, 3, h, w)
    l_high, l_low = losses(c_hat, event_frame, rgb)
    assert l_high.ndim == 0 and l_low.ndim == 0

    print("[OK] CTA fusion smoke test passed")


if __name__ == "__main__":
    main()

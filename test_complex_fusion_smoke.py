#!/usr/bin/env python3
"""Smoke test for experimental complex RGB+event fusion modules.

Usage:
  python tools/test_complex_fusion_smoke.py
"""

import torch

from network.CompEvent_arch import CompEventFusion, sample_fused_feature_for_nerf


def main() -> None:
    torch.manual_seed(0)

    b, t, c, h, w = 2, 4, 16, 32, 32
    bins = 8
    n = 128

    rgb_feats = torch.randn(b, t, c, h, w)
    event_voxels = torch.randn(b, t, bins, h, w)

    fusion = CompEventFusion(rgb_channels=c, event_bins=bins, hidden_channels=12)
    fused = fusion(rgb_feats, event_voxels)

    assert fused.shape == (b, c, h, w), f"unexpected fused shape: {fused.shape}"
    assert torch.isfinite(fused).all(), "fused has non-finite values"

    uv = torch.rand(b, n, 2) * 2 - 1  # normalized to [-1, 1]
    sampled = sample_fused_feature_for_nerf(fused, uv)

    assert sampled.shape == (b, n, c), f"unexpected sampled shape: {sampled.shape}"
    assert torch.isfinite(sampled).all(), "sampled has non-finite values"

    print("[OK] complex fusion smoke test passed")
    print(f"fused={tuple(fused.shape)} sampled={tuple(sampled.shape)}")


if __name__ == "__main__":
    main()
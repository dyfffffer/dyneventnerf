from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.ComplexBiGRU import ComplexBiGRU


class ComplexMixer(nn.Module):
    """Local (depth-wise conv) + frequency-domain complex fusion."""

    def __init__(self, channels: int):
        super().__init__()
        self.local_real = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.local_imag = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z):
            raise ValueError("ComplexMixer expects a complex tensor")

        real = self.local_real(z.real)
        imag = self.local_imag(z.imag)
        z_local = torch.complex(real, imag)

        z_fft = torch.fft.fft2(z_local, dim=(-2, -1), norm="ortho")
        gate = self.freq_gate(torch.cat([z_fft.real, z_fft.imag], dim=1))
        gate_real, gate_imag = gate.chunk(2, dim=1)
        z_fft = torch.complex(z_fft.real * gate_real, z_fft.imag * gate_imag)
        return torch.fft.ifft2(z_fft, dim=(-2, -1), norm="ortho")


class EventVoxelizer(nn.Module):
    """Build event voxel grids for each time window.

    events: [B, N, 4] where columns are (x, y, t, p) with x/y pixel coords.
    """

    def __init__(self, height: int, width: int, bins: int):
        super().__init__()
        self.height = height
        self.width = width
        self.bins = bins

    @torch.no_grad()
    def forward(self, events: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        b, _, _ = events.shape
        out = events.new_zeros((b, self.bins, self.height, self.width), dtype=torch.float32)

        x = events[..., 0].long().clamp(0, self.width - 1)
        y = events[..., 1].long().clamp(0, self.height - 1)
        t = events[..., 2]
        p = events[..., 3].float()

        denom = (t_end - t_start).clamp(min=1e-6).unsqueeze(-1)
        tau = ((t - t_start.unsqueeze(-1)) / denom).clamp(0.0, 1.0 - 1e-6)
        bin_idx = (tau * self.bins).long().clamp(0, self.bins - 1)

        flat_hw = self.height * self.width
        for bi in range(b):
            linear = bin_idx[bi] * flat_hw + y[bi] * self.width + x[bi]
            out[bi].view(-1).index_add_(0, linear, p[bi])

        return out


class CompEventFusion(nn.Module):
    """Pipeline:
    1) event voxel per window
    2) RGB real + Event imaginary
    3) ComplexBiGRU temporal aggregation
    4) ComplexMixer local+frequency fusion
    5) fused feature map per view
    """

    def __init__(self, rgb_channels: int, event_bins: int, hidden_channels: int):
        super().__init__()
        self.event_encoder = nn.Sequential(
            nn.Conv2d(event_bins, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, rgb_channels, kernel_size=3, padding=1),
        )
        self.temporal = ComplexBiGRU(input_size=rgb_channels, hidden_size=hidden_channels)
        self.to_rgb_channels = nn.Conv2d(hidden_channels * 2, rgb_channels, kernel_size=1)
        self.mixer = ComplexMixer(rgb_channels)

    def forward(self, rgb_feats: torch.Tensor, event_voxels: torch.Tensor) -> torch.Tensor:
        """Args:
        rgb_feats: [B, T, C, H, W]
        event_voxels: [B, T, Be, H, W]
        Returns:
            fused_per_view: [B, C, H, W]
        """
        b, t, c, h, w = rgb_feats.shape
        event_encoded = self.event_encoder(event_voxels.view(b * t, *event_voxels.shape[2:]))
        event_encoded = event_encoded.view(b, t, c, h, w)

        z = torch.complex(rgb_feats, event_encoded)

        z_seq = z.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c)
        z_seq = self.temporal(z_seq)

        c2 = z_seq.shape[-1]
        z_last = z_seq[:, -1].reshape(b, h, w, c2).permute(0, 3, 1, 2).contiguous()
        z_last = torch.complex(self.to_rgb_channels(z_last.real), self.to_rgb_channels(z_last.imag))

        z_fused = self.mixer(z_last)
        return z_fused.real


def sample_fused_feature_for_nerf(
    fused_feature: torch.Tensor,
    uv: torch.Tensor,
    align_corners: bool = False,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample fused feature map at projected UV for NeRF.

    fused_feature: [B, C, H, W]
    uv: [B, N, 2] in normalized [-1, 1] coordinates
    returns: [B, N, C]
    """
    grid = uv.unsqueeze(2)
    sampled = F.grid_sample(fused_feature, grid, mode=mode, padding_mode="zeros", align_corners=align_corners)
    return sampled.squeeze(-1).permute(0, 2, 1).contiguous()


class CompEventNeRFAdapter(nn.Module):
    """A tiny adapter showing how to obtain per-sample fused features for NeRF."""

    def __init__(self, fusion: CompEventFusion):
        super().__init__()
        self.fusion = fusion

    def forward(
        self,
        rgb_feats: torch.Tensor,
        event_voxels: torch.Tensor,
        proj_uv: torch.Tensor,
        cached_fused: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fused = cached_fused if cached_fused is not None else self.fusion(rgb_feats, event_voxels)
        return sample_fused_feature_for_nerf(fused, proj_uv)
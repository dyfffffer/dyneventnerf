from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventVoxelizer(nn.Module):
    """Build event voxels from events in a time window.

    events: [B, N, 4] with (x, y, t, p).
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


class FrameCNN(nn.Module):
    """Lightweight feature extractor for RGB or Event voxel input."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGRUCell(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.reset_gate = nn.Conv2d(ch * 2, ch, 3, padding=1)
        self.update_gate = nn.Conv2d(ch * 2, ch, 3, padding=1)
        self.out_gate = nn.Conv2d(ch * 2, ch, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(xh))
        z = torch.sigmoid(self.update_gate(xh))
        q = torch.tanh(self.out_gate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * q


class CTAGRU(nn.Module):
    """Context-aware temporal aggregation over (t-1, t, t+1)."""

    def __init__(self, ch: int):
        super().__init__()
        self.ctx = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, 3, 1),
        )
        self.cell = ConvGRUCell(ch)

    def forward(self, z_prev: torch.Tensor, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        logits = self.ctx(torch.cat([z_prev, z_t, z_next], dim=1))
        w = torch.softmax(logits, dim=1)
        fused = w[:, 0:1] * z_prev + w[:, 1:2] * z_t + w[:, 2:3] * z_next
        return self.cell(fused, z_t)


class ThreeFrameCTAFusion(nn.Module):
    """Implements the user-defined pipeline:

    F_rgb = CNN(I_t)
    F_event = CNN(E_t)
    Z_t = concat(F_rgb, F_event)
    F_t = CTA_GRU(Z_{t-1}, Z_t, Z_{t+1})
    """

    def __init__(self, rgb_ch: int = 3, event_bins: int = 8, feat_ch: int = 32):
        super().__init__()
        self.rgb_encoder = FrameCNN(rgb_ch, feat_ch)
        self.event_encoder = FrameCNN(event_bins, feat_ch)
        self.cta_gru = CTAGRU(ch=feat_ch * 2)

    def encode_frame(self, rgb: torch.Tensor, event_voxel: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_encoder(rgb)
        f_event = self.event_encoder(event_voxel)
        return torch.cat([f_rgb, f_event], dim=1)

    def forward(
        self,
        rgb_prev: torch.Tensor,
        rgb_t: torch.Tensor,
        rgb_next: torch.Tensor,
        event_prev_to_t: torch.Tensor,
        event_t_to_next: torch.Tensor,
    ) -> torch.Tensor:
        # Use previous event voxel for t-1, avg for t, next event voxel for t+1.
        z_prev = self.encode_frame(rgb_prev, event_prev_to_t)
        z_t = self.encode_frame(rgb_t, 0.5 * (event_prev_to_t + event_t_to_next))
        z_next = self.encode_frame(rgb_next, event_t_to_next)
        return self.cta_gru(z_prev, z_t, z_next)


def sample_feature_by_uv(feature_map: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample per-ray feature from F_t.

    feature_map: [B, C, H, W]
    uv: [B, N, 2] normalized in [-1, 1]
    returns: [B, N, C]
    """
    sampled = F.grid_sample(feature_map, uv.unsqueeze(2), mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1).permute(0, 2, 1).contiguous()


class NeRFFiLMModulator(nn.Module):
    """Apply FiLM modulation to NeRF base hidden state.

    h' = gamma(f_t) * h + beta(f_t)
    """

    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp_gamma = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.mlp_beta = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        gamma = self.mlp_gamma(f_t)
        beta = self.mlp_beta(f_t)
        return gamma * h + beta


class FusionLosses(nn.Module):
    """High/low-frequency constraints.

    L_high = || grad(C_hat) - Event_frame ||
    L_low = || blur(C_hat) - RGB ||
    """

    def __init__(self, blur_kernel: int = 5):
        super().__init__()
        self.blur_kernel = blur_kernel

    @staticmethod
    def _gradient_mag(x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, c_hat: torch.Tensor, event_frame: torch.Tensor, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_c = self._gradient_mag(c_hat)
        if event_frame.shape[1] != grad_c.shape[1]:
            event_frame = event_frame.mean(dim=1, keepdim=True).expand_as(grad_c)
        l_high = F.l1_loss(grad_c, event_frame)

        c_blur = F.avg_pool2d(c_hat, kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel // 2)
        l_low = F.l1_loss(c_blur, rgb)
        return l_high, l_low
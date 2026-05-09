import torch
import torch.nn as nn


class ComplexBiGRU(nn.Module):
    """Complex-valued bi-directional GRU implemented with paired real GRUs.

    Input/Output tensors are complex64/complex128 with shape [B, T, N, C].
    Temporal aggregation runs along T.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru_real = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_imag = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise ValueError("ComplexBiGRU expects a complex tensor")

        b, t, n, c = x.shape
        x_real = x.real.reshape(b * n, t, c)
        x_imag = x.imag.reshape(b * n, t, c)

        rr, _ = self.gru_real(x_real)
        ri, _ = self.gru_real(x_imag)
        ir, _ = self.gru_imag(x_real)
        ii, _ = self.gru_imag(x_imag)

        out_real = rr - ii
        out_imag = ri + ir

        out = torch.complex(out_real, out_imag)
        out = out.reshape(b, n, t, out.shape[-1]).permute(0, 2, 1, 3).contiguous()
        return out
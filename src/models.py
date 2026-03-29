"""
models.py — time-conditioned MLP for latent flow matching in R^{d0}.

Architecture is fixed across all settings; only the input/output
dimension changes with d0 (the latent/intrinsic dimension).
"""

import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for a scalar time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "Embedding dim must be even."
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, 1) or (B,)
        Returns:
            emb: (B, dim)
        """
        t = t.reshape(-1)                  # (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / (half - 1)
        )                                  # (half,)
        args = t[:, None] * freqs[None, :] # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class VelocityMLP(nn.Module):
    """
    Time-conditioned MLP for latent flow matching.

    Input:  concatenation of h_bar (latent coords in R^{d0}) and a
            sinusoidal time embedding.
    Output: latent velocity vector in R^{d0}.

    Architecture template (fixed across all settings):
      - 4 hidden layers (default)
      - width 256 (default), comfortably overparameterized for d0 <= 16
      - SiLU activations
      - 64-d sinusoidal time embedding
    """

    def __init__(
        self,
        d0: int,
        time_emb_dim: int = 64,
        hidden_width: int = 256,
        hidden_layers: int = 4,
    ):
        super().__init__()
        self.d0 = d0

        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, time_emb_dim)

        in_dim = d0 + time_emb_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_width), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.SiLU()]
        layers.append(nn.Linear(hidden_width, d0))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, d0)  latent coordinates (= x_t @ U)
            t: (B, 1) or (B,)
        Returns:
            latent_velocity: (B, d0)
        """
        t_emb = self.time_emb(t)          # (B, time_emb_dim)
        t_emb = self.time_proj(t_emb)     # (B, time_emb_dim)
        inp = torch.cat([h, t_emb], dim=-1)
        return self.net(inp)

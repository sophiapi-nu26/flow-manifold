"""
train.py — training utilities for latent conditional flow matching.

The model operates in R^{d0} (latent space). Training projects both the
model input and the regression target to latent coordinates via U.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def make_dataloader(data: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------------------------------------------------------
# Per-batch loss
# ---------------------------------------------------------------------------

def _fm_loss(
    model: nn.Module,
    x1: torch.Tensor,
    dx: int,
    d0: int,
    U: torch.Tensor,
    eps: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Projected conditional flow-matching loss for one batch.

    Samples (x0, t) fresh for each call; target is (x1 - x0) @ U (latent
    component of the conditional velocity). Model input is xt @ U.

    Loss = mean over batch of  ||v_hat(xt @ U, t) - (x1 - x0) @ U||_2^2
    (sum over d0 latent coordinates, mean over batch).

    Note: training on (x1 - x0) @ U is exactly equivalent to training on the
    full target minus the analytically known normal component; no approximation
    is involved. See experiment_refactor_plan.md for the derivation.

    Args:
        model: latent MLP, takes (B, d0) and returns (B, d0)
        x1:    (B, dx) ambient data points in col(U)
        dx:    ambient dimension (for x0 sampling)
        d0:    latent dimension (output dim)
        U:     (dx, d0) orthonormal embedding matrix, on device
        eps:   time clipping (avoid t=0 and t=1)
        device: torch device
    """
    B = x1.shape[0]
    x0 = torch.randn(B, dx, device=device)
    t = eps + (1.0 - 2.0 * eps) * torch.rand(B, 1, device=device)
    xt = t * x1 + (1.0 - t) * x0
    h_bar = xt @ U                # (B, d0) — model input
    target = (x1 - x0) @ U       # (B, d0) — projected conditional target
    pred = model(h_bar, t)
    return ((pred - target) ** 2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Train / val epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dx: int,
    d0: int,
    U: torch.Tensor,
    eps: float,
    grad_clip: float,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss (per sample)."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for (x1,) in loader:
        x1 = x1.to(device)
        loss = _fm_loss(model, x1, dx, d0, U, eps, device)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x1.shape[0]
        total_samples += x1.shape[0]

    return total_loss / total_samples


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    dx: int,
    d0: int,
    U: torch.Tensor,
    eps: float,
    device: torch.device,
) -> float:
    """Run one validation epoch. Returns mean loss (per sample)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for (x1,) in loader:
        x1 = x1.to(device)
        loss = _fm_loss(model, x1, dx, d0, U, eps, device)
        total_loss += loss.item() * x1.shape[0]
        total_samples += x1.shape[0]

    return total_loss / total_samples

"""
data.py — synthetic data generation for the flow-manifold experiment.

All geometry (U, GMM) and datasets are generated deterministically from seeds
that are fixed per (dx, d0) setting and never resampled across training seeds.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Subspace matrix
# ---------------------------------------------------------------------------

def make_U(dx: int, d0: int, geom_seed: int) -> torch.Tensor:
    """
    Random dx x d0 orthonormal matrix via QR decomposition.

    Returns:
        U: (dx, d0) float32 tensor with orthonormal columns
    """
    rng = np.random.RandomState(geom_seed)
    A = rng.randn(dx, d0)
    Q, _ = np.linalg.qr(A)
    return torch.tensor(Q[:, :d0], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Latent GMM
# ---------------------------------------------------------------------------

def make_gmm_params(d0: int, gmm_seed: int, M: int = 4):
    """
    Generate latent GMM parameters.

    Construction recipe (fixed by gmm_seed so it is the same across all dx
    values that share the same d0):
      - uniform mixture weights
      - means ~ N(0, 4 I_{d0})  (std=2)
      - diagonal covariances, entries ~ Uniform[0.5, 1.5]

    Returns:
        pis:         (M,)     float32 mixture weights
        mus:         (M, d0)  float32 component means
        diag_Sigmas: (M, d0)  float32 diagonal variance entries
    """
    rng = np.random.RandomState(gmm_seed)
    pis = np.ones(M, dtype=np.float32) / M
    mus = (rng.randn(M, d0) * 2.0).astype(np.float32)          # N(0, 4I)
    diag_Sigmas = rng.uniform(0.5, 1.5, size=(M, d0)).astype(np.float32)
    return (
        torch.from_numpy(pis),
        torch.from_numpy(mus),
        torch.from_numpy(diag_Sigmas),
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _sample_gmm_np(
    n: int,
    pis_np: np.ndarray,
    mus_np: np.ndarray,
    diag_Sigmas_np: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample n latent points from the GMM. Returns (n, d0) float32 array."""
    M, d0 = mus_np.shape
    comp = rng.choice(M, size=n, p=pis_np)
    out = np.zeros((n, d0), dtype=np.float32)
    for m in range(M):
        idx = np.where(comp == m)[0]
        if len(idx) == 0:
            continue
        std_m = np.sqrt(diag_Sigmas_np[m])          # (d0,)
        z = rng.randn(len(idx), d0).astype(np.float32)
        out[idx] = mus_np[m] + z * std_m
    return out


def make_datasets(
    dx: int,
    d0: int,
    U: torch.Tensor,
    pis: torch.Tensor,
    mus: torch.Tensor,
    diag_Sigmas: torch.Tensor,
    data_seed: int,
    n_train_master: int = 50_000,
    n_val: int = 2_000,
    n_test: int = 5_000,
):
    """
    Generate master train, validation, and test datasets for one (dx, d0) setting.

    The three n_train levels {2000, 10000, 50000} are nested prefixes of
    x_train_master, making sample-size comparisons cleaner.

    Returns:
        x_train_master: (n_train_master, dx) float32
        x_val:          (n_val, dx)           float32
        x_test:         (n_test, dx)          float32
    """
    rng = np.random.RandomState(data_seed)

    pis_np = pis.numpy()
    mus_np = mus.numpy()
    diag_Sigmas_np = diag_Sigmas.numpy()
    U_np = U.numpy()                          # (dx, d0)

    total = n_train_master + n_val + n_test
    h_all = _sample_gmm_np(total, pis_np, mus_np, diag_Sigmas_np, rng)
    x_all = h_all @ U_np.T                   # (total, dx)

    x_train_master = torch.from_numpy(x_all[:n_train_master])
    x_val = torch.from_numpy(x_all[n_train_master : n_train_master + n_val])
    x_test = torch.from_numpy(x_all[n_train_master + n_val :])
    return x_train_master, x_val, x_test

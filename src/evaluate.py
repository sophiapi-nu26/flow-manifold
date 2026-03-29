"""
evaluate.py — held-out evaluation metrics for the oracle-U latent ablation.

Primary metric:
    tangent oracle MSE (per-dim and total)
        = E || v_hat(h_bar, t) - v*(h_bar, t) ||_2^2  [total]
        = total / d0                                    [per-dim]

Secondary / verification metrics:
    normal MSE (software correctness check — should be < 1e-5)
    latent scatter data (for Figure 2A)
    latent endpoint samples (for Figure 2C)
    sliced Wasserstein distance (for Figure 2C scalar metric)

Evaluation randomness is pinned via local CPU torch.Generator seeded by
eval_seed. This makes all reported numbers exactly reproducible regardless
of global RNG state.
"""

import numpy as np
import torch
import torch.nn as nn

from .oracle import oracle_tangent_latent, proj_perp, theoretical_normal


# ---------------------------------------------------------------------------
# Shared sampling helper
# ---------------------------------------------------------------------------

def _sample_latent_batch(
    x_test: torch.Tensor,
    U: torch.Tensor,
    B: int,
    dx: int,
    eps: float,
    device: torch.device,
    gen: torch.Generator,
):
    """
    Sample a batch of (h_bar, t) from the test set in latent coordinates.

    Draws x1 with replacement from x_test, samples fresh x0 ~ N(0, I_dx),
    forms xt = t*x1 + (1-t)*x0, then projects to h_bar = xt @ U.

    All randomness goes through `gen` for exact reproducibility.
    """
    n_test = x_test.shape[0]
    idx = torch.randint(0, n_test, (B,), generator=gen)
    x1 = x_test[idx].to(device)
    x0 = torch.randn(B, dx, generator=gen).to(device)
    t  = (eps + (1.0 - 2.0 * eps) * torch.rand(B, 1, generator=gen)).to(device)
    xt = t * x1 + (1.0 - t) * x0
    h_bar = xt @ U    # (B, d0)
    return h_bar, t, xt


# ---------------------------------------------------------------------------
# Online tangent oracle MSE (used for Panel B and during training eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_tangent_oracle_mse(
    model: nn.Module,
    x_test: torch.Tensor,
    U: torch.Tensor,
    pis: torch.Tensor,
    mus: torch.Tensor,
    diag_Sigmas: torch.Tensor,
    device: torch.device,
    eps: float,
    n_pairs: int = 10_000,
    batch_size: int = 256,
    eval_seed: int = 42,
) -> tuple[float, float]:
    """
    Compute per-dim and total tangent oracle MSE online from x_test.

    Returns:
        (tangent_oracle_mse_per_dim, tangent_oracle_mse_total)
    """
    model.eval()
    d0 = U.shape[1]
    dx = x_test.shape[1]
    gen = torch.Generator()
    gen.manual_seed(eval_seed)
    total_mse = 0.0
    done = 0

    while done < n_pairs:
        B = min(batch_size, n_pairs - done)
        h_bar, t, _ = _sample_latent_batch(x_test, U, B, dx, eps, device, gen)

        v_hat = model(h_bar, t)
        v_star = oracle_tangent_latent(h_bar, t, pis, mus, diag_Sigmas)
        total_mse += ((v_hat - v_star) ** 2).sum(dim=1).mean().item() * B
        done += B

    total = total_mse / n_pairs
    per_dim = total / d0
    return per_dim, total


# ---------------------------------------------------------------------------
# Panel A: build and load cached eval set
# ---------------------------------------------------------------------------

def _sample_gmm_torch(
    n: int,
    pis: torch.Tensor,
    mus: torch.Tensor,
    diag_Sigmas: torch.Tensor,
    gen: torch.Generator,
) -> torch.Tensor:
    """Sample n points from latent GMM using a torch Generator. Returns (n, d0)."""
    M, d0 = mus.shape
    comp = torch.multinomial(pis, n, replacement=True, generator=gen)  # (n,)
    h = torch.zeros(n, d0)
    for m in range(M):
        idx = (comp == m).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        std_m = diag_Sigmas[m].sqrt()  # (d0,)
        z = torch.randn(len(idx), d0, generator=gen)
        h[idx] = mus[m] + z * std_m
    return h


def build_panel_a_eval_cache(
    pis: torch.Tensor,
    mus: torch.Tensor,
    diag_Sigmas: torch.Tensor,
    d0: int,
    n_points: int = 10_000,
    eval_seed: int = 42,
    eps: float = 0.01,
    gmm_seed: int = None,
    save_path: str = None,
    version: str = "1.0",
) -> dict:
    """
    Build the Panel A evaluation cache: pre-computed (h_bar, t, v*(h_bar,t)) tuples.

    h_bar is formed from latent samples directly:
        h1 ~ GMM in R^{d0}
        h0 ~ N(0, I_{d0})  (distribution of x0 @ U for orthonormal U)
        t  ~ Uniform(eps, 1-eps)
        h_bar = t * h1 + (1-t) * h0

    Using the latent path directly (rather than ambient x and then projecting)
    gives the exact same marginal distribution and avoids any dx-dependence.

    If save_path is provided, the cache is saved to disk.
    Returns the cache dict.
    """
    gen = torch.Generator()
    gen.manual_seed(eval_seed)

    # Sample latent endpoints h1 ~ GMM
    h1 = _sample_gmm_torch(n_points, pis, mus, diag_Sigmas, gen)  # (N, d0)

    # Sample latent noise h0 ~ N(0, I_{d0})
    h0 = torch.randn(n_points, d0, generator=gen)                  # (N, d0)

    # Sample times
    t = (eps + (1.0 - 2.0 * eps) * torch.rand(n_points, 1, generator=gen))  # (N, 1)

    # Interpolate to latent h_bar
    h_bar = t * h1 + (1.0 - t) * h0                               # (N, d0)

    # Compute oracle tangent velocity
    with torch.no_grad():
        v_star = oracle_tangent_latent(h_bar, t, pis, mus, diag_Sigmas)  # (N, d0)

    cache = {
        "h_bar": h_bar,
        "t": t,
        "v_star": v_star,
        "metadata": {
            "d0": d0,
            "eval_seed": eval_seed,
            "n_points": n_points,
            "eps": eps,
            "gmm_seed": gmm_seed,
            "version": version,
        },
    }

    if save_path is not None:
        torch.save(cache, save_path)

    return cache


@torch.no_grad()
def compute_tangent_oracle_mse_from_cache(
    model: nn.Module,
    h_bar_cache: torch.Tensor,
    t_cache: torch.Tensor,
    v_star_cache: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, float]:
    """
    Compute per-dim and total tangent oracle MSE from a pre-built eval cache.

    Uses the exact same (h_bar, t, v*) tuples regardless of dx — this is
    what makes Panel A a controlled invariance check.

    Returns:
        (tangent_oracle_mse_per_dim, tangent_oracle_mse_total)
    """
    model.eval()
    d0 = h_bar_cache.shape[1]
    n_total = h_bar_cache.shape[0]
    total_mse = 0.0
    done = 0

    while done < n_total:
        B = min(batch_size, n_total - done)
        h = h_bar_cache[done : done + B].to(device)
        t = t_cache[done : done + B].to(device)
        v_s = v_star_cache[done : done + B].to(device)

        v_hat = model(h, t)
        total_mse += ((v_hat - v_s) ** 2).sum(dim=1).mean().item() * B
        done += B

    total = total_mse / n_total
    per_dim = total / d0
    return per_dim, total


# ---------------------------------------------------------------------------
# Normal-component MSE (software correctness check)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_normal_mse(
    model: nn.Module,
    x_test: torch.Tensor,
    U: torch.Tensor,
    device: torch.device,
    eps: float,
    n_pairs: int = 10_000,
    batch_size: int = 256,
    eval_seed: int = 43,
) -> float:
    """
    Verify that the reconstructed ambient velocity's normal component equals
    the analytical normal.

    Full ambient velocity: u_hat = v_hat @ U.T + theoretical_normal(xt, t, U)
    Normal component:      (I - UU^T) u_hat = theoretical_normal(xt, t, U)
                           (since v_hat @ U.T lies in col(U))

    This should return < 1e-5 for any correctly implemented latent model.
    Any non-trivial value indicates a bug in the analytical normal reconstruction
    or the U orthonormality check.

    U must already be on `device`.
    """
    model.eval()
    dx = x_test.shape[1]
    gen = torch.Generator()
    gen.manual_seed(eval_seed)
    total_mse = 0.0
    done = 0

    while done < n_pairs:
        B = min(batch_size, n_pairs - done)
        h_bar, t, xt = _sample_latent_batch(x_test, U, B, dx, eps, device, gen)

        v_hat = model(h_bar, t)                         # (B, d0)
        u_hat_tangent = v_hat @ U.T                     # (B, dx) — lies in col(U)
        theory_n = theoretical_normal(xt, t, U)         # (B, dx)
        u_hat = u_hat_tangent + theory_n                # (B, dx)

        pred_perp = proj_perp(u_hat, U)
        total_mse += ((pred_perp - theory_n) ** 2).sum(dim=1).mean().item() * B
        done += B

    return total_mse / n_pairs


# ---------------------------------------------------------------------------
# Scatter data for Figure 2A
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_latent_scatter_data(
    model: nn.Module,
    x_test: torch.Tensor,
    U: torch.Tensor,
    pis: torch.Tensor,
    mus: torch.Tensor,
    diag_Sigmas: torch.Tensor,
    device: torch.device,
    eps: float,
    n_pairs: int = 5_000,
    batch_size: int = 256,
    eval_seed: int = 44,
):
    """
    Collect per-coordinate velocity values for the Figure 2A scatter plot.

    Returns three flattened arrays (each of length n_pairs * d0):
        v_star_flat:   oracle tangent velocity coordinates
        v_hat_flat:    learned tangent velocity coordinates
        coord_idx_flat: integer coordinate index (0 .. d0-1)
    """
    model.eval()
    d0 = U.shape[1]
    dx = x_test.shape[1]
    gen = torch.Generator()
    gen.manual_seed(eval_seed)
    v_star_list, v_hat_list, coord_list = [], [], []
    done = 0

    while done < n_pairs:
        B = min(batch_size, n_pairs - done)
        h_bar, t, _ = _sample_latent_batch(x_test, U, B, dx, eps, device, gen)

        v_hat = model(h_bar, t)                                           # (B, d0)
        v_star = oracle_tangent_latent(h_bar, t, pis, mus, diag_Sigmas)  # (B, d0)

        v_star_list.append(v_star.cpu())
        v_hat_list.append(v_hat.cpu())
        # Coordinate indices: repeat 0..d0-1 for each point in batch
        coord_list.append(
            torch.arange(d0).unsqueeze(0).expand(B, -1)  # (B, d0)
        )
        done += B

    v_star_all = torch.cat(v_star_list, dim=0)   # (n_pairs, d0)
    v_hat_all  = torch.cat(v_hat_list,  dim=0)   # (n_pairs, d0)
    coord_all  = torch.cat(coord_list,  dim=0)   # (n_pairs, d0)

    return (
        v_star_all.numpy().flatten(),
        v_hat_all.numpy().flatten(),
        coord_all.numpy().flatten(),
    )


# ---------------------------------------------------------------------------
# Latent ODE generation for Figure 2C
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_latent_samples(
    model: nn.Module,
    d0: int,
    device: torch.device,
    n_samples: int = 2_000,
    n_steps: int = 100,
    t_start: float = 0.01,
    t_end: float = 0.99,
    eval_seed: int = 45,
) -> np.ndarray:
    """
    Generate latent endpoint samples by integrating the learned ODE in R^{d0}:
        dh/dt = v_hat(h, t)

    Starts from h0 ~ N(0, I_{d0}) at t=t_start, uses Euler integration.

    Returns:
        h_generated: (n_samples, d0) numpy array of latent endpoints at t=t_end
    """
    model.eval()
    gen = torch.Generator()
    gen.manual_seed(eval_seed)
    h = torch.randn(n_samples, d0, generator=gen).to(device)

    times = np.linspace(t_start, t_end, n_steps + 1)
    dt = (t_end - t_start) / n_steps

    for i in range(n_steps):
        t_val = float(times[i])
        t_tensor = torch.full((n_samples, 1), t_val, device=device)
        v = model(h, t_tensor)
        h = h + dt * v

    return h.cpu().numpy()


# ---------------------------------------------------------------------------
# Sliced Wasserstein distance for Figure 2C
# ---------------------------------------------------------------------------

def compute_sliced_wasserstein(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_projections: int = 500,
    seed: int = 0,
) -> float:
    """
    Sliced Wasserstein distance between two sets of samples.

    Uses n_projections random directions on the unit sphere, generated
    once with a fixed seed. Projection directions are not resampled per call.

    Args:
        samples_a: (N, d) array
        samples_b: (M, d) array  (must have same d; N and M can differ)
        n_projections: number of random projection directions
        seed: RNG seed for projection directions

    Returns:
        SWD: mean W1 distance averaged over projections (lower = more similar)
    """
    assert samples_a.shape[1] == samples_b.shape[1], "samples must have same dimension"
    d = samples_a.shape[1]
    Na, Nb = samples_a.shape[0], samples_b.shape[0]

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_projections, d))
    directions = raw / np.linalg.norm(raw, axis=1, keepdims=True)  # (P, d)

    pa = samples_a @ directions.T  # (Na, P)
    pb = samples_b @ directions.T  # (Nb, P)
    pa.sort(axis=0)
    pb.sort(axis=0)

    if Na != Nb:
        # Interpolate the smaller set to match the larger
        x_a = np.linspace(0, 1, Na)
        x_b = np.linspace(0, 1, Nb)
        x_common = np.linspace(0, 1, max(Na, Nb))
        pa_interp = np.stack(
            [np.interp(x_common, x_a, pa[:, j]) for j in range(n_projections)], axis=1
        )
        pb_interp = np.stack(
            [np.interp(x_common, x_b, pb[:, j]) for j in range(n_projections)], axis=1
        )
        return float(np.mean(np.abs(pa_interp - pb_interp)))

    return float(np.mean(np.abs(pa - pb)))

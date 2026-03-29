"""
oracle.py — oracle marginal velocity and latent GMM score for the
flow-manifold experiment.

All functions follow the row-vector convention used throughout the codebase:
    y = x @ U    corresponds to  U^T x  in math notation
    parallel @ U.T  maps latent vectors back to ambient space
"""

import torch


# ---------------------------------------------------------------------------
# Latent GMM score
# ---------------------------------------------------------------------------

def score_latent_gmm(
    y: torch.Tensor,
    t: torch.Tensor,
    pis: torch.Tensor,
    mus: torch.Tensor,
    diag_Sigmas: torch.Tensor,
) -> torch.Tensor:
    """
    Score  nabla_y log p_t^h(y)  of the time-marginal latent GMM.

    Under the linear path X_t = t X_1 + (1-t) X_0 with X_0 ~ N(0,I_{dx}),
    the projected latent marginal  h_bar_t = U^T X_t  is a GMM with:
        m_m(t)  = t * mu_m
        S_m(t)  = t^2 * Sigma_m + (1-t)^2 * I_{d0}   (diagonal)

    The score is:
        nabla_y log p_t^h(y) = sum_m  r_m(y,t) * [-S_m(t)^{-1} (y - m_m(t))]

    Responsibilities are computed via numerically stable log-sum-exp.

    Args:
        y:           (B, d0)  latent points (= x @ U in ambient coords)
        t:           (B, 1)   time values in (0, 1)
        pis:         (M,)     mixture weights
        mus:         (M, d0)  component means
        diag_Sigmas: (M, d0)  diagonal variance entries of the data-space GMM

    Returns:
        score: (B, d0)
    """
    t_sq = t ** 2                    # (B, 1)
    one_minus_t_sq = (1.0 - t) ** 2 # (B, 1)

    # Time-dependent diagonal variances  S_m(t)_jj = t^2 * sigma_mj + (1-t)^2
    # Shape: (B, M, d0)
    S_diag = (
        t_sq.unsqueeze(1) * diag_Sigmas.unsqueeze(0)
        + one_minus_t_sq.unsqueeze(1)
    )

    # Time-dependent means  m_m(t) = t * mu_m;  shape: (B, M, d0)
    m_t = t.unsqueeze(1) * mus.unsqueeze(0)

    # Residuals  (y - m_m(t));  shape: (B, M, d0)
    diff = y.unsqueeze(1) - m_t

    # Diagonal Mahalanobis distance  (B, M)
    mahal = (diff ** 2 / S_diag).sum(dim=-1)

    # Log determinant  log|S_m(t)| = sum_j log S_m(t)_jj;  shape: (B, M)
    log_det = S_diag.log().sum(dim=-1)

    # Unnormalized log responsibilities (constant 0.5*d0*log(2pi) cancels)
    log_resp = -0.5 * mahal - 0.5 * log_det + pis.log().unsqueeze(0)  # (B, M)

    # Normalise via log-sum-exp
    log_Z = torch.logsumexp(log_resp, dim=1, keepdim=True)  # (B, 1)
    resps = torch.exp(log_resp - log_Z)                     # (B, M)

    # Per-component score  -S_m(t)^{-1} (y - m_m(t)) = -diff / S_diag
    score_m = -diff / S_diag   # (B, M, d0)

    # Mixture score
    score = (resps.unsqueeze(-1) * score_m).sum(dim=1)  # (B, d0)
    return score


# ---------------------------------------------------------------------------
# Oracle marginal velocity
# ---------------------------------------------------------------------------

def oracle_velocity(
    x: torch.Tensor,
    t: torch.Tensor,
    U: torch.Tensor,
    pis: torch.Tensor,
    mus: torch.Tensor,
    diag_Sigmas: torch.Tensor,
) -> torch.Tensor:
    """
    Oracle marginal velocity for the linear flow-matching path.

    Formula (Section 5 of the handoff):
        u*_t(x) = U [ (1/t) h_bar + ((1-t)/t) nabla log p_t^h(h_bar) ]
                  - (1/(1-t)) (I - UU^T) x
    where h_bar = U^T x.

    All tensors must be on the same device.

    Args:
        x:           (B, dx)
        t:           (B, 1)   clipped away from 0 and 1
        U:           (dx, d0)
        pis:         (M,)
        mus:         (M, d0)
        diag_Sigmas: (M, d0)

    Returns:
        velocity: (B, dx)
    """
    y = x @ U   # (B, d0)   ≡  U^T x

    score = score_latent_gmm(y, t, pis, mus, diag_Sigmas)  # (B, d0)

    # Tangent (on-subspace) component
    parallel_latent = (1.0 / t) * y + ((1.0 - t) / t) * score  # (B, d0)
    parallel = parallel_latent @ U.T                              # (B, dx)

    # Normal (off-subspace) component
    x_perp = proj_perp(x, U)                    # (B, dx)
    normal = -(1.0 / (1.0 - t)) * x_perp        # (B, dx)

    return parallel + normal


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def proj_parallel(x: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Project x onto col(U).  Returns (B, dx)."""
    return (x @ U) @ U.T


def proj_perp(x: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Project x onto the orthogonal complement of col(U).  Returns (B, dx)."""
    return x - proj_parallel(x, U)


def theoretical_normal(x: torch.Tensor, t: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Theoretical normal component of the oracle velocity:
        kappa_t (I - UU^T) x  =  -(1/(1-t)) x_perp

    Args:
        x: (B, dx)
        t: (B, 1)
        U: (dx, d0)

    Returns:
        (B, dx)
    """
    return -(1.0 / (1.0 - t)) * proj_perp(x, U)


# ---------------------------------------------------------------------------
# Latent oracle tangent velocity
# ---------------------------------------------------------------------------

def oracle_tangent_latent(
    h_bar: torch.Tensor,
    t: torch.Tensor,
    pis: torch.Tensor,
    mus: torch.Tensor,
    diag_Sigmas: torch.Tensor,
) -> torch.Tensor:
    """
    Oracle tangent velocity in latent space.

    This is the d0-dimensional quantity the latent model is trained to predict:
        v*(h_bar, t) = (1/t) * h_bar + ((1-t)/t) * nabla_{h_bar} log p_t^h(h_bar)

    This is the latent component of the full oracle ambient velocity:
        u*(x_t, t) = v*(h_bar, t) @ U^T  +  analytical_normal(x_t, t, U)

    Args:
        h_bar:       (B, d0)  latent coordinates (= x_t @ U)
        t:           (B, 1)   time values in (0, 1), clipped away from boundaries
        pis:         (M,)     mixture weights
        mus:         (M, d0)  component means
        diag_Sigmas: (M, d0)  diagonal variance entries

    Returns:
        v_star: (B, d0)
    """
    score = score_latent_gmm(h_bar, t, pis, mus, diag_Sigmas)  # (B, d0)
    return (1.0 / t) * h_bar + ((1.0 - t) / t) * score

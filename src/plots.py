"""
plots.py — Figure 1 and Figure 2 for the oracle-U latent ablation experiment.

Both functions are called from scripts/make_figures.py after all runs finish.
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Figure 1: intrinsic-vs-ambient scaling
# ---------------------------------------------------------------------------

def plot_figure1(results_csv: str, output_dir: str) -> None:
    """
    Two-panel figure showing per-dimension tangent oracle MSE vs n_train.

    Panel A: d0=4 fixed, one curve per dx in {32, 128, 512}
             — oracle latent invariance check (curves should overlap)
    Panel B: dx=128 fixed, one curve per d0 in {2, 4, 8, 16}
             — intrinsic-dimension dependence

    Primary y-axis: tangent_oracle_mse_per_dim = total_tangent_MSE / d0
    Total tangent MSE is shown in a small inset (Panel B only).

    Mean +/- std over 3 seeds.
    """
    df = pd.read_csv(results_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Support both old (oracle_mse) and new (tangent_oracle_mse_total) column names
    if "tangent_oracle_mse_per_dim" in df.columns:
        per_dim_col = "tangent_oracle_mse_per_dim"
        total_col   = "tangent_oracle_mse_total"
    else:
        # Legacy fallback: old column names (oracle_mse / dx)
        per_dim_col = None
        total_col   = "oracle_mse"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Panel A: fix d0=4, vary dx ----
    ax = axes[0]
    sub_a = df[df["d0"] == 4].copy()
    dx_vals = sorted(sub_a["dx"].unique())
    cmap_a = plt.cm.viridis(np.linspace(0.15, 0.85, len(dx_vals)))

    for dx_val, color in zip(dx_vals, cmap_a):
        rows = sub_a[sub_a["dx"] == dx_val].copy()
        if per_dim_col is not None:
            rows["_y"] = rows[per_dim_col]
        else:
            rows["_y"] = rows[total_col] / 4.0   # legacy: d0=4 is fixed
        grp = (
            rows
            .groupby("n_train")["_y"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("n_train")
        )
        ax.errorbar(
            grp["n_train"], grp["mean"], yerr=grp["std"],
            label=f"$d_x={dx_val}$", color=color, marker="o",
            capsize=3, linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size $n$", fontsize=12)
    ax.set_ylabel("Per-dim tangent oracle MSE  $(1/d_0)\\|\\hat{v} - v^*\\|^2$", fontsize=11)
    ax.set_title("(A)  Vary ambient dim  ($d_0 = 4$ fixed)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)

    # ---- Panel B: fix dx=128, vary d0 ----
    ax = axes[1]
    sub_b = df[df["dx"] == 128].copy()
    d0_vals = sorted(sub_b["d0"].unique())
    cmap_b = plt.cm.plasma(np.linspace(0.15, 0.85, len(d0_vals)))

    for d0_val, color in zip(d0_vals, cmap_b):
        rows = sub_b[sub_b["d0"] == d0_val].copy()
        if per_dim_col is not None:
            rows["_y"] = rows[per_dim_col]
        else:
            rows["_y"] = rows[total_col] / d0_val
        grp = (
            rows
            .groupby("n_train")["_y"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("n_train")
        )
        ax.errorbar(
            grp["n_train"], grp["mean"], yerr=grp["std"],
            label=f"$d_0={d0_val}$", color=color, marker="o",
            capsize=3, linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size $n$", fontsize=12)
    ax.set_ylabel("Per-dim tangent oracle MSE  $(1/d_0)\\|\\hat{v} - v^*\\|^2$", fontsize=11)
    ax.set_title("(B)  Vary intrinsic dim  ($d_x = 128$ fixed)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(
        0.98, 0.03,
        "Note: per-dim MSE; total MSE scales linearly with $d_0$",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        color="gray",
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"figure1.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved to {output_dir}/figure1.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 2: tangent component diagnostics
# ---------------------------------------------------------------------------

def plot_figure2(
    v_star_flat: np.ndarray,
    v_hat_flat: np.ndarray,
    coord_idx_flat: np.ndarray,
    oracle_mse_history: list,
    train_loss_history: list,
    latent_generated: np.ndarray,
    latent_reference: np.ndarray,
    swd: float,
    d0: int,
    run_label: str,
    output_dir: str,
) -> None:
    """
    Three-panel diagnostic figure for the oracle-U latent ablation.

    (A) Scatter: oracle vs learned latent velocity, colored by coordinate index.
        Coordinate-specific failure modes are visible as per-color deviations.

    (B) Convergence: per-dim tangent oracle MSE and training loss vs epoch.

    (C) Latent endpoint distribution: 2D PCA scatter of generated vs reference
        samples, plus SWD scalar metric.

    Args:
        v_star_flat:       (N*d0,) oracle tangent velocity coordinates
        v_hat_flat:        (N*d0,) learned tangent velocity coordinates
        coord_idx_flat:    (N*d0,) int, coordinate index 0..d0-1
        oracle_mse_history: list of {"epoch": int, "oracle_mse": float}
        train_loss_history: list of float (one per epoch)
        latent_generated:  (n_samples, d0) generated latent samples at t~1
        latent_reference:  (n_samples, d0) reference GMM samples
        swd:               sliced Wasserstein distance (scalar)
        d0:                latent dimension
        run_label:         string identifier for title/filename
        output_dir:        directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- (A) Velocity scatter colored by coordinate ----
    ax = axes[0]
    n_show = min(8_000, len(v_star_flat))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(v_star_flat), n_show, replace=False)

    cmap_coords = plt.cm.tab10
    unique_coords = np.arange(d0)
    for ci in unique_coords:
        mask = (coord_idx_flat[idx] == ci)
        if not mask.any():
            continue
        ax.scatter(
            v_star_flat[idx][mask], v_hat_flat[idx][mask],
            alpha=0.25, s=2.0, rasterized=True,
            color=cmap_coords(ci / max(d0, 10)),
            label=f"coord {ci}",
        )

    # Per-coordinate MSE breakdown (text annotation)
    coord_mse_lines = []
    for ci in unique_coords:
        mask = (coord_idx_flat == ci)
        if mask.any():
            mse_ci = float(np.mean((v_star_flat[mask] - v_hat_flat[mask]) ** 2))
            coord_mse_lines.append(f"c{ci}: {mse_ci:.3f}")
    ax.text(
        0.02, 0.98, "\n".join(coord_mse_lines),
        transform=ax.transAxes, fontsize=7, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )

    lim = np.abs(v_star_flat[idx]).max() * 1.15
    lim = max(lim, np.abs(v_hat_flat[idx]).max() * 1.15)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0, alpha=0.6, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Oracle $v^*(h, t)$", fontsize=11)
    ax.set_ylabel("Learned $\\hat{v}(h, t)$", fontsize=11)
    ax.set_title(f"(A) Latent velocity scatter\n{run_label}", fontsize=11)
    if d0 <= 10:
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # ---- (B) Training convergence ----
    ax = axes[1]
    epochs_eval = [e["epoch"] for e in oracle_mse_history]
    mse_vals    = [e["oracle_mse"] for e in oracle_mse_history]

    ax.plot(epochs_eval, mse_vals, "b-o", markersize=3, linewidth=1.5,
            label="Oracle MSE (per-dim)")

    if train_loss_history:
        epoch_range = range(1, len(train_loss_history) + 1)
        ax_r = ax.twinx()
        ax_r.plot(list(epoch_range), train_loss_history, "r--", linewidth=1.0,
                  alpha=0.6, label="Train loss")
        ax_r.set_ylabel("Training loss", fontsize=10, color="red")
        ax_r.tick_params(axis="y", labelcolor="red")
        ax_r.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Per-dim tangent oracle MSE", fontsize=10)
    ax.set_title(f"(B) Convergence\n{run_label}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    # ---- (C) Latent endpoint distribution ----
    ax = axes[2]
    # Joint PCA on union so neither distribution is privileged
    combined = np.concatenate([latent_generated, latent_reference], axis=0)
    if combined.shape[1] > 2:
        pca = PCA(n_components=2)
        pca.fit(combined)
        gen_2d = pca.transform(latent_generated)
        ref_2d = pca.transform(latent_reference)
        xlabel = "PCA dim 1"
        ylabel = "PCA dim 2"
    else:
        gen_2d = latent_generated
        ref_2d = latent_reference
        xlabel = "Latent dim 1"
        ylabel = "Latent dim 2"

    n_show_c = min(1500, len(gen_2d), len(ref_2d))
    ax.scatter(ref_2d[:n_show_c, 0], ref_2d[:n_show_c, 1],
               alpha=0.3, s=3.0, color="steelblue", label="Reference GMM",
               rasterized=True)
    ax.scatter(gen_2d[:n_show_c, 0], gen_2d[:n_show_c, 1],
               alpha=0.3, s=3.0, color="darkorange", label="Generated",
               rasterized=True)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"(C) Latent endpoint distribution\n{run_label}\nSWD = {swd:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    suffix = run_label.replace("/", "_")
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"figure2_{suffix}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved to {output_dir}/figure2_{suffix}.{{png,pdf}}")

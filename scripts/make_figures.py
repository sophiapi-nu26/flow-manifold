#!/usr/bin/env python
"""
make_figures.py — generate Figure 1 and Figure 2 after the sweep is complete.

Usage:
    python scripts/make_figures.py \
        --sweep_dir experiments/flow_manifold_synth \
        --figures_dir experiments/flow_manifold_synth/figures

Figure 1 requires results.csv (produced by aggregate_results.py).
Figure 2 requires diagnostic arrays saved by run_one_config.py for the
representative setting (run_diagnostics: true in config).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plots import plot_figure1, plot_figure2


# Representative run for Figure 2
PRIMARY_DIAG = "dx128_d04_n50000_seed0"


def load_diag_arrays(run_dir: str):
    """Load Figure-2 diagnostic arrays saved by run_one_config.py."""
    v_star_flat    = np.load(os.path.join(run_dir, "latent_scatter_theory.npy"))
    v_hat_flat     = np.load(os.path.join(run_dir, "latent_scatter_learned.npy"))
    coord_idx_flat = np.load(os.path.join(run_dir, "latent_scatter_coord_idx.npy"))
    latent_generated = np.load(os.path.join(run_dir, "latent_generated.npy"))
    latent_reference = np.load(os.path.join(run_dir, "latent_reference.npy"))

    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    oracle_mse_history = metrics.get("oracle_mse_history", [])
    train_loss_history = metrics.get("train_loss", [])

    with open(os.path.join(run_dir, "diagnostics.json")) as f:
        diag = json.load(f)
    swd = diag["swd"]
    d0  = diag["d0"]

    return (
        v_star_flat, v_hat_flat, coord_idx_flat,
        oracle_mse_history, train_loss_history,
        latent_generated, latent_reference,
        swd, d0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_dir", default="experiments/flow_manifold_synth",
    )
    parser.add_argument(
        "--figures_dir", default=None,
        help="Output dir for figures (default: sweep_dir/figures)",
    )
    args = parser.parse_args()

    if args.figures_dir is None:
        args.figures_dir = os.path.join(args.sweep_dir, "figures")
    os.makedirs(args.figures_dir, exist_ok=True)

    # ---- Figure 1 ----
    results_csv = os.path.join(args.sweep_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"results.csv not found at {results_csv}")
        print("Run:  python scripts/aggregate_results.py  first.")
    else:
        plot_figure1(results_csv, args.figures_dir)

    # ---- Figure 2 ----
    primary_dir = os.path.join(args.sweep_dir, PRIMARY_DIAG)
    scatter_path = os.path.join(primary_dir, "latent_scatter_theory.npy")

    if not os.path.exists(scatter_path):
        print(f"Diagnostic arrays not found for {PRIMARY_DIAG}.")
        print("Make sure run_diagnostics=true was set for that config and the run completed.")
    else:
        (
            v_star_flat, v_hat_flat, coord_idx_flat,
            oracle_mse_history, train_loss_history,
            latent_generated, latent_reference,
            swd, d0,
        ) = load_diag_arrays(primary_dir)

        plot_figure2(
            v_star_flat, v_hat_flat, coord_idx_flat,
            oracle_mse_history, train_loss_history,
            latent_generated, latent_reference,
            swd, d0,
            run_label=PRIMARY_DIAG,
            output_dir=args.figures_dir,
        )

    print(f"Figures written to {args.figures_dir}")


if __name__ == "__main__":
    main()

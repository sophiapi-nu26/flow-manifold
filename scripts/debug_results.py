#!/usr/bin/env python
"""
debug_results.py — check whether results.csv is consistent with theory.

Usage:
    python scripts/debug_results.py \
        --results experiments/flow_manifold_synth/results.csv

Checks performed
----------------
1. Per-dimension oracle MSE across dx (Panel A, d0=4 fixed).
   Theory: should be roughly CONSTANT across dx values when d0 is fixed.

2. Decomposition into on-subspace and normal-component errors.
   oracle_mse = on_subspace_mse + normal_mse  (approximately, by Pythagoras).
   Theory: on-subspace error / dx should be ~constant across dx;
           normal-component error / (dx - d0) should be ~constant and small.

3. Null-model test: is normal_mse ≈ (dx - d0)?
   If the model outputs zero for the normal component, then
       normal_mse ≈ E[||-(1/(1-t))*(I-UU^T)*x_t||^2] ≈ (dx - d0).
   A ratio normal_mse / (dx - d0) near 1.0 means the model is NOT learning
   the normal component (capacity issue: hidden width < dx).

4. Per-dimension oracle MSE across d0 (Panel B, dx=128 fixed).
   Theory: should INCREASE with d0, and more steeply than Panel A varies with dx.
"""

import argparse

import numpy as np
import pandas as pd


SEP = "-" * 70


def mean_std(series):
    return series.mean(), series.std()


def summarise(df, groupby_col, vary_label, n_train_val, extra_filter=None):
    sub = df[df["n_train"] == n_train_val].copy()
    if extra_filter is not None:
        for k, v in extra_filter.items():
            sub = sub[sub[k] == v]

    sub["oracle_per_dim"]  = sub["oracle_mse"] / sub["dx"]
    sub["normal_per_dim"]  = sub["normal_mse"] / (sub["dx"] - sub["d0"]).clip(lower=1)
    sub["onsubspace_mse"]  = (sub["oracle_mse"] - sub["normal_mse"]).clip(lower=0)
    sub["onsubspace_per_dim"] = sub["onsubspace_mse"] / sub["d0"]
    # Null-model ratio: 1.0 means model outputs zero for normal component
    sub["null_ratio"] = sub["normal_mse"] / (sub["dx"] - sub["d0"]).clip(lower=1)

    grp = (
        sub.groupby(groupby_col)
        .agg(
            oracle_mean=("oracle_mse", "mean"),
            oracle_per_dim_mean=("oracle_per_dim", "mean"),
            normal_per_dim_mean=("normal_per_dim", "mean"),
            onsubspace_per_dim_mean=("onsubspace_per_dim", "mean"),
            null_ratio_mean=("null_ratio", "mean"),
        )
        .reset_index()
        .sort_values(groupby_col)
    )
    return grp


def check_panel_a(df):
    print(SEP)
    print("CHECK 1 — Panel A: fix d0=4, vary dx")
    print("Theory: oracle_mse/dx should be roughly CONSTANT across dx.")
    print()

    for n in sorted(df["n_train"].unique()):
        grp = summarise(df, "dx", "dx", n, extra_filter={"d0": 4})
        print(f"  n_train = {n}")
        print(f"  {'dx':>6}  {'oracle_mse':>12}  {'oracle/dx':>10}  "
              f"{'onsubspace/d0':>14}  {'normal/(dx-d0)':>15}  {'null_ratio':>10}")
        for _, row in grp.iterrows():
            print(f"  {int(row['dx']):>6}  {row['oracle_mean']:>12.3f}  "
                  f"{row['oracle_per_dim_mean']:>10.4f}  "
                  f"{row['onsubspace_per_dim_mean']:>14.4f}  "
                  f"{row['normal_per_dim_mean']:>15.4f}  "
                  f"{row['null_ratio_mean']:>10.3f}")
        print()

    print("  null_ratio near 1.0 => model outputs ~zero for normal component")
    print("  (likely cause: model hidden width < dx, insufficient capacity)")


def check_panel_b(df):
    print(SEP)
    print("CHECK 2 — Panel B: fix dx=128, vary d0")
    print("Theory: oracle_mse/dx should INCREASE with d0 (more than Panel A varies).")
    print()

    for n in sorted(df["n_train"].unique()):
        grp = summarise(df, "d0", "d0", n, extra_filter={"dx": 128})
        print(f"  n_train = {n}")
        print(f"  {'d0':>4}  {'oracle_mse':>12}  {'oracle/dx':>10}  "
              f"{'onsubspace/d0':>14}  {'normal/(dx-d0)':>15}  {'null_ratio':>10}")
        for _, row in grp.iterrows():
            print(f"  {int(row['d0']):>4}  {row['oracle_mean']:>12.3f}  "
                  f"{row['oracle_per_dim_mean']:>10.4f}  "
                  f"{row['onsubspace_per_dim_mean']:>14.4f}  "
                  f"{row['normal_per_dim_mean']:>15.4f}  "
                  f"{row['null_ratio_mean']:>10.3f}")
        print()


def check_a_vs_b(df):
    print(SEP)
    print("CHECK 3 — Relative sensitivity: does d0 drive more variation than dx?")
    print("Theory: spread across d0 values >> spread across dx values (per-dim).")
    print()

    n = df["n_train"].max()
    sub = df[df["n_train"] == n].copy()
    sub["oracle_per_dim"] = sub["oracle_mse"] / sub["dx"]

    panel_a = sub[sub["d0"] == 4].groupby("dx")["oracle_per_dim"].mean()
    panel_b = sub[sub["dx"] == 128].groupby("d0")["oracle_per_dim"].mean()

    ratio_a = panel_a.max() / panel_a.min()
    ratio_b = panel_b.max() / panel_b.min()

    print(f"  n_train = {n} (largest dataset, best-trained models)")
    print(f"  Panel A (vary dx, d0=4):   max/min oracle_mse/dx = {ratio_a:.2f}x")
    print(f"  Panel B (vary d0, dx=128): max/min oracle_mse/dx = {ratio_b:.2f}x")
    print()
    if ratio_b > ratio_a:
        print("  OK: d0 drives more variation than dx (consistent with theory).")
    else:
        print("  WARNING: dx drives MORE variation than d0 — inconsistent with theory.")
        print("  Likely cause: model does not learn the normal component for large dx.")
        print("  Fix: make model hidden width scale with dx (e.g. max(256, dx)).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="experiments/flow_manifold_synth/results.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")
    print(f"dx values:      {sorted(df['dx'].unique())}")
    print(f"d0 values:      {sorted(df['d0'].unique())}")
    print(f"n_train values: {sorted(df['n_train'].unique())}")
    print(f"seeds:          {sorted(df['seed'].unique())}")

    check_panel_a(df)
    check_panel_b(df)
    check_a_vs_b(df)
    print(SEP)


if __name__ == "__main__":
    main()

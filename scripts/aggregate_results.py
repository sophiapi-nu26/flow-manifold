#!/usr/bin/env python
"""
aggregate_results.py — collect per-run result.json files into results.csv.

Run this after the sweep (or partial sweep) to produce a single CSV
that plots.py can read.

Usage:
    python scripts/aggregate_results.py \
        --sweep_dir experiments/flow_manifold_synth \
        --output    experiments/flow_manifold_synth/results.csv
"""

import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_dir", default="experiments/flow_manifold_synth",
        help="Root directory containing per-run subdirectories",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: sweep_dir/results.csv)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.sweep_dir, "results.csv")

    rows = []
    missing = []

    for run_name in sorted(os.listdir(args.sweep_dir)):
        run_dir = os.path.join(args.sweep_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        result_path = os.path.join(run_dir, "result.json")
        if not os.path.exists(result_path):
            missing.append(run_name)
            continue
        with open(result_path) as f:
            row = json.load(f)
        rows.append(row)

    if not rows:
        print("No result.json files found — nothing to aggregate.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Aggregated {len(rows)} runs → {args.output}")

    if missing:
        print(f"Missing result.json in {len(missing)} run dirs:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()

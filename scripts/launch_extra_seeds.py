#!/usr/bin/env python
"""
launch_extra_seeds.py — generate configs for additional training seeds (3 and 4)
to supplement the original 3-seed sweep.

This script reuses the existing panel_a_latent_data.pt file (already on the
cluster) and writes only the 36 new configs (2 extra seeds × 18 unique settings).

Usage:
  # Generate configs only
  python scripts/launch_extra_seeds.py --configs_dir configs --output_root experiments/flow_manifold_synth

  # Generate and print Slurm submission command
  python scripts/launch_extra_seeds.py --configs_dir configs --output_root experiments/flow_manifold_synth --slurm
"""

import argparse
import itertools
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.launch_sweep import (
    PANEL_A, PANEL_B, N_TRAIN_LIST, DEFAULTS,
    gmm_seed, geom_seed, data_seed,
    build_config,
)

EXTRA_SEEDS = [3, 4]


def generate_extra_configs(output_root: str, panel_a_data_path: str):
    seen = set()
    configs = []

    # Panel A: d0=4, vary dx
    for dx, n_train, seed in itertools.product(PANEL_A["dx_list"], N_TRAIN_LIST, EXTRA_SEEDS):
        d0 = PANEL_A["d0"]
        key = (dx, d0, n_train, seed)
        if key not in seen:
            seen.add(key)
            configs.append(build_config(dx, d0, n_train, seed, output_root,
                                        panel_a_data_path=panel_a_data_path))

    # Panel B: dx=128, vary d0
    for d0, n_train, seed in itertools.product(PANEL_B["d0_list"], N_TRAIN_LIST, EXTRA_SEEDS):
        dx = PANEL_B["dx"]
        key = (dx, d0, n_train, seed)
        if key not in seen:
            seen.add(key)
            # dx=128, d0=4 is Panel A — use shared data path
            if d0 == PANEL_A["d0"]:
                configs.append(build_config(dx, d0, n_train, seed, output_root,
                                            panel_a_data_path=panel_a_data_path))
            else:
                configs.append(build_config(dx, d0, n_train, seed, output_root))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="experiments/flow_manifold_synth")
    parser.add_argument("--configs_dir", default="configs")
    parser.add_argument("--slurm", action="store_true",
                        help="Print sbatch commands after writing configs")
    args = parser.parse_args()

    # panel_a_latent_data.pt must already exist (generated during original sweep)
    panel_a_data_path = os.path.abspath(
        os.path.join(args.configs_dir, "panel_a_latent_data.pt")
    )
    if not os.path.exists(panel_a_data_path):
        print(f"ERROR: {panel_a_data_path} not found.")
        print("Run launch_sweep.py first to generate it, then run this script.")
        return

    configs = generate_extra_configs(args.output_root, panel_a_data_path)
    print(f"Extra configs to generate: {len(configs)}")

    os.makedirs(args.configs_dir, exist_ok=True)
    config_paths = []
    for cfg in configs:
        path = os.path.join(args.configs_dir, f"{cfg['run_name']}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        config_paths.append(path)
        print(f"  {cfg['run_name']}")

    # Write a config list for this extra batch
    list_path = os.path.join(args.configs_dir, "extra_seeds_runs.txt")
    with open(list_path, "w") as f:
        for p in config_paths:
            f.write(p + "\n")
    print(f"\nConfig list written to {list_path}")

    if args.slurm:
        n = len(config_paths) - 1
        print(f"\nTo submit to Slurm (all {len(config_paths)} fit within the 50-job limit):")
        print(f"  sbatch --array=0-{n}%8 slurm/run_extra_seeds.sbatch")


if __name__ == "__main__":
    main()

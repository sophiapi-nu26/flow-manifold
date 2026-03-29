#!/usr/bin/env python
"""
launch_sweep.py — generate all sweep configs and optionally launch them.

Seed policy
-----------
  gmm_seed  = d0 * 1000 + 42               # fixed per d0; same GMM across all dx
  geom_seed = dx * 10000 + d0 * 100 + 17   # fixed per (dx, d0); determines U
  data_seed = dx * 10000 + d0 * 100 + 99   # fixed per (dx, d0); determines datasets
  train_seed = 0, 1, 2                      # varies across the 3 seeds per config

Panel A (d0=4, vary dx) uses a shared latent data file so that all three dx
values see the same latent GMM, the same training h-samples, and the same
evaluation (h_bar, t, v*) cache.  The shared file is written to
  configs/panel_a_latent_data.pt
and all Panel A configs include panel_a_data_path pointing to it.

(dx=128, d0=4) appears in both Panel A and Panel B — same config, same runs.
Total unique configs: 54  (27 + 36 - 9 overlap).

Usage:
  # Generate configs only (dry run)
  python scripts/launch_sweep.py --output_root experiments/flow_manifold_synth --dry_run

  # Generate configs and launch sequentially (local)
  python scripts/launch_sweep.py --output_root experiments/flow_manifold_synth --launch sequential

  # Generate configs and submit Slurm job array
  python scripts/launch_sweep.py --output_root experiments/flow_manifold_synth --launch slurm
"""

import argparse
import itertools
import os
import subprocess
import sys

import numpy as np
import torch
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import make_gmm_params
from src.evaluate import build_panel_a_eval_cache, _sample_gmm_torch


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------

PANEL_A = {   # fix d0=4, vary dx
    "d0": 4,
    "dx_list": [32, 128, 512],
}

PANEL_B = {   # fix dx=128, vary d0
    "dx": 128,
    "d0_list": [2, 4, 8, 16],
}

N_TRAIN_LIST = [2_000, 10_000, 50_000]
SEEDS = [0, 1, 2]

# Representative settings for Figure-2 diagnostics
DIAGNOSTIC_RUNS = {
    ("dx128", "d04", "n50000", "seed0"),
}

# Default hyper-parameters (shared across all runs)
DEFAULTS = {
    "M":              4,
    "n_train_master": 50_000,
    "n_val":          2_000,
    "n_test":         5_000,
    "time_emb_dim":   64,
    "hidden_width":   256,
    "hidden_layers":  4,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "batch_size":     256,
    "epochs":         150,
    "eps":            0.01,
    "grad_clip":      1.0,
    "eval_every":     10,
    "n_eval_pairs":   10_000,
    "eval_seed":      42,
}


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def gmm_seed(d0: int) -> int:
    return d0 * 1000 + 42

def geom_seed(dx: int, d0: int) -> int:
    return dx * 10_000 + d0 * 100 + 17

def data_seed(dx: int, d0: int) -> int:
    return dx * 10_000 + d0 * 100 + 99


# ---------------------------------------------------------------------------
# Panel A shared data generation
# ---------------------------------------------------------------------------

def build_panel_a_data(configs_dir: str, dry_run: bool = False) -> str:
    """
    Generate and save the Panel A shared latent data file.

    Contains:
      - pis, mus, diag_Sigmas: GMM params for d0=4 (gmm_seed=4042)
      - h_train_master:  (50000, 4) — shared training latent samples
      - h_val:           (2000,  4) — shared validation latent samples
      - h_test:          (5000,  4) — shared test latent samples
      - eval_cache:      {h_bar, t, v_star} — pre-computed Panel A eval tuples
      - metadata:        version/provenance info

    The training samples use data_seed(128, 4) so they exactly match the
    dx=128, d0=4 runs (which are shared between Panel A and Panel B).

    Returns the path to the saved file.
    """
    d0       = PANEL_A["d0"]    # 4
    M        = DEFAULTS["M"]    # 4
    eps      = DEFAULTS["eps"]  # 0.01
    eval_seed_val = DEFAULTS["eval_seed"]   # 42
    g_seed   = gmm_seed(d0)     # 4042
    d_seed   = data_seed(128, d0)   # data_seed(128,4) = 1280499

    panel_a_path = os.path.abspath(os.path.join(configs_dir, "panel_a_latent_data.pt"))

    if dry_run:
        print(f"  [dry_run] Would write Panel A data to {panel_a_path}")
        return panel_a_path

    print(f"  Building Panel A shared latent data (d0={d0}, gmm_seed={g_seed}) ...")

    pis, mus, diag_Sigmas = make_gmm_params(d0, g_seed, M=M)

    # Sample latent training/val/test data using same seed as dx=128, d0=4
    rng = np.random.RandomState(d_seed)
    pis_np = pis.numpy()
    mus_np = mus.numpy()
    diag_Sigmas_np = diag_Sigmas.numpy()
    M_val = pis_np.shape[0]

    def _sample_gmm_np(n):
        comp = rng.choice(M_val, size=n, p=pis_np)
        out = np.zeros((n, d0), dtype=np.float32)
        for m in range(M_val):
            idx = np.where(comp == m)[0]
            if len(idx) == 0:
                continue
            std_m = np.sqrt(diag_Sigmas_np[m])
            z = rng.randn(len(idx), d0).astype(np.float32)
            out[idx] = mus_np[m] + z * std_m
        return out

    n_train_master = DEFAULTS["n_train_master"]
    n_val          = DEFAULTS["n_val"]
    n_test         = DEFAULTS["n_test"]
    total          = n_train_master + n_val + n_test

    h_all         = _sample_gmm_np(total)
    h_train_master = torch.from_numpy(h_all[:n_train_master])
    h_val          = torch.from_numpy(h_all[n_train_master : n_train_master + n_val])
    h_test         = torch.from_numpy(h_all[n_train_master + n_val :])

    # Build eval cache
    eval_cache_dict = build_panel_a_eval_cache(
        pis, mus, diag_Sigmas, d0,
        n_points=DEFAULTS["n_eval_pairs"],
        eval_seed=eval_seed_val,
        eps=eps,
        gmm_seed=g_seed,
        save_path=None,   # we'll embed it in the main file below
        version="1.0",
    )

    panel_data = {
        "pis":             pis,
        "mus":             mus,
        "diag_Sigmas":     diag_Sigmas,
        "h_train_master":  h_train_master,
        "h_val":           h_val,
        "h_test":          h_test,
        "eval_cache":      eval_cache_dict,
        "metadata": {
            "d0":              d0,
            "gmm_seed":        g_seed,
            "data_seed":       d_seed,
            "eval_seed":       eval_seed_val,
            "n_train_master":  n_train_master,
            "n_val":           n_val,
            "n_test":          n_test,
            "n_eval_cache":    DEFAULTS["n_eval_pairs"],
            "eps":             eps,
            "version":         "1.0",
        },
    }
    torch.save(panel_data, panel_a_path)
    print(f"  Panel A data saved to {panel_a_path}")
    return panel_a_path


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def make_run_name(dx: int, d0: int, n_train: int, seed: int) -> str:
    return f"dx{dx}_d0{d0}_n{n_train}_seed{seed}"


def build_config(
    dx: int,
    d0: int,
    n_train: int,
    seed: int,
    output_root: str,
    panel_a_data_path: str = None,
) -> dict:
    run_name = make_run_name(dx, d0, n_train, seed)

    key = (f"dx{dx}", f"d0{d0}", f"n{n_train}", f"seed{seed}")
    run_diagnostics = key in DIAGNOSTIC_RUNS

    cfg = {
        # Geometry
        "dx":        dx,
        "d0":        d0,
        # Seeds
        "gmm_seed":  gmm_seed(d0),
        "geom_seed": geom_seed(dx, d0),
        "data_seed": data_seed(dx, d0),
        "train_seed": seed,
        # Data
        "n_train":   n_train,
        # Diagnostics
        "run_diagnostics": run_diagnostics,
        # Output
        "output_root": output_root,
        "run_name":    run_name,
    }
    cfg.update(DEFAULTS)

    # Panel A configs use shared latent data
    if panel_a_data_path is not None:
        cfg["panel_a_data_path"] = panel_a_data_path

    return cfg


def generate_all_configs(output_root: str, panel_a_data_path: str = None):
    """Return a list of all unique (dx, d0, n_train, seed) configs."""
    seen = set()
    configs = []

    # Panel A: d0=4 fixed, vary dx — use shared latent data
    for dx, n_train, seed in itertools.product(PANEL_A["dx_list"], N_TRAIN_LIST, SEEDS):
        d0 = PANEL_A["d0"]
        key = (dx, d0, n_train, seed)
        if key not in seen:
            seen.add(key)
            configs.append(build_config(
                dx, d0, n_train, seed, output_root,
                panel_a_data_path=panel_a_data_path,
            ))

    # Panel B: dx=128 fixed, vary d0
    for d0, n_train, seed in itertools.product(PANEL_B["d0_list"], N_TRAIN_LIST, SEEDS):
        dx = PANEL_B["dx"]
        key = (dx, d0, n_train, seed)
        if key not in seen:
            seen.add(key)
            # (dx=128, d0=4) overlap runs are already included with panel_a_data_path
            configs.append(build_config(dx, d0, n_train, seed, output_root))

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate sweep configs and optionally launch.")
    parser.add_argument(
        "--output_root", default="experiments/flow_manifold_synth",
        help="Root directory for all run outputs",
    )
    parser.add_argument(
        "--configs_dir", default="configs",
        help="Directory to write generated YAML configs",
    )
    parser.add_argument(
        "--launch", choices=["none", "sequential", "slurm"], default="none",
        help="How to launch jobs after config generation",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print configs without writing files",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Pass --overwrite to run_one_config.py (re-run completed runs)",
    )
    args = parser.parse_args()

    if not args.dry_run:
        os.makedirs(args.configs_dir, exist_ok=True)

    # ---- Build Panel A shared latent data ----
    panel_a_data_path = build_panel_a_data(args.configs_dir, dry_run=args.dry_run)

    configs = generate_all_configs(args.output_root, panel_a_data_path=panel_a_data_path)
    print(f"Total unique configs: {len(configs)}")

    if args.dry_run:
        for cfg in configs:
            panel_flag = " [panel_a]" if cfg.get("panel_a_data_path") else ""
            print(f"  {cfg['run_name']}{panel_flag}")
        return

    # Write config files
    config_paths = []
    for cfg in configs:
        path = os.path.join(args.configs_dir, f"{cfg['run_name']}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        config_paths.append(path)

    # Write config list file (for Slurm array)
    list_path = os.path.join(args.configs_dir, "all_runs.txt")
    with open(list_path, "w") as f:
        for p in config_paths:
            f.write(p + "\n")
    print(f"Config list written to {list_path}")

    # Also write a debug config
    debug_cfg = build_config(32, 4, 2_000, 0, args.output_root,
                             panel_a_data_path=panel_a_data_path)
    debug_cfg["epochs"] = 5
    debug_cfg["eval_every"] = 1
    debug_cfg["n_eval_pairs"] = 200
    debug_cfg["run_diagnostics"] = True
    debug_cfg["run_name"] = "debug"
    debug_path = os.path.join(args.configs_dir, "debug.yaml")
    with open(debug_path, "w") as f:
        yaml.dump(debug_cfg, f, default_flow_style=False)
    print(f"Debug config written to {debug_path}")

    if args.launch == "none":
        print(f"\nConfigs written to {args.configs_dir}/")
        print("To run one config locally:")
        print(f"  python scripts/run_one_config.py --config {config_paths[0]}")
        print("\nTo submit to Slurm (babel, 50-job QOS limit):")
        n = len(config_paths)
        print(f"  sbatch --array=0-{min(49, n-1)}%8 slurm/run_array.sbatch")
        if n > 50:
            print(f"  # Then after queue drops below limit:")
            print(f"  sbatch --array=50-{n-1}%8 slurm/run_array.sbatch")
        return

    run_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_one_config.py"
    )
    overwrite_flag = ["--overwrite"] if args.overwrite else []

    if args.launch == "sequential":
        print(f"\nLaunching {len(config_paths)} runs sequentially ...")
        for i, path in enumerate(config_paths):
            print(f"\n=== Run {i+1}/{len(config_paths)}: {path} ===")
            ret = subprocess.run(
                [sys.executable, run_script, "--config", path] + overwrite_flag
            )
            if ret.returncode != 0:
                print(f"WARNING: run failed with code {ret.returncode}")

    elif args.launch == "slurm":
        slurm_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "slurm", "run_array.sbatch"
        )
        n = len(config_paths) - 1
        cmd = ["sbatch", f"--array=0-{n}%8", slurm_script]
        print(f"\nSubmitting Slurm array: {' '.join(cmd)}")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()

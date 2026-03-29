#!/usr/bin/env python
"""
run_one_config.py — run a single oracle-U latent ablation experiment from a YAML config.

Usage:
    python scripts/run_one_config.py --config configs/debug.yaml
    python scripts/run_one_config.py --config configs/dx128_d04_n50000_seed0.yaml
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import make_U, make_gmm_params, make_datasets
from src.evaluate import (
    compute_normal_mse,
    compute_tangent_oracle_mse,
    compute_tangent_oracle_mse_from_cache,
    compute_latent_scatter_data,
    generate_latent_samples,
    compute_sliced_wasserstein,
    _sample_gmm_torch,
)
from src.models import VelocityMLP
from src.train import make_dataloader, set_seed, train_epoch, val_epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device_from_cfg(cfg: dict) -> torch.device:
    if cfg.get("device"):
        return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_or_generate_geometry(cfg: dict, run_dir: str):
    """
    Generate U, GMM params, and all datasets for non-Panel-A runs.
    Geometry is determined entirely by (dx, d0, gmm_seed, geom_seed, data_seed).
    """
    dx = cfg["dx"]
    d0 = cfg["d0"]
    M = cfg.get("M", 4)

    U = make_U(dx, d0, cfg["geom_seed"])
    pis, mus, diag_Sigmas = make_gmm_params(d0, cfg["gmm_seed"], M=M)
    x_train_master, x_val, x_test = make_datasets(
        dx, d0, U, pis, mus, diag_Sigmas,
        data_seed=cfg["data_seed"],
        n_train_master=cfg.get("n_train_master", 50_000),
        n_val=cfg.get("n_val", 2_000),
        n_test=cfg.get("n_test", 5_000),
    )

    # Sanity-check: data lies in the subspace (||(I-UU^T)x|| ≈ 0)
    U_np = U.numpy()
    x_check = x_test[:100].numpy()
    proj = x_check @ U_np @ U_np.T
    residual = np.abs(x_check - proj).max()
    if residual > 1e-4:
        print(f"WARNING: subspace residual is large: {residual:.2e} — check U generation")

    return U, pis, mus, diag_Sigmas, x_train_master, x_val, x_test


def _load_panel_a_geometry(cfg: dict, run_dir: str):
    """
    Load shared latent geometry for Panel A runs.

    U is generated fresh for this dx (as usual), but pis/mus/diag_Sigmas and
    the latent samples h_train/h_val/h_test are loaded from the shared Panel A
    data file so that all dx values see the same latent distributions.
    """
    dx = cfg["dx"]
    d0 = cfg["d0"]
    panel_a_path = cfg["panel_a_data_path"]

    panel_data = torch.load(panel_a_path, weights_only=False)

    pis          = panel_data["pis"]
    mus          = panel_data["mus"]
    diag_Sigmas  = panel_data["diag_Sigmas"]
    h_train_master = panel_data["h_train_master"]   # (50000, d0)
    h_val          = panel_data["h_val"]            # (2000, d0)
    h_test         = panel_data["h_test"]           # (5000, d0)

    # Generate U specific to this (dx, d0) pair
    U = make_U(dx, d0, cfg["geom_seed"])

    # Embed latent samples to ambient space
    x_train_master = h_train_master @ U.T           # (50000, dx)
    x_val          = h_val          @ U.T           # (2000,  dx)
    x_test         = h_test         @ U.T           # (5000,  dx)

    # Save U for this run
    geom_path = os.path.join(run_dir, "geometry.pt")
    if not os.path.exists(geom_path):
        torch.save({"U": U}, geom_path)

    eval_cache = panel_data.get("eval_cache")
    return U, pis, mus, diag_Sigmas, x_train_master, x_val, x_test, eval_cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing run directory (default: skip if metrics.json exists)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ setup
    run_dir = os.path.join(cfg["output_root"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path) and not args.overwrite:
        print(f"Run already complete: {run_dir}  (use --overwrite to redo)")
        return

    # Save config copy
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    device = _device_from_cfg(cfg)
    print(f"[{cfg['run_name']}]  device={device}  dx={cfg['dx']}  d0={cfg['d0']}  "
          f"n_train={cfg['n_train']}  seed={cfg['train_seed']}")

    # --------------------------------------------------------- geometry / data
    is_panel_a = bool(cfg.get("panel_a_data_path"))
    eval_cache = None

    if is_panel_a:
        U, pis, mus, diag_Sigmas, x_train_master, x_val, x_test, eval_cache = \
            _load_panel_a_geometry(cfg, run_dir)
    else:
        U, pis, mus, diag_Sigmas, x_train_master, x_val, x_test = \
            _load_or_generate_geometry(cfg, run_dir)
        # Save geometry once (idempotent across seeds)
        geom_path = os.path.join(run_dir, "geometry.pt")
        if not os.path.exists(geom_path):
            torch.save(
                {"U": U, "pis": pis, "mus": mus, "diag_Sigmas": diag_Sigmas},
                geom_path,
            )

    n_train = cfg["n_train"]
    x_train = x_train_master[:n_train]   # nested prefix

    dx = cfg["dx"]
    d0 = cfg["d0"]

    # --------------------------------- move GMM/U to device
    U_dev           = U.to(device)
    pis_dev         = pis.to(device)
    mus_dev         = mus.to(device)
    diag_Sigmas_dev = diag_Sigmas.to(device)

    if eval_cache is not None:
        h_bar_cache  = eval_cache["h_bar"].to(device)
        t_cache      = eval_cache["t"].to(device)
        v_star_cache = eval_cache["v_star"].to(device)

    # -------------------------------------------- set training seed, build model
    set_seed(cfg["train_seed"])

    model = VelocityMLP(
        d0=d0,
        time_emb_dim=cfg.get("time_emb_dim", 64),
        hidden_width=cfg.get("hidden_width", 256),
        hidden_layers=cfg.get("hidden_layers", 4),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # ---------------------------------------------------- training hyper-params
    eps        = cfg.get("eps", 0.01)
    grad_clip  = cfg.get("grad_clip", 1.0)
    batch_size = cfg.get("batch_size", 256)
    epochs     = cfg.get("epochs", 150)
    eval_every = cfg.get("eval_every", 10)
    n_eval_pairs = cfg.get("n_eval_pairs", 10_000)
    eval_seed  = cfg.get("eval_seed", 42)

    train_loader = make_dataloader(x_train, batch_size, shuffle=True)
    val_loader   = make_dataloader(x_val,   batch_size, shuffle=False)

    # --------------------------------------------------------- training loop
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "oracle_mse_history": [],  # list of {"epoch": int, "oracle_mse": float}
    }

    t_wall_start = time.time()

    for epoch in range(epochs):
        tr_loss = train_epoch(
            model, train_loader, optimizer,
            dx, d0, U_dev, eps, grad_clip, device,
        )
        vl_loss = val_epoch(
            model, val_loader,
            dx, d0, U_dev, eps, device,
        )
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            if eval_cache is not None:
                per_dim, _ = compute_tangent_oracle_mse_from_cache(
                    model, h_bar_cache, t_cache, v_star_cache,
                    device, batch_size,
                )
            else:
                per_dim, _ = compute_tangent_oracle_mse(
                    model, x_test, U_dev, pis_dev, mus_dev, diag_Sigmas_dev,
                    device, eps, n_pairs=n_eval_pairs, batch_size=batch_size,
                    eval_seed=eval_seed,
                )
            history["oracle_mse_history"].append(
                {"epoch": epoch + 1, "oracle_mse": per_dim}
            )
            print(
                f"  epoch {epoch+1:>4}/{epochs}  "
                f"train={tr_loss:.5f}  val={vl_loss:.5f}  "
                f"oracle_mse_per_dim={per_dim:.5f}"
            )
        else:
            print(f"  epoch {epoch+1:>4}/{epochs}  train={tr_loss:.5f}  val={vl_loss:.5f}")

    wall_time = time.time() - t_wall_start

    # --------------------------------------------- final evaluation
    if eval_cache is not None:
        final_per_dim, final_total = compute_tangent_oracle_mse_from_cache(
            model, h_bar_cache, t_cache, v_star_cache,
            device, batch_size,
        )
    else:
        final_per_dim, final_total = compute_tangent_oracle_mse(
            model, x_test, U_dev, pis_dev, mus_dev, diag_Sigmas_dev,
            device, eps, n_pairs=n_eval_pairs, batch_size=batch_size,
            eval_seed=eval_seed,
        )

    final_normal_mse = compute_normal_mse(
        model, x_test, U_dev, device, eps,
        n_pairs=n_eval_pairs, batch_size=batch_size,
        eval_seed=eval_seed + 1,
    )

    history["final_tangent_oracle_mse_per_dim"] = final_per_dim
    history["final_tangent_oracle_mse_total"]   = final_total
    history["final_normal_mse"]                 = final_normal_mse
    history["wall_time_seconds"]                = wall_time

    # --------------------------------------------- save artifacts
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint.pt"))

    result_row = {
        "run_name":                      cfg["run_name"],
        "dx":                            dx,
        "d0":                            d0,
        "n_train":                       n_train,
        "seed":                          cfg["train_seed"],
        "tangent_oracle_mse_per_dim":    final_per_dim,
        "tangent_oracle_mse_total":      final_total,
        "normal_mse":                    final_normal_mse,
        "wall_time_seconds":             wall_time,
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result_row, f, indent=2)

    # ----------------------------------------- Figure-2 diagnostics (optional)
    if cfg.get("run_diagnostics", False):
        print("  Running Figure-2 diagnostics ...")

        # (A) Velocity scatter data
        v_star_flat, v_hat_flat, coord_idx_flat = compute_latent_scatter_data(
            model, x_test, U_dev, pis_dev, mus_dev, diag_Sigmas_dev,
            device, eps, n_pairs=5_000, batch_size=batch_size,
            eval_seed=eval_seed + 2,
        )
        np.save(os.path.join(run_dir, "latent_scatter_theory.npy"),    v_star_flat)
        np.save(os.path.join(run_dir, "latent_scatter_learned.npy"),   v_hat_flat)
        np.save(os.path.join(run_dir, "latent_scatter_coord_idx.npy"), coord_idx_flat)

        # (C) Latent endpoint distribution
        latent_generated = generate_latent_samples(
            model, d0, device, n_samples=2_000, n_steps=100,
            t_start=0.01, t_end=0.99, eval_seed=eval_seed + 3,
        )
        # Reference samples from the GMM
        gen_ref = torch.Generator()
        gen_ref.manual_seed(eval_seed + 4)
        latent_reference = _sample_gmm_torch(
            2_000, pis, mus, diag_Sigmas, gen_ref
        ).numpy()

        np.save(os.path.join(run_dir, "latent_generated.npy"),  latent_generated)
        np.save(os.path.join(run_dir, "latent_reference.npy"),  latent_reference)

        swd = compute_sliced_wasserstein(
            latent_generated, latent_reference, n_projections=500, seed=0
        )
        with open(os.path.join(run_dir, "diagnostics.json"), "w") as f:
            json.dump({"swd": swd, "d0": d0}, f, indent=2)

        print(f"  Diagnostic arrays saved.  SWD={swd:.4f}")

    print(
        f"  DONE  tangent_mse_per_dim={final_per_dim:.5f}  "
        f"normal_mse={final_normal_mse:.2e}  "
        f"time={wall_time:.0f}s\n"
        f"  Outputs: {run_dir}"
    )


if __name__ == "__main__":
    main()

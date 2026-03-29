# How to Run the Oracle-U Latent Ablation Experiment

This document covers the full workflow: generating configs, running a smoke test locally, submitting to the cluster, and producing figures.

---

## Overview

The experiment runs 54 jobs across two panels:

| Panel | Fixed | Varying | n_train | Seeds | Runs |
|---|---|---|---|---|---|
| A | d0=4 | dx ∈ {32, 128, 512} | {2k, 10k, 50k} | 3 | 27 |
| B | dx=128 | d0 ∈ {2, 4, 8, 16} | {2k, 10k, 50k} | 3 | 36 |

dx=128, d0=4 is shared between panels → **54 unique runs total**.

Panel A uses a shared latent data file (`configs/panel_a_latent_data.pt`) so all three dx values train on the same latent GMM and the same latent samples, making the ambient-dimension invariance comparison fully controlled.

---

## Step 1 — Set up the environment (cluster, once only)

```bash
conda create -n flowmanifold python=3.11 -y
conda activate flowmanifold
pip install -r requirements.txt
```

---

## Step 2 — Generate configs and Panel A shared data

Run this **once**, either locally or on the cluster. It writes:
- 54 YAML config files to `configs/`
- `configs/all_runs.txt` (one config path per line, read by the Slurm array)
- `configs/panel_a_latent_data.pt` (shared GMM + latent samples + eval cache for Panel A)
- `configs/debug.yaml` (5-epoch smoke test)

```bash
python scripts/launch_sweep.py \
    --output_root experiments/flow_manifold_synth \
    --configs_dir configs
```

To preview what will be generated without writing anything:

```bash
python scripts/launch_sweep.py --dry_run
```

---

## Step 3 — Smoke test (optional but recommended)

Run the debug config (5 epochs, dx=32, d0=4, n=2000) to verify everything works before submitting 54 jobs:

```bash
python scripts/run_one_config.py --config configs/debug.yaml
```

Expected output:
```
[debug]  device=cuda  dx=32  d0=4  n_train=2000  seed=0
  epoch    1/5  train=...  val=...  oracle_mse_per_dim=...
  ...
  DONE  tangent_mse_per_dim=...  normal_mse=...e-11  time=...s
```

Two things to verify:
- `oracle_mse_per_dim` decreases across the 5 epochs.
- `normal_mse` is very small (< 1e-5, typically ~1e-11). Any larger value indicates a bug.

---

## Step 4 — Transfer files to the cluster

From your local machine (Windows), use `scp`:

```bash
scp -r src scripts slurm configs requirements.txt sophiapi@babel:/home/sophiapi/flow-manifold/
```

**Important:** `configs/panel_a_latent_data.pt` must be included — it is referenced by absolute path in the Panel A YAML configs. If you regenerate configs on the cluster instead, the path baked into those YAMLs will be correct automatically.

If you generated configs locally and are copying them to the cluster, double-check that the `panel_a_data_path` key in any Panel A config points to the correct cluster path:

```bash
grep panel_a_data_path configs/dx32_d04_n2000_seed0.yaml
```

It should resolve to somewhere under `/home/sophiapi/flow-manifold/configs/`. If it shows your local Windows path, regenerate configs on the cluster instead:

```bash
# On the cluster
python scripts/launch_sweep.py \
    --output_root experiments/flow_manifold_synth \
    --configs_dir configs
```

---

## Step 5 — Submit the sweep (babel cluster)

Babel's QOS limit is 50 submitted jobs at a time. Submit in two batches:

```bash
mkdir -p logs

# First batch: jobs 0–49
sbatch --array=0-49%8 slurm/run_array.sbatch

# Check how many jobs are still in the queue
squeue -u $USER | wc -l    # subtract 1 for the header

# When that number is below 50, submit the remaining 4
sbatch --array=50-53%8 slurm/run_array.sbatch
```

The `%8` cap means at most 8 jobs run simultaneously. Each job takes 2–10 minutes on GPU depending on n_train and dx.

---

## Step 6 — Monitor progress

```bash
# Jobs currently in queue
squeue -u $USER

# How many runs have completed (target: 54)
find experiments/flow_manifold_synth -name result.json | wc -l

# Live-tail the output of a specific job
tail -f logs/flowarray_<JOBID>_<TASKID>.out

# Quick check of a completed run
cat experiments/flow_manifold_synth/dx128_d04_n50000_seed0/result.json
```

A completed result looks like:
```json
{
  "run_name": "dx128_d04_n50000_seed0",
  "dx": 128, "d0": 4, "n_train": 50000, "seed": 0,
  "tangent_oracle_mse_per_dim": ...,
  "tangent_oracle_mse_total": ...,
  "normal_mse": 1.4e-11,
  "wall_time_seconds": ...
}
```

---

## Step 7 — Re-run any failed jobs

Find which runs are missing a result:

```bash
python scripts/aggregate_results.py \
    --sweep_dir experiments/flow_manifold_synth
```

This prints the names of any run directories without a `result.json`. Re-run each one:

```bash
# Locally
python scripts/run_one_config.py --config configs/<run_name>.yaml --overwrite

# Or as a single Slurm job
sbatch slurm/run_single.sbatch configs/<run_name>.yaml
```

---

## Step 8 — Aggregate results and generate figures

Once all 54 runs are complete:

```bash
# Collect per-run result.json files into one CSV
python scripts/aggregate_results.py \
    --sweep_dir experiments/flow_manifold_synth \
    --output    experiments/flow_manifold_synth/results.csv

# Generate Figure 1 and Figure 2
python scripts/make_figures.py \
    --sweep_dir   experiments/flow_manifold_synth \
    --figures_dir experiments/flow_manifold_synth/figures
```

Outputs:
- `figures/figure1.{png,pdf}` — Panel A (vary dx, d0=4 fixed) and Panel B (vary d0, dx=128 fixed), y-axis is per-dimension tangent oracle MSE
- `figures/figure2_dx128_d04_n50000_seed0.{png,pdf}` — Three-panel diagnostic: velocity scatter, training convergence, latent endpoint distribution

Figure 2 requires that the representative run (`dx128_d04_n50000_seed0`) had `run_diagnostics: true` in its config (set automatically by `launch_sweep.py`).

---

## Step 9 — Copy results back locally

From your local machine:

```bash
scp -r sophiapi@babel:/home/sophiapi/flow-manifold/experiments/flow_manifold_synth/figures \
    "C:/Users/sophi/Documents/magics/fw-transformers-ldls/experiments/flow_manifold_synth/"

# Or copy the full sweep directory (includes result JSONs for further analysis)
scp -r sophiapi@babel:/home/sophiapi/flow-manifold/experiments/flow_manifold_synth \
    "C:/Users/sophi/Documents/magics/fw-transformers-ldls/experiments/"
```

---

## Quick reference

```bash
# On the cluster, from /home/sophiapi/flow-manifold:

conda activate flowmanifold

# 1. Generate configs + Panel A data (once)
python scripts/launch_sweep.py --output_root experiments/flow_manifold_synth

# 2. Smoke test
python scripts/run_one_config.py --config configs/debug.yaml

# 3. Submit sweep
mkdir -p logs
sbatch --array=0-49%8 slurm/run_array.sbatch
# (submit 50-53 separately after queue drops below 50)

# 4. Monitor
squeue -u $USER
find experiments/flow_manifold_synth -name result.json | wc -l

# 5. After all 54 complete
python scripts/aggregate_results.py --sweep_dir experiments/flow_manifold_synth
python scripts/make_figures.py --sweep_dir experiments/flow_manifold_synth
```

---

## Sanity checks to run before the full sweep

Before submitting all 54 jobs, it is worth running a few quick spot-checks to catch bugs early. Run one representative config (e.g. dx=128, d0=4, n=2000, seed=0) for ~10 epochs and verify:

1. **Normal MSE < 1e-5** — printed as `normal_mse=...` in the `DONE` line.
2. **Per-dim tangent MSE decreases** — `oracle_mse_per_dim` should drop across logged epochs.
3. **No NaN losses** — any `nan` in train/val loss indicates a numerical issue.

```bash
# Run a quick 10-epoch spot check
python -c "
import yaml
with open('configs/dx128_d04_n2000_seed0.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['epochs'] = 10
cfg['run_name'] = 'spotcheck'
cfg['eval_every'] = 2
with open('configs/spotcheck.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
python scripts/run_one_config.py --config configs/spotcheck.yaml
```

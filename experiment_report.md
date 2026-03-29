# Experiment Implementation Report
## Synthetic Validation for *Learning Manifold Data with Flow Matching*

**Date:** 2026-03-27
**Status:** Code complete, smoke test passed, ready for full sweep

---

## 1. What Was Built

### 1.1 Overview

A complete, reproducible experiment codebase that generates two figures for the paper:

- **Figure 1** — Oracle velocity MSE vs training set size, showing that performance depends much more on intrinsic dimension `d0` than ambient dimension `dx`.
- **Figure 2** — Tangent/normal diagnostic, showing that the learned velocity field's off-subspace component matches the theoretically predicted linear contraction.

The code runs identically locally and on a Slurm cluster. Only the config file changes between runs.

---

### 1.2 File Structure

```
fw-transformers-ldls/
  src/
    data.py          generate U, GMM parameters, and datasets
    oracle.py        GMM score function, oracle velocity, projection helpers
    models.py        VelocityMLP (sinusoidal time embedding + MLP)
    train.py         FM training utilities (set_seed, train/val epoch)
    evaluate.py      oracle MSE, normal MSE, scatter data, ODE trajectory
    plots.py         Figure 1 and Figure 2
  scripts/
    run_one_config.py    run a single config end-to-end
    launch_sweep.py      generate all 54 configs and optionally launch
    aggregate_results.py collect result.json files into results.csv
    make_figures.py      generate both figures after the sweep
  slurm/
    run_single.sbatch    single-job Slurm script
    run_array.sbatch     job array for full sweep (54 tasks, 8 concurrent max)
    submit_sweep.sh      convenience wrapper to generate configs + submit array
  configs/
    debug.yaml           5-epoch smoke test (dx=32, d0=4, n_train=2000)
  experiments/
    flow_manifold_synth/ all run outputs land here
  requirements.txt
  README.md
  experiment_report.md   this file
```

---

### 1.3 Data Generation (`src/data.py`)

**Subspace matrix `U`:**
- Shape `(dx, d0)` with orthonormal columns
- Generated via QR decomposition of a random Gaussian matrix
- Seeded by `geom_seed = dx * 10000 + d0 * 100 + 17` — fixed per `(dx, d0)` setting

**Latent GMM:**
- `M = 4` components, uniform weights
- Means sampled from `N(0, 4 I_{d0})` (std = 2)
- Diagonal covariances with entries sampled uniformly from `[0.5, 1.5]`
- Seeded by `gmm_seed = d0 * 1000 + 42` — fixed per `d0` only, so the same GMM is shared across all `dx` values in the ambient sweep

**Datasets:**
- Master train set of 50,000 points generated once per `(dx, d0)` setting
- `n_train ∈ {2000, 10000, 50000}` are nested prefixes of the master train set
- Validation set: 2,000 points; test set: 5,000 points
- Seeded by `data_seed = dx * 10000 + d0 * 100 + 99`

All geometry is fixed across the 3 training seeds. Only model initialization, minibatch order, and sampled noise `x0` and time `t` vary across seeds.

---

### 1.4 Oracle Velocity (`src/oracle.py`)

The oracle marginal velocity for the linear path is:

```
u*(x, t) = U [ (1/t) h_bar + ((1-t)/t) ∇ log p_t^h(h_bar) ]
           - (1/(1-t)) (I - UU^T) x
```

where `h_bar = U^T x` (in code: `y = x @ U`).

**GMM score computation:**

The latent marginal `p_t^h` is a GMM with time-dependent parameters:
- `m_m(t) = t * mu_m`
- `S_m(t) = t^2 * Sigma_m + (1-t)^2 * I_{d0}` (diagonal)

The score is computed as a responsibility-weighted sum:
```
∇_y log p_t^h(y) = -∑_m r_m(y,t) * S_m(t)^{-1} (y - m_m(t))
```

**Implementation details:**
- Fully vectorized: operations are `(B, M, d0)` shaped, no Python loops over components
- Responsibilities computed via log-sum-exp for numerical stability
- Diagonal structure of `S_m(t)` exploited — inverse is elementwise division, log-det is sum of logs
- No numerical issues because time is clipped to `[eps, 1-eps]` with `eps = 0.01`

---

### 1.5 Model (`src/models.py`)

`VelocityMLP` — fixed architecture across all `(dx, d0)` settings:
- **Input:** `cat(x_t, time_emb)` where `x_t ∈ R^{dx}` and `time_emb ∈ R^{64}`
- **Time embedding:** 64-dimensional sinusoidal embedding + linear projection
- **Hidden layers:** 4 layers, width 256, SiLU activations
- **Output:** velocity vector in `R^{dx}`

Only the input/output dimension changes with `dx`. All other architecture parameters are fixed.

---

### 1.6 Training (`src/train.py`, `scripts/run_one_config.py`)

**Conditional FM loss per batch:**
```python
x0 = randn(B, dx)
t  = Uniform[eps, 1-eps]
xt = t * x1 + (1-t) * x0
target = x1 - x0
loss = mean over batch of ||u_theta(xt, t) - target||_2^2
```

**Optimizer:** AdamW, lr = 3e-4, weight decay = 1e-4
**Batch size:** 256
**Epochs:** 150
**Gradient clipping:** max norm 1.0

**Logging:** train loss and val loss every epoch; oracle velocity MSE on the test set every 10 epochs and at the final epoch.

---

### 1.7 Evaluation (`src/evaluate.py`)

**Oracle velocity MSE (main metric):**
```
E || u_hat(x_t, t) - u*(x_t, t) ||_2^2
```
Computed over 10,000 `(x1, x0, t)` triples where `x1` is sampled from the held-out test set. This is the marginal oracle velocity, not the conditional training target `x1 - x0`.

**Normal-component MSE (Figure 2 scalar):**
```
E || (I-UU^T) u_hat(x_t,t) - kappa_t (I-UU^T) x_t ||_2^2
```

**Scatter data (Figure 2 panel A):**
Flattened coordinates of the theoretical and learned normal component across 5,000 test points.

**Generation ODE trajectory (Figure 2 panel C):**
Euler integration of `dx/dt = u_theta(x, t)` from `t = 0.01` to `t = 0.99`, starting from Gaussian noise `z ~ N(0, I_{dx})`. Tracks mean off-subspace norm `||(I-UU^T) x_t||` over time, which should decrease as the flow pushes points toward the data subspace.

---

### 1.8 Sweep Design (`scripts/launch_sweep.py`)

| Panel | Fixed | Varied |
|-------|-------|--------|
| A | d0 = 4 | dx ∈ {32, 128, 512} |
| B | dx = 128 | d0 ∈ {2, 4, 8, 16} |

`n_train ∈ {2000, 10000, 50000}`, 3 seeds each.

`(dx=128, d0=4)` appears in both panels and is deduplicated — **54 unique training runs total** (the handoff's count of 63 double-counts the 9 overlapping configs).

**Representative diagnostic runs** (Figure 2):
- `dx=128, d0=4, n_train=50000, seed=0` — primary
- `dx=512, d0=4, n_train=50000, seed=0` — optional secondary

---

### 1.9 Seed Policy Summary

| Seed | Formula | What it controls |
|------|---------|-----------------|
| `gmm_seed` | `d0 * 1000 + 42` | GMM means and covariances |
| `geom_seed` | `dx * 10000 + d0 * 100 + 17` | Subspace matrix U |
| `data_seed` | `dx * 10000 + d0 * 100 + 99` | Dataset sampling |
| `train_seed` | 0, 1, or 2 | Model init, minibatch order, x0 and t sampling |

---

## 2. Smoke Test Results

Run: `python scripts/run_one_config.py --config configs/debug.yaml`
Setting: `dx=32, d0=4, n_train=2000, 5 epochs`

| Check | Result | Expected |
|-------|--------|----------|
| Subspace residual `||(I-UU^T)x||` | 7.15e-07 | ~machine epsilon |
| Oracle velocity finite, no NaN | ✓ | — |
| Train loss decreasing | 55.2 → 47.1 | decreasing |
| Oracle MSE decreasing | 43.6 → 35.9 | decreasing |
| Normal component shape | `(50, 32)` | `(B, dx)` |
| Normal component magnitude (at x_t) | mean norm ≈ 5.2 | ≈ 5.3 (theoretical) |
| Diagnostic arrays saved | ✓ | scatter, ODE, geometry |
| Wall time | 1.6s | — |

All sanity checks pass.

---

## 3. Output Structure Per Run

```
experiments/flow_manifold_synth/{run_name}/
  config.yaml          exact config used
  geometry.pt          U, pis, mus, diag_Sigmas (torch tensors)
  checkpoint.pt        trained model weights
  metrics.json         train_loss[], val_loss[], oracle_mse_history[], final metrics
  result.json          one-row summary: dx, d0, n_train, seed, oracle_mse, normal_mse, wall_time

  # only for run_diagnostics=true configs:
  scatter_theory.npy
  scatter_learned.npy
  ode_times.npy
  ode_perp_norms.npy
```

After all runs:
```
experiments/flow_manifold_synth/
  results.csv          one row per run (aggregated by aggregate_results.py)
  figures/
    figure1.png / .pdf
    figure2.png / .pdf
```

---

## 4. Next Steps

### 4.1 Before launching the full sweep

**Step 1 — Verify a realistic pilot locally.**
Run one or two configs with full epochs (150) but small settings:
```bash
# Edit debug.yaml to set epochs: 150, then:
python scripts/run_one_config.py --config configs/debug.yaml --overwrite
```
Check that:
- Oracle MSE continues to decrease substantially after epoch 10
- The ODE trajectory shows off-subspace norm clearly decreasing (not flat)
- Wall time per run is acceptable for your hardware

**Step 2 — Estimate wall time for dx=512.**
The largest configs (`dx=512`) will be slowest. Run one to time it:
```bash
python scripts/run_one_config.py \
    --config configs/dx512_d04_n50000_seed0.yaml   # after generating configs
```
If >4 hours on a single GPU, consider reducing `epochs` from 150 to 100 (but report this change).

---

### 4.2 Generate all configs

```bash
python scripts/launch_sweep.py --output_root experiments/flow_manifold_synth
```

This writes:
- 54 YAML files to `configs/`
- `configs/all_runs.txt` (one path per line, for the Slurm array)
- `configs/debug.yaml` (refreshed)

---

### 4.3 Launch the full sweep

**On a Slurm cluster:**
```bash
bash slurm/submit_sweep.sh
```

Or manually:
```bash
sbatch --array=0-53%8 slurm/run_array.sbatch
```

The `%8` limits concurrent jobs to 8. Adjust based on cluster policy.

**Locally (sequential, for a smaller pilot):**
```bash
python scripts/launch_sweep.py \
    --output_root experiments/flow_manifold_synth \
    --launch sequential
```

**Monitor progress** by counting completed result.json files:
```bash
find experiments/flow_manifold_synth -name result.json | wc -l
# should reach 54 when done
```

---

### 4.4 After the sweep — generate figures

```bash
# Aggregate all result.json files into one CSV
python scripts/aggregate_results.py \
    --sweep_dir experiments/flow_manifold_synth \
    --output    experiments/flow_manifold_synth/results.csv

# Generate Figure 1 and Figure 2
python scripts/make_figures.py \
    --sweep_dir   experiments/flow_manifold_synth \
    --figures_dir experiments/flow_manifold_synth/figures
```

Figures are saved as both `.png` (for checking) and `.pdf` (for the paper).

---

### 4.5 Things to check in the figures

**Figure 1 — what success looks like:**
- Panel A (vary `dx`, fix `d0=4`): the three curves should be close together — ambient dimension barely affects MSE
- Panel B (vary `d0`, fix `dx=128`): the four curves should spread out clearly — intrinsic dimension strongly affects MSE
- Both panels: MSE should decrease as `n_train` increases
- Log-log scale helps readability; slopes need not be fitted exactly

**Figure 2 — what success looks like:**
- Scatter plot: points clustered near the `y=x` diagonal
- Normal MSE: a small number (much smaller than overall oracle MSE)
- ODE trajectory: off-subspace norm starts high (~`sqrt(dx - d0)`) and decreases toward 0 as `t → 1`

If Figure 2 does not look clean, the most likely causes are:
1. The model hasn't trained long enough (try more epochs for the representative setting)
2. The representative setting is too small (try `n_train=50000`)

---

### 4.6 If any runs fail

Check failed run logs in `experiments/flow_manifold_synth/{run_name}/` — the metrics.json will be absent. Rerun failed configs with:
```bash
python scripts/run_one_config.py --config configs/{run_name}.yaml --overwrite
```

Common failure modes to check:
- NaN in oracle MSE (numerical instability — check that time clipping `eps=0.01` is applied everywhere)
- OOM on `dx=512` runs — reduce `batch_size` in that config's YAML
- Missing checkpoint (run timed out before finishing) — resubmit with a longer `--time` limit

---

### 4.7 Optional: secondary metric

If time permits, add Sinkhorn-2 or MMD as a secondary endpoint metric comparing generated samples to test data. This would go in `src/evaluate.py` as a new function and be called at the end of `run_one_config.py`. Do not add this until Figure 1 and Figure 2 are finalized.

---

### 4.8 Cluster environment setup

The Slurm scripts assume a conda environment named `flowmanifold`. Create it with:
```bash
conda create -n flowmanifold python=3.11 -y
conda activate flowmanifold
pip install -r requirements.txt
```

Verify the environment with:
```bash
python scripts/run_one_config.py --config configs/debug.yaml
```

---

## 5. Known Limitations and Design Choices

- **54 vs 63 runs:** The handoff counts 63 runs (27 + 36) but `(dx=128, d0=4)` appears in both panels and is run only once. The 9 shared runs are used in both Figure 1 panels. This is strictly cleaner and not a deviation from the science.

- **Euler ODE integration:** The generation ODE uses simple Euler steps (100 steps over `[0.01, 0.99]`). This is sufficient for the trajectory visualization. For higher-fidelity generation quality (e.g., a secondary Wasserstein metric), switch to a 4th-order Runge-Kutta solver — `scipy.integrate.solve_ivp` with `method='RK45'` is one option.

- **Oracle MSE evaluation noise:** Each call to `compute_oracle_mse` re-samples `x0` and `t` randomly, so there is some Monte Carlo variance in the reported numbers. With `n_eval_pairs=10000` this variance is small but not zero. The 3-seed mean/std in Figure 1 reflects training variance, not this evaluation noise.

- **Model capacity:** Width 256 is fixed across all settings including `dx=512`. For `dx=512` with `n_train=50000`, the model may be mildly underfitting. If oracle MSE fails to decrease beyond a plateau for the largest settings, consider widening to 512 — but check with mentors before changing.

- **No early stopping:** Training runs the full 150 epochs. This is intentional for consistent comparison. If runtime is too long, reduce epochs uniformly and document the change.

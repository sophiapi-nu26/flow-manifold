# Flow Manifold: Oracle-U Latent Ablation

Synthetic experiment validating the intrinsic-dimension scaling behavior predicted by the tangent/normal decomposition theorem. The experiment tests whether, once oracle access to the subspace $U$ is provided, tangent estimation difficulty depends on intrinsic dimension $d_0$ rather than ambient dimension $d_x$.

## Experiment summary

Two panels, 90 runs total (5 seeds each):

- **Panel A** — fix $d_0 = 4$, vary $d_x \in \{32, 128, 512\}$: curves nearly identical across ambient dimensions, confirming latent invariance
- **Panel B** — fix $d_x = 128$, vary $d_0 \in \{2, 4, 8, 16\}$: MSE increases with intrinsic dimension, with $d_0 = 2$ consistently easiest and $d_0 = 16$ consistently hardest

## Setup

```bash
conda create -n flowmanifold python=3.11 -y
conda activate flowmanifold
pip install -r requirements.txt
```

## Running

See [`how_to_run.md`](how_to_run.md) for the full walkthrough including cluster submission.

Quick start:

```bash
# Generate configs (also builds shared Panel A data)
python scripts/launch_sweep.py --output_root experiments/flow_manifold_synth

# Smoke test (5 epochs, dx=32, d0=4)
python scripts/run_one_config.py --config configs/debug.yaml

# After all runs complete: aggregate and plot
python scripts/aggregate_results.py --sweep_dir experiments/flow_manifold_synth
python scripts/make_figures.py --sweep_dir experiments/flow_manifold_synth
```

## Repository layout

```
src/
  data.py       — GMM and dataset generation
  models.py     — latent MLP (input/output in R^{d0})
  oracle.py     — oracle tangent velocity, GMM score
  train.py      — projected FM training loop
  evaluate.py   — tangent oracle MSE, Panel A cache, SWD
  plots.py      — Figure 1 (scaling) and Figure 2 (diagnostics)

scripts/
  launch_sweep.py        — generate all 54 configs + Panel A shared data
  launch_extra_seeds.py  — add seeds 3 & 4 (36 additional runs)
  run_one_config.py      — run a single experiment from a YAML config
  aggregate_results.py   — collect result.json files into results.csv
  make_figures.py        — generate figures from completed sweep

slurm/
  run_array.sbatch       — Slurm array job for full sweep
  run_extra_seeds.sbatch — Slurm array job for extra seeds
  run_single.sbatch      — single job for reruns
```

## Writeup

The experiment is reported in [`writeup.tex`](writeup.tex).

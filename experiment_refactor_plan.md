# Oracle-U Latent-Space Ablation: Experiment Plan

## Overview and framing

This document describes a controlled ablation experiment to run **alongside** the original
ambient-space experiment, not as a replacement for it.

The two experiments serve complementary roles:

- **Original ambient experiment** (practical baseline): a generic MLP trained in R^{dx} predicts
  the full velocity u(xt, t) ∈ R^{dx} without access to U. This reflects the full difficulty a
  practitioner faces, including the cost of learning the manifold structure from data.

- **Oracle-U latent-space ablation** (controlled isolation): the model is given oracle access to U,
  projects xt to latent coordinates h_bar = xt @ U ∈ R^{d0}, and is trained only on the tangent
  velocity component. This is designed to isolate the d0-dimensional score estimation problem
  highlighted by the decomposition theorem, by removing the confounding cost of subspace recovery.

The contrast between the two is part of the empirical story: practical models may still incur
some ambient-dimension burden, but once subspace discovery is removed, the remaining
tangent-learning problem should depend much more on d0 than on dx. The oracle-U experiment tests
whether this qualitative pattern is visible under controlled conditions.

---

## What the decomposition theorem says

The paper proves that the oracle marginal velocity decomposes as:

```
u*(xt, t) = U * [(1/t) * h_bar + ((1-t)/t) * ∇_{h_bar} log p_t^h(h_bar)]   ← tangent term
           + (-(1/(1-t))) * (I - UU^T) * xt                                   ← normal term
```

where `h_bar = U^T xt ∈ R^{d0}`.

The normal term is a deterministic linear function of xt and t — given U it can be computed
exactly. The tangent term is a d0-dimensional estimation problem: it depends on the data
distribution only through the latent density p^h in R^{d0}.

The decomposition theorem suggests that, once the normal component is handled analytically, the
dominant remaining difficulty should depend primarily on d0 rather than dx. The oracle-U
experiment is designed to test whether this pattern is visible in practice. This should not be
read as a direct implementation of the theorem: the theory does not assume oracle U, and this
experiment does not eliminate every possible source of dx-dependence.

---

## Why the ambient experiment may show more dx-dependence

A generic ambient MLP must handle two tasks simultaneously:

1. **Subspace recovery**: learn to separate on-subspace variation from off-subspace variation, and
   recover the relevant d0-dimensional subspace geometry inside R^{dx}. Intuitively, this becomes
   harder as dx grows and can plausibly impose an additional dx-dependent burden on the ambient
   estimator. This is a heuristic motivation, not a formal rate claim.

2. **On-manifold score estimation**: given h_bar = U^T xt, estimate ∇ log p_t^h(h_bar). This is a
   purely d0-dimensional problem.

The oracle-U ablation removes task 1 by giving the model U directly, allowing a cleaner test of
whether task 2 has the d0-dependent behavior the decomposition theorem suggests.

---

## Training vs. evaluation distinction

**Training** uses conditional flow-matching targets:

```
target = (x1 - x0) @ U    ∈ R^{d0}   (tangent component of the conditional velocity)
```

This is the standard flow-matching regression objective, projected to latent coordinates.

**Evaluation** uses the oracle marginal velocity in latent space:

```
v*(h_bar, t) = (1/t) * h_bar + ((1-t)/t) * ∇_{h_bar} log p_t^h(h_bar)
```

This is a different quantity — the expectation of the conditional target over x0 and x1 given xt,
evaluated exactly using the known latent GMM. The model is trained on noisy conditional samples
and evaluated against the noiseless oracle. This distinction must be maintained clearly in both
the code and any writeup.

---

## Why projecting the training target is exact

Because x1 ∈ col(U):

```
(I - UU^T)(x1 - x0) = -(I - UU^T)x0
```

Also, since xt = t*x1 + (1-t)*x0:

```
(I - UU^T)xt = (1-t)(I - UU^T)x0
```

Therefore:

```
-(1/(1-t)) * (I - UU^T) * xt = -(I - UU^T) * x0
```

The analytical normal term exactly equals the normal component of the conditional target for every
individual sample. Training on `(x1 - x0) @ U` is exactly equivalent to training on the full
target minus the analytically known normal component — no approximation is involved.

---

## Model architecture

The latent MLP takes `(h_bar, t)` where `h_bar = xt @ U ∈ R^{d0}` and outputs `v_hat ∈ R^{d0}`.
The full reconstructed velocity is:

```
u_hat(xt, t) = v_hat @ U.T  +  analytical_normal(xt, t, U)
analytical_normal(xt, t, U) = -(1/(1-t)) * (I - UU^T) * xt
```

Architecture (fixed across all settings):
- Input dim: d0 + 64 (latent coordinate + sinusoidal time embedding)
- 4 hidden layers, width 256, SiLU activations
- Output dim: d0

Width 256 is expected to be comfortably overparameterized relative to the latent dimensions
considered in the sweep.

---

## Training objective

```python
x0     = torch.randn(B, dx)
t      = eps + (1 - 2*eps) * torch.rand(B, 1)
xt     = t * x1 + (1 - t) * x0

h_bar  = xt @ U              # (B, d0) — model input
target = (x1 - x0) @ U      # (B, d0) — tangent conditional target

v_hat  = model(h_bar, t)     # (B, d0)
loss   = ((v_hat - target)**2).sum(dim=1).mean()
```

---

## Evaluation metrics

### Primary: per-dimension tangent oracle MSE

```
tangent_oracle_mse_per_dim = (1/d0) * E || v_hat(h_bar, t) - v*(h_bar, t) ||_2^2
```

Dividing by d0 is necessary for Panel B comparability: without normalization, total MSE grows
mechanically with d0 simply because there are more output coordinates.

Note: per-dimension tangent MSE is not a perfectly pure measure of intrinsic dimension
difficulty. The latent GMM changes across d0 values, so Panel B curves reflect both the change
in intrinsic dimension and the change in the specific canonical latent problem. Panel B should
be interpreted as comparing a family of latent problems indexed by d0, not as a controlled
experiment holding everything else fixed.

### Secondary: total tangent oracle MSE

```
tangent_oracle_mse_total = E || v_hat(h_bar, t) - v*(h_bar, t) ||_2^2
```

Report alongside per-dimension MSE for reference and for comparison with the ambient experiment.

### Verification: analytical normal MSE

Report `normal_mse` as a software correctness check. Since the normal component is inserted
analytically, this should be near zero (floating-point residual only). Any non-trivial value
indicates a bug.

---

## Panel A: shared latent data across dx values

For Panel A (fix d0=4, vary dx ∈ {32, 128, 512}), the latent model only ever sees
d0-dimensional h_bar. To make the comparison as clean as possible:

- Use the **same latent GMM** (same pis, mus, diag_Sigmas with d0=4) across all three dx values.
- Use the **same latent samples h** (same master train, val, and test sets in R^{d0}) across all
  three dx values.
- Use the **same training seeds** across all three dx values (seed ∈ {0, 1, 2} for each).
- Use **different U matrices** for each dx value. These are applied only at the embedding step
  `x = h @ U.T` and are not visible to the latent model.

**Cached evaluation set for Panel A**: cache `(h_bar, t, v*(h_bar, t))` once for d0=4, using the
shared latent GMM and a fixed eval seed (42). Reuse the exact same cached file across all dx
settings and seeds in Panel A. The implementation must not recompute oracle values dynamically,
generate a fresh evaluation set per run, or cache any other tuple format. This makes the
invariance comparison fully reproducible and eliminates Monte Carlo noise from the comparison.

**Implementation**: the shared latent GMM and shared latent samples for Panel A are generated
once in `scripts/launch_sweep.py` and saved to disk (e.g. `configs/panel_a_latent_data.pt`).
Individual per-dx config files reference this shared file. This is an explicit, auditable
record of what was shared and makes reruns reproducible. No changes to `src/data.py` are needed
for this; the sharing is enforced at the config-generation level.

**Interpretation**: Panel A should be described as an oracle-latent invariance check — it shows
that once the relevant subspace is provided and the estimator operates only on latent coordinates,
increasing the ambient embedding dimension does not materially change the tangent estimation
problem. This is a controlled and defensible claim. It should not be stated as proof that ambient
dimension never matters in general; the collapse is partly built into the design by construction.

---

## Panel B: vary intrinsic dimension

For Panel B (fix dx=128, vary d0 ∈ {2, 4, 8, 16}), use one canonical GMM per d0 value as in
the original spec.

Report per-dimension tangent oracle MSE as the primary y-axis, with total tangent oracle MSE as
a secondary panel or inset.

**Expected behavior**: higher d0 should tend to make the latent score estimation problem harder
on average. Exact monotonicity is not guaranteed — with one randomly sampled GMM per d0 value,
it is entirely possible for d0=8 to produce slightly lower per-dim error than d0=4 at some
n_train values. Additionally, fixing the number of mixture components at M=4 across all d0 values may reduce
how strongly the higher-dimensional settings separate from the lower-dimensional ones.

**Fallback if results are unstable**: if the qualitative ordering across d0 is unstable across
seeds or difficult to interpret, rerun Panel B with multiple GMM draws per d0, average results,
and show error bars over both training seeds and GMM samples. This prevents the fallback from
feeling ad hoc after the fact.

---

## Figure 2: tangent component diagnostics

Since the normal component is inserted analytically, plotting learned vs. theoretical normal
coordinates would only verify the software implementation. Figure 2 instead focuses on the
learned tangent component.

### (A) Learned vs oracle latent velocity scatter

Plot learned tangent velocity coordinates against oracle tangent velocity coordinates in R^{d0}.

- x-axis: oracle v*(h_bar, t), flattened across coordinates and test points
- y-axis: learned v_hat(h_bar, t), flattened correspondingly
- Points colored by coordinate index (d0=4 allows 4 distinguishable colors), so that
  coordinate-specific failure modes are visible rather than hidden by the aggregate cloud
- Include a per-coordinate MSE breakdown (small bar chart or table) alongside the scatter
- Representative setting: dx=128, d0=4, n_train=50000, seed=0

### (B) Tangent oracle MSE over training (convergence diagnostic)

- x-axis: training epoch
- y-axis: per-dimension tangent oracle MSE on the fixed cached test set
- Representative setting: dx=128, d0=4, n_train=50000, seed=0

This is a training-dynamics panel. It should be framed as showing that the latent model
converges and that the per-dimension tangent oracle MSE metric improves over training (not just
the conditional training loss). Both the training loss and oracle MSE curves should be shown for
comparison.

### (C) Latent endpoint distribution comparison

Generate latent samples by integrating the ODE `dh/dt = v_hat(h, t)` in latent space directly
(not in ambient space followed by projection), starting from `h0 ~ N(0, I_{d0})` at t=0,
using Euler integration with 100 steps from t=0.01 to t=0.99.

Generate n=2000 samples. Compare the distribution of generated h1 against a reference sample of
2000 draws from the true latent GMM.

**Primary scalar metric**: sliced Wasserstein distance (SWD) between generated and reference
latent samples. Use 500 random projection directions, generated once with a fixed seed (seed=0)
and reused identically for all runs and seeds — not resampled per evaluation. Lower is better.
Report the mean SWD averaged across 3 training seeds.

**Secondary qualitative**: a 2D scatter of the first two PCA coordinates, comparing generated
vs. reference samples side by side. Fit PCA jointly on the union of the reference and generated
samples used in this panel, so that neither distribution is visually privileged.

---

## Sweep settings

| Panel | Fixed | Varying | n_train | Seeds | Runs |
|---|---|---|---|---|---|
| A | d0=4 | dx ∈ {32, 128, 512} | {2000, 10000, 50000} | 3 | 27 |
| B | dx=128 | d0 ∈ {2, 4, 8, 16} | {2000, 10000, 50000} | 3 | 36 |

dx=128, d0=4 is shared between panels → 54 unique runs total.

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | unchanged |
| Learning rate | 3e-4 | unchanged |
| Weight decay | 1e-4 | unchanged |
| Batch size | 256 | unchanged |
| Epochs | 150 | latent model is small; no dx-scaling needed |
| Time epsilon | 0.01 | unchanged |
| Hidden width | 256 | expected to be comfortably overparameterized for d0 ≤ 16 |
| Hidden layers | 4 | unchanged |
| Time embedding dim | 64 | unchanged |

---

## What changes in the codebase

| File | Change |
|---|---|
| `src/models.py` | Input dim `dx + time_emb_dim` → `d0 + time_emb_dim`; output dim `dx` → `d0`; remove `max(hidden_width, dx)` line |
| `src/train.py` | Target `x1 - x0` → `(x1 - x0) @ U`; model input `xt` → `xt @ U`; pass U to training functions |
| `src/oracle.py` | Add `oracle_tangent_latent(h_bar, t, pis, mus, diag_Sigmas)` returning d0-dimensional oracle tangent velocity |
| `src/evaluate.py` | Replace `compute_oracle_mse` with `compute_tangent_oracle_mse` (per-dim and total, against cached eval set); keep `compute_normal_mse` as a verification-only check; add `compute_sliced_wasserstein` for Figure 2C |
| `src/plots.py` | Figure 1: per-dim tangent MSE on y-axis; Figure 2: redesigned per above |
| `scripts/run_one_config.py` | Pass U to model/trainer; project xt → h_bar; log per-dim and total tangent MSE; load cached eval set for Panel A configs |
| `scripts/launch_sweep.py` | Generate and save shared latent GMM and latent data to `configs/panel_a_latent_data.pt` for Panel A; write Panel A configs referencing this shared file; reuse same training seeds across dx values in Panel A |

No changes needed to Slurm scripts or `src/data.py`.

---

## Sanity checks before the full sweep

**Check 1 — Shapes**: for dx=128, d0=4, verify model input is (B, 68) and output is (B, 4).

**Check 2 — Normal MSE ≈ 0**: `compute_normal_mse` should return < 1e-5 on any trained model.
A non-trivial value indicates a bug in the analytical normal reconstruction.

**Check 3 — Oracle MSE decreases during training**: for at least one representative setting
(dx=128, d0=4, n_train=2000), verify that per-dimension tangent oracle MSE decreases over
epochs — not just the conditional training loss. This ensures the model improves on the actual
scientific quantity of interest.

**Check 4 — Panel A relative variation**: at n_train=50000, per-dim tangent MSE across
dx ∈ {32, 128, 512} should show substantially less variation than Panel B shows across
d0 ∈ {2, 4, 8, 16}. These are qualitative comparisons; no hard thresholds should be applied.

**Check 5 — Panel B broad trend**: per-dim tangent MSE at n_train=50000 should broadly worsen
as d0 increases. Exact monotonicity is not required and should not be expected given that each
d0 value uses an independently sampled canonical GMM.

**Check 6 — Training loss decreases**: verify the per-dimension tangent training loss decreases
over epochs for at least one representative setting (dx=128, d0=4, n_train=2000, epochs=10).

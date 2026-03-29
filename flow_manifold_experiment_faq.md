# FAQ: Synthetic Flow-Manifold Experiment

This document answers common implementation questions for the synthetic experiment validating the paper’s theory.

---

## 1) What hardware should I plan for?

See flow_manifold_experiment_handoff_updated.md

---

## 2) What packages should I use?

Use a simple, standard Python stack.

### Recommended packages
- `torch`
- `numpy`
- `matplotlib`
- `tqdm`
- `pandas`
- `pyyaml` or `json`
- optionally `scipy` for ODE helpers or numerical utilities

### Package philosophy
Do **not** build this on a complicated ecosystem. The goal is a small clean synthetic experiment, not an infrastructure-heavy project.

### Default framework
**PyTorch is the intended framework.**

---

## 3) What does “same latent GMM construction recipe across all runs” mean?

For the main reported figures, this should be interpreted **strictly**, not loosely.

The goal is to make differences in performance come from changes in:
- intrinsic dimension `d0`, or
- ambient dimension `dx`,

rather than from changes in the underlying synthetic data distribution.

### Recommended policy

#### Ambient sweep
For the sweep where:
- `d0 = 4` is fixed
- `dx in {32, 128, 512}` varies

use **one canonical latent GMM** shared across all those `dx` values.

#### Intrinsic sweep
For the sweep where:
- `dx = 128` is fixed
- `d0 in {2, 4, 8, 16}` varies

use **one canonical latent GMM per value of `d0`**.

### Across seeds
Do **not** resample the GMM across the 3 seeds.

### Short answer
For a given reported setting, keep the latent distribution fixed and only vary training randomness across seeds.

---

## 4) Should the subspace matrix \(U\) be re-sampled across seeds?

No. For the main reported figures, **fix one `U` per `(dx, d0)` setting** and share it across all 3 seeds.

### Why?
This keeps the geometry of the synthetic problem fixed so that differences across seeds reflect only:
- model initialization,
- minibatch order,
- sampled `t`,
- sampled Gaussian noise `x0`,
- and related training randomness.

### Recommended policy
For each `(dx, d0)`:
- sample one orthonormal matrix `U`
- keep it fixed across all seeds
- keep it fixed across all dataset sizes for that setting as well

### Even cleaner recommendation
Also fix:
- one master train set
- one validation set
- one test set

for each `(dx, d0)` setting, and let
- `n_train = 2000`
- `n_train = 10000`
- `n_train = 50000`

be nested subsets of the same master train set.

That makes comparisons cleaner.

---

## 5) Should the 63 runs be launched sequentially or through a sweep launcher?

Use a **sweep launcher**.

### Recommended structure
Have:
1. one script that runs **a single config**
2. one launcher that enumerates all configs
3. optional parallel dispatch across GPUs or processes

### Why?
A single giant monolithic script is harder to debug, harder to resume, and harder to parallelize.

### Best practice
Implement something like:

- `run_one_config.py`
- `launch_sweep.py`

### Execution policy
- **1 GPU:** sequential queue is fine
- **multiple GPUs:** assign at most one run per GPU
- **CPU-only:** parallelize cautiously

### Bottom line
Even if you ultimately run sequentially, still build a sweep launcher.

---

## 6) Where should outputs go?

Create a fresh experiment directory dedicated to this project.

## Recommended directory structure

```text
experiments/flow_manifold_synth/
  configs/
  logs/
  checkpoints/
  metrics/
  figures/
  summaries/
```

### What to save for each run
- config file
- random seed
- GMM parameters
- matrix `U`
- train/validation metrics
- final oracle velocity MSE
- any normal-component diagnostics
- optionally checkpoints

### What to save globally
- final combined CSV or JSON summary
- figure-ready aggregates
- final paper figures

### Why this matters
The experiment is small, but reproducibility still matters. It should be easy to trace every point in the final figure back to a config and seed.

---

## 7) Is the interpretation of \(U^T x_t\) with projected Gaussian noise correct?

Yes. That is the intended interpretation.

---

## 8) More explicitly: what is happening in the oracle formula?

The forward conditional path is

\[
X_t = \mu_t X_1 + \sigma_t X_0,
\qquad X_0 \sim \mathcal{N}(0, I_{d_x}),
\qquad X_1 = U h.
\]

For the linear path we use

\[
\mu_t = t,
\qquad \sigma_t = 1-t,
\]

so

\[
X_t = t X_1 + (1-t) X_0.
\]

Now project this into the latent subspace:

\[
U^\top X_t = t U^\top X_1 + (1-t) U^\top X_0.
\]

Since `X1 = U h` and `U` has orthonormal columns,

\[
U^\top X_1 = h.
\]

So this becomes

\[
U^\top X_t = t h + (1-t) \xi,
\qquad \xi := U^\top X_0.
\]

Because
- `X0 ~ N(0, I_{dx})`, and
- `U` has orthonormal columns,

we get

\[
\xi \sim \mathcal{N}(0, I_{d0}).
\]

### Meaning
Yes: `x0` is a full ambient Gaussian, and `U^T x0` is simply its projection into the latent coordinates.

This is exactly what makes the latent marginal tractable.

---

## 9) Why does that projected-noise interpretation matter?

It matters because the oracle velocity uses the latent score term evaluated at

\[
\bar h = U^\top x_t.
\]

If the latent data distribution is a Gaussian mixture model in `h`, then after mixing with projected Gaussian noise under the affine path, the latent marginal at time `t` is still a Gaussian mixture model.

That is why the latent score
\[
\nabla_{\bar h} \log p_t^h(\bar h)
\]
can be computed explicitly.

So the experiment relies on this exact interpretation.

---

## 10) What object should be used for evaluation?

Use the **marginal oracle velocity** evaluated at `x_t`, not the raw conditional target `x1 - x0`.

### Training target
During training, the model predicts the conditional flow-matching target:
\[
x_1 - x_0.
\]

### Evaluation target
For reporting the main metric, compare the learned velocity to the **oracle marginal velocity**
\[
u_t^\star(x_t).
\]

### Why?
The paper’s theory is about the oracle velocity field, so that is the right object for the main empirical validation.

### Important warning
Do **not** accidentally report training loss as the main result.

---

## 11) What should vary across the 3 seeds?

For each fixed `(dx, d0, n_train)` configuration, the 3 seeds should vary only the stochastic parts of training.

### Good things to vary across seeds
- model initialization
- minibatch order
- sampled time `t`
- sampled Gaussian `x0`
- dataloader shuffling

### Things that should stay fixed across seeds
- GMM parameters
- matrix `U`
- train/val/test datasets for the setting

This keeps seed variance interpretable.

---

## 12) Should the architecture change across settings?

No. Keep the architecture template fixed unless explicitly told otherwise.

### Why?
The point of the experiment is to study sensitivity to:
- intrinsic dimension
- ambient dimension

not sensitivity to model redesign.

### Allowed differences
The input and output dimensions necessarily change with `dx`, but the overall template should stay the same:
- same number of layers
- same hidden width
- same activation
- same time embedding style
- same optimizer settings unless there is a strong reason to change them

---

## 13) What is the intended training setup?

Use standard first-order conditional flow matching on synthetic vector data.

For each batch:
1. sample `x1` from the synthetic data distribution
2. sample `x0 ~ N(0, I_dx)`
3. sample `t ~ Uniform[eps, 1-eps]`
4. form
   \[
   x_t = t x_1 + (1-t) x_0
   \]
5. use target
   \[
   x_1 - x_0
   \]
6. train with MSE

The model should take `(x_t, t)` as input and output a vector in `R^{dx}`.

---

## 14) What is the intended model type?

Use a small, time-conditioned vector model.

### Practical recommendation
Use a small **MLP** unless someone explicitly insists on a transformer.

### Why an MLP is okay
The data is just vectors in `R^{dx}`. There is no need to force a more complicated architecture if a simple one can test the theory.

### Suggested default
- 4 hidden layers
- width 256 or 512
- SiLU or GELU
- sinusoidal time embedding

The important thing is to keep the architecture fixed across settings.

---

## 15) What should Figure 1 show?

Figure 1 is the intrinsic-vs-ambient scaling figure.

### Panel A
Fix `d0 = 4` and vary:
- `dx in {32, 128, 512}`

### Panel B
Fix `dx = 128` and vary:
- `d0 in {2, 4, 8, 16}`

### In both panels
Use:
- `n_train in {2000, 10000, 50000}`
- 3 seeds
- y-axis = held-out oracle velocity MSE
- x-axis = training set size

### Intended qualitative outcome
- increasing ambient dimension should affect performance **a little**
- increasing intrinsic dimension should affect performance **more strongly**

This is a qualitative validation, not an exact asymptotic slope-fitting exercise.

---

## 16) What should Figure 2 show?

Figure 2 is the tangent/normal diagnostic.

For a trained model, decompose the learned velocity into:
- parallel / tangent component
- perpendicular / normal component

using the known matrix `U`.

### Include:
1. **scatter plot**
   - learned normal component vs theoretical normal component

2. **scalar normal MSE**
   - error between learned normal term and predicted normal contraction

3. **trajectory plot**
   - track the off-subspace norm during reverse ODE rollout

### Intended qualitative outcome
The normal component of the learned field should resemble the theoretically predicted orthogonal contraction.

---

## 17) Which model should be used for the decomposition diagnostic?

Use one or two representative well-trained models, not the entire sweep.

### Recommended choices
- `dx=128, d0=4, n_train=50000`
- optionally `dx=512, d0=4, n_train=50000`

The point is to show the structure clearly, not to create another large grid.

---

## 18) What are the main reproducibility requirements?

For each run, save:
- full config
- seed
- GMM parameters
- `U`
- key metrics
- logs
- figure ingredients if useful

Also save:
- combined summaries across runs
- final figures in `.png` and `.pdf`

### Nice habit
Save one machine-readable `results.csv` or `results.jsonl` with one row per run.

---

## 19) What are the main implementation pitfalls?

### Pitfall 1
Sampling times too close to `t=1`, causing instability because terms like `1/(1-t)` appear.

### Pitfall 2
Using unstable GMM responsibility calculations instead of log-sum-exp.

### Pitfall 3
Evaluating against the wrong target (`x1 - x0` instead of oracle velocity).

### Pitfall 4
Resampling `U` or the GMM across seeds, making the plots noisy and harder to interpret.

### Pitfall 5
Changing model architecture across settings and muddying the comparison.

---

## 20) What should I do first before launching the full sweep?

Before running all 63 jobs, do a small pilot such as:

- `dx=32`
- `d0=4`
- `n_train=2000`

Check that:
- the data lies in the subspace
- the training loss decreases
- oracle velocity code runs correctly
- oracle MSE improves with training
- logging and saving work properly

Only then launch the full sweep.

---

## 21) If runtime becomes an issue, what should I do?

Do **not** silently shrink the experiment.

Instead:
1. run a pilot
2. measure runtime per config
3. report the bottleneck
4. ask before changing the experimental grid

The reported figures are supposed to match the intended plan, so changes should be deliberate.

---

## 22) What is the shortest version of the execution policy?

If you need the shortest possible summary, use this:

- Use **PyTorch**
- Assume **GPU if available**
- Fix the **latent GMM** within each reported setting
- Fix **`U`** within each `(dx, d0)` setting
- Keep **datasets fixed** within each setting
- Vary only **training randomness** across seeds
- Use a **sweep launcher**
- Save outputs under `experiments/flow_manifold_synth/`
- Evaluate using the **oracle marginal velocity**
- Yes, `U^T x0` is the projected ambient Gaussian noise

---

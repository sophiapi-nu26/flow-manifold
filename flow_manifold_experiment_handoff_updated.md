# Handoff: Minimal Synthetic Experiments for **Learning Manifold Data with Flow Matching**
## Updated version with local-development + Slurm cluster workflow

## 0. Why you are doing this

The paper is mostly theoretical. Reviewers asked for a **small empirical experiment** showing that, in practice:

1. performance depends much more on the **intrinsic dimension** `d0` than the **ambient dimension** `dx`, and
2. the learned velocity field has the predicted **tangent / normal decomposition**.

Your job is **not** to build a large benchmark suite. Your job is to build **one clean synthetic setup** and produce **two figures**.

This document is written so you can implement the experiment even if you have not read the paper closely.

---

## 1. Big picture in plain English

The paper studies data that really lives in a low-dimensional linear subspace of a higher-dimensional ambient space.

Concretely:

- latent variable: `h in R^{d0}`
- ambient observation: `x = U h in R^{dx}`
- `U` is a `dx x d0` matrix with orthonormal columns

So although `x` is `dx`-dimensional, it only has `d0` true degrees of freedom.

The paper’s core theoretical claim is that under this setup, the optimal flow-matching velocity field splits into:

- an **on-subspace / tangent** term (the hard statistical part), and
- an **off-subspace / normal** term (a simple linear contraction back toward the subspace).

The reviewers want a small synthetic experiment showing that this structure is visible in practice.

---

## 2. What you need to deliver

## Main deliverables

1. **Figure 1: Intrinsic-vs-ambient scaling**
   - two panels:
     - panel A: fix `d0=4`, vary `dx in {32, 128, 512}`
     - panel B: fix `dx=128`, vary `d0 in {2, 4, 8, 16}`
   - for each setting, vary dataset size `n_train in {2000, 10000, 50000}`
   - y-axis: held-out **oracle velocity MSE**
   - x-axis: `n_train`
   - use 3 seeds, plot mean ± std

2. **Figure 2: Tangent/normal diagnostic**
   - using representative trained models, check whether the learned normal component matches the theoretically predicted one
   - include:
     - scatter plot of learned normal component vs theoretical normal component
     - scalar normal-component MSE
     - one trajectory plot showing off-subspace norm along the reverse ODE

3. **A short README / notes file**
   - exact hyperparameters
   - exact random seeds
   - exact commands for local runs
   - exact commands for cluster runs
   - where figures are saved
   - which runs succeeded / failed

## Nice-to-have but optional

- endpoint Sinkhorn-2, approximate Wasserstein-type metric, or MMD as a **secondary** metric only
- a small table of final scalar results corresponding to the plots

## Do not spend time on

- real data
- higher-order flow matching
- split-head architectures
- elaborate ablations
- exact exponent fitting

---

## 3. Minimal theory you need

## 3.1 Data model

We generate synthetic data from a latent Gaussian mixture model (GMM):

\[
h \sim \sum_{m=1}^M \pi_m \, \mathcal{N}(\mu_m, \Sigma_m),
\qquad x = U h.
\]

Here:
- `h` is the latent variable in `R^{d0}`
- `x` is the ambient vector in `R^{dx}`
- `U` has orthonormal columns

We will use `M = 4` mixture components.

## 3.2 Flow-matching path

Use the standard affine conditional path:

\[
X_t = \mu_t X_1 + \sigma_t X_0,
\qquad X_0 \sim \mathcal{N}(0, I_{dx}),
\qquad X_1 \sim P_1.
\]

For simplicity, use the **linear path**:

\[
\mu_t = t, \qquad \sigma_t = 1-t,
\]

so

\[
X_t = t X_1 + (1-t) X_0.
\]

Sample time from

\[
t \sim \mathrm{Unif}[\varepsilon, 1-\varepsilon]
\]

with a small epsilon, e.g. `eps = 1e-2`.

## 3.3 Training target for conditional FM

For the linear path,

\[
\dot{\mu}_t = 1, \qquad \dot{\sigma}_t = -1.
\]

So the conditional target velocity is

\[
\dot{\mu}_t X_1 + \dot{\sigma}_t X_0 = X_1 - X_0.
\]

That means for a training example:

- sample `X1` from data
- sample `X0 ~ N(0, I)`
- sample `t`
- form `Xt = t X1 + (1-t) X0`
- train the model to predict `X1 - X0` from `(Xt, t)`

This is the standard first-order conditional flow-matching regression loss.

---

## 4. Why we use a latent GMM

We want to evaluate against the **oracle marginal velocity** \(u_t^\star(x)\), not just training loss.

Using a latent GMM makes the latent marginal at time `t` tractable.

If

\[
h \sim \sum_{m=1}^M \pi_m \mathcal{N}(\mu_m, \Sigma_m),
\]

then the projected latent variable at time `t` is

\[
\bar h_t = U^\top X_t = t h + (1-t)\xi,
\qquad \xi \sim \mathcal{N}(0, I_{d0}),
\]

where \(\xi = U^\top X_0\) is the projection of the ambient Gaussian noise into the latent coordinates.

So \(\bar h_t\) is again a GMM with components

\[
\bar h_t \sim \sum_{m=1}^M \pi_m \,
\mathcal{N}\!\left(t\mu_m, \; t^2 \Sigma_m + (1-t)^2 I_{d0}\right).
\]

That lets us compute the latent density and its score exactly.

---

## 5. The oracle velocity formula you should use

This is the most important implementation detail.

The theory says the oracle marginal velocity decomposes as

\[
u_t^\star(x)
=
U\left[\alpha_t \bar h + \beta_t \nabla_{\bar h}\log p_t^h(\bar h)\right]
+
\kappa_t (I-UU^\top)x,
\qquad \bar h = U^\top x.
\]

For the **linear path** `mu_t=t`, `sigma_t=1-t`, these coefficients become:

\[
\kappa_t = \frac{\dot{\sigma}_t}{\sigma_t} = -\frac{1}{1-t},
\]

\[
\lambda_t = \dot{\mu}_t - \mu_t \kappa_t
= 1 - t\left(-\frac{1}{1-t}\right)
= \frac{1}{1-t},
\]

\[
\alpha_t = \kappa_t + \frac{\lambda_t}{\mu_t}
= -\frac{1}{1-t} + \frac{1}{t(1-t)}
= \frac{1}{t},
\]

\[
\beta_t = \frac{\lambda_t \sigma_t^2}{\mu_t}
= \frac{1}{1-t}\cdot \frac{(1-t)^2}{t}
= \frac{1-t}{t}.
\]

So for our experiments the oracle velocity is

\[
u_t^\star(x)
=
U\left[
\frac{1}{t}\bar h
+
\frac{1-t}{t}\nabla_{\bar h}\log p_t^h(\bar h)
\right]
-
\frac{1}{1-t}(I-UU^\top)x.
\]

This is the formula you should implement.

---

## 6. How to compute the latent score \(\nabla \log p_t^h(\bar h)\)

Let

\[
p_t^h(y) = \sum_{m=1}^M \pi_m \, \mathcal{N}(y; m_m(t), S_m(t))
\]

with

\[
m_m(t) = t\mu_m, \qquad S_m(t)=t^2\Sigma_m + (1-t)^2 I.
\]

Then

\[
\nabla_y \log p_t^h(y)
=
\sum_{m=1}^M r_m(y,t)\,\nabla_y \log \mathcal{N}(y; m_m(t), S_m(t)),
\]

where the posterior responsibilities are

\[
r_m(y,t)
=
\frac{\pi_m \mathcal{N}(y; m_m(t), S_m(t))}
{\sum_{j=1}^M \pi_j \mathcal{N}(y; m_j(t), S_j(t))}.
\]

For a Gaussian,

\[
\nabla_y \log \mathcal{N}(y; m, S) = -S^{-1}(y-m).
\]

Therefore,

\[
\nabla_y \log p_t^h(y)
=
-\sum_{m=1}^M r_m(y,t)\, S_m(t)^{-1}(y-m_m(t)).
\]

## Implementation advice

Implement a function like

```python
score_latent_gmm(y, t, pis, mus, Sigmas) -> score
```

where:
- `y` has shape `(batch, d0)`
- `t` has shape `(batch,)` or `(batch, 1)`
- output has shape `(batch, d0)`

Use numerically stable log-sum-exp logic for responsibilities.

---

## 7. Exact experiment design

## 7.1 Synthetic data generation

For each `(dx, d0)` setting:

1. Sample a random orthonormal matrix `U in R^{dx x d0}`
   - easiest way: sample a Gaussian matrix and run QR decomposition
   - keep the first `d0` columns

2. Define a latent 4-component GMM
   - mixture weights: uniform (`pi_m = 1/4`)
   - means: choose random vectors in `R^{d0}`, scaled so components are separated but not absurdly far apart
   - covariances: diagonal positive-definite matrices with moderate variance

## Concrete recommendation

Use:
- `M = 4`
- `pi = [0.25, 0.25, 0.25, 0.25]`
- component means: sample each `mu_m ~ N(0, 4 I_{d0})`
- component covariances: diagonal entries sampled uniformly from `[0.5, 1.5]`

Then sample `h`, and set `x = U h`.

---

## 8. IMPORTANT reproducibility policy for GMM and U

This section overrides any looser interpretation.

## 8.1 GMM policy

The latent GMM should be **fixed within each reported setting**, not resampled across seeds.

### Ambient sweep
For:
- `d0 = 4` fixed
- `dx in {32, 128, 512}` varying

use **one canonical latent GMM** shared across all those `dx` values.

### Intrinsic sweep
For:
- `dx = 128` fixed
- `d0 in {2, 4, 8, 16}` varying

use **one canonical latent GMM per `d0`**.

### Across seeds
Do **not** resample the GMM across the 3 seeds.

## 8.2 U policy

For each `(dx, d0)` setting:
- sample **one** orthonormal matrix `U`
- keep it fixed across all 3 seeds
- keep it fixed across all dataset sizes for that setting

This makes the geometry fixed, so seeds only capture training randomness.

## 8.3 Dataset policy

For each `(dx, d0)` setting, also fix:

- one master train set
- one validation set
- one test set

Then let:

- `n_train = 2000`
- `n_train = 10000`
- `n_train = 50000`

be nested subsets of the same master train set.

This makes comparisons much cleaner.

---

## 9. Dataset sizes

Use

```text
n_train in {2000, 10000, 50000}
```

For each setting also make:
- `n_val = 2000`
- `n_test = 5000`

The **test set** is where you compute held-out oracle velocity MSE.

---

## 10. Sweep settings

## Panel A: vary ambient dimension
Fix:
- `d0 = 4`

Vary:
- `dx in {32, 128, 512}`

## Panel B: vary intrinsic dimension
Fix:
- `dx = 128`

Vary:
- `d0 in {2, 4, 8, 16}`

For each `(dx, d0)` and each `n_train`, train with **3 seeds**.

Total number of training runs:

- panel A: `3 ambient settings * 3 dataset sizes * 3 seeds = 27`
- panel B: `4 intrinsic settings * 3 dataset sizes * 3 seeds = 36`

Total = **63 runs**

If runtime becomes a serious issue, do **not** silently shrink the grid. First report the measured runtime bottleneck.

---

## 11. Environment and package assumptions

Use a simple standard stack.

## Recommended packages

- `torch`
- `numpy`
- `matplotlib`
- `tqdm`
- `pandas`
- `json` or `pyyaml`
- optionally `scipy`

## Framework

Use **PyTorch**.

## Hardware policy

Write the code so it works on either CPU or GPU, but the intended full sweep should assume **GPU if available**.

### Practical meaning

- local debugging and smoke tests can be CPU or laptop GPU
- the full 63-run sweep is intended to run on cluster GPUs
- local development should focus on correctness, not on finishing the full sweep

---

## 12. Model choice

Use a **small time-conditioned vector model** for first-order flow matching.

## Recommendation

Use a small **MLP** unless someone explicitly insists on a transformer.

The data is just vectors, so an MLP is the simplest and least error-prone option.

## Suggested default

- 4 hidden layers
- width 256 or 512
- SiLU or GELU activations
- 64-dimensional sinusoidal time embedding
- output dimension `dx`

Keep the architecture template fixed across all settings.

---

## 13. Training objective

For each batch:

1. sample `x1` from the synthetic dataset
2. sample `x0 ~ N(0, I_dx)`
3. sample `t ~ Uniform[eps, 1-eps]`
4. form
   \[
   x_t = t x_1 + (1-t) x_0
   \]
5. target
   \[
   v_{\text{target}} = x_1 - x_0
   \]
6. predict
   \[
   \hat u_\theta(x_t, t)
   \]
7. train with mean squared error

## Suggested defaults

- optimizer: `AdamW`
- learning rate: `3e-4`
- weight decay: `1e-4`
- batch size: `256`
- epochs: `150`
- time epsilon: `1e-2`

Use early stopping only if it is easy to implement, but keep the reported protocol consistent.

---

## 14. Evaluation metric 1: Held-out oracle velocity MSE

This is the main metric.

For test points:

1. sample `x1` from held-out synthetic data
2. sample `x0 ~ N(0, I)`
3. sample `t ~ Uniform[eps, 1-eps]`
4. form `x_t = t x1 + (1-t) x0`
5. compute model prediction \(\hat u_\theta(x_t, t)\)
6. compute oracle velocity \(u_t^\star(x_t)\)
7. report

\[
\mathrm{MSE}_{\text{oracle}}
=
\mathbb{E}\|\hat u_\theta(X_t,t)-u_t^\star(X_t)\|_2^2.
\]

## Important warning

Do **not** use `x1 - x0` as the reported evaluation target. That is only the **conditional** training target. The main reported metric must compare against the **marginal oracle velocity**.

---

## 15. Figure 1: what success looks like

Figure 1 should show:

- when `d0` is fixed and `dx` increases, curves change **a little**
- when `dx` is fixed and `d0` increases, curves change **much more**

This is a qualitative validation. You do **not** need to fit exact slopes or claim exact asymptotic exponents.

## Plot format

- x-axis: number of training samples (`n_train`)
- y-axis: oracle velocity MSE
- one curve per `dx` in panel A
- one curve per `d0` in panel B
- mean ± std over 3 seeds

A log scale on x and/or y is fine if it helps readability.

---

## 16. Tangent / normal diagnostic

The theory says the oracle velocity splits as

\[
u_t^\star(x) = u_\parallel^\star(x,t) + u_\perp^\star(x,t)
\]

with

\[
u_\perp^\star(x,t) = \kappa_t (I-UU^\top)x.
\]

For the linear path,

\[
\kappa_t = -\frac{1}{1-t},
\]

so the theoretical normal term is explicit.

## 16.1 Post-hoc projection of learned velocity

After training, for any predicted velocity \(\hat u(x,t)\), define

\[
\hat u_\parallel = UU^\top \hat u,
\qquad
\hat u_\perp = (I-UU^\top)\hat u.
\]

Since we know the true `U`, this is easy.

## 16.2 Diagnostic outputs

For one representative model (or a very small subset), make:

### (A) Scatter plot
- x-axis: coordinates of theoretical normal term
- y-axis: coordinates of learned normal term

You can flatten coordinates across many examples into one big scatter plot.

### (B) Scalar normal-component MSE

\[
\mathbb{E}\left[\left\|
\hat u_\perp(x,t)
-
\kappa_t (I-UU^\top)x
\right\|_2^2\right].
\]

### (C) Reverse ODE trajectory plot
Starting from Gaussian samples, integrate the reverse ODE using the learned field and track

\[
\|(I-UU^\top)x_t\|_2
\]

over time.

## Recommended representative settings

Use one or two well-trained settings, for example:
- `dx=128, d0=4, n_train=50000`
- optionally `dx=512, d0=4, n_train=50000`

Do **not** expand this into another large grid.

---

## 17. Local development vs cluster execution

This is important.

The intended workflow is:

- **intern develops locally**
- **cluster is used only for larger validation runs and the final full sweep**

## Intern owns locally

The intern should implement and test:

- synthetic data generator
- latent GMM score function
- oracle velocity function
- model and training loop
- plotting
- logging and checkpointing
- tiny pilot runs

## You own on the cluster

You will handle:

- cluster environment setup
- Slurm scripts
- job submission
- monitoring
- reruns of failed jobs
- full 63-run sweep

The intern does **not** need cluster access to do most of the work.

---

## 18. What the intern should achieve locally

Treat local development as a correctness phase.

## Stage 1: tiny smoke test

Run something like:
- `dx=32`
- `d0=4`
- `n_train=2000`
- `epochs=5`

Check:
- shapes are correct
- data lies in the subspace
- oracle velocity code runs
- train loss decreases
- output files save correctly

## Stage 2: small pilot

Run one or two more realistic pilots locally, e.g.
- `dx=32, d0=4, n_train=2000, epochs=20`
- `dx=128, d0=4, n_train=2000, epochs=20`

Check:
- oracle MSE is finite
- oracle MSE improves with training
- plots render
- no numerical instability near `t=1`

## Stage 3: cluster-ready state

Before handoff, the intern should be able to provide:
- a working environment file
- a command for one run
- a command for the sweep
- a short README
- a tiny regression-test config

---

## 19. Code structure

Use a clean layout like:

```text
project/
  configs/
  scripts/
    run_one_config.py
    launch_sweep.py
  src/
    data.py
    models.py
    oracle.py
    train.py
    evaluate.py
    plots.py
  slurm/
    run_single.sbatch
    run_array.sbatch
    submit_sweep.sh
  experiments/
    flow_manifold_synth/
```

## Cluster-specific principle

The Python code should be cluster-agnostic.

Only the `slurm/` files should be cluster-specific.

---

## 20. Required command-line interface

The same Python script should run both locally and on the cluster.

## Example local command

```bash
python scripts/run_one_config.py --config configs/debug.yaml
```

## Example cluster command

```bash
python scripts/run_one_config.py --config configs/exp_dx128_d04_n50000_seed0.yaml
```

Do **not** create separate “local logic” and “cluster logic” in the training code. Only the config should change.

---

## 21. Slurm workflow

Your cluster uses **Slurm**.

That means the cleanest setup is:

- Python code is developed locally
- you wrap it with thin Slurm job scripts

## Recommended approach

### During debugging
Use **one Slurm job per config**.

### For the full sweep
Use a **Slurm job array**.

---

## 22. Example Slurm scripts

## 22.1 Single-config Slurm script

```bash
#!/bin/bash
#SBATCH --job-name=flowmanifold
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source ~/.bashrc
conda activate flowmanifold

cd /path/to/project

python scripts/run_one_config.py --config "$1"
```

Example submission:

```bash
sbatch slurm/run_single.sbatch configs/exp_dx128_d04_n50000_seed0.yaml
```

## 22.2 Job-array Slurm script

Create a file like:

```text
configs/all_runs.txt
```

with one config path per line.

Then:

```bash
#!/bin/bash
#SBATCH --job-name=flowarray
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-62%8

source ~/.bashrc
conda activate flowmanifold

cd /path/to/project

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" configs/all_runs.txt)
python scripts/run_one_config.py --config "$CONFIG"
```

This example limits the array to at most 8 simultaneous tasks.

---

## 23. Output directory policy

Create a fresh experiment directory, e.g.

```text
experiments/flow_manifold_synth/
  configs/
  logs/
  checkpoints/
  metrics/
  figures/
  summaries/
```

For each run, save:
- config copy
- seed
- GMM parameters
- matrix `U`
- train and validation curves
- final oracle velocity MSE
- any diagnostic metrics
- checkpoint if useful

Recommended per-run directory format:

```text
experiments/flow_manifold_synth/dx128_d04_n50000_seed0/
```

---

## 24. Reproducibility requirements

For each run, save:

- full config
- seed
- GMM parameters
- matrix `U`
- train metrics
- validation metrics
- test metrics
- wall-clock time

Also save one global `results.csv` or `results.jsonl` with one row per run.

Save final paper figures in both:
- `.png`
- `.pdf`

---

## 25. Sanity checks before the full sweep

Before launching all 63 jobs, verify the following on one small setting, e.g. `dx=32, d0=4, n_train=2000`.

### Sanity check 1: data really lies in the subspace

\[
\|(I-UU^\top)x\| \approx 0
\]

for generated data, up to floating-point error.

### Sanity check 2: training loss decreases

Verify the conditional FM training loss decreases over time.

### Sanity check 3: oracle velocity code runs stably

No NaNs, no shape mismatches, no numerical explosions.

### Sanity check 4: oracle MSE improves with training

On a pilot setting, verify that longer training improves oracle velocity MSE.

### Sanity check 5: normal component formula behaves sensibly

\[
u_\perp^\star(x,t) = -\frac{1}{1-t}(I-UU^\top)x
\]

should have the right shape and sensible magnitudes.

---

## 26. Pseudocode for one training batch

```python
# x1: batch from synthetic dataset, shape (B, dx)
x0 = torch.randn(B, dx, device=device)
t = eps + (1 - 2 * eps) * torch.rand(B, 1, device=device)

xt = t * x1 + (1 - t) * x0
target = x1 - x0

pred = model(xt, t)
loss = ((pred - target) ** 2).sum(dim=1).mean()
```

---

## 27. Pseudocode for oracle velocity

```python
def oracle_velocity(x, t, U, pis, mus, Sigmas):
    # x: (B, dx)
    # t: (B, 1)

    y = x @ U   # corresponds to U^T x in batched row-vector convention

    score = score_latent_gmm(y, t, pis, mus, Sigmas)   # (B, d0)

    parallel_latent = (1.0 / t) * y + ((1.0 - t) / t) * score
    parallel = parallel_latent @ U.T

    proj = x @ U @ U.T
    x_perp = x - proj
    normal = -(1.0 / (1.0 - t)) * x_perp

    return parallel + normal
```

Be consistent about row-vector vs column-vector conventions.

---

## 28. Main implementation pitfalls

### Pitfall 1
Sampling times too close to `t=1`, causing instability from terms like `1/(1-t)`.

### Pitfall 2
Using unstable GMM responsibility calculations instead of log-sum-exp.

### Pitfall 3
Reporting training loss against `x1 - x0` as the main experiment result.

### Pitfall 4
Resampling the GMM or `U` across seeds and muddying the comparison.

### Pitfall 5
Changing architecture across settings and making the plots harder to interpret.

### Pitfall 6
Hardcoding local or cluster paths.

### Pitfall 7
Making the cluster wrapper control experiment logic rather than just submitting configs.

---

## 29. What to hand back before cluster launch

Before the full cluster run, please provide:

1. `requirements.txt` or `environment.yml`
2. a working debug config
3. a working command for one run
4. a working sweep command or dry-run enumerator
5. a short README
6. local pilot results
7. one machine-readable results file from pilot runs

At that point, the code should be cluster-ready.

---

## 30. Short execution summary

If you need the shortest possible version:

- use **PyTorch**
- develop and debug **locally**
- assume **GPU on the Slurm cluster** for the full sweep
- fix the **latent GMM** within each reported setting
- fix **`U`** within each `(dx, d0)` setting
- keep datasets fixed within each setting
- vary only **training randomness** across seeds
- use config-driven scripts
- use **single-job Slurm scripts** for debugging
- use a **Slurm job array** for the full sweep
- evaluate using **oracle marginal velocity MSE**
- include the tangent/normal diagnostic
- save all outputs under `experiments/flow_manifold_synth/`

---

## 31. If you get stuck

When reporting a problem, include:

1. the exact file / function
2. the exact traceback or error
3. one concrete tensor-shape example
4. whether the issue is in:
   - data generation
   - oracle velocity
   - training
   - evaluation
   - plotting
   - Slurm submission

That will make debugging much faster.

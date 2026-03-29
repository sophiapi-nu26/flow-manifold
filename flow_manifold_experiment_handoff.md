# Handoff: Minimal Synthetic Experiments for **Learning Manifold Data with Flow Matching**

## 0. Why you are doing this

The paper is mostly theoretical. Reviewers asked for **a small empirical experiment** showing that, in practice:

1. performance depends much more on the **intrinsic dimension** `d0` than the **ambient dimension** `dx`, and
2. the learned velocity field has the predicted **tangent / normal decomposition**.

Your job is **not** to build a big benchmark suite. Your job is to build **one clean synthetic setup** and produce **two figures**.

This document is written so you can implement the experiment even if you have not read the paper closely.

---

## 1. The big picture in plain English

The paper studies data that really lives in a low-dimensional linear subspace of a higher-dimensional ambient space.

Concretely:

- latent variable: `h in R^{d0}`
- ambient observation: `x = U h in R^{dx}`
- `U` is a `dx x d0` matrix with orthonormal columns

So although `x` is a `dx`-dimensional vector, it only has `d0` true degrees of freedom.

The paper's core theoretical claim is that under this setup, the optimal flow-matching velocity field splits into:

- an **on-subspace / tangent** term (the hard/statistical part), and
- an **off-subspace / normal** term (a simple linear contraction back toward the subspace).

The reviewers want a small experiment showing that this structure is visible in practice.

---

## 2. What you need to deliver

Please produce the following:

### Main deliverables

1. **Figure 1: Intrinsic-vs-ambient scaling**
   - two panels:
     - panel A: fix `d0=4`, vary `dx in {32, 128, 512}`
     - panel B: fix `dx=128`, vary `d0 in {2, 4, 8, 16}`
   - for each setting, vary dataset size `n in {2000, 10000, 50000}`
   - y-axis: held-out **oracle velocity MSE**
   - x-axis: `n`
   - 3 random seeds, plot mean +/- std

2. **Figure 2: Tangent/normal diagnostic**
   - using the same trained models, check whether the learned normal component matches the theoretically predicted one
   - include:
     - scatter plot of learned normal component vs theoretical normal component
     - scalar normal-component MSE
     - one trajectory plot showing the off-subspace norm along the reverse ODE

3. **A short README / notes file**
   - exact hyperparameters
   - exact random seeds
   - which runs succeeded / failed
   - where figures are saved

### Nice-to-have but optional

- endpoint Sinkhorn-2 or approximate Wasserstein-type metric as a **secondary** metric only
- a table of final numbers corresponding to the plots

Do **not** spend time on:
- real data
- higher-order flow matching
- split-head architectures
- elaborate ablations
- exact exponent fitting

---

## 3. Minimal theory you need

### 3.1 Data model

We generate synthetic data from a latent Gaussian mixture model (GMM):

\[
h \sim \sum_{m=1}^M \pi_m \, \mathcal{N}(\mu_m, \Sigma_m),
\qquad x = U h.
\]

Here:
- `h` is the latent variable in `R^{d0}`
- `x` is the ambient vector in `R^{dx}`
- `U` has orthonormal columns

We will use `M=4` mixture components.

### 3.2 Flow-matching path

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

with a small epsilon, e.g. `eps = 1e-3` or `1e-2`.

### 3.3 Training target for conditional FM

For the linear path,

\[
\dot{\mu}_t = 1, \qquad \dot{\sigma}_t = -1.
\]

So the conditional target velocity is simply

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

so \(\bar h_t\) is again a GMM with components

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

### Implementation advice

Implement a function like:

```python
score_latent_gmm(y, t, pis, mus, Sigmas) -> score
```

where:
- `y` has shape `(batch, d0)`
- `t` has shape `(batch,)` or `(batch, 1)`
- output has shape `(batch, d0)`

Use numerically stable log-sum-exp logic when computing responsibilities.

---

## 7. Exact experiment design

## 7.1 Synthetic data generation

For each `(dx, d0)` setting:

1. Sample a random orthonormal matrix `U in R^{dx x d0}`.
   - easiest way: sample a Gaussian matrix and run QR decomposition
   - keep the first `d0` columns

2. Define a latent 4-component GMM:
   - mixture weights: uniform (`pi_m = 1/4`)
   - means: choose random vectors in `R^{d0}`, scaled so components are separated but not absurdly far apart
   - covariances: diagonal positive-definite matrices with moderate variance

### Concrete recommendation

Use:
- `M = 4`
- `pi = [0.25, 0.25, 0.25, 0.25]`
- component means:
  - sample each `mu_m ~ N(0, 4 I_{d0})`
- component covariances:
  - diagonal entries sampled uniformly from `[0.5, 1.5]`

Then sample `h`, and set `x = U h`.

### Important
Use the **same latent GMM construction recipe** across all runs so that differences are driven by `d0` and `dx`, not by wildly different data difficulty.

---

## 7.2 Dataset sizes

Use

```text
n_train in {2000, 10000, 50000}
```

For each run also make:
- `n_val = 2000`
- `n_test = 5000`

The **test set** is where you compute held-out oracle velocity MSE.

---

## 7.3 Sweep settings

### Panel A: vary ambient dimension
Fix:
- `d0 = 4`

Vary:
- `dx in {32, 128, 512}`

### Panel B: vary intrinsic dimension
Fix:
- `dx = 128`

Vary:
- `d0 in {2, 4, 8, 16}`

For each `(dx, d0)` and each `n_train`, train with **3 seeds**.

Total number of training runs:

- panel A: `3 ambient settings * 3 dataset sizes * 3 seeds = 27`
- panel B: `4 intrinsic settings * 3 dataset sizes * 3 seeds = 36`

Total = **63 runs**

That is a bit large but still manageable if the model is small.

If runtime becomes a problem, talk to us **before** reducing the grid.

---

## 8. Model choice

## Preferred choice
Use a **small time-conditioned vector model** for first-order flow matching.

The proposal originally said “standard FM transformer.” Since the data is just vectors, the architecture does not need to be fancy. The important thing is that:
- it maps `(x_t, t)` to a vector in `R^{dx}`
- the same architecture template is used across all settings
- capacity is large enough that we are not badly underfitting every run

## Practical recommendation
Use one of these:

### Option A: small MLP (easiest and acceptable unless told otherwise)
- input: concatenate `x_t` with a time embedding
- 4 hidden layers
- width 256 or 512
- SiLU / GELU activations
- output dimension `dx`

### Option B: lightweight transformer-style model
Only do this if you already have a clean implementation path.
Because the input is a single vector rather than an image or token sequence, an MLP is much simpler and less error-prone.

## My recommendation
Start with **Option A (MLP)** unless your mentor explicitly insists on a transformer.

The empirical claim being checked is the geometry/scaling behavior, not an architecture novelty claim.

---

## 9. Time embedding

Use a standard time embedding.

Simplest version:
- sinusoidal embedding of scalar `t`
- then a linear layer
- concatenate to `x_t`

Example:
- 32- or 64-dimensional sinusoidal embedding

This does not need to be fancy.

---

## 10. Training objective

For each batch:

1. sample `x1` from the synthetic dataset
2. sample `x0 ~ N(0, I_dx)`
3. sample `t ~ Uniform[e, 1-e]`
4. form
   \[
   x_t = t x_1 + (1-t)x_0
   \]
5. target:
   \[
   v_{\text{target}} = x_1 - x_0
   \]
6. predict
   \[
   \hat u_\theta(x_t, t)
   \]
7. train with mean squared error:
   \[
   \|\hat u_\theta(x_t, t) - (x_1 - x_0)\|_2^2
   \]

### Optimizer
Use AdamW.

### Suggested defaults
- learning rate: `1e-3` or `3e-4`
- batch size: `256` or `512`
- epochs: enough for convergence, e.g. `100` to `200`
- weight decay: small, e.g. `1e-4`

Use early stopping on validation loss if helpful.

---

## 11. Evaluation metric 1: Held-out oracle velocity MSE

This is the main metric.

For test points:

1. sample `x1` from held-out synthetic data
2. sample `x0 ~ N(0, I)`
3. sample `t ~ Uniform[e, 1-e]`
4. form `x_t = t x1 + (1-t) x0`
5. compute model prediction \(\hat u_\theta(x_t, t)\)
6. compute oracle velocity \(u_t^\star(x_t)\) using the formula in Section 5
7. report

\[
\mathrm{MSE}_{\text{oracle}}
=
\mathbb{E}\|\hat u_\theta(X_t,t)-u_t^\star(X_t)\|_2^2.
\]

### Important
This is **not** the same as training loss against `x1 - x0`.
We care about the **marginal oracle velocity**, because that is the object from the paper.

---

## 12. Figure 1: what success looks like

The figure should show:

- when `d0` is fixed and `dx` increases, the curves should change **a little**
- when `dx` is fixed and `d0` increases, the curves should change **much more**

You do **not** need to fit slopes or claim exact rates.
This is intended to be a **qualitative validation**.

### Plot format
- x-axis: number of training samples (`n_train`)
- y-axis: oracle velocity MSE **divided by `dx`** (per-dimension MSE)
- one curve per `dx` (left panel) or per `d0` (right panel)
- mean +/- std over 3 seeds

A log scale on the x-axis is fine.
A log scale on the y-axis is also fine if it helps readability.

### Important: normalize by `dx`

The oracle MSE is a sum over all `dx` output dimensions, so raw MSE scales with `dx` regardless of
intrinsic difficulty. Dividing by `dx` gives a per-dimension error that is comparable across settings.

Without this normalization, Panel A will show curves that differ by orders of magnitude simply because
`dx` is larger — obscuring the theoretical prediction that intrinsic dimension `d0` drives difficulty,
not ambient dimension `dx`. Similarly, Panel B's `d0`-dependent signal gets buried under the
`(dx - d0)`-dimensional normal-component baseline.

---

## 13. Tangent / normal diagnostic

This is the second figure.

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

### 13.1 Post-hoc projection of the learned velocity

After training, for any predicted velocity \(\hat u(x,t)\), define

\[
\hat u_\parallel = UU^\top \hat u,
\qquad
\hat u_\perp = (I-UU^\top)\hat u.
\]

Since we know the true `U` from data generation, this is easy.

### 13.2 Diagnostic outputs

For one representative model (or a small subset of representative models), make:

#### (A) Scatter plot
Plot

- x-axis: coordinates of the theoretical normal term
  \[
  \kappa_t (I-UU^\top)x
  \]
- y-axis: coordinates of the learned normal term
  \[
  \hat u_\perp(x,t)
  \]

You can flatten coordinates across many examples into one big scatter plot.

What we hope to see:
- points roughly on the diagonal

#### (B) Scalar normal-component MSE
Compute

\[
\mathbb{E}\left[\left\|
\hat u_\perp(x,t)
-
\kappa_t (I-UU^\top)x
\right\|_2^2\right].
\]

Report this as a number, or as a small bar plot / table.

#### (C) Reverse ODE trajectory plot
Starting from Gaussian samples, integrate the reverse ODE using the learned field and track

\[
\|(I-UU^\top)x_t\|_2
\]

over time.

Intuition:
- the off-subspace norm should shrink as we reverse toward the data distribution

### Which trained model should you use for this figure?
Use one or two representative well-trained settings, for example:
- `dx=128, d0=4, n=50000`
- maybe also `dx=512, d0=4, n=50000`

Do **not** make this part huge.

---

## 14. Reverse ODE details for the trajectory plot

We want to visualize how a point moves back toward the subspace.

### Suggested procedure

1. sample initial points `z ~ N(0, I_dx)`
2. solve the reverse-time ODE from `t = 0.999` down to `t = eps`
3. at each step compute
   \[
   \|(I-UU^\top)x_t\|_2
   \]
4. average over a batch of trajectories and plot mean vs time

### Numerical method
Use a simple ODE solver:
- Euler is acceptable
- RK4 is better if easy

Keep it simple and stable.

### Important sign convention
Be careful with time direction.
If you define the forward ODE as

\[
\frac{dx}{dt} = u_\theta(x,t),
\]

then for reverse integration from high `t` to low `t`, you can just step backward in time using negative `dt`.

Test this on a few samples first.

---

## 15. Secondary metric (optional)

If you have time, add a **secondary endpoint metric** comparing generated terminal samples to true data samples.

This could be:
- Sinkhorn-2 distance
- approximate Wasserstein-2
- MMD

But this is **optional** and should not delay the main experiment.

The primary metric remains oracle velocity MSE.

---

## 16. Sanity checks you must do before launching the full sweep

Before running all 63 jobs, verify the following on one small setting, e.g. `dx=32, d0=4, n=2000`.

### Sanity check 1: data really lies in the subspace
For generated data `x = U h`, verify numerically that

\[
\|(I-UU^\top)x\| \approx 0
\]

up to floating-point error.

### Sanity check 2: oracle velocity code
For random `(x,t)`, compute:
- oracle velocity from the theorem formula
- and separately, if feasible, a Monte Carlo estimate of `E[X1 - X0 | X_t=x]`

These do not need to match perfectly with a noisy MC estimate, but they should be in the same ballpark.
If MC is too annoying, skip this, but at least unit test shapes / stability carefully.

### Sanity check 3: training loss decreases
Verify the conditional FM training loss decreases over time.

### Sanity check 4: oracle MSE improves with more training
On a single setting, verify that training longer reduces oracle velocity MSE.

### Sanity check 5: normal component formula
Check that for random `x`,
\[
u_\perp^\star(x,t) = -\frac{1}{1-t}(I-UU^\top)x
\]
has the right shape and sensible magnitude.

---

## 17. Suggested code structure

A clean structure would be:

```text
project/
  data.py
  models.py
  train.py
  oracle.py
  evaluate.py
  plots.py
  configs/
  outputs/
```

### `data.py`
- generate random orthonormal `U`
- generate latent GMM parameters
- sample `h`
- form `x = U h`
- build train/val/test datasets

### `oracle.py`
- latent GMM density helpers
- latent GMM score function
- oracle velocity function
- projection helpers:
  - `proj_parallel(x, U)`
  - `proj_perp(x, U)`

### `models.py`
- time embedding
- MLP model (or transformer if you choose)

### `train.py`
- conditional FM training loop
- save checkpoints
- log train/val loss

### `evaluate.py`
- held-out oracle velocity MSE
- normal-component MSE
- reverse ODE trajectory evaluation

### `plots.py`
- figure 1
- figure 2

---

## 18. Pseudocode for one training batch

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

## 19. Pseudocode for oracle velocity

```python
def oracle_velocity(x, t, U, pis, mus, Sigmas):
    # x: (B, dx)
    # t: (B, 1)

    y = x @ U          # if U is (dx, d0), then y = U^T x in row-vector convention
                       # equivalently y = x @ U in code

    score = score_latent_gmm(y, t, pis, mus, Sigmas)   # shape (B, d0)

    parallel = (1.0 / t) * y + ((1.0 - t) / t) * score
    parallel = parallel @ U.T

    proj = x @ U @ U.T
    x_perp = x - proj
    normal = -(1.0 / (1.0 - t)) * x_perp

    return parallel + normal
```

### Important note on shapes
Be consistent about row-vector vs column-vector conventions.
In code with batched row vectors:
- `y = x @ U` corresponds to \(U^\top x\) in math notation
- `parallel @ U.T` maps latent vectors back into ambient space

---

## 20. Hyperparameter defaults

Use these unless you have a reason to change them:

### Data
- number of GMM components: `M=4`
- epsilon for time sampling: `1e-2`

### Model
- time embedding dim: `64`
- hidden width: `256`
- hidden layers: `4`
- activation: `SiLU`

### Training
- optimizer: `AdamW`
- learning rate: `3e-4`
- weight decay: `1e-4`
- batch size: `256`
- epochs: `150`
- gradient clipping: optional, e.g. `1.0`

### Evaluation
- `n_test_eval_pairs = 10000` if feasible
- 3 seeds per configuration

---

## 21. Reproducibility requirements

Please save:

- all random seeds
- `U`
- latent GMM parameters (`pi`, `mu`, `Sigma`)
- config for each run
- final metrics as `.json` or `.csv`
- plots as `.png` and `.pdf`

For each run, log:
- train loss
- validation loss
- oracle velocity MSE
- wall-clock time

---

## 22. Common failure modes

### Failure mode 1: numerical issues near `t=1`
Because terms like `1/(1-t)` appear, avoid sampling too close to 1.
That is why we clip time to `[eps, 1-eps]`.

### Failure mode 2: unstable GMM score computation
Compute GMM log-densities carefully using log-sum-exp.

### Failure mode 3: comparing to the wrong target
Do **not** use `x1 - x0` as the evaluation oracle.
That is only the **conditional** target used in training.
For evaluation, use the **marginal oracle velocity** formula from Section 5.

### Failure mode 4: architecture changes across settings
Do not quietly change model size across experiments.
Keep the model template fixed unless asked otherwise.

### Failure mode 5: overcomplicating the project
Remember: the goal is two clear figures, not a thesis.

---

## 23. What to write in the paper once results are ready

The intended take-away is something like:

> On synthetic data supported on a `d0`-dimensional linear subspace, standard first-order flow matching is substantially more sensitive to intrinsic dimension than ambient dimension. Moreover, the learned velocity field exhibits the predicted tangent/normal structure: its off-subspace component closely matches the theoretically derived orthogonal contraction.

Do not claim:
- exact asymptotic exponents
- broad real-world validation
- higher-order conclusions

This section is meant to be a **targeted synthetic validation** of the paper's theory.

---

## 24. Priority order if time is tight

If you are short on time, do things in this order:

1. get the synthetic data + training loop working
2. implement the oracle velocity correctly
3. produce Figure 1
4. produce the scalar normal MSE
5. produce the scatter plot
6. produce the trajectory plot
7. only then consider any secondary metric

---

## 25. Final checklist

Before you say the experiment is done, make sure you have:

- [ ] synthetic data generator with known `U`
- [ ] latent GMM score function
- [ ] oracle velocity function
- [ ] conditional FM training loop
- [ ] 63-run sweep completed (or documented reason if not)
- [ ] Figure 1 with mean +/- std over 3 seeds
- [ ] Figure 2 with scatter + scalar error + trajectory plot
- [ ] saved configs, seeds, and metrics
- [ ] short README describing how to reproduce the runs

---

## 26. If you get stuck

If you are blocked, the first things to report are:

1. the exact function / file where the issue occurs
2. a minimal error message or traceback
3. one concrete tensor shape example
4. whether the issue is:
   - data generation
   - training
   - oracle velocity computation
   - plotting
   - ODE rollout

That will make it much easier to help you quickly.

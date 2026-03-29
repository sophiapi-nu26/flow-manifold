# Cluster Transfer Guide

Everything you need to copy to the cluster and run the full sweep.

---

## Step 1 — Copy files to the cluster

You only need the code, not the experiment outputs. From your local machine, run:

```bash
rsync -av --progress \
  --exclude='experiments/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.git/' \
  /c/Users/sophi/Documents/magics/fw-transformers-ldls/ \
  YOUR_USERNAME@YOUR_CLUSTER_ADDRESS:/path/to/fw-transformers-ldls/
```

Replace `YOUR_USERNAME`, `YOUR_CLUSTER_ADDRESS`, and `/path/to/` with your actual cluster login and desired destination path.

```bash
rsync -av --progress --exclude='experiments/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.git/' /c/Users/sophi/Documents/magics/fw-transformers-ldls/ sophiapi@babel:/home/sophiapi/flow-manifold
```

```bash
ssh sophiapi@babel "mkdir -p /home/sophiapi/flow-manifold"
scp -r src scripts slurm configs requirements.txt README.md sophiapi@babel:/home/sophiapi/flow-manifold/
```

**What gets copied:**
```
src/
scripts/
slurm/
configs/           (including all 54 generated sweep configs + debug.yaml)
requirements.txt
README.md
```

**What is excluded** (not needed on the cluster):
```
experiments/       (outputs will be generated there)
__pycache__/
```

> **If rsync is not available on Windows**, use [WinSCP](https://winscp.net) to drag-and-drop, or run the rsync from inside WSL.

> **If configs/ hasn't been generated yet** (i.e. you haven't run `launch_sweep.py` locally), either do it locally first or run it on the cluster in Step 4 below.

---

## Step 2 — SSH into the cluster

```bash
ssh YOUR_USERNAME@YOUR_CLUSTER_ADDRESS
cd /path/to/fw-transformers-ldls
```

---

## Step 3 — Set up the conda environment

Only needs to be done once.

```bash
conda create -n flowmanifold python=3.11 -y
conda activate flowmanifold
pip install -r requirements.txt
```

Verify the setup with a 5-epoch smoke test:

```bash
python scripts/run_one_config.py --config configs/debug.yaml
```

You should see loss decreasing over 5 epochs and a final print line with `oracle_mse` and `time=`.

---

## Step 4 — Generate sweep configs (if not already done locally)

If you transferred the configs from local, skip this. Otherwise run:

```bash
python scripts/launch_sweep.py --output_root experiments/flow_manifold_synth
```

This writes 54 YAML files to `configs/` and `configs/all_runs.txt`.

---

## Step 5 — Edit the Slurm scripts

Open `slurm/run_array.sbatch` and `slurm/run_single.sbatch` and confirm:

- `--partition=gpu` matches your cluster's GPU partition name
- `conda activate flowmanifold` matches the environment name you created

If your cluster uses `module load` instead of conda, replace the `conda activate` line accordingly.

---

## Step 6 — Submit the full sweep

```bash
bash slurm/submit_sweep.sh
```

This generates configs and submits the first 50 jobs (babel's QOS limit is 50 submitted jobs at a time).

**Then submit the remaining 4 once your queue drops below 50:**

```bash
# Check how many jobs you currently have submitted
squeue -u $USER | wc -l   # subtract 1 for the header line

# When that number is below 50, submit the tail:
sbatch --array=50-53%8 slurm/run_array.sbatch
```

You don't need to wait for jobs to *finish* — just for enough to leave the queue (completed jobs disappear from `squeue`). With `%8` running at a time and 8-hour jobs, you'll typically have headroom within an hour or two.

---

## Step 7 — Monitor progress

```bash
# Check job status
squeue -u $USER

# Count completed runs (target: 54)
find experiments/flow_manifold_synth -name result.json | wc -l

# Check a specific run's output
cat experiments/flow_manifold_synth/dx128_d04_n50000_seed0/metrics.json | python -m json.tool | tail -10
```

---

## Step 8 — If any jobs fail

Find which runs are missing:

```bash
python scripts/aggregate_results.py --sweep_dir experiments/flow_manifold_synth
# Will print a list of run dirs missing result.json
```

Rerun each failed config:

```bash
python scripts/run_one_config.py --config configs/<run_name>.yaml --overwrite
```

Or resubmit as a single Slurm job:

```bash
sbatch slurm/run_single.sbatch configs/<run_name>.yaml
```

---

## Step 9 — Generate figures (after all 54 runs complete)

```bash
# Collect results into one CSV
python scripts/aggregate_results.py \
    --sweep_dir experiments/flow_manifold_synth \
    --output    experiments/flow_manifold_synth/results.csv

# Generate Figure 1 and Figure 2
python scripts/make_figures.py \
    --sweep_dir   experiments/flow_manifold_synth \
    --figures_dir experiments/flow_manifold_synth/figures
```

---

## Step 10 — Copy figures back to your local machine

```bash
# Run this locally
rsync -av \
  YOUR_USERNAME@YOUR_CLUSTER_ADDRESS:/path/to/fw-transformers-ldls/experiments/flow_manifold_synth/ \
  /c/Users/sophi/Documents/magics/fw-transformers-ldls/experiments/flow_manifold_synth/
```

Figures will be in `experiments/flow_manifold_synth/figures/` as both `.png` and `.pdf`.

---

## Quick reference — full sequence of commands on the cluster

```bash
# After SSH and cd to project directory:
conda activate flowmanifold
python scripts/run_one_config.py --config configs/debug.yaml        # smoke test
bash slurm/submit_sweep.sh                                          # submit sweep
squeue -u $USER                                                     # monitor
find experiments/flow_manifold_synth -name result.json | wc -l     # check progress
python scripts/aggregate_results.py                                 # after all 54 done
python scripts/make_figures.py                                      # generate figures
```

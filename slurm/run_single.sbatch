#!/bin/bash
#SBATCH --job-name=flowmanifold
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --chdir=/home/sophiapi/flow-manifold
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Usage:
#   sbatch slurm/run_single.sbatch configs/exp_dx128_d04_n50000_seed0.yaml

source ~/.bashrc
conda activate flowmanifold

mkdir -p logs

python scripts/run_one_config.py --config "$1"

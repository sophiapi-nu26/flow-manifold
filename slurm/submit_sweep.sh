#!/bin/bash
# submit_sweep.sh — generate configs and submit the full Slurm array.
#
# Usage:
#   bash slurm/submit_sweep.sh

set -e

cd "$(dirname "$(dirname "$(realpath "$0")")")"   # project root

echo "=== Generating sweep configs ==="
python scripts/launch_sweep.py --output_root experiments/flow_manifold_synth

echo ""
echo "=== Submitting Slurm array ==="
# Submit first 50 jobs (QOS limit is 50 submitted jobs at a time)
sbatch --array=0-49%8 slurm/run_array.sbatch
echo ""
echo "NOTE: 4 remaining jobs (50-53) must be submitted once the queue drops below 50."
echo "  Run: sbatch --array=50-53%8 slurm/run_array.sbatch"

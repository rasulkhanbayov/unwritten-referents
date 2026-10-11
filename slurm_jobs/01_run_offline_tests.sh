#!/bin/bash -l
#SBATCH -J skc_offline_tests
#SBATCH -o offline_tests_%j.out
#SBATCH -e offline_tests_%j.err
#SBATCH -p cpu-all
#SBATCH -c 2
#SBATCH --mem 4000MB
#SBATCH -t 00:20:00

# STEP 0: the two offline self-tests. CPU ONLY -- do not submit this to
# gpu-all/gpu-short/gpu-A100/gpu-H200; per the Panther End User Guide, CPU-only
# jobs belong in cpu-all so GPU queue slots are not wasted on work that does
# not touch a GPU. Both suites need only numpy. Must both print PASSED
# before anything below in this directory is submitted -- see skc_README.md.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
cd "$REPO/src"

echo "=== skc_metrics.py --smoke-test ==="
python3 skc_metrics.py --smoke-test

echo
echo "=== skc_pipeline_test.py ==="
python3 skc_pipeline_test.py

echo
echo "Both suites passed. Safe to submit GPU jobs in this directory."

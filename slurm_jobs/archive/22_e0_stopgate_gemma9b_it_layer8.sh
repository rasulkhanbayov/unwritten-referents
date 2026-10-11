#!/bin/bash -l
#SBATCH -J skc_e0_gemma9b_it_l8
#SBATCH -o e0_gemma9b_it_l8_%j.out
#SBATCH -e e0_gemma9b_it_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00

# STEP 9-PREREQUISITE, gemma-2-9b-it, ROUND 2: layer 8, not 16.
#
# WHY: 21_e0_stopgate_gemma2b9b_it.sh (layer 16, job 382412) FAILED, but
# it was the BEST result of any attempt so far, on either model -- BOTH
# estimators passed individually (diffmeans 0.725, probe 0.693, both
# clearly above the 0.65 threshold), only cross-estimator agreement
# (0.162) missed the 0.5 bar. Trend across all four real attempts so far
# (see EXECUTION_PLAN.md for the full table):
#   2b-it / layer 16 / n=20000: diffmeans pass, probe FAIL, agreement -0.072
#   2b-it / layer  8 / n=20000: diffmeans pass, probe pass, agreement  0.365
#   2b-it / layer  8 / n=30000: diffmeans pass, probe FAIL, agreement  0.187
#   9b-it / layer 16 / n=20000: diffmeans pass, probe pass, agreement  0.162
#
# Layer 8 is the base 2B model's OWN working layer (agreement 0.620
# there) and was also 2b-it's best result among its own three attempts
# (agreement 0.365, though still short) -- trying it here follows the
# same reasoned progression already used for 2b-it, on a model where
# layer 16 has now also been tried and come closer than any 2b-it attempt.
#
# Same --n 20000/--fit-frac 0.5/--seed 0/--skip-sae as every other E0 run
# this session. Output to its own directory (e0_gemma2b9b_it_layer8) --
# does not touch the layer-16 result already on disk.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see other scripts' identical comment.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e0_gemma2b9b_it_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-9b-it \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-9b-it, layer 8) PASSED. frozen_gate_*.json files are in $OUT."
    echo "Update 10_e6_cot_gemma2b_it.sh (or a new copy) to use --model gemma-2-9b-it,"
    echo "--gate-dir $OUT, --layer 8 before running E6."
else
    echo "E0 (gemma-2-9b-it, layer 8) FAILED (exit $exit_code)." >&2
    echo "This would be the FIFTH failed attempt across two models -- discuss with the" >&2
    echo "user before trying anything further (another layer, a bigger n, a different" >&2
    echo "model, or accepting E6 cannot run with this gate-fitting procedure). Do not" >&2
    echo "keep sweeping unprompted." >&2
fi
exit "$exit_code"

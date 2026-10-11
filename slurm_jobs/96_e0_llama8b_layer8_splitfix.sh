#!/bin/bash -l
#SBATCH -J skc_e0_llama8b_l8_splitfix
#SBATCH -o e0_llama8b_l8_splitfix_%j.out
#SBATCH -e e0_llama8b_l8_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# VERSION3 era, C7 split-leakage fix, extended to Llama-3.1-8B.
#
# Same leak as Gemma-2-2b (see EXECUTION_PLAN.md's full account): the
# fit/held-out entity split in skc_run_e0.py's split_entities() only ever
# used data/onehop_queries.json's own head entities, with no cross-check
# against data/twohop_queries.json's bridge/target entities (402/847,
# 47.5% overlap, measured directly against the real query files --
# model-independent, since it's purely a property of the shared query
# JSONs, not of any model's own activations). skc_run_e0.py's
# --named-bridge-queries flag is now REQUIRED and the fix
# (exclude_from_fit) is baked into the driver script itself -- this job
# automatically gets the fix with no extra flags needed.
#
# Matches job 44's exact invocation (--n 20000, --skip-sae, --fit-frac
# 0.5, --seed 0) -- llama-3.1-8b's own headline layer/estimator choice,
# unchanged. Only the output dir changes, so the old leaky-split result
# (e0_llama8b_layer8_v2) stays on disk for comparison.

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
OUT="$REPO/results/e0_llama8b_layer8_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model llama-3.1-8b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (llama-3.1-8b, layer 8, SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Check $OUT/E0.json's own gate_open field. Compare its agreement against"
    echo "the OLD e0_llama8b_layer8_v2/E0.json (agreement=0.613, leaky split, PASSED)."
else
    echo "E0 (llama-3.1-8b, layer 8, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

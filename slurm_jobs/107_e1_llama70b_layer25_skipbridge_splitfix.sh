#!/bin/bash -l
#SBATCH -J skc_e1_llama70b_l25_skipbridge
#SBATCH -o e1_llama70b_l25_skipbridge_%j.out
#SBATCH -e e1_llama70b_l25_skipbridge_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# VERSION3 era: E1 for Llama-3.1-70B under the SPLIT-LEAKAGE FIX, WITH
# --skip-bridge-resolution.
#
# WHY THIS JOB EXISTS: job 105 (attempting full bridge resolution at
# layer 25, this model's own best E0 gate layer) got 0/494 Patchscopes
# resolutions -- a clean null, same failure signature (literal template
# scaffolding, no entity substituted) as every other (model, layer)
# tried except Gemma-2-2b/layer-16. See EXECUTION_PLAN.md's own summary:
# Patchscopes has now been tried at 4 combinations across this project
# and succeeded exactly once. --skip-bridge-resolution (K1=1 only) is
# therefore the only way to get a usable E1 result for Llama-3.1-70b at
# all, the same disclosed methodological weakening as job 106
# (gemma-2-9b) and every other layer-8 Llama run this project has done.
#
# This result will NOT be directly comparable to Gemma-2-2b/layer-16's
# own gs-vs-gb finding (full K1=1+bridge-resolved filter) -- report it
# with that caveat stated alongside it, same as job 106's own gemma-2-9b
# result.
#
# Uses e0_llama70b_layer25_splitfix (job 103, PASSED at agreement=0.609,
# the most robust gate of any model in the project) as --gate-dir.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see other scripts' identical comment.

export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=1

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e1_llama70b_layer25_skipbridge_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (llama-3.1-70b, layer 25, SPLIT-LEAKAGE FIX, K1=1-only) completed (exit 0)."
    echo "Check $OUT/E1.json's auroc.gate_to_M. NOTE: this uses the WEAKER"
    echo "K1=1-only filter (job 105 confirmed 0 bridge resolutions at this"
    echo "layer) -- not directly comparable to Gemma-2-2b/layer-16's own"
    echo "full-filter gs-vs-gb finding without that caveat stated alongside it."
else
    echo "E1 (llama-3.1-70b, layer 25, K1=1-only) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

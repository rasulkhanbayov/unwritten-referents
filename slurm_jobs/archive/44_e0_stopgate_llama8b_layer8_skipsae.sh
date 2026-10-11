#!/bin/bash -l
#SBATCH -J skc_e0_llama8b_l8_nosae
#SBATCH -o e0_llama8b_l8_nosae_%j.out
#SBATCH -e e0_llama8b_l8_nosae_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 21: E0 on Llama 3.1 8B, layer 8, WITHOUT SAE.
#
# WHY: job 385048 (all three estimators) FAILED -- diffmeans (0.853)
# and probe (0.827) both individually pass and AGREE WELL WITH EACH
# OTHER (0.613), but SAE (0.642) does not even clear 0.65 on its own,
# and drags min_pairwise_spearman down to 0.219. Same pattern as
# Gemma-2-2b and Gemma-2-9b (see job 43's identical comment) -- three
# for three across model families, a genuine structural SAE-disagreement
# finding, not a per-model bug. User's explicit decision: accept a
# --skip-sae (2-estimator) gate for this model too.
#
# Output to a NEW directory (e0_llama8b_layer8_v2) -- the old
# e0_llama8b_layer8 (3-estimator, failed) result stays on disk as a
# historical record.

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
OUT="$REPO/results/e0_llama8b_layer8_v2"
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
    echo "E0 (llama-3.1-8b, layer 8, diffmeans+probe only) PASSED."
    echo "Continue the E1/E3/E5 chain at this layer for this model."
else
    echo "E0 (llama-3.1-8b, layer 8, diffmeans+probe only) FAILED (exit $exit_code)." >&2
    echo "This would be a NEW finding (diffmeans/probe themselves failed to" >&2
    echo "agree without SAE in the mix) -- report back with the specific" >&2
    echo "numbers before trying another layer." >&2
fi
exit "$exit_code"

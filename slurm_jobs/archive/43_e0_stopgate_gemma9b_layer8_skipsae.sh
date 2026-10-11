#!/bin/bash -l
#SBATCH -J skc_e0_gemma9b_l8_nosae
#SBATCH -o e0_gemma9b_l8_nosae_%j.out
#SBATCH -e e0_gemma9b_l8_nosae_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 20: E0 on Gemma 2 9B BASE, layer 8, WITHOUT SAE.
#
# WHY: job 384006 (all three estimators) FAILED -- diffmeans (0.861)
# and probe (0.824) both individually pass and AGREE WELL WITH EACH
# OTHER (0.534), but SAE (0.699) drags min_pairwise_spearman down to
# 0.241, well below the 0.50 bar. This is the SAME pattern seen on
# Gemma-2-2b (job 384005) and Llama-3.1-8b (job 385048) -- three for
# three across model families -- confirmed a genuine, structural
# property of comparing a single max-min-selected SAE latent against
# full-residual-stream linear directions, not a per-model bug. User's
# explicit decision: accept a --skip-sae (2-estimator) gate for this
# model, matching the pattern already used for Gemma-2-2b
# (e0_gemma2b_layer8_v2) and Llama-3.1-70b (e0_llama70b_layer8).
#
# Output to a NEW directory (e0_gemma9b_layer8_v2) -- the old
# e0_gemma9b_layer8 (3-estimator, failed) result stays on disk as a
# historical record documenting the SAE-disagreement finding, not
# overwritten.

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
OUT="$REPO/results/e0_gemma9b_layer8_v2"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-9b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-9b BASE, layer 8, diffmeans+probe only) PASSED."
    echo "Continue the E1/E3/E5 chain at this layer for this model."
else
    echo "E0 (gemma-2-9b BASE, layer 8, diffmeans+probe only) FAILED (exit $exit_code)." >&2
    echo "This would be a NEW finding (diffmeans/probe themselves failed to" >&2
    echo "agree without SAE in the mix) -- report back with the specific" >&2
    echo "numbers before trying another layer." >&2
fi
exit "$exit_code"

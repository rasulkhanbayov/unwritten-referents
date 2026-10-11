#!/bin/bash -l
#SBATCH -J skc_e1_gemma9b_l8_skipbridge
#SBATCH -o e1_gemma9b_l8_skipbridge_%j.out
#SBATCH -e e1_gemma9b_l8_skipbridge_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# VERSION3 era: E1 for Gemma-2-9B under the SPLIT-LEAKAGE FIX, WITH
# --skip-bridge-resolution.
#
# WHY THIS JOB EXISTS: job 104 (attempting full bridge resolution,
# without --skip-bridge-resolution) got 0/346 Patchscopes resolutions at
# layer 8 -- a clean, confirmed null, same as Gemma-2-2b's own layer-8
# result (job 95). Gemma-2-9b's only passing gate is layer 8 (this model
# was not part of the 3-model full-layer-sweep request, so no deeper
# candidate layer has been checked) -- --skip-bridge-resolution (K1=1
# only) is therefore the only way to get a usable E1 result for this
# model at all, the same disclosed methodological weakening as
# Gemma-2-2b's OLD (pre-layer-16) result and every Llama layer-8 run.
#
# This result will NOT be directly comparable to Gemma-2-2b/layer-16's
# own gs-vs-gb finding (which used the FULL K1=1+bridge-resolved filter)
# -- report it as a K1=1-only number alongside that caveat, the same way
# job 95 already established that layer 8 (any Gemma model, so far) and
# the full filter are mutually exclusive.
#
# Uses e0_gemma9b_layer8_splitfix (job 97, PASSED at agreement=0.505 --
# thin margin, already flagged) as --gate-dir.

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
OUT="$REPO/results/e1_gemma9b_layer8_skipbridge_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-9b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma9b_layer8_splitfix" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (gemma-2-9b, layer 8, SPLIT-LEAKAGE FIX, K1=1-only) completed (exit 0)."
    echo "Check $OUT/E1.json's auroc.gate_to_M. NOTE: this uses the WEAKER"
    echo "K1=1-only filter (job 104 confirmed 0 bridge resolutions at this"
    echo "layer/model) -- not directly comparable to Gemma-2-2b/layer-16's own"
    echo "full-filter gs-vs-gb finding without that caveat stated alongside it."
else
    echo "E1 (gemma-2-9b, layer 8, K1=1-only) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

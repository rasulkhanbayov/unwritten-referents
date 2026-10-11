#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_l12_splitfix
#SBATCH -o e0_gemma2b_l12_splitfix_%j.out
#SBATCH -e e0_gemma2b_l12_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# VERSION3 era, C7 split-leakage fix, layer sweep. Job 89
# (e0_gemma2b_layer8_splitfix) re-ran E0 at layer 8 under the corrected
# split (split_entities() now excludes every head/bridge/target entity in
# data/twohop_queries.json from the fit set -- see that job's own header
# for the full account of the 402/847 leaked-bridge-entity finding this
# fixes). RESULT: agreement dropped from 0.547 (leaky split, passing) to
# 0.317 (fixed split, FAILING the 0.50 bar) -- both estimators still
# individually clear AUROC 0.65 comfortably (0.880/0.844), but they now
# disagree substantially more about WHICH entities are known/unknown once
# neither can lean on leaked bridge/target entities.
#
# This job checks layer 12 under the SAME fix, to see whether the leak
# was propping up agreement specifically at layer 8, or whether the
# problem is broader. Historical context (leaky split, for comparison):
# layer 12's own OLD result (e0_gemma2b_layer12/E0.json) already had
# agreement=0.273 (failing) even WITH the leak -- worse than layer 8's own
# leaky-split 0.547. If the fixed split makes layer 12 even worse, that is
# consistent with the leak having inflated agreement more at some layers
# than others, not evidence layer 12 was ever a good candidate.
#
# Matches job 72's own invocation exactly (all three estimators attempted,
# no --skip-sae, --n 20000, --fit-frac 0.5, --seed 0) except for the split
# fix itself, so the two runs are otherwise apples-to-apples.

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
OUT="$REPO/results/e0_gemma2b_layer12_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layer 12 \
    --fit-frac 0.5 \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b, layer 12, SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Check $OUT/E0.json's own gate_open field -- exit 0 does not mean the"
    echo "positive control passed, only that the run completed. Compare against"
    echo "the OLD e0_gemma2b_layer12/E0.json (agreement=0.273, leaky split) to see"
    echo "whether the fix made this layer's own agreement better or worse."
else
    echo "E0 (gemma-2-2b, layer 12, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

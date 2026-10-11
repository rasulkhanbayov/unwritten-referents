#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_l7_splitfix
#SBATCH -o e0_gemma2b_l7_splitfix_%j.out
#SBATCH -e e0_gemma2b_l7_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# VERSION3 era, C7 split-leakage fix, layer sweep (companion to job 92's
# own layer-19 attempt; see that job's header and job 89's for the full
# leak/fix account).
#
# Layer 7 has the SECOND-highest diffmeans~probe agreement of all 26
# layers under the leaky split (0.632, from
# results/e0_layersweep_gemma2b/e0_layersweep_full.json), right behind
# layer 19 (0.661) and ahead of layer 20 (0.623) and layer 8 itself
# (0.547). Adjacent to layer 8, so likely similar SAE-dictionary
# availability and Patchscopes behavior if either of those becomes
# relevant again once a layer clears the positive control.

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
OUT="$REPO/results/e0_gemma2b_layer7_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layer 7 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b, layer 7, SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Check $OUT/E0.json's own gate_open field. Compare its agreement against"
    echo "the leaky-split sweep's own layer-7 diffmeans~probe value (0.632)."
else
    echo "E0 (gemma-2-2b, layer 7, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

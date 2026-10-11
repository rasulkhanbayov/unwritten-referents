#!/bin/bash -l
#SBATCH -J skc_e1_gemma2b_l16_splitfix
#SBATCH -o e1_gemma2b_l16_splitfix_%j.out
#SBATCH -e e1_gemma2b_l16_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# VERSION3 era, C7 split-leakage fix: E1 at the NEW headline layer.
#
# WHY LAYER 16, NOW: layer 8 was originally chosen over layer 16 (job 27)
# specifically because layer 16's E0 failed cross-estimator agreement
# under the OLD leaky split (-0.227). Under the FIXED split (job 91,
# results/e0_gemma2b_layer16_splitfix), that result reverses completely:
# layer 16 now has the BEST agreement of any layer tried (0.706,
# diffmeans=0.856 probe=0.872), while layer 8 itself now FAILS under the
# fix (0.317, job 89). User confirmed: adopt layer 16 as the new headline
# layer for Gemma-2-2b.
#
# NOT using --skip-bridge-resolution here (unlike job 27's layer-8 run):
# layer 16 is exactly the layer where Patchscopes was already confirmed to
# decode the bridge cleanly (see job 27's own header). Layer 8 needed
# --skip-bridge-resolution as a disclosed methodological weakening ONLY
# because Patchscopes decoded to template noise there. That constraint does
# not apply at layer 16, so this run recovers full bridge-resolution
# evidence (K1=1 AND Patchscopes agreement) rather than K1=1 alone.
#
# Uses the NEW gate dir (e0_gemma2b_layer16_splitfix) exclusively -- per
# job 89/91's own guidance, never point E1 at a pre-fix gate directory
# again.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see job 27's own header
# for the full undefined-symbol/nvJitLink incident this avoids.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e1_gemma2b_layer16_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-2b \
    --layer 16 \
    --gate-dir "$REPO/results/e0_gemma2b_layer16_splitfix" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (gemma-2-2b, layer 16, SPLIT-LEAKAGE FIX applied, full bridge"
    echo "resolution i.e. NOT --skip-bridge-resolution) completed (exit 0)."
    echo "Check $OUT/E1.json against the OLD e1_gemma2b_layer8 result to see"
    echo "how the headline-layer switch changes the reported finding."
else
    echo "E1 (gemma-2-2b, layer 16, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

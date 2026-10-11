#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_l16_splitfix
#SBATCH -o e0_gemma2b_l16_splitfix_%j.out
#SBATCH -e e0_gemma2b_l16_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# VERSION3 era, C7 split-leakage fix, layer sweep (companion to job 90's
# own layer-12 attempt; see that job's header for the full account of the
# leak and job 89's own layer-8 result under the fix: agreement dropped
# from 0.547 to 0.317, FAILING the 0.50 bar).
#
# Layer 16's own OLD result (e0_gemma2b_layer16/E0.json, leaky split)
# already had agreement=-0.227 -- the worst of any layer tried, including
# under the leak's own inflation. Layer 16 is also the layer where
# Patchscopes decodes cleanly (per the old project's own diagnostic
# trail), so it is worth checking under the fix even though it is the
# least likely candidate to pass, purely so the sweep covers every layer
# with a real historical result rather than skipping the one most
# relevant to bridge-resolution confirmation.
#
# Matches job 04/18's own invocation (--skip-sae, --n 20000, --fit-frac
# 0.5, --seed 0) except for the split fix itself.

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
OUT="$REPO/results/e0_gemma2b_layer16_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layer 16 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b, layer 16, SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Check $OUT/E0.json's own gate_open field. Compare against the OLD"
    echo "e0_gemma2b_layer16/E0.json (agreement=-0.227, leaky split, already the"
    echo "worst layer tried) to see whether the fix changed this."
else
    echo "E0 (gemma-2-2b, layer 16, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

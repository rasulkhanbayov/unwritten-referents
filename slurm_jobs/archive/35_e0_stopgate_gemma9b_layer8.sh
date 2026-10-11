#!/bin/bash -l
#SBATCH -J skc_e0_gemma9b_l8
#SBATCH -o e0_gemma9b_l8_%j.out
#SBATCH -e e0_gemma9b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 12: E0 on Gemma 2 9B BASE (not instruction-tuned -- that is
# e0_gemma2b9b_it_layer8, already passed and used for E6), layer 8, all
# three estimators including SAE (now wired up -- see
# skc_harness.py's load_sae() docstring and job 34's comment for the
# fixes that unblocked it).
#
# LAYER CHOICE: 8, same starting point that worked for Gemma-2-2b base
# (e0_gemma2b_layer8_v2, agreement 0.547) and for Gemma-2-9b-it
# (e0_gemma2b9b_it_layer8, agreement 0.588) -- neither of those
# transfers is a guarantee for the BASE 9B model, which has never had
# its own E0 attempt this session. Treat this as a genuine first attempt,
# not a re-confirmation; if it fails, try a second layer next per the
# same iterative pattern used for 2B and 9B-it earlier, reporting back
# before re-running rather than guessing further unprompted.
#
# --mem 60000MB: 9B needs meaningfully more memory than 2B for the same
# batch sizes, same reasoning as the existing 9B-it E6 script.

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
OUT="$REPO/results/e0_gemma9b_layer8"
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
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-9b BASE, layer 8) PASSED. Continue the E1/E3/E5 chain at"
    echo "this layer for this model."
else
    echo "E0 (gemma-2-9b BASE, layer 8) FAILED (exit $exit_code)." >&2
    echo "Try a different layer next -- report back with the specific" >&2
    echo "AUROC/agreement numbers before re-running, same discipline as" >&2
    echo "the 2B and 9B-it E0 attempts earlier this session." >&2
fi
exit "$exit_code"

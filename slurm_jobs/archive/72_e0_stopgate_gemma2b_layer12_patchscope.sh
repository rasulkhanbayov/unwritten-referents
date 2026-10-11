#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_l12
#SBATCH -o e0_gemma2b_l12_%j.out
#SBATCH -e e0_gemma2b_l12_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# GPU item 1 of 9 remaining paper TORUNs: retry the Patchscopes-confirmed
# bridge-resolution chain on Gemma 2 2B at a NEW layer, since both layers
# tried so far have their own real problem: layer 8 works for the E0
# gate (agreement 0.547) but has never had bridge resolution checked
# with Patchscopes; layer 16 DOES let Patchscopes decode cleanly, but
# its own E0 cross-estimator agreement is NEGATIVE (-0.227, confirmed
# directly from results/e0_gemma2b_layer16/E0.json) -- far below the
# 0.50 bar, so layer 16 cannot supply a trustworthy gate even though
# Patchscopes works there. Re-trying layer 16 itself would not help;
# this tries layer 12 instead, a genuinely new bet between the two
# already-tried layers.
#
# If this passes BOTH bars (E0 agreement >= 0.50 AND Patchscopes
# decodes a clean bridge entity), submit
# 73_e1_main_gemma2b_layer12_patchscope.sh next, WITHOUT
# --skip-bridge-resolution, to get the real (not --skip-bridge-resolution
# weakened) E1 chain the paper's own text flags as still missing.
#
# If this fails again, report which bar failed (E0 agreement, or
# Patchscopes decode quality) before trying a fourth layer -- do not
# guess a new layer blind a second time.

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
OUT="$REPO/results/e0_gemma2b_layer12"
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
    echo "E0 (gemma-2-2b, layer 12) PASSED. Check e0_gemma2b_l12_*.out for the"
    echo "actual agreement value before proceeding -- 'exit 0' means the run"
    echo "completed, gate_open may still be false. If gate_open is true, submit"
    echo "73_e1_main_gemma2b_layer12_patchscope.sh next (WITHOUT"
    echo "--skip-bridge-resolution) to test whether Patchscopes ALSO decodes"
    echo "cleanly at this layer -- both bars must clear for this TORUN to close."
else
    echo "E0 (gemma-2-2b, layer 12) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

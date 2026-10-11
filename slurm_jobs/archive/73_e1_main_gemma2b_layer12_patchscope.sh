#!/bin/bash -l
#SBATCH -J skc_e1_gemma2b_l12
#SBATCH -o e1_gemma2b_l12_%j.out
#SBATCH -e e1_gemma2b_l12_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00

# GPU item 1, part 2: E1 on Gemma 2 2B at layer 12, WITHOUT
# --skip-bridge-resolution -- runs the REAL Patchscopes-based
# independent bridge-resolution check skc_runbook.txt calls for,
# closing the disclosed methodological weakening every other E1 run in
# this project has carried. ONLY SUBMIT once
# 72_e0_stopgate_gemma2b_layer12_patchscope.sh has PASSED with
# gate_open=true -- if E0 failed at this layer, this run's own gate
# would be untrustworthy before Patchscopes is even checked.
#
# -t 16:00:00 (up from the --skip-bridge-resolution runs' usual
# 12:00:00): the real Patchscopes decode-and-check path does more work
# per query than the --skip-bridge-resolution path (which only checks
# K1), so budget more time for the first real attempt at this layer.

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
OUT="$REPO/results/e1_gemma2b_layer12_patchscope"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-2b \
    --layer 12 \
    --gate-dir "$REPO/results/e0_gemma2b_layer12" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"

echo ""
echo "This run does NOT pass --skip-bridge-resolution -- if it completes, the"
echo "real Patchscopes-confirmed bridge-resolution filter was applied,"
echo "closing the disclosed methodological weakening every OTHER E1 run in"
echo "this project has carried. Report retained_fraction and the real"
echo "K1=1+bridge-resolved filtered query count against the --skip-bridge-"
echo "resolution numbers already in the paper."

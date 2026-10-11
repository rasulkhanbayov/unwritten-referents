#!/bin/bash -l
#SBATCH -J skc_e4_gemma2b_l8
#SBATCH -o e4_gemma2b_l8_%j.out
#SBATCH -e e4_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# STEP 7, SECONDARY TRACK, RE-RUN AFTER THE PADDING-SIDE FIX AND THE
# LAYER-8-VS-16 TRADEOFF DECISION: E4 at layer 8.
#
# WHY: --queries now points at results/e1_gemma2b_layer8/queries_filtered.json
# -- the K1=1-only (--skip-bridge-resolution) filtered set from the new
# layer-8 E1 run (job 382900), NOT the old Patchscopes-filtered
# layer-16 set. This means E4's own baseline comparisons inherit the
# SAME caveat E1's layer-8 run carries: no independent confirmation the
# bridge was resolved internally, only that K1=1 in isolation. Disclose
# this alongside E4's own results, same as E1's.
#
# --gate-dir now points at e0_gemma2b_layer8_v2 (the first genuinely
# trustworthy E0 result this session, WITH the padding-side fix
# applied).
#
# --run-self-ask still ON deliberately -- see the original layer-16
# script's own comment for why (skc_REDFLAGS.md/skc_runbook.txt call it
# mandatory).
#
# Output to a NEW directory (e4_gemma2b_layer8) -- the old
# e4_gemma2b result (pre-padding-fix, Patchscopes-filtered queries)
# stays on disk as a historical record.

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
OUT="$REPO/results/e4_gemma2b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

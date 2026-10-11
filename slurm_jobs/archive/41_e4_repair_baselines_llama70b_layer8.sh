#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_l8
#SBATCH -o e4_llama70b_l8_%j.out
#SBATCH -e e4_llama70b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# STEP 18: E4 (repair baselines) on Llama 3.1 70B (4-bit), layer 8.
#
# --queries points at results/e1_llama70b_layer8/queries_filtered.json
# (job 385284's own output, n=494, K1=1-only / --skip-bridge-resolution
# filtered set) -- inherits the SAME disclosed caveat as every other
# --skip-bridge-resolution result this session: no independent
# Patchscopes confirmation, only K1=1 in isolation. Report this
# alongside E4's own numbers here, same as everywhere else.
#
# --gate-dir points at e0_llama70b_layer8 (job 384352, PASSED cleanly
# on the FIRST genuine attempt -- no second-layer caveat needed for
# this model, unlike Gemma-2-2b/9b or Llama-3.1-8b).
#
# --run-self-ask ON, same as every other E4 run this session --
# skc_REDFLAGS.md/skc_runbook.txt call the self-ask baseline mandatory.
#
# --mem/-c match every other 70B job this session.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see other scripts' identical comment.

export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=1

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e4_llama70b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_llama70b_layer8/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_llama70b_layer8" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

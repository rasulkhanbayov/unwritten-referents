#!/bin/bash -l
#SBATCH -J skc_e3_qwen32b_l24
#SBATCH -o e3_qwen32b_l24_%j.out
#SBATCH -e e3_qwen32b_l24_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# *** DISCLOSED LIMITATION -- see 69_e1_main_qwen32b_layer24.sh's own
# comment for the full explanation. Short version: E0 FAILED its own
# stop-gate at this model/layer (diffmeans/probe both individually
# strong -- AUROC 0.9070/0.8816 -- but disagree with each other,
# Spearman=0.2963, required >=0.5, essentially unchanged from layer 8's
# 0.2966). User's own explicit choice: proceed anyway with prominent
# disclosure, same standard as HaloScope's own disclosed-uncertain-
# reconstruction flag elsewhere in this project. ***
#
# --primary-estimator diffmeans: same consistency reasoning as every
# other model.

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
OUT="$REPO/results/e3_qwen32b_layer24"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model qwen3-32b \
    --gate-dir "$REPO/results/e0_qwen32b_layer24" \
    --primary-estimator diffmeans \
    --layers 24 \
    --position-offsets -1 0 1 \
    --n-perm 1000 \
    --seed 0 \
    --out "$OUT"

echo "Peak (layer, position) is in $OUT/e3_full_map.json."
echo ""
echo "REMINDER: this model's E0 did NOT pass its own stop-gate -- see"
echo "69_e1_main_qwen32b_layer24.sh's own comment for the full disclosure."

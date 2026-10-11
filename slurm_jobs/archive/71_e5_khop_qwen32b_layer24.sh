#!/bin/bash -l
#SBATCH -J skc_e5_qwen32b_l24
#SBATCH -o e5_qwen32b_l24_%j.out
#SBATCH -e e5_qwen32b_l24_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# *** DISCLOSED LIMITATION -- see 69_e1_main_qwen32b_layer24.sh's own
# comment for the full explanation. Short version: E0 FAILED its own
# stop-gate at this model/layer (diffmeans/probe both individually
# strong -- AUROC 0.9070/0.8816 -- but disagree with each other,
# Spearman=0.2963, required >=0.5, essentially unchanged from layer 8's
# 0.2966). User's own explicit choice: proceed anyway with prominent
# disclosure, same standard as HaloScope's own disclosed-uncertain-
# reconstruction flag elsewhere in this project. ***
#
# --skip-bridge-resolution: same reasoning as every other non-Gemma
# model. -t 14:00:00: E5 does roughly double E1's own labeling work
# (k=2 truncation + full k=3 chain), same reasoning as every other
# model's own E5 script.

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
OUT="$REPO/results/e5_qwen32b_layer24"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e5.py \
    --queries "$REPO/data/threehop_queries.json" \
    --model qwen3-32b \
    --layer 24 \
    --gate-dir "$REPO/results/e0_qwen32b_layer24" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"

echo ""
echo "REMINDER: this model's E0 did NOT pass its own stop-gate -- see"
echo "69_e1_main_qwen32b_layer24.sh's own comment for the full disclosure."

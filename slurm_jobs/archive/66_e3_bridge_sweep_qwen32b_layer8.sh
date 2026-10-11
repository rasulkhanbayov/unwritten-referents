#!/bin/bash -l
#SBATCH -J skc_e3_qwen32b_l8
#SBATCH -o e3_qwen32b_l8_%j.out
#SBATCH -e e3_qwen32b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# E3 on Qwen3-32B (4-bit), layer 8 -- does the frozen gate read K2 above
# chance at the bridge position on the composed prompt? Same model/layer
# as E0 (job 64) and E1 (job 65) -- only submit once both have PASSED.
#
# --mem/-c match E0/E1's own allocation for this 4-bit 32B model.
#
# HF_HUB_* env vars: same download-robustness fix as E0/E1's own
# scripts -- if E0/E1 already ran on this same cache/node the download
# will be a cache hit; if not, expect a fresh download.

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
OUT="$REPO/results/e3_qwen32b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model qwen3-32b \
    --gate-dir "$REPO/results/e0_qwen32b_layer8" \
    --primary-estimator diffmeans \
    --layers 8 \
    --position-offsets -1 0 1 \
    --n-perm 1000 \
    --seed 0 \
    --out "$OUT"

echo "Peak (layer, position) is in $OUT/e3_full_map.json."

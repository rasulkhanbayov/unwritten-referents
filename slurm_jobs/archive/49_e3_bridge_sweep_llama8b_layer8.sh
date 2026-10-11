#!/bin/bash -l
#SBATCH -J skc_e3_llama8b_l8
#SBATCH -o e3_llama8b_l8_%j.out
#SBATCH -e e3_llama8b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 10:00:00

# STEP 26: E3 on Llama 3.1 8B, layer 8. Gate: e0_llama8b_layer8_v2
# (job 387017, PASSED).

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
OUT="$REPO/results/e3_llama8b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model llama-3.1-8b \
    --gate-dir "$REPO/results/e0_llama8b_layer8_v2" \
    --primary-estimator diffmeans \
    --layers 8 \
    --position-offsets -1 0 1 \
    --n-perm 1000 \
    --seed 0 \
    --out "$OUT"

echo "Peak (layer, position) is in $OUT/e3_full_map.json."

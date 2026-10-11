#!/bin/bash -l
#SBATCH -J skc_e5_qwen32b_l8
#SBATCH -o e5_qwen32b_l8_%j.out
#SBATCH -e e5_qwen32b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# E5 (compounding, three-hop) on Qwen3-32B (4-bit), layer 8. Same
# model/layer as E0/E1/E3 -- only submit once all three have PASSED.
# --skip-bridge-resolution for the SAME reason as E1 (job 65): Patchscopes
# has never been verified on Qwen (or Llama) at any layer.
#
# --mem/-c match every other Qwen3-32B job.
#
# -t 14:00:00: up from E1's 12h -- E5 labels BOTH the k=2 truncation and
# the full k=3 chain (data/threehop_queries.json, n=3961), roughly double
# the labeling work of E1's single k=2 pass, same reasoning as
# Llama-3.1-70B's own E5 script (job 40).

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
OUT="$REPO/results/e5_qwen32b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e5.py \
    --queries "$REPO/data/threehop_queries.json" \
    --model qwen3-32b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_qwen32b_layer8" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"

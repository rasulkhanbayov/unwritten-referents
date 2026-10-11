#!/bin/bash -l
#SBATCH -J skc_e1_gemma9b_l8
#SBATCH -o e1_gemma9b_l8_%j.out
#SBATCH -e e1_gemma9b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# STEP 22: E1 on Gemma 2 9B (base), layer 8, --skip-bridge-resolution.
# Gate: e0_gemma9b_layer8_v2 (job 387016, PASSED, diffmeans+probe only,
# agreement 0.534).
#
# --skip-bridge-resolution: Patchscopes has only ever been confirmed to
# decode cleanly on Gemma at layer 16+; this base-model chain uses
# layer 8 (where the gate itself is trustworthy), so the same
# disclosed methodological weakening as every other layer-8 chain this
# session applies here too.
#
# data/onehop_correct.json reused unchanged as --singlehop-correct-queries
# -- it is a query pool re-labeled live against whichever --model is
# passed, not baked-in answers for a specific model (confirmed when
# reused for Llama-3.1-70b's own E1 run).

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
OUT="$REPO/results/e1_gemma9b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-9b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma9b_layer8_v2" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"

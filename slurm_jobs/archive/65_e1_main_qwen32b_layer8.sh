#!/bin/bash -l
#SBATCH -J skc_e1_qwen32b_l8
#SBATCH -o e1_qwen32b_l8_%j.out
#SBATCH -e e1_qwen32b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# E1 on Qwen3-32B (4-bit), layer 8, --skip-bridge-resolution. Only
# submit once E0 (job 64) has PASSED at this model/layer -- if E0 needed
# a different layer, change --layer/--gate-dir below to match before
# submitting this.
#
# WHY --skip-bridge-resolution: Patchscopes-based bridge resolution has
# only ever been confirmed to decode cleanly on GEMMA at layer 16+ --
# it has NEVER been tested on Llama OR Qwen at any layer, and nothing
# licenses assuming it transfers (same reasoning already applied to
# Llama-3.1-70B's own E1, job 38). Same K1=1-only, disclosed-weakening
# caveat as every other --skip-bridge-resolution run this project.
#
# --singlehop-correct-queries reuses data/onehop_correct.json UNCHANGED
# -- confirmed (when this was first done for Llama-3.1-70B, job 38's own
# comment) that this file is only a QUERY POOL, not baked-in answers for
# any specific model; skc_run_e1.py re-labels every query live against
# whichever --model is passed, so reuse across models is correct.
#
# --mem/-c/HF_HUB_* env vars: same as this model's own E0 script (job
# 64) -- no reason this step needs less, and the download-robustness
# fix applies identically regardless of which step first triggers the
# download.

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
OUT="$REPO/results/e1_qwen32b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model qwen3-32b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_qwen32b_layer8" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"

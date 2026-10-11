#!/bin/bash -l
#SBATCH -J skc_e1_llama70b_l8
#SBATCH -o e1_llama70b_l8_%j.out
#SBATCH -e e1_llama70b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# STEP 15: E1 on Llama 3.1 70B (4-bit), layer 8, --skip-bridge-resolution.
#
# WHY THIS MODEL/LAYER: E0 job 384352 PASSED cleanly at this
# model/layer combination on the FIRST genuine attempt (diffmeans
# AUROC=0.771, probe AUROC=0.752, agreement=0.514) -- no second layer
# needed, unlike every other model this session.
#
# WHY --skip-bridge-resolution: Patchscopes-based bridge resolution has
# only ever been confirmed to decode cleanly on GEMMA at layer 16+, via
# an exhaustive sweep earlier this session -- it has NEVER been tested
# on Llama at any layer, and nothing licenses assuming it transfers.
# This is not a choice being re-litigated here, it is a hard requirement
# given the current state of verification: --skip-bridge-resolution is
# the only option that does not silently assume something unverified.
# Same K1=1-only, disclosed-weakening caveat as every other
# --skip-bridge-resolution run this session.
#
# --singlehop-correct-queries reuses data/onehop_correct.json UNCHANGED:
# confirmed by reading skc_run_e1.py directly that this file is only a
# QUERY POOL (which entity/relation pairs to re-ask), not baked-in
# Gemma answers -- the script re-labels every one of these queries live
# against whichever --model is passed (label_queries_batched(model, tok,
# singlehop_q, ...)), so reusing the same pool across models is correct,
# not a Gemma-specific artifact being smuggled into a Llama run.
#
# --mem 120000MB, -c 16: matches E0's own allocation for this 4-bit 70B
# model (job 37's script) -- no reason this step needs less.
#
# HF_HUB_ETAG_TIMEOUT/DOWNLOAD_TIMEOUT/HF_HUB_ENABLE_HF_TRANSFER: same
# fix as E0's own script (job 37) -- the model cache was FULLY CLEARED
# this session (see EXECUTION_PLAN.md, ~323GB deleted across the whole
# account including this model's own 149GB), so this run WILL trigger a
# fresh full download, not a cache hit. Expect this run to take
# meaningfully longer than a cached run would, and do not mistake slow
# download progress for a hang.

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
OUT="$REPO/results/e1_llama70b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model llama-3.1-70b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_llama70b_layer8" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"

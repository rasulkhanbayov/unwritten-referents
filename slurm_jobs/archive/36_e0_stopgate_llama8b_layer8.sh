#!/bin/bash -l
#SBATCH -J skc_e0_llama8b_l8
#SBATCH -o e0_llama8b_l8_%j.out
#SBATCH -e e0_llama8b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 13: E0 on Llama 3.1 8B (base), layer 8, all three estimators
# including the Llama Scope SAE (llama_scope_lxa_32x release,
# id format "l{layer}a_32x" -- a COMPLETELY DIFFERENT naming scheme from
# Gemma Scope's "layer_{layer}/width_16k/canonical", confirmed by reading
# sae_lens's own pretrained_saes.yaml directly; see skc_harness.py's
# SAE_RELEASE_IDS for both).
#
# LAYER CHOICE: 8, same starting point tried for Gemma 2 9B base (job 35)
# -- Llama 3.1 8B has never had its own E0 attempt this session, and
# nothing about Gemma's working layer transfers to a different model
# family by assumption. Treat this as a genuine first attempt; if it
# fails, try a different layer next, reporting back before re-running.
#
# meta-llama/Llama-3.1-8B is ALSO a gated model on HuggingFace, same
# HF_TOKEN mechanism as every Gemma checkpoint -- confirm the token's
# account has accepted Meta's license for this repo before this job
# runs, or it will fail with the same GatedRepoError/401 class of error
# as an unaccepted Gemma license would.

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
OUT="$REPO/results/e0_llama8b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model llama-3.1-8b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (llama-3.1-8b, layer 8) PASSED. Continue the E1/E3/E5 chain at"
    echo "this layer for this model."
else
    echo "E0 (llama-3.1-8b, layer 8) FAILED (exit $exit_code)." >&2
    echo "Check whether the failure is GatedRepoError/401 (license not yet" >&2
    echo "accepted for this specific repo) before assuming it is an" >&2
    echo "agreement/AUROC failure -- these look different in the log and" >&2
    echo "need different fixes." >&2
fi
exit "$exit_code"

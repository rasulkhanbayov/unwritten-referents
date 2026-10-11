#!/bin/bash -l
#SBATCH -J skc_e0_llama70b_l8
#SBATCH -o e0_llama70b_l8_%j.out
#SBATCH -e e0_llama70b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# STEP 14: E0 on Llama 3.1 70B, 4-bit (bitsandbytes, installed and
# confirmed importable -- version 0.50.0), layer 8, --skip-sae (no
# public SAE dictionary exists past 9B; skc_runbook.txt is explicit
# this scale check is difference-of-means/probe only).
#
# --mem 120000MB, -c 16 (up from the usual 40-60GB/8-core): a 70B model
# even in 4-bit needs meaningfully more memory and the quantization
# kernels benefit from more CPU threads than the smaller models this
# session has run. This is a starting budget -- watch the first run's
# actual peak memory and adjust if it OOMs or if the allocation is
# wastefully large for what the job actually uses.
#
# LAYER CHOICE: 8, same starting point as the other new-model E0
# attempts (jobs 35, 36) -- a 70B model's layer 8 (out of 80) is a much
# smaller FRACTIONAL depth than layer 8 was for the 26-layer Gemma-2-2b,
# so this is a genuinely different bet, not a proportional match. If it
# fails, a deeper layer (proportional to depth, e.g. ~layer 24 for 80
# layers to match 2B's 8/26 fraction) is the natural next attempt --
# report back with the actual numbers before re-running.
#
# meta-llama/Llama-3.1-70B is a GATED model on HuggingFace, same
# HF_TOKEN mechanism as every other model here -- confirm the token's
# account has accepted Meta's license for THIS SPECIFIC repo (separate
# from the 8B repo's own license acceptance).

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see other scripts' identical comment.

# RE-RUN AFTER A REAL FAILURE (job 384036): the first attempt spent nearly
# all of its 16-hour budget stuck downloading this model's 30 shard files
# from HuggingFace, repeatedly hitting "Read timed out. (read
# timeout=10)" on plain HEAD requests, and was killed by Slurm's own
# time limit before the model ever finished loading -- E0 itself never
# started. Confirmed the "10" in that error is huggingface_hub's own
# DEFAULT_ETAG_TIMEOUT (10 seconds, verified directly against the
# installed huggingface_hub==0.36.2's own constants module) -- an
# unreasonably short timeout for HEAD requests to a large, sometimes
# congested repo, not a sign of a broken connection. Two real fixes,
# not a blind timeout bump:
#   - HF_HUB_ETAG_TIMEOUT / HF_HUB_DOWNLOAD_TIMEOUT raised from the
#     10s/10s defaults to 120s, so a slow-but-working HEAD/GET has room
#     to complete instead of being torn down and retried from scratch.
#   - HF_HUB_ENABLE_HF_TRANSFER=1, using the hf_transfer Rust backend
#     (installed this session: hf_transfer==0.1.9) for the actual large
#     file transfers -- faster and more resilient to exactly the kind of
#     transient slowness that triggered the original failures.
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
OUT="$REPO/results/e0_llama70b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model llama-3.1-70b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (llama-3.1-70b, 4-bit, layer 8) PASSED. Continue the E1/E3/E5"
    echo "chain at this layer, difference-of-means/probe only (no SAE row"
    echo "for this model in Table tab:gap, marked with its own dagger footnote)."
else
    echo "E0 (llama-3.1-70b, 4-bit, layer 8) FAILED (exit $exit_code)." >&2
    echo "Check whether the failure is OOM, GatedRepoError/401, or a real" >&2
    echo "AUROC/agreement failure -- these need different fixes and look" >&2
    echo "different in the log." >&2
fi
exit "$exit_code"

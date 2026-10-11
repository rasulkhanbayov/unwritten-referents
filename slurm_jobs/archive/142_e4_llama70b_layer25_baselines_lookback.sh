#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_ll
#SBATCH -o e4_llama70b_l25_ll_%j.out
#SBATCH -e e4_llama70b_l25_ll_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# Phase 5 / M2/M3: Lookback Lens + ICR-probe on Llama-3.1-70B -- SEPARATE
# job from 143/144's own "main" batches, same eager-attention-isolation
# reason as job 140 (gemma-2-9b's companion).
#
# REBUILT after a REAL failure on the first attempt (job 142, run 1):
# "ValueError: Some modules are dispatched on the CPU or the disk. Make
# sure you have enough GPU RAM to fit the quantized model." -- loading a
# SECOND, eager-attention 4-bit copy of this 70B model while the FIRST
# (default-backend) 4-bit copy was still resident exceeded available GPU
# memory, and bitsandbytes' 4-bit quantizer has NO degraded fallback for
# that (it refuses a CPU/disk-offloaded device_map outright, rather than
# silently working slower). This is a genuinely new finding: Lookback
# Lens's eager-attention load had never previously been tested on a 4-bit
# model in this project (every prior run was gemma-2-2b, full precision),
# and two simultaneous 4-bit 70B copies do not both fit even though two
# simultaneous 4-bit 9B copies (job 140) did.
#
# FIX: --sequential-model-load (new flag, added to skc_run_e4.py this
# session) frees the default model instance BEFORE loading the eager
# instance, reloading it afterward only if a later-requested baseline
# still needs it (self-ask here does, so it will be reloaded once more
# after Lookback Lens/ICR-probe finish -- one extra load/reload cycle,
# cheap since weights are already cached locally, not re-downloaded).
#
# Same --queries/--gate-dir/--saplma-train-queries as job 141/143/144.

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
OUT="$REPO/results/e4_llama70b_layer25_lookback"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-lookback-lens \
    --run-icr-probe \
    --sequential-model-load \
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_lookback.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, Lookback Lens + ICR-probe) completed (exit 0)."
    echo "Verify surface_gate/BGR-min/BGR-lr numbers here match job 143/144's own"
    echo "EXACTLY before trusting the isolation held on this 4-bit model too."
else
    echo "E4 (llama-3.1-70b, layer 25, Lookback Lens + ICR-probe) FAILED (exit $exit_code)." >&2
    echo "If this is a quantization/eager-attention incompatibility, that is a" >&2
    echo "real, reportable finding, not something to route around." >&2
fi
exit "$exit_code"

#!/bin/bash -l
#SBATCH -J skc_e0_sweep_llama70b
#SBATCH -o e0_sweep_llama70b_%j.out
#SBATCH -e e0_sweep_llama70b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 200000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 30:00:00

# VERSION3 era: full per-layer E0 sweep for Llama-3.1-70B, under the
# SPLIT-LEAKAGE FIX, per the user's explicit choice (full sweeps for all
# 3 models still failing their headline layer).
#
# WHY THIS JOB EXISTS: job 99 (layer 8, splitfix) FAILED
# (agreement=0.463, down from the leaky split's own 0.514) -- same
# reversal pattern as the other models. No historical per-layer sweep
# exists for this model.
#
# Uses skc_run_e0_layersweep.py's now-fixed split (see job 100's own
# header for the full account) and its relaxed --model choices --
# llama-3.1-70b is in FOUR_BIT_MODELS (skc_harness.py), no public SAE at
# any layer; load_sae() raises a clean ValueError for it, caught by the
# sweep's own per-layer try/except, so every layer degrades to
# diffmeans+probe only, matching this model's own existing --skip-sae
# convention (job 37/99).
#
# 4-BIT LOADING: same bitsandbytes quantization as jobs 37/99's own
# single-layer attempts -- skc_harness.load_model() already branches on
# FOUR_BIT_MODELS internally, no extra flag needed.
#
# LAYER COUNT: meta-llama/Llama-3.1-70B publicly documented as 80
# transformer layers (job 37's own header already states "out of 80"),
# but per this project's "confirm directly, don't assume" discipline,
# --layers below is generated from a live AutoConfig read at the START
# of this job, not hardcoded.
#
# --mem 200000MB / -t 30:00:00: this is the single most expensive job in
# this project to date -- 80 layers of a 70B (4-bit) model's activations
# held simultaneously (fit + held-out + named-bridge splits), on top of
# the model's own large resident footprint even quantized. Job 37's
# single-layer attempt already needed 120000MB/20:00:00 AND hit a real
# HF-download-timeout failure on its first try (see job 37's own header
# for the HF_HUB_* fix this reuses) -- budgeting generously here since a
# second real failure this expensive would be costly to diagnose and
# re-run. Watch actual peak memory/time on this first attempt and report
# back if it OOMs or runs long, rather than assuming this budget is
# exactly right.
#
# Reuses job 37/99's own --n 20000 / --fit-frac 0.5 / --seed 0 protocol
# and HF_HUB_* download-robustness env vars.

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
OUT="$REPO/results/e0_layersweep_llama70b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"

echo "Confirming meta-llama/Llama-3.1-70B's actual layer count via AutoConfig ..."
N_LAYERS=$(python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('meta-llama/Llama-3.1-70B')
print(c.num_hidden_layers)
")
echo "AutoConfig reports num_hidden_layers=$N_LAYERS"
LAYERS=$(seq 0 $((N_LAYERS - 1)))

python3 skc_run_e0_layersweep.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model llama-3.1-70b \
    --layers $LAYERS \
    --fit-frac 0.5 \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "Llama-3.1-70B full layer sweep (SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Report which layers (if any) clear gate_open from"
    echo "$OUT/e0_layersweep_full.json, ranked by cross-estimator agreement."
else
    echo "Llama-3.1-70B full layer sweep FAILED (exit $exit_code)." >&2
    echo "Check whether this is OOM, a GatedRepoError/401/download timeout, or" >&2
    echo "a real AUROC/agreement failure -- these need different fixes." >&2
fi
exit "$exit_code"

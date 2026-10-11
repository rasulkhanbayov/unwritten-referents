#!/bin/bash -l
#SBATCH -J skc_e0_sweep_qwen32b
#SBATCH -o e0_sweep_qwen32b_%j.out
#SBATCH -e e0_sweep_qwen32b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 160000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 24:00:00

# VERSION3 era: full per-layer E0 sweep for Qwen3-32B, under the
# SPLIT-LEAKAGE FIX, per the user's explicit choice (full sweeps for all
# 3 models still failing their headline layer).
#
# WHY THIS JOB EXISTS: this model failed BOTH tried layers even before
# the split fix (layer 8: agreement=0.297, job 64; layer 24: 0.296, job
# 68) -- and job 98 (layer 24, splitfix) failed again, WORSE (0.248).
# Unlike the other 3 models, this one's positive control was never
# actually passing at any point in this project, leaky split or not. A
# full sweep is the only way to find out whether ANY layer of this model
# clears the bar, or whether Qwen3-32B is a genuine null result for this
# pipeline.
#
# Uses skc_run_e0_layersweep.py's now-fixed split (see job 100's own
# header for the full account of the layersweep script's OWN
# split-leakage bug, just fixed, separate from and in addition to
# skc_run_e0.py's earlier fix) and its relaxed --model choices (this
# model was previously excluded; the per-layer SAE fit's own try/except
# handles the "no public SAE dictionary" ValueError skc_harness.load_sae
# raises for this FOUR_BIT_MODELS entry the same way it handles a
# per-layer coverage gap -- confirmed by reading load_sae() directly, not
# assumed).
#
# 4-BIT LOADING: same bitsandbytes quantization as job 68's own
# single-layer attempts -- skc_harness.load_model() already branches on
# FOUR_BIT_MODELS internally, no extra flag needed here.
#
# LAYER COUNT: Qwen/Qwen3-32B confirmed directly via AutoConfig in this
# project already (num_hidden_layers=64, see job 68's own header) -- this
# job re-confirms it live at runtime anyway, for the same "don't assume,
# even when previously confirmed" discipline as job 100.
#
# --mem 160000MB / -t 24:00:00: substantially larger than job 68's own
# single-layer 120000MB/20:00:00 -- 64 layers of a 32B model's hidden
# activations held simultaneously (fit + held-out + named-bridge splits)
# is the largest memory footprint of any sweep in this project so far.
# Watch actual peak usage on the first attempt and adjust if this proves
# wastefully large or, worse, insufficient.
#
# Reuses job 68's own --n 20000 / --fit-frac 0.5 / --seed 0 protocol and
# HF_HUB_* download-robustness env vars.

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
OUT="$REPO/results/e0_layersweep_qwen32b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"

echo "Confirming Qwen/Qwen3-32B's actual layer count via AutoConfig ..."
N_LAYERS=$(python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen3-32B')
print(c.num_hidden_layers)
")
echo "AutoConfig reports num_hidden_layers=$N_LAYERS"
LAYERS=$(seq 0 $((N_LAYERS - 1)))

python3 skc_run_e0_layersweep.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model qwen3-32b \
    --layers $LAYERS \
    --fit-frac 0.5 \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "Qwen3-32B full layer sweep (SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Report which layers (if any) clear gate_open from"
    echo "$OUT/e0_layersweep_full.json. If ZERO layers pass, that is itself a"
    echo "real, reportable finding for this model -- it never passed at any"
    echo "tried layer, leaky split or not."
else
    echo "Qwen3-32B full layer sweep FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

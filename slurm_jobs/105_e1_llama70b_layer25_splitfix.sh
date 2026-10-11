#!/bin/bash -l
#SBATCH -J skc_e1_llama70b_l25_splitfix
#SBATCH -o e1_llama70b_l25_splitfix_%j.out
#SBATCH -e e1_llama70b_l25_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# VERSION3 era: E1 for Llama-3.1-70B under the SPLIT-LEAKAGE FIX, at the
# NEW headline layer (25, from job 102's own 80-layer sweep -- the
# widest, most robust passing band of any model in the project), and
# attempting full bridge resolution (NOT --skip-bridge-resolution).
#
# RUN THIS ONLY AFTER JOB 103 (e0_llama70b_layer25_splitfix) HAS
# COMPLETED AND PASSED -- this job's --gate-dir depends on job 103's own
# frozen_gate_diffmeans.json existing. Confirm job 103's own agreement
# (~0.609 expected, matching the sweep) before running this.
#
# WHY NOT --skip-bridge-resolution, UNLIKE job 38's own layer-8 run:
# job 38 used --skip-bridge-resolution because Patchscopes had NEVER been
# tested on Llama at all (not because it was confirmed to fail) --
# nothing licensed assuming it worked OR that it didn't. Layer 25 (37.5%
# of this model's own 80 layers -- deeper than layer 8's 10%) is also a
# meaningfully different bet than the shallow layer 8 attempt. This job
# actually tries it rather than continuing to assume the weakening is
# required. If 0 items survive the bridge-resolved filter, skc_run_e1.py
# will SystemExit cleanly and that is itself the answer for this
# model/layer, same discipline as job 95's real Gemma-2-2b/layer-8 null.
#
# --singlehop-correct-queries reuses data/onehop_correct.json unchanged
# (confirmed reusable across models, see job 38's own comment).
#
# HF_HUB_* robustness env vars: same as jobs 37/38/99/102 for this
# gated, large-download model.

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
OUT="$REPO/results/e1_llama70b_layer25_bridgeresolved_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (llama-3.1-70b, layer 25, SPLIT-LEAKAGE FIX, full bridge resolution)"
    echo "completed (exit 0). Check $OUT/E1.json's auroc.gate_to_M -- compare"
    echo "against Gemma-2-2b/layer-16's own gs-vs-gb finding (gate_to_M=0.374"
    echo "surface / 0.822 bridge) to see whether this generalizes to Llama too."
else
    echo "E1 (llama-3.1-70b, layer 25, bridge-resolved filter) FAILED or 0 items" >&2
    echo "survived (exit $exit_code). If '0 queries survived the K1=1 +" >&2
    echo "bridge-resolved filter', that answers whether Patchscopes decodes on" >&2
    echo "Llama at this layer at all -- the FIRST such test on this model" >&2
    echo "family -- report it, do not route around it." >&2
fi
exit "$exit_code"

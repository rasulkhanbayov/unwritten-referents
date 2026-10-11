#!/bin/bash -l
#SBATCH -J skc_e7ac_llama70b
#SBATCH -o e7ac_llama70b_l25_%j.out
#SBATCH -e e7ac_llama70b_l25_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 20:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos -- see jobs 145-149's own headers.

# VERSION3 era: named-bridge (a) and length-matched distractor (c)
# controls for Llama-3.1-70B -- companion to job 150 (Gemma-2-9B's own
# identical pair). See that job's own header for the full account of
# why this is needed (the abstract's own headline availability-recovery
# claim, never run on either in-scope model under the fixed split).
#
# Matches job 42's own llama-70b invocation pattern (same HF_HUB_*
# robustness env vars, same 4-bit convention via skc_harness's own
# FOUR_BIT_MODELS), pointed at this model's own confirmed-stable gate
# (e0_llama70b_layer25_splitfix, agreement 0.60-0.63 across all 5 seeds,
# the widest passing band of any model in the project) at its NEW
# headline layer 25, not the old layer 8.

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
GATE="$REPO/results/e0_llama70b_layer25_splitfix/frozen_gate_diffmeans.json"
OUT="$REPO/results/e7ac_llama70b_layer25_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"

echo "=== (a) named-bridge control ==="
python3 skc_run_e7_ablations.py named-bridge \
    --queries "$REPO/data/twohop_queries.json" \
    --model llama-3.1-70b --layer 25 --gate-path "$GATE" --seed 0 --out "$OUT"

echo "=== (c) length-matched distractor control ==="
python3 skc_run_e7_ablations.py distractor \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-path "$GATE" \
    --seed 0 \
    --out "$OUT"

echo "Results in $OUT/e7a_named_bridge.json and $OUT/e7c_distractor.json"

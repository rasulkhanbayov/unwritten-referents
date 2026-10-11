#!/bin/bash -l
#SBATCH -J skc_e0_llama70b_l25_splitfix
#SBATCH -o e0_llama70b_l25_splitfix_%j.out
#SBATCH -e e0_llama70b_l25_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# VERSION3 era: single-layer E0 refit at Llama-3.1-70B's NEW headline
# layer (25), to produce a frozen_gate_diffmeans.json / frozen_gate_probe.json
# that skc_run_e1.py can actually load.
#
# WHY THIS JOB: job 102's full 80-layer sweep found a wide, robust
# passing band at layers 17-35 (19 layers all clearing agreement>=0.50),
# with layer 25 the single best (agreement=0.609, diffmeans AUROC=0.816,
# probe AUROC=0.802) -- see EXECUTION_PLAN.md for the full per-layer
# ranking. But skc_run_e0_layersweep.py (the sweep driver) never
# persists per-layer frozen gates to disk, only the aggregate E0 dict --
# so E1 cannot be pointed at the sweep's own output directory. This job
# re-fits E0 at layer 25 alone, using the ORDINARY skc_run_e0.py (which
# DOES save frozen_gate_*.json), so the numbers reproduce the sweep's own
# layer-25 result (same protocol, same fixed split) and E1 has something
# to load.
#
# --skip-sae: matches this model's own established convention
# (FOUR_BIT_MODELS, no public SAE at any layer) -- same as jobs 37/99.
#
# Uses the SAME --n 20000 / --fit-frac 0.5 / --seed 0 protocol as every
# other E0 run and the sweep itself, so this is a faithful single-layer
# reproduction, not a new bet.

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
OUT="$REPO/results/e0_llama70b_layer25_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model llama-3.1-70b \
    --layer 25 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (llama-3.1-70b, layer 25, SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Confirm agreement matches the sweep's own layer-25 result (~0.609)"
    echo "before trusting frozen_gate_diffmeans.json for the next E1 run."
else
    echo "E0 (llama-3.1-70b, layer 25, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

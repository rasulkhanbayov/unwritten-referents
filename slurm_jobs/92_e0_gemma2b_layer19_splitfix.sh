#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_l19_splitfix
#SBATCH -o e0_gemma2b_l19_splitfix_%j.out
#SBATCH -e e0_gemma2b_l19_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# VERSION3 era, C7 split-leakage fix, layer sweep (see job 89's own header
# for the full leak/fix account: layer 8 agreement dropped from 0.547
# (leaky, passing) to 0.317 (fixed, FAILING the 0.50 bar)).
#
# WHY LAYER 19, NOT ANOTHER GUESS: the old project already ran a full
# 26-layer E0 refit sweep under the LEAKY split
# (results/e0_layersweep_gemma2b/e0_layersweep_full.json) and reported
# "zero of 26 layers clear the full THREE-estimator bar" -- but that
# sweep's own per-layer diffmeans~probe (TWO-estimator) agreement was
# never separately ranked until this session did so while deciding where
# to look next under the fix. Layer 19 has the HIGHEST diffmeans~probe
# agreement of all 26 layers under the leaky split (0.661), ahead of
# layer 7 (0.632), layer 20 (0.623), and layer 8 itself (0.547) -- the
# layer layer 8's own headline number actually uses. If the leak's own
# inflation is roughly uniform across layers (not yet known), layer 19 is
# the single best candidate to clear 0.50 even after the fix removes
# whatever boost the leak was providing.
#
# Uses --skip-sae (matching layer 8's own convention, since SAE was the
# weakest/most leak-sensitive estimator in every prior sweep and is not
# needed to test the two-estimator bar this job is actually checking).

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
OUT="$REPO/results/e0_gemma2b_layer19_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layer 19 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b, layer 19, SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Check $OUT/E0.json's own gate_open field. Compare its agreement against"
    echo "the leaky-split sweep's own layer-19 diffmeans~probe value (0.661) to see"
    echo "how much of that agreement the leak was accounting for."
else
    echo "E0 (gemma-2-2b, layer 19, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

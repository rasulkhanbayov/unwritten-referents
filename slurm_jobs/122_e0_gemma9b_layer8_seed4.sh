#!/bin/bash -l
#SBATCH -J skc_e0_gemma9b_l8_seed4
#SBATCH -o e0_gemma9b_l8_seed4_%j.out
#SBATCH -e e0_gemma9b_l8_seed4_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# VERSION3 era, Phase 4 / C3: 5-seed reruns. REDFLAGS.md C3 requires seed
# variance reported alongside every number ("A bootstrap over queries
# estimates sampling noise in the evaluation set and says nothing about
# variance from probe initialization ... or cross-fitting folds"). Seed 0
# for this model/layer already exists and PASSED the fixed-split positive
# control (e0_gemma9b_layer8_splitfix, agreement=0.505) --
# this job is seed 4 of the same protocol, same model/layer, so E0's
# own entity split AND the probe's own CV folds vary (both are seeded by
# --seed in skc_run_e0.py), which is exactly the variance C3 asks to be
# captured, not just re-running the same split 5 times.
#
# gate_open may legitimately differ by seed -- report per-seed results
# honestly rather than assuming this seed must also pass just because
# seed 0 did.

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
OUT="$REPO/results/e0_gemma9b_layer8_seed4"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-9b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 4 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-9b, layer 8, seed 4) completed (exit 0)."
    echo "Check $OUT/E0.json's own gate_open field -- report honestly even if"
    echo "this seed disagrees with seed 0's own result."
else
    echo "E0 (gemma-2-9b, layer 8, seed 4) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

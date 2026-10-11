#!/bin/bash -l
#SBATCH -J skc_e7pop_gemma9bit
#SBATCH -o e7_popularity_gemma9bit_%j.out
#SBATCH -e e7_popularity_gemma9bit_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 06:00:00

# GPU item 4 of 9, part 2: PREREQUISITE for the coefficient-sweep search
# on gemma-2-9b-it (78 -> 79 -> 80). Mirrors
# 32_e7_popularity_match_gemma2b_layer8.sh exactly, just on the
# instruction-tuned 9B checkpoint -- reuses the SAME model-independent
# split files (data/twohop_val.json, data/twohop_test.json,
# data/pageviews_twohop.json are all query-level/pageview data, not
# tied to any one model); only the live K1-labeling pass
# (meta.bridge_known) needs re-running per model.

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
OUT_TEST="$REPO/results/e7_popularity_gemma9b_it_test"
OUT_VAL="$REPO/results/e7_popularity_gemma9b_it_val"
mkdir -p "$OUT_TEST" "$OUT_VAL"

cd "$REPO/src/driver_scripts"

echo "=== popularity-matching the TEST split ==="
python3 skc_run_e7_ablations.py popularity \
    --queries "$REPO/data/twohop_test.json" \
    --model gemma-2-9b-it \
    --pageviews "$REPO/data/pageviews_twohop.json" \
    --seed 0 \
    --out "$OUT_TEST"

echo "=== popularity-matching the VALIDATION split ==="
python3 skc_run_e7_ablations.py popularity \
    --queries "$REPO/data/twohop_val.json" \
    --model gemma-2-9b-it \
    --pageviews "$REPO/data/pageviews_twohop.json" \
    --seed 0 \
    --out "$OUT_VAL"

echo "Saved to $OUT_TEST/queries_with_deciles.json and $OUT_VAL/queries_with_deciles.json"
echo "-- pass these as --queries and --val-queries respectively to skc_run_e2.py."

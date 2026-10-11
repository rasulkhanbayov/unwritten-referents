#!/bin/bash -l
#SBATCH -J skc_e2_gemma9bit
#SBATCH -o e2_gemma9bit_%j.out
#SBATCH -e e2_gemma9bit_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 10:00:00

# GPU item 4 of 9, part 4 (final): E2 causal tests / coefficient sweep
# on Gemma 2 9B Instruct, layer 8 -- the search for a model/template
# combination where refusal actually varies with steering coefficient,
# per the paper's own disclosed gap ("state a non-degenerate selection
# criterion once a model/template combination is found where refusal
# actually varies with coefficient"). ONLY SUBMIT once 78 (popularity
# match) and 79 (E3) have BOTH passed.
#
# Same 7-point coefficient grid as the original gemma-2-2b sweep
# ({-4,-2,-1,1,2,4,8}, skc_run_e2.py's own default) -- kept identical so
# a comparison between the two models' own refusal-rate-vs-coefficient
# curves is apples-to-apples, not confounded by a different grid.
#
# IF THIS ALSO SHOWS REFUSAL RATE 0.000 AT EVERY COEFFICIENT: that is a
# REAL, reportable finding in its own right (an instruction-tuned model
# STILL not exhibiting refusal-lexicon behavior on this specific
# composed-prompt template would be evidence the null is about the
# TEMPLATE, not about instruction-tuning specifically) -- report it
# exactly as honestly as the gemma-2-2b null was reported, do not treat
# a second null as a failed job. If it DOES show real variation, that
# closes the TORUN with a genuine non-degenerate selection criterion
# (e.g. the coefficient maximizing validation-split refusal rate,
# assuming it is not tied across multiple values the way the all-zero
# case was).

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
OUT="$REPO/results/e2_gemma9b_it_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e2.py \
    --queries "$REPO/results/e7_popularity_gemma9b_it_test/queries_with_deciles.json" \
    --val-queries "$REPO/results/e7_popularity_gemma9b_it_val/queries_with_deciles.json" \
    --model gemma-2-9b-it \
    --layer 8 \
    --e3-map "$REPO/results/e3_gemma9b_it_layer8/e3_full_map.json" \
    --gate-dir "$REPO/results/e0_gemma2b9b_it_layer8" \
    --primary-estimator diffmeans \
    --coeffs -4 -2 -1 1 2 4 8 \
    --seed 0 \
    --out "$OUT"

echo "Full E2 result (C1+C2+C3) is in $OUT/e2_full.json. Report the refusal-rate-"
echo "vs-coefficient curve alongside gemma-2-2b's own all-zero result, whichever"
echo "way this one comes out."

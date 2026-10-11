#!/bin/bash -l
#SBATCH -J skc_e3_gemma9bit_l8
#SBATCH -o e3_gemma9bit_l8_%j.out
#SBATCH -e e3_gemma9bit_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# GPU item 4 of 9, part 3: E3 on Gemma 2 9B Instruct, layer 8 --
# PREREQUISITE for the coefficient-sweep search (skc_run_e2.py refuses
# to run without an E3 peak map at the SAME layer/model, by design: "do
# not guess one independently").
#
# WHY THIS MODEL: gemma-2-2b's own coefficient sweep found refusal rate
# EXACTLY 0.000 at every coefficient tried (three independent
# measurements -- C1, C2, C3 -- all agreeing this model does not
# exhibit refusal-lexicon behavior on the composed-prompt template at
# all; see EXECUTION_PLAN.md, job 383331). An INSTRUCTION-TUNED model
# is a reasonable next bet: base models tend to confabulate confident
# wrong answers rather than decline, while instruction tuning
# specifically trains models to refuse/hedge in some circumstances.
# gemma-2-9b-it already has a passing E0 gate in this project (layer 8,
# used for E6) -- reused here rather than fitting a new one.
#
# Uses the SAME gate as E6 (results/e0_gemma2b9b_it_layer8).

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
OUT="$REPO/results/e3_gemma9b_it_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-9b-it \
    --gate-dir "$REPO/results/e0_gemma2b9b_it_layer8" \
    --primary-estimator diffmeans \
    --layers 8 \
    --position-offsets -1 0 1 \
    --n-perm 1000 \
    --seed 0 \
    --out "$OUT"

echo "Peak (layer, position) is in $OUT/e3_full_map.json -- required before"
echo "submitting 80_e2_coefficientsweep_gemma9b_it.sh."

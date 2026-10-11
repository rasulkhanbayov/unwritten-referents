#!/bin/bash -l
#SBATCH -J skc_e0_gemma9b_it
#SBATCH -o e0_gemma9b_it_%j.out
#SBATCH -e e0_gemma9b_it_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00

# STEP 9-PREREQUISITE, NEW MODEL: E0 on gemma-2-9b-it, for E6.
#
# WHY THIS MODEL, NOT gemma-2-2b-it: three independent attempts to get an
# agreeing gate on gemma-2-2b-it all failed (layer 16: agreement -0.072;
# layer 8/n=20000: agreement 0.365; layer 8/n=30000: agreement 0.187 --
# non-monotonic in sample size, probe swinging between clearly-passing
# and clearly-failing across otherwise-similar runs). Per skc_runbook.txt's
# own FLAG-BACK TRIGGERS, this was reported to the user rather than
# chased with a fourth combination -- see EXECUTION_PLAN.md for the full
# trail. User chose to switch E6 to a DIFFERENT instruction-tuned model
# instead: gemma-2-9b-it, over llama-3.1-8b-it, specifically because it
# is the SAME architecture family as everything already validated in
# this pipeline (E0/E1/E3/E5 all ran on Gemma-2) and is already confirmed
# to download/load cleanly on this cluster -- it is E7(k)'s own judge
# model.
#
# --mem bumped to 60GB (vs 40GB for the 2b models) -- 9B in bf16 is
# meaningfully larger; sized to what this job actually needs, not copied
# from the 2b scripts. Still well within a single A100's 80GB, no 4-bit
# quantization needed (FOUR_BIT_MODELS in skc_harness.py is only
# llama-3.1-70b/qwen3-32b).
#
# --layer 16: starting point, matching the CURRENT pipeline default
# (skc_run_e1.py/e3/e5 all use layer 16 on the base 2b model) rather than
# guessing at a new value for a genuinely new model. If this fails the
# same way the 2b-it attempts did, the next step is the SAME layer-8
# fallback already tried for 2b-it, not a NEW kind of guess -- but do not
# auto-escalate past that without checking in, per the same discipline
# used for gemma-2-2b-it above.
#
# Output to its OWN directory (e0_gemma2b9b_it_layer16) -- entirely
# separate from every gemma-2-2b(-it) result already on disk.

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
OUT="$REPO/results/e0_gemma2b9b_it_layer16"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-9b-it \
    --layer 16 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-9b-it, layer 16) PASSED. frozen_gate_*.json files are in $OUT."
    echo "Update 10_e6_cot_gemma2b_it.sh (or a new copy) to use --model gemma-2-9b-it,"
    echo "--gate-dir $OUT, --layer 16 before running E6."
else
    echo "E0 (gemma-2-9b-it, layer 16) FAILED (exit $exit_code)." >&2
    echo "Try layer 8 next, same as the fallback already used for gemma-2-2b-it -- do" >&2
    echo "not guess at a third layer or a different sample size without checking in" >&2
    echo "first if layer 8 also fails." >&2
fi
exit "$exit_code"

#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_it
#SBATCH -o e0_gemma2b_it_%j.out
#SBATCH -e e0_gemma2b_it_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 9-PREREQUISITE: E0 refit on gemma-2-2b-it, specifically for
# 10_e6_cot_gemma2b_it.sh to consume.
#
# WHY THIS EXISTS: 10_e6_cot_gemma2b_it.sh's own comment flagged an
# unresolved validity gap -- it was pointing --gate-dir at E0's BASE-model
# (gemma-2-2b) run, but E6 scores gemma-2-2b-IT's activations with that
# gate. Instruction tuning can shift residual-stream geometry enough that
# a base-model-fit gate direction does not transfer -- using it anyway
# would risk a confidently wrong E6 result rather than an error. The
# correct fix (per that script's own comment, confirmed with the user) is
# to run E0 again, this time on gemma-2-2b-it, and point E6's --gate-dir
# at THIS run's output instead.
#
# skc_run_e0.py's --model choices previously did not include ANY -it
# variant (only "gemma-2-2b", "gemma-2-9b", "llama-3.1-8b") even though
# skc_harness.MODEL_IDS/load_model already support them (confirmed:
# 10_e6_cot_gemma2b_it.sh itself already passes --model gemma-2-2b-it to
# skc_run_e6.py, a different script) -- fixed by adding the three -it
# keys to skc_run_e0.py's own argparse choices before this script could
# be written at all.
#
# Same --n/--layer/--fit-frac/--seed/--skip-sae as the base-model E0 run
# (04_e0_stopgate_gemma2b.sh) for a like-for-like comparison -- --skip-sae
# is not a new gap here: the SAE branch is unimplemented regardless of
# which model is passed (skc_run_e0.py raises NotImplementedError there
# unconditionally), so every E0 run this session has used --skip-sae.
#
# Output goes to a NEW directory (e0_gemma2b_it_layer16) -- the base
# model's own layer-16 result (e0_gemma2b_layer16) must not be
# overwritten; it is still what E1/E3/E5 use.

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
OUT="$REPO/results/e0_gemma2b_it_layer16"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b-it \
    --layer 16 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b-it) PASSED. frozen_gate_*.json files are in $OUT."
    echo "Point 10_e6_cot_gemma2b_it.sh's --gate-dir at $OUT before running E6."
else
    echo "E0 (gemma-2-2b-it) FAILED (exit $exit_code). Do NOT proceed to E6 with a" >&2
    echo "gate fit on the wrong model -- fix this before running E6, do not fall" >&2
    echo "back to the base-model gate as a workaround. See skc_runbook.txt's" >&2
    echo "FLAG-BACK TRIGGERS." >&2
fi
exit "$exit_code"

#!/bin/bash -l
#SBATCH -J skc_e1_gemma9b_l8_splitfix
#SBATCH -o e1_gemma9b_l8_splitfix_%j.out
#SBATCH -e e1_gemma9b_l8_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# VERSION3 era: E1 for Gemma-2-9B under the SPLIT-LEAKAGE FIX, attempting
# full bridge resolution (NOT --skip-bridge-resolution), per the user's
# choice to run the same gs-vs-gb E1 comparison (that surfaced at
# Gemma-2-2b/layer-16) on the other 2 passing models before deciding the
# paper's final scope.
#
# WHY NOT --skip-bridge-resolution, UNLIKE job 45's own layer-8 run:
# Patchscopes has only been CONFIRMED to decode cleanly on Gemma at
# layer 16+ (an exhaustive sweep on Gemma-2-2b specifically) -- it has
# NEVER actually been tried on Gemma-2-9B at layer 8. Rather than assume
# it fails the way Gemma-2-2b's own layer 8 did (a different model, even
# if same family), this job actually attempts it. If 0 items survive the
# bridge-resolved filter (as happened for Gemma-2-2b/layer 8, job 95),
# skc_run_e1.py will SystemExit cleanly and that is itself the answer --
# not a bug to route around, same discipline as job 95.
#
# Uses e0_gemma9b_layer8_splitfix (job 97, PASSED at agreement=0.505 --
# see EXECUTION_PLAN.md's own note that this is a thin, 0.005 margin,
# worth treating with extra caution) as --gate-dir, per the project's
# standing rule to never point E1 at a pre-fix gate directory.
#
# --singlehop-correct-queries reuses data/onehop_correct.json unchanged
# (confirmed reusable across models, see job 45's own comment -- it is a
# query pool re-labeled live, not baked-in answers).

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
OUT="$REPO/results/e1_gemma9b_layer8_bridgeresolved_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-9b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma9b_layer8_splitfix" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (gemma-2-9b, layer 8, SPLIT-LEAKAGE FIX, full bridge resolution)"
    echo "completed (exit 0). Check $OUT/E1.json's auroc.gate_to_M -- compare"
    echo "against Gemma-2-2b/layer-16's own gs-vs-gb finding (gate_to_M=0.374"
    echo "surface / 0.822 bridge) to see whether this generalizes."
else
    echo "E1 (gemma-2-9b, layer 8, bridge-resolved filter) FAILED or 0 items" >&2
    echo "survived (exit $exit_code). If '0 queries survived the K1=1 +" >&2
    echo "bridge-resolved filter', that itself answers whether Patchscopes" >&2
    echo "decodes at THIS model's layer 8 -- report it, do not route around it." >&2
fi
exit "$exit_code"

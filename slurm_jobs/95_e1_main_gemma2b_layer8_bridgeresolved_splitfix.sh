#!/bin/bash -l
#SBATCH -J skc_e1_gemma2b_l8_bridgeres
#SBATCH -o e1_gemma2b_l8_bridgeres_%j.out
#SBATCH -e e1_gemma2b_l8_bridgeres_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# VERSION3 era: clean-comparison run for the gate_to_M inversion finding.
#
# WHY THIS JOB: job 94 (layer 16, fixed split, K1=1+bridge-resolved filter,
# n=158) found AUROC(gate->M) = 0.374 (below chance, bootstrap 95% CI
# 0.281-0.469 -- confirmed real, not small-n noise). The existing layer-8
# comparison point (results/e1_gemma2b_layer8/E1.json, AUROC(gate->M)=0.616)
# used a DIFFERENT filter (K1=1 ONLY, --skip-bridge-resolution, n=239) --
# not apples-to-apples with layer 16's filter. This job re-runs layer 8
# with the SAME filter as layer 16 (K1=1 + Patchscopes bridge-resolved) so
# the two layers can be compared on identical population-selection terms.
#
# EXPECTED CAVEAT (not a bug if it happens): skc_run_e1.py's own
# --skip-bridge-resolution help text states Patchscopes was "confirmed" not
# to decode cleanly at layer 8 (see EXECUTION_PLAN.md's exhaustive layer
# sweep, independent of the split-leak fix -- a property of activation
# depth, not of E0's split). If that holds, this job may retain very few
# or zero items after the bridge-resolved filter, and skc_run_e1.py will
# SystemExit cleanly with "0 queries survived the K1=1 + bridge-resolved
# filter" rather than silently producing a degenerate number. That outcome
# is itself informative (confirms layer 8 cannot be honestly compared this
# way) and should be reported back, not treated as a job failure to
# work around.
#
# Uses the FIXED-split gate (e0_gemma2b_layer8_splitfix, job 89) as
# --gate-dir, consistent with the project's standing rule to never point
# E1 at a pre-fix gate directory again.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see job 27's own header.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e1_gemma2b_layer8_bridgeresolved_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_splitfix" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (gemma-2-2b, layer 8, SPLIT-FIX gate, K1=1+bridge-resolved filter,"
    echo "SAME filter as job 94's layer 16 run) completed (exit 0)."
    echo "Compare $OUT/E1.json's auroc.gate_to_M and n against"
    echo "results/e1_gemma2b_layer16_splitfix/E1.json (0.374, n=158) for a"
    echo "clean, same-filter comparison."
else
    echo "E1 (gemma-2-2b, layer 8, bridge-resolved filter) FAILED or 0 items" >&2
    echo "survived the filter (exit $exit_code). If the error says '0 queries" >&2
    echo "survived the K1=1 + bridge-resolved filter', that itself is the" >&2
    echo "expected/confirmatory result per this script's own header -- report" >&2
    echo "it, do not treat it as a bug to route around." >&2
fi
exit "$exit_code"

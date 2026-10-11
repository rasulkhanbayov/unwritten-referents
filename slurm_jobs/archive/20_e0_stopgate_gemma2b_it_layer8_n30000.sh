#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_it_l8_n30k
#SBATCH -o e0_gemma2b_it_l8_n30k_%j.out
#SBATCH -e e0_gemma2b_it_l8_n30k_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 18:00:00

# STEP 9-PREREQUISITE, ROUND 3: E0 on gemma-2-2b-it, layer 8, n=30000
# (up from n=20000).
#
# WHY: 19_e0_stopgate_gemma2b_it_layer8.sh (n=20000) got much closer than
# the layer-16 attempt but still failed -- BOTH estimators individually
# PASS (diffmeans AUROC 0.756, probe AUROC 0.741, both clearly above the
# 0.65 threshold) but cross-estimator agreement is 0.365, below the 0.5
# bar (not negative like layer 16's -0.072, a real, different, closer
# result). This is the SAME shape of failure the base model went through
# earlier this session: agreement (not either AUROC) was the unstable
# quantity across sample sizes there too, before the fit_diffmeans
# outlier-standardization fix (already applied here, so this run
# benefits from it) and before landing on n=20000/layer 8 as a stable
# pass (agreement 0.620). Trying a LARGER sample here, same layer, same
# fix already in place, tests whether the same "just needs more data"
# pattern resolves this the way it eventually did for the base model --
# --n 30000 (up from 20000), same --fit-frac/--seed/--skip-sae otherwise.
#
# Runs longer than the n=20000 attempt (more queries to label + capture
# activations for) -- -t 18:00:00 gives headroom over the n=20000 run's
# -t 14:00:00.
#
# Output to a NEW directory (e0_gemma2b_it_layer8_n30000) -- neither the
# layer-16 nor the layer-8/n=20000 attempt's results are overwritten; all
# three are legitimate records of what was tried.

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
OUT="$REPO/results/e0_gemma2b_it_layer8_n30000"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 30000 \
    --model gemma-2-2b-it \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b-it, layer 8, n=30000) PASSED. frozen_gate_*.json files are in $OUT."
    echo "Point 10_e6_cot_gemma2b_it.sh's --gate-dir at $OUT and --layer at 8 before running E6."
else
    echo "E0 (gemma-2-2b-it, layer 8, n=30000) ALSO FAILED (exit $exit_code)." >&2
    echo "Three attempts (layer 16; layer 8/n=20000; layer 8/n=30000) have now failed to" >&2
    echo "produce an agreeing gate for gemma-2-2b-it. Flag back per skc_runbook.txt" >&2
    echo "FLAG-BACK TRIGGERS -- do not try a fourth combination without discussing this" >&2
    echo "with the user first; this is starting to look like a real property of the" >&2
    echo "instruction-tuned model's residual-stream geometry, not a sampling artifact." >&2
fi
exit "$exit_code"

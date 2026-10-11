#!/bin/bash -l
#SBATCH -J skc_e4_gemma9b_main
#SBATCH -o e4_gemma9b_l8_main_%j.out
#SBATCH -e e4_gemma9b_l8_main_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# Phase 5 / M2/M3: re-score the 13-baseline comparison on Gemma-2-9B (the
# model's own confirmed-stable, 5-seed-verified fixed-split gate), needed
# to rewrite Table tab:bgr/tab:bgr-full honestly onto the paper's now-
# narrowed 2-model scope (gemma-2-9b, llama-3.1-70b), since ALL existing
# baseline scores (jobs 08/29/41/51-59/85-88) are on gemma-2-2b, which
# dropped from scope after C7/C3's own multi-seed instability finding.
#
# THIS JOB: every baseline that uses the DEFAULT model load (no separate
# eager-attention instance needed) -- self-ask, SAPLMA, entity-token
# probe, semantic-entropy-probe, self-consistency, selfcheckgpt,
# semantic-entropy, p-true, operadic-consistency. Lookback Lens and
# ICR-probe are DELIBERATELY EXCLUDED from this job -- see job 140's own
# header for why they need a separate job (both need a SECOND, eager-
# attention model instance, confirmed non-numerically-equivalent to the
# default SDPA backend on bfloat16 -- mean abs logit diff 0.168, found
# and fixed earlier this project by isolating them onto their own model
# load; combining them into a shared-model job would silently corrupt
# every OTHER selector's own score the same way it did before that fix).
# HaloScope/INSIDE/SEP all use the DEFAULT model load (confirmed by
# reading skc_run_e4.py directly, not assumed) -- safe here.
#
# --queries points at THIS model's own eval-side queries_filtered.json
# (job 106, K1=1-only/--skip-bridge-resolution -- Patchscopes never
# resolves at this model/layer, per job 104's confirmed 0/346 result;
# same disclosed caveat as every other number for this model).
# --gate-dir points at e0_gemma9b_layer8_splitfix (the confirmed-stable
# gate, agreement 0.50-0.52 across all 5 seeds).
# --saplma-train-queries points at THIS session's own newly-built
# data/onehop_fit_gemma9b_layer8_splitfix.json (9110 queries, matched
# exactly against fit_raw_generations.json from e0_gemma9b_layer8_splitfix
# -- same fit split this model's diffmeans/probe were themselves fit on).
#
# Also exports per-item records (--export-records/--export-split, this
# session's own new addition to skc_run_e4.py) so scripts/controls.py and
# any future baseline-fairness reanalysis can run on real per-item
# baseline scores, not just this job's own aggregate AUROC numbers.

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
OUT="$REPO/results/e4_gemma9b_layer8_main"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma9b_layer8_skipbridge_splitfix/queries_filtered.json" \
    --model gemma-2-9b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma9b_layer8_splitfix" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-saplma \
    --run-entity-token-probe \
    --run-haloscope \
    --run-inside \
    --run-semantic-entropy-probe \
    --run-self-consistency \
    --run-selfcheckgpt \
    --run-semantic-entropy \
    --run-p-true \
    --run-operadic-consistency \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma9b_layer8_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/gemma-2-9b/f1_twohop_baselines_main.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (gemma-2-9b, layer 8, main baseline batch) completed (exit 0)."
    echo "Cost report: $OUT/e4_cost_report.json. Combine with job 140's own"
    echo "Lookback Lens/ICR-probe results before rewriting Table tab:bgr-full."
else
    echo "E4 (gemma-2-9b, layer 8, main baseline batch) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

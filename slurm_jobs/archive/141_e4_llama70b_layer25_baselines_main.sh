#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_main
#SBATCH -o e4_llama70b_l25_main_%j.out
#SBATCH -e e4_llama70b_l25_main_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 24:00:00

# Phase 5 / M2/M3: re-score the 13-baseline comparison on Llama-3.1-70B
# (the model's own confirmed-stable, 5-seed-verified fixed-split gate,
# at its NEW headline layer 25 -- the widest, most robust positive
# control in the whole project). See job 139's own header (gemma-2-9b's
# companion job) for the full account of why this is needed.
#
# THIS JOB: every baseline using the DEFAULT model load -- same set as
# job 139. Lookback Lens/ICR-probe excluded here too, in their own job
# (142), for the same eager-attention-isolation reason.
#
# --queries: this model's own eval-side queries_filtered.json (job 107,
# K1=1-only/--skip-bridge-resolution -- Patchscopes never resolves at
# this model/layer either, per job 105's confirmed 0/494 result).
# --gate-dir: e0_llama70b_layer25_splitfix (confirmed-stable, agreement
# 0.60-0.63 across all 5 seeds).
# --saplma-train-queries: data/onehop_fit_llama70b_layer25_splitfix.json
# (9110 queries, matched exactly against this model's own
# fit_raw_generations.json).
#
# HF_HUB_* robustness env vars: same as every other 70B job this
# project (gated, large download).
#
# --mem 120000MB / -c 16 / -t 24:00:00: matches this model's own
# established 4-bit-70B budget, generously extended in time since this
# job runs ~9 baselines (more than any single 70B E4 job attempted
# before -- watch actual wall-clock and split further if this proves
# too tight).

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see other scripts' identical comment.

export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=1

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e4_llama70b_layer25_main"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
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
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_main.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, main baseline batch) completed (exit 0)."
    echo "Cost report: $OUT/e4_cost_report.json. Combine with job 142's own"
    echo "Lookback Lens/ICR-probe results before rewriting Table tab:bgr-full."
else
    echo "E4 (llama-3.1-70b, layer 25, main baseline batch) FAILED (exit $exit_code)." >&2
    echo "Check whether this is OOM, GatedRepoError/401, or a real scoring" >&2
    echo "failure -- these need different fixes and look different in the log." >&2
fi
exit "$exit_code"

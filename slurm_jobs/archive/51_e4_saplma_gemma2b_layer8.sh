#!/bin/bash -l
#SBATCH -J skc_e4_saplma_gemma2b
#SBATCH -o e4_saplma_gemma2b_l8_%j.out
#SBATCH -e e4_saplma_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# STEP 28 (Phase 2, item 5 of 12 E4 baselines): SAPLMA (Azaria & Mitchell
# 2023), Gemma 2 2B, layer 8. First real GPU run of this baseline --
# offline-verified only (synthetic XOR-learning test) before this.
#
# --saplma-train-queries points at data/onehop_fit_gemma2b_layer8_v2.json,
# built from results/e0_gemma2b_layer8_v2/fit_raw_generations.json's own
# recorded qids (9985 items, exact match, confirmed) -- the SAME fit
# split E0's diffmeans/probe estimators were themselves trained on, kept
# consistent with this pipeline's own fit/held-out discipline. SAPLMA
# trains its own MLP on this split, then is scored on the SAME E1-filtered
# test queries as every other E4 baseline (results/e1_gemma2b_layer8/
# queries_filtered.json) -- never trained on what it is scored on.
#
# Runs together with --run-self-ask (already-verified, mandatory per
# skc_REDFLAGS.md) in the same job, since both need the same loaded model
# and this keeps GPU idle time down between them -- SAPLMA training adds
# real time (an extra generation+capture pass over ~10k fit queries plus
# MLP training) on top of the already-slow Self-Ask baseline, so -t is
# raised from the original E4 script's own budget.

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
OUT="$REPO/results/e4_gemma2b_layer8_saplma"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-saplma \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma2b_layer8_v2.json" \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

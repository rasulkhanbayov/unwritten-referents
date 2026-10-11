#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_partA
#SBATCH -o e4_llama70b_l25_partA_%j.out
#SBATCH -e e4_llama70b_l25_partA_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 24:00:00

# Phase 5 / M2/M3: REBUILD of job 141, which was CANCELLED at the 24h
# time limit -- confirmed from its own .out log it was still inside
# HaloScope's fit step, having spent the ENTIRE 24h budget relabeling the
# 9110-query fit split THREE separate times (once each for SAPLMA,
# entity-token-probe, HaloScope -- skc_run_e4.py's own fit_* functions
# each independently called label_queries_batched() on the identical fit
# split, with no sharing). NOTHING was saved to results/ -- total loss,
# not a partial result to resume from.
#
# TWO FIXES APPLIED BEFORE THIS REBUILD:
# 1. skc_run_e4.py now labels the fit split ONCE and shares it across
#    SAPLMA/entity-token-probe/HaloScope via a new fit_labels= parameter
#    on each fit_* function (see main()'s own new comment block for the
#    full account) -- cuts this job's own redundant relabeling from 3x to
#    1x on a 70B model, the single largest cost this job pays.
# 2. Also split into 2 smaller jobs (this one = partA, job 144 = partB)
#    rather than trust one 24h budget for everything job 139 covered on
#    gemma-2-9b in ~20h -- llama-3.1-70b's own forward-pass cost is
#    substantially higher and the label-cache fix's real speedup on THIS
#    model has never been measured, so a conservative split avoids paying
#    for ANOTHER full 24h loss if the fix under-delivers.
#
# THIS JOB (partA): p-true, SAPLMA, entity-token-probe, HaloScope,
# self-ask -- the DEFAULT-model baselines that share the new label cache,
# plus the two cheapest standalone baselines (p-true, self-ask).
# INSIDE/SEP/self-consistency/selfcheckgpt/semantic-entropy/operadic-
# consistency moved to job 144 (partB) -- see that job's own header.
#
# Same --queries/--gate-dir/--saplma-train-queries as job 141.

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
OUT="$REPO/results/e4_llama70b_layer25_partA"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --run-p-true \
    --run-saplma \
    --run-entity-token-probe \
    --run-haloscope \
    --run-self-ask \
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_partA.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, partA) completed (exit 0)."
    echo "Check how much of the 24h budget this actually used -- report back so"
    echo "job 144 (partB)'s own budget can be sanity-checked against real timing."
else
    echo "E4 (llama-3.1-70b, layer 25, partA) FAILED or timed out (exit $exit_code)." >&2
fi
exit "$exit_code"

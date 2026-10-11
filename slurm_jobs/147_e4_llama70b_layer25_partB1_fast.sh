#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_partB1
#SBATCH -o e4_llama70b_l25_partB1_%j.out
#SBATCH -e e4_llama70b_l25_partB1_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 20:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos, NOT gpu-A100 -- all A100 nodes were
# reported full at submission time; user confirmed h200_hbku/h200_hbku_qos are
# valid sacctmgr associations for this account. H200 is also a faster GPU, which
# helps directly with this job's own timeout history.

# Phase 5 / M2/M3: REBUILD of job 144 (partB), which timed out at 24h
# while fitting Semantic Entropy Probe -- confirmed from its own .out
# log it HAD already completed labeling, activation capture, and INSIDE
# scoring before dying mid-SEP-fit. Total loss again -- but confirms
# INSIDE (fully unsupervised, no fit split) is fast on this model, and
# that SEP's own k=10-sample + up-to-90-NLI-call-per-fit-query cost is
# the actual long pole, not the other baselines bundled with it.
#
# SPLIT FURTHER: this job (partB1) = self-ask, INSIDE, self-consistency,
# SelfCheckGPT, semantic-entropy, operadic-consistency -- everything
# EXCEPT Semantic Entropy Probe, which gets its own dedicated job (148,
# partB2) with a longer budget.
#
# Same --queries/--gate-dir/--saplma-train-queries as job 144.

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
OUT="$REPO/results/e4_llama70b_layer25_partB1"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-inside \
    --run-self-consistency \
    --run-selfcheckgpt \
    --run-semantic-entropy \
    --run-operadic-consistency \
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_partB1.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, partB1) completed (exit 0)."
    echo "Verify surface_gate/BGR-min/BGR-lr match job 142/145's own exactly."
else
    echo "E4 (llama-3.1-70b, layer 25, partB1) FAILED or timed out (exit $exit_code)." >&2
fi
exit "$exit_code"

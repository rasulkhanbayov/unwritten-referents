#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_partA1
#SBATCH -o e4_llama70b_l25_partA1_%j.out
#SBATCH -e e4_llama70b_l25_partA1_%j.err
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

# Phase 5 / M2/M3: REBUILD of job 143 (partA), which timed out at 24h
# while fitting HaloScope -- confirmed from its own .out log it HAD
# already completed labeling (once, thanks to the label-cache fix),
# SAPLMA fit+score, and entity-token-probe fit+score before dying mid-
# HaloScope-fit. Total loss again (skc_run_e4.py only saves at the very
# end) -- but this confirms the label-cache fix itself worked (only ONE
# 9110-item labeling pass this time, not three like job 141), and that
# p-true/SAPLMA/entity-token-probe are all comfortably fast on this
# model. HaloScope itself is the part that did not finish in the
# remaining time.
#
# SPLIT FURTHER: this job (partA1) = p-true, SAPLMA, entity-token-probe,
# self-ask -- everything partA proved fast, MINUS HaloScope, which gets
# its own dedicated job (146, partA2) so a slow HaloScope fit cannot
# threaten the baselines already confirmed to finish quickly.
#
# Same --queries/--gate-dir/--saplma-train-queries as job 143.

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
OUT="$REPO/results/e4_llama70b_layer25_partA1"
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
    --run-self-ask \
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_partA1.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, partA1) completed (exit 0)."
    echo "Compare surface_gate/BGR-min/BGR-lr against job 142's own"
    echo "(0.495/0.539/0.666) and job 145's own future companions for consistency."
else
    echo "E4 (llama-3.1-70b, layer 25, partA1) FAILED or timed out (exit $exit_code)." >&2
fi
exit "$exit_code"

#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_partA2
#SBATCH -o e4_llama70b_l25_partA2_%j.out
#SBATCH -e e4_llama70b_l25_partA2_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 30:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos, NOT gpu-A100 -- all A100 nodes were
# reported full at submission time; user confirmed h200_hbku/h200_hbku_qos are
# valid sacctmgr associations for this account. H200 is also a faster GPU, which
# helps directly with this job's own timeout history.

# Phase 5 / M2/M3: HaloScope ALONE for llama-3.1-70b -- split out of
# job 143/145 after job 143 timed out at 24h mid-HaloScope-fit (see job
# 145's own header for the full account). Giving HaloScope its own job
# with a longer, dedicated budget (30h, up from the 24h shared budget
# that was not enough alongside 3 other baselines) rather than bundling
# it with anything else that could be threatened by its own long
# runtime.
#
# self-ask included too (cheap, uses the default model already loaded,
# no reason not to get it "for free" alongside HaloScope) -- purely for
# the cross-job surface_gate/BGR-min/BGR-lr consistency check, same
# discipline as every other split E4 job.
#
# Same --queries/--gate-dir/--saplma-train-queries as job 143/145.

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
OUT="$REPO/results/e4_llama70b_layer25_partA2"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-haloscope \
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_partA2.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, partA2 -- HaloScope) completed (exit 0)."
    echo "Verify surface_gate/BGR-min/BGR-lr match job 142/145's own exactly."
else
    echo "E4 (llama-3.1-70b, layer 25, partA2) FAILED or timed out (exit $exit_code)." >&2
    echo "If this STILL times out at 30h, HaloScope's own fit cost on this model" >&2
    echo "needs a real look (e.g. reducing k or the fit-pool size) rather than" >&2
    echo "another blind budget increase." >&2
fi
exit "$exit_code"

#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_partB
#SBATCH -o e4_llama70b_l25_partB_%j.out
#SBATCH -e e4_llama70b_l25_partB_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 24:00:00

# Phase 5 / M2/M3: partB of the 70B baseline batch (see job 143's own
# header for the full account of why job 141 was rebuilt and split).
#
# THIS JOB: INSIDE, Semantic Entropy Probe, self-consistency,
# SelfCheckGPT, semantic-entropy, operadic-consistency -- all baselines
# that do NOT need the shared SAPLMA-style fit-label cache (INSIDE is
# fully unsupervised at fit time; SEP/self-consistency/semantic-entropy
# pay their real, unavoidable k=10-sampling cost regardless; operadic-
# consistency likewise). Also includes --run-self-ask again (cheap,
# needed here too since this job does not depend on partA having run
# first -- these two jobs are INDEPENDENT, not sequential, despite the
# partA/partB naming, so surface_gate/BGR-min/BGR-lr should also be
# checked to match job 143's own numbers exactly, same cross-job sanity
# check as every other split E4 job this project has used).
#
# Same --queries/--gate-dir/--saplma-train-queries as job 141/143
# (SEP needs --saplma-train-queries even though it does not share the
# label cache, since it needs the fit split's own query POOL to sample
# from -- confirmed by reading skc_run_e4.py's own
# --run-semantic-entropy-probe requirement check).

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
OUT="$REPO/results/e4_llama70b_layer25_partB"
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
    --run-semantic-entropy-probe \
    --run-self-consistency \
    --run-selfcheckgpt \
    --run-semantic-entropy \
    --run-operadic-consistency \
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_partB.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, partB) completed (exit 0)."
    echo "Verify surface_gate/BGR-min/BGR-lr here match job 143's own EXACTLY"
    echo "(same seed, same gate -- should be byte-identical)."
else
    echo "E4 (llama-3.1-70b, layer 25, partB) FAILED or timed out (exit $exit_code)." >&2
fi
exit "$exit_code"

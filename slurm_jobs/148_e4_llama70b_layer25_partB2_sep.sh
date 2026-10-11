#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_partB2
#SBATCH -o e4_llama70b_l25_partB2_%j.out
#SBATCH -e e4_llama70b_l25_partB2_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 36:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos, NOT gpu-A100 -- all A100 nodes were
# reported full at submission time; user confirmed h200_hbku/h200_hbku_qos are
# valid sacctmgr associations for this account. H200 is also a faster GPU, which
# helps directly with this job's own timeout history.

# Phase 5 / M2/M3: Semantic Entropy Probe ALONE for llama-3.1-70b --
# split out of job 144/147 after job 144 timed out at 24h mid-SEP-fit
# (see job 147's own header for the full account). This is the single
# most expensive fit step of any baseline in this project: k=10 sampled
# generations + up to 90 pairwise NLI calls PER EACH of 9110 fit
# queries, on the slowest (70B, 4-bit) model tested. Giving it its own
# job with a long, dedicated budget (36h) rather than guessing a smaller
# increment and risking a THIRD total loss.
#
# self-ask included too (cheap, cross-job consistency check).
#
# IF THIS STILL TIMES OUT AT 36H: do not just raise the budget again --
# investigate whether SEP's own fit pool needs to be SUBSAMPLED for this
# model specifically (e.g. 2000-3000 of the 9110 fit queries, still
# comfortably enough to fit a linear probe) rather than assuming the
# full pool is necessary at 70B scale the same way it was at 2B/9B
# scale, where the per-query cost is far cheaper. Report back with
# exactly how far it got (check for a progress marker in fit_semantic_
# entropy_probe() -- currently none exists, which is itself worth fixing
# if this needs a third attempt, so future runs are not fully blind
# again).
#
# Same --queries/--gate-dir/--saplma-train-queries as job 144/147.

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
OUT="$REPO/results/e4_llama70b_layer25_partB2"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-semantic-entropy-probe \
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_partB2.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, partB2 -- SEP) completed (exit 0)."
    echo "Verify surface_gate/BGR-min/BGR-lr match job 142/145's own exactly."
else
    echo "E4 (llama-3.1-70b, layer 25, partB2) FAILED or timed out (exit $exit_code)." >&2
fi
exit "$exit_code"

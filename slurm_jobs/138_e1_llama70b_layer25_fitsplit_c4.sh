#!/bin/bash -l
#SBATCH -J skc_e1_llama70b_l25_fitsplit
#SBATCH -o e1_llama70b_l25_fitsplit_%j.out
#SBATCH -e e1_llama70b_l25_fitsplit_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# VERSION3 era, Phase 1 / C4: score the DISJOINT fit-split composed queries
# for llama-3.1-70b, to standardize g_S/g_B before scripts/ablations.py
# can run the bridge-only test.
#
# See job 137's own header (gemma-2-9b's identical companion job) for the
# full account of why a disjoint fit split was needed and how
# data/c4_fit_queries.json was built (1253 real, live-Wikidata-verified
# two-hop chains, the k=2 prefix of data/threehop_queries.json, confirmed
# entity-disjoint from data/twohop_queries.json's own eval-side
# population).
#
# --skip-bridge-resolution: matches this model/layer's own established
# convention (job 105 confirmed 0/494 Patchscopes resolution at layer 25
# on Llama-3.1-70b) -- same selection criterion as the eval-side run,
# different disjoint entities.
#
# Uses e0_llama70b_layer25_splitfix (the model's own confirmed-stable,
# 5-seed-verified gate) as --gate-dir.

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
OUT="$REPO/results/e1_llama70b_layer25_fitsplit_c4"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/c4_fit_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_fitsplit.jsonl" \
    --export-split fit \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (llama-3.1-70b, layer 25, FIT-SPLIT c4_fit_queries.json) completed (exit 0)."
    echo "Records exported to VERSION3/data/records/llama-3.1-70b/f1_twohop_fitsplit.jsonl."
    echo "This run's own E1.json is NOT the number that matters here -- only the"
    echo "exported g_S/g_B records are needed, for ablations.py's Standardizer."
else
    echo "E1 (llama-3.1-70b, layer 25, FIT-SPLIT) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

#!/bin/bash -l
#SBATCH -J skc_e7pop_gemma2b_l8
#SBATCH -o e7_popularity_gemma2b_l8_%j.out
#SBATCH -e e7_popularity_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00

# PREREQUISITE FOR E2 (STEP 6), NOT E2 ITSELF: skc_run_e7_ablations.py's
# `popularity` subcommand, run separately on the val and test splits
# (data/twohop_val.json, data/twohop_test.json -- built by
# driver_scripts/skc_split_val_test.py, a disjoint-by-qid partition of
# data/twohop_queries.json, seed 0, 20% validation). This assigns
# popularity_decile (from data/pageviews_twohop.json, fetched live via
# skc_data.fetch_pageviews against the public Wikimedia Pageviews REST
# API -- see driver_scripts/skc_fetch_pageviews.py) and meta.bridge_known
# (from a live model K1 pass) to each query, which is what
# skc_run_e2.py's C2 donor matching (match_on_decile) needs and could not
# run without.
#
# Needs a GPU (loads gemma-2-2b for the live K1 pass), unlike the
# pageviews fetch itself (CPU/network-only, already run separately).
#
# Runs on BOTH splits in one job (val is cheap relative to model load
# time; no reason to burn two separate queue slots for this).

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here. torch==2.4.1+cu124's wheel
# bundles its own complete CUDA 12.4 runtime (cusparse, nvJitLink, etc.) and
# needs nothing from the system toolkit -- nothing in this pipeline compiles
# a custom CUDA extension. `module load cuda12.1/toolkit/12.1.1` was tried
# here originally and caused a real failure: it prepends an OLDER cusparse/
# nvJitLink onto LD_LIBRARY_PATH ahead of torch's own bundled cu124 libs,
# producing `undefined symbol: __nvJitLinkComplete_12_4` on the first real
# GPU job. Do not re-add a module load line to "fix" a CUDA-looking error
# without checking whether the module itself is the cause first.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }

# google/gemma-2-2b (and every other Gemma checkpoint) is a GATED model on
# HuggingFace -- Google requires an authenticated, license-accepted account
# to download it, or transformers fails with GatedRepoError/401 even though
# the network path itself works. .env holds HF_TOKEN=... (never echo this
# file's contents or commit/zip it into the submission -- see EXECUTION_PLAN.md).
set -a; source "$REPO/.env"; set +a
OUT_TEST="$REPO/results/e7_popularity_gemma2b_layer8_test"
OUT_VAL="$REPO/results/e7_popularity_gemma2b_layer8_val"
mkdir -p "$OUT_TEST" "$OUT_VAL"

cd "$REPO/src/driver_scripts"

echo "=== popularity-matching the TEST split ==="
python3 skc_run_e7_ablations.py popularity \
    --queries "$REPO/data/twohop_test.json" \
    --model gemma-2-2b \
    --pageviews "$REPO/data/pageviews_twohop.json" \
    --seed 0 \
    --out "$OUT_TEST"

echo "=== popularity-matching the VALIDATION split ==="
python3 skc_run_e7_ablations.py popularity \
    --queries "$REPO/data/twohop_val.json" \
    --model gemma-2-2b \
    --pageviews "$REPO/data/pageviews_twohop.json" \
    --seed 0 \
    --out "$OUT_VAL"

echo "Saved to $OUT_TEST/queries_with_deciles.json and $OUT_VAL/queries_with_deciles.json"
echo "-- pass these as --queries and --val-queries respectively to skc_run_e2.py."

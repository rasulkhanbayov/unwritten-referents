#!/bin/bash -l
#SBATCH -J skc_e2_gemma2b_l8
#SBATCH -o e2_gemma2b_l8_%j.out
#SBATCH -e e2_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# STEP 6, SECONDARY TRACK, NOW UNBLOCKED: E2 causal tests (C1 surface
# steer, C2 bridge patch -- THE DECISIVE test, C3 gate inject) at layer 8.
#
# WHY LAYER 8, NOT 16: e0_gemma2b_layer16's cross-estimator agreement
# problem was investigated exhaustively this session (dimension-exclusion
# hypothesis tested and disproven -- see EXECUTION_PLAN.md) and accepted
# as a genuine, disclosed estimator-robustness property, not something to
# keep chasing. Layer 8 (e0_gemma2b_layer8_v2, agreement 0.547) is the
# pipeline's only currently-trustworthy gate for the base gemma-2-2b
# model, and E3/E4/E5/E7 were all already re-run against it this session.
# --e3-map below points at e3_gemma2b_layer8_v2 (job 382903) to match.
#
# BOTH DATA BLOCKERS NOW RESOLVED:
#   - popularity_decile / meta.bridge_known: built by
#     32_e7_popularity_match_gemma2b_layer8.sh, itself fed by
#     data/pageviews_twohop.json (live Wikimedia Pageviews REST API pull,
#     driver_scripts/skc_fetch_pageviews.py -- 847 unique bridge entities
#     from data/twohop_queries.json, real HTTP calls, 404s on any surface
#     form that doesn't match a real article title dropped rather than
#     imputed -- see that script's own printed drop count).
#   - disjoint validation split: driver_scripts/skc_split_val_test.py,
#     seed 0, 20% validation / 80% test, disjointness asserted in code
#     (data/twohop_val.json, data/twohop_test.json).
#   Run 32_e7_popularity_match_gemma2b_layer8.sh BEFORE this script --
#   --queries/--val-queries below point at ITS output, not the plain
#   split files.
#
# --patch-layer-window omitted: defaults to a window centered on
# --e3-map's own peak layer (L8_pos1, see job 382903's result), same
# default behavior as skc_run_e2.py uses elsewhere in this pipeline.

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
OUT="$REPO/results/e2_gemma2b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e2.py \
    --queries "$REPO/results/e7_popularity_gemma2b_layer8_test/queries_with_deciles.json" \
    --val-queries "$REPO/results/e7_popularity_gemma2b_layer8_val/queries_with_deciles.json" \
    --model gemma-2-2b \
    --layer 8 \
    --e3-map "$REPO/results/e3_gemma2b_layer8_v2/e3_full_map.json" \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"

echo "Full E2 result (C1+C2+C3) is in $OUT/e2_full.json -- E2's dissociation ratio is"
echo "the decisive causal test (C2 in particular); report it alongside E1/E3's"
echo "correlational results, all at the same layer 8, same caveats disclosed."

#!/bin/bash -l
#SBATCH -J skc_e2_gemma2b
#SBATCH -o e2_gemma2b_%j.out
#SBATCH -e e2_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# STEP 6: E2, causal tests -- C1 (surface steer), C2 (bridge patch, THE
# DECISIVE test), and C3 (gate inject) ALL implemented now (see
# EXECUTION_PLAN.md for the design and the user's own confirmation to use
# E3's composed-prompt bridge-resolution position for C2/C3, rather than
# skc_harness.bridge_resolved_patchscope()'s during-generation site that
# E1 uses -- those are two different, deliberate choices for different
# consumers, not the same "the bridge position" every time).
#
# STILL BLOCKED ON DATA, NOT CODE, right now:
#   - --queries needs popularity_decile AND meta.bridge_known set, which
#     needs skc_run_e7_ablations.py's `popularity` subcommand, which
#     itself needs a --pageviews file (Wikipedia pageview API access /
#     dump parsing) -- NOT fetched anywhere in this bundle (flagged
#     repeatedly, still out of scope as of this comment).
#   - --val-queries needs a query file DISJOINT (no shared qids) from
#     --queries, for selecting C1's steering coefficient without tuning
#     on the test set -- not yet built either.
#   data/twohop_queries_with_deciles.json and data/twohop_queries_val.json
#   (referenced below) do NOT exist yet -- this script is NOT ready to run
#   until both are built. Do not point it at a substitute file to make it
#   "run"; a wrong/missing decile match measures popularity, not
#   knowledge, and a non-disjoint validation split tunes the coefficient
#   on the test set -- both are explicitly named failure modes in
#   skc_runbook.txt/skc_run_e2.py's own docstring.
#
# --e3-map (new, REQUIRED): points at 06's e3_full_map.json. Its peak
# (layer, offset) is what C2/C3 patch/inject at -- must be an E3 run at
# the SAME layer as --layer/--gate-dir below, or the position and the
# gate direction refer to different residual streams (skc_run_e2.py
# checks this and refuses to proceed if they mismatch).
#
# --patch-layer-window is now OPTIONAL: if omitted, defaults to a small
# window centered on --e3-map's own peak layer. Pass it explicitly only
# to override that default.

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
OUT="$REPO/results/e2_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e2.py \
    --queries "$REPO/data/twohop_queries_with_deciles.json" \
    --val-queries "$REPO/data/twohop_queries_val.json" \
    --model gemma-2-2b \
    --layer 16 \
    --e3-map "$REPO/results/e3_gemma2b_layer16/e3_full_map.json" \
    --gate-dir "$REPO/results/e0_gemma2b_layer16" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"

echo "Full E2 result (C1+C2+C3) is in $OUT/e2_full.json."

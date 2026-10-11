#!/bin/bash -l
#SBATCH -J skc_e1_gemma2b_l8
#SBATCH -o e1_gemma2b_l8_%j.out
#SBATCH -e e1_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# STEP 4, RE-RUN AFTER THE PADDING-SIDE FIX AND THE LAYER-8-VS-16
# TRADEOFF DECISION: E1 at layer 8, with --skip-bridge-resolution.
#
# WHY LAYER 8, NOT 16: E0 at layer 16 (with the padding-side fix
# correctly applied -- capture_activations() now forces right-padding
# internally, see skc_gates.py) FAILED on cross-estimator agreement
# (-0.227), a real, unresolved diffmeans outlier-sensitivity issue at
# that depth (traced to genuine "massive activation" dimensions, not a
# bug -- mean-centering fit_diffmeans was verified, both mathematically
# and empirically, NOT to fix this). E0 at layer 8 PASSED cleanly
# (agreement 0.547, job 382897) -- the FIRST genuinely trustworthy E0
# result of the whole session.
#
# WHY --skip-bridge-resolution: Patchscopes only decodes the bridge
# cleanly at layer 16+ on this model (confirmed via an exhaustive sweep
# earlier this session) -- layer 8 decodes to template noise at every
# position tested. Continuing at layer 8 (where the gate itself is
# trustworthy) therefore means giving up Patchscopes-based bridge
# resolution, which skc_runbook.txt's own filter spec wants alongside
# K1=1. CONFIRMED WITH THE USER: this is a deliberate, disclosed
# methodological weakening, not an oversight -- E1 filters on K1=1 ONLY
# for this run. See skc_run_e1.py's own module docstring and
# EXECUTION_PLAN.md for the full reasoning and tradeoff. K1=1 filtering
# ITSELF is never skipped, regardless -- only the ADDITIONAL Patchscopes
# evidence layer.
#
# data/onehop_correct.json does NOT need rebuilding for this layer
# change: it was built from K1 LABELS (generate_greedy(), which always
# explicitly set its own required padding_side and was never affected by
# the capture_activations() bug), not from activations -- K1 labeling is
# layer-independent, only activation CAPTURE is layer-specific.
#
# Output to a NEW directory (e1_gemma2b_layer8) -- the old
# e1_gemma2b_layer16 result (pre-padding-fix, Patchscopes-based) is left
# on disk as a historical record, not overwritten.

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
OUT="$REPO/results/e1_gemma2b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"

#!/bin/bash -l
#SBATCH -J skc_e5_gemma2b_l8
#SBATCH -o e5_gemma2b_l8_%j.out
#SBATCH -e e5_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# STEP 8, SECONDARY TRACK, RE-RUN AFTER THE PADDING-SIDE FIX AND THE
# LAYER-8-VS-16 TRADEOFF DECISION: E5 at layer 8.
#
# WHY: --gate-dir now points at e0_gemma2b_layer8_v2 (the first genuinely
# trustworthy E0 result this session, agreement 0.547, WITH the
# capture_activations() padding-side fix applied). The old
# e5_gemma2b result (if it ran at all) predates the padding fix and
# pointed at the layer-16 gate -- not reused.
#
# --skip-bridge-resolution (NEW FLAG, added to skc_run_e5.py to mirror
# skc_run_e1.py's identical flag): Patchscopes only decodes cleanly at
# layer 16+ on gemma-2-2b, confirmed during E1's investigation this
# session. Running skc_run_e5.py's Patchscopes bridge-resolution check
# at layer 8 unmodified would silently produce unreliable
# bridge-resolved labels rather than an error. This flag skips that
# check and filters on K1=1 alone instead -- the SAME disclosed
# methodological weakening as the layer-8 E1 run (job 382900), applied
# here for the identical reason. Report this caveat alongside E5's
# results, same as E1's.
#
# Output to a NEW directory (e5_gemma2b_layer8) -- any earlier
# layer-16 E5 result stays on disk as a historical record, not
# overwritten.

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
OUT="$REPO/results/e5_gemma2b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e5.py \
    --queries "$REPO/data/threehop_queries.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"

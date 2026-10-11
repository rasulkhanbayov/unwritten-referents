#!/bin/bash -l
#SBATCH -J skc_e1_gemma2b
#SBATCH -o e1_gemma2b_%j.out
#SBATCH -e e1_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# STEP 4: E1, THE MAIN RESULT. Run only after:
#   1. 04_e0_stopgate_gemma2b.sh exits 0 (DONE -- layer 16, agreement 0.513)
#
# 06_e3_bridge_sweep_gemma2b.sh / e3_full_map.json is NO LONGER a
# prerequisite for THIS script. bridge_resolved_patchscope() used to need
# E3's static-prompt (layer, offset) peak; it was rebuilt (see
# EXECUTION_PLAN.md's "round 4" entry) to instead capture DURING the
# model's own hop1-only generation and locate the bridge-resolution site
# per query in that generation's own text -- validated to decode the
# bridge correctly on K1=1 queries, which the old E3-peak-based site never
# did (an exhaustive static-prompt sweep found it undecoded at ANY layer
# or offset). --e3-map was removed from this script's own arguments to
# match; E3 remains a valid, separate experiment in its own right (it
# still answers "where does the frozen gate read K2 above chance"), it
# just no longer feeds E1's bridge-resolved filter.
#
# NOT an --array job (unlike the original 3-seed design this was modeled
# on): there is no seeded multi-dataset generation step in this project
# (data/twohop_queries_seed{N}.json was never built -- that pattern was
# carried over from a template and never actually implemented here). This
# runs ONCE against the real, existing data/twohop_queries.json (3503
# queries) and data/onehop_correct.json (1772 K1=1 items, built offline by
# skc_build_singlehop_correct.py from E0's own already-collected
# generations -- no new GPU time spent labeling them again).
#
# Patchscopes (skc_harness.bridge_resolved_patchscope) is implemented and
# validated on a small sample (see EXECUTION_PLAN.md) but this is still
# the FIRST run of the rebuilt during-generation version against the FULL
# 3503-query set. Treat this as a debugging run: read the printed
# retained_fraction and a sample of decoded Patchscopes text
# (patchscope_sample.json) before trusting the E1 numbers.

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
OUT="$REPO/results/e1_gemma2b_layer16"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-2b \
    --layer 16 \
    --gate-dir "$REPO/results/e0_gemma2b_layer16" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"

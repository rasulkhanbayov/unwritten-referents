#!/bin/bash -l
#SBATCH -J skc_e4_gemma2b
#SBATCH -o e4_gemma2b_%j.out
#SBATCH -e e4_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# STEP 7: E4, repair + baselines (Table 3). Requires E0 passed and E1's
# filtered query set (K1=1 + bridge-resolved) as --queries, so this is also
# blocked on Patchscopes being implemented for real E1 output -- see
# 05_e1_main_gemma2b.sh's header.
#
# --run-self-ask is ON here DELIBERATELY. skc_REDFLAGS.md and
# skc_runbook.txt both call the Class-3 decomposition-pipeline comparison
# (Self-Ask etc.) MANDATORY, not optional -- it is the comparison most
# likely to sink the paper's practical claim if skipped, because "just
# prompt the model to decompose the question" is the obvious reviewer
# objection and now has published methods behind it. This makes the job
# ~3x slower (3 forward passes per query in Self-Ask alone) -- the -t 12:00:00
# walltime budget accounts for that; do not shrink it to save queue time.
#
# --run-p-true is OFF by default (uncomment to enable) -- it is not named as
# mandatory anywhere in skc_runbook.txt/skc_REDFLAGS.md, unlike Self-Ask.

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
OUT="$REPO/results/e4_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer16/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 16 \
    --gate-dir "$REPO/results/e0_gemma2b_layer16" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

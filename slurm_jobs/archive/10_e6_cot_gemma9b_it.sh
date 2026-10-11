#!/bin/bash -l
#SBATCH -J skc_e6_gemma9b_it
#SBATCH -o e6_gemma9b_it_%j.out
#SBATCH -e e6_gemma9b_it_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# STEP 9: E6, chain-of-thought gate restoration. Uses an INSTRUCTION-TUNED
# checkpoint, not a base model -- CoT prompting on a base model is
# unreliable, and skc_run_e6.py's docstring notes an instruction- or
# reasoning-tuned model is expected here.
#
# MODEL, gemma-2-9b-it (NOT gemma-2-2b-it): three independent E0 attempts
# on gemma-2-2b-it all failed to reach the required cross-estimator
# agreement (layer 16: -0.072; layer 8/n=20000: 0.365; layer 8/n=30000:
# 0.187 -- non-monotonic in sample size, probe swinging between clearly
# passing and clearly failing across otherwise-similar runs; full trail
# in EXECUTION_PLAN.md). Flagged back per skc_runbook.txt's FLAG-BACK
# TRIGGERS rather than chased with a fourth combination. User chose to
# switch E6 to a DIFFERENT instruction-tuned model instead:
# gemma-2-9b-it, over llama-3.1-8b-it, specifically because it is the
# SAME architecture family as everything already validated in this
# pipeline and was already confirmed to download/load cleanly on this
# cluster (it is E7(k)'s own judge model).
#
# LAYER, 8 (NOT 16): gemma-2-9b-it's own E0 needed two attempts too --
# layer 16 (job 382412) got close (both estimators individually passed,
# diffmeans 0.725/probe 0.693) but agreement (0.162) still missed the 0.5
# bar. LAYER 8 (job 382551) PASSED CLEANLY: diffmeans 0.721, probe 0.666,
# agreement 0.588 -- comfortably above 0.5, gate_open: true. This gate
# (results/e0_gemma2b9b_it_layer8/frozen_gate_diffmeans.json) is what E6
# uses below.
#
# --mem 60000MB (up from the 2b-it version's 40000MB): 9B needs
# meaningfully more memory than 2B for the same batch sizes.
#
# Runs longer than most other single-GPU jobs here (200 new tokens per CoT
# trace x however many queries, on a larger 9B model) -- -t 12:00:00 is a
# starting budget (up from the 2b-it version's 10h), watch the first run's
# wall-clock and adjust.

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

# google/gemma-2-9b-it (and every other Gemma checkpoint) is a GATED model
# on HuggingFace -- Google requires an authenticated, license-accepted
# account to download it, or transformers fails with GatedRepoError/401
# even though the network path itself works. .env holds HF_TOKEN=... (never
# echo this file's contents or commit/zip it into the submission -- see
# EXECUTION_PLAN.md).
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e6_gemma9b_it"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e6.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-9b-it \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b9b_it_layer8" \
    --primary-estimator diffmeans \
    --cot-max-new-tokens 200 \
    --seed 0 \
    --out "$OUT"

echo "MANUALLY AUDIT $OUT/e6_cot_traces_sample.json before trusting the self-written-"
echo "bridge span heuristic (see skc_run_e6.py's module docstring)."

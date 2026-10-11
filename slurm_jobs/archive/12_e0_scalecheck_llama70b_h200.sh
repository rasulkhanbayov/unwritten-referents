#!/bin/bash -l
#SBATCH -J skc_e0_llama70b
#SBATCH -o e0_llama70b_%j.out
#SBATCH -e e0_llama70b_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 3-00:00:00

# STEP (scale check, run in parallel with the main gemma-2-2b track, not
# sequentially after it): "does this only hold on small models?"
# skc_runbook.txt names Llama-3.1-70B and Qwen3-32B, 4-bit, PROBE-ONLY (no
# SAE -- no public SAE dictionary exists at this scale; --skip-sae is
# required here, not optional).
#
# PARTITION CHOICE: gpu-H200, and this is the job that actually justifies
# using H200 rather than A100. Llama-3.1-70B in 4-bit (via bitsandbytes)
# needs roughly 35-40GB for weights alone, before activations/KV cache for
# batched capture_activations() calls -- comfortably fits an 80GB A100, but
# H200's 141GB gives real headroom for larger batch sizes during activation
# capture (skc_runbook.txt: "the expensive parts are activation caching and
# the patching sweeps"), and this job's -t 3-00:00:00 (the partition's max)
# means slow, cramped batching directly costs wall-clock against that
# 3-day ceiling. Use H200 here; save A100 for the 2B/8B/9B jobs above that
# do not need the extra memory.
#
# --account/--qos: CONFIRMED via `sacctmgr show assoc user=$USER
# format=account,qos` on panther-login -- this account's real H200 grant is
# h200_hbku / h200_hbku_qos, matching the prior ACL project's convention
# (slurm_jobs/05_full_qwen3vl30b_H200.sh there), not the Panther End User
# Guide PDF's generic H200/h200_qos example, which does not apply here.
#
# gpu-H200 has a 3-day max walltime and max 4 concurrent jobs per user
# (Panther guide) -- this single job alone should not exceed either, but do
# not stack more than 3 other H200 jobs concurrently with this one.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- same reasoning as
# 03_pilot_gemma2b.sh: torch's cu124 wheel is self-contained, and loading a
# system CUDA toolkit module ahead of it caused a real
# `undefined symbol: __nvJitLinkComplete_12_4` failure on the A100 track's
# first real GPU job. bitsandbytes (needed here for 4-bit) also ships its
# own bundled CUDA kernels for this reason; do not add a module load back
# to "help" it.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }

# meta-llama/Llama-3.1-70B (like every Llama checkpoint) is ALSO gated on
# HuggingFace, same as Gemma -- see 03_pilot_gemma2b.sh's comment for the
# full explanation. .env holds HF_TOKEN=... (never echo/commit/zip it).
set -a; source "$REPO/.env"; set +a

OUT="$REPO/results/e0_llama70b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --model llama-3.1-70b \
    --layer 40 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 PASSED for llama-3.1-70b (probe-only, no SAE per skc_runbook.txt). Proceed"
    echo "to a 70B E1 run using the same --skip-sae convention."
else
    echo "E0 FAILED for llama-3.1-70b (exit $exit_code). This is itself informative for" >&2
    echo "the 'does this hold at scale' question -- do not discard it, report it." >&2
fi
exit "$exit_code"

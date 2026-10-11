#!/bin/bash -l
#SBATCH -J skc_e0_qwen32b_l24_splitfix
#SBATCH -o e0_qwen32b_l24_splitfix_%j.out
#SBATCH -e e0_qwen32b_l24_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# VERSION3 era, C7 split-leakage fix, extended to Qwen3-32B.
#
# Same leak as Gemma-2-2b (see EXECUTION_PLAN.md's full account) --
# model-independent, purely a property of the shared query JSONs. Fix is
# baked into skc_run_e0.py itself (--named-bridge-queries required,
# exclude_from_fit applied automatically) -- no extra flags needed here.
#
# NOTE: qwen3-32b's own layer-24 result (job 68, e0_qwen32b_layer24) was
# already the LEAKY split's own FAILURE case (agreement=0.296, gate_open
# false) -- layer 8 failed too (job 64, agreement=0.297 per job 68's own
# header). Running this job anyway for completeness/historical
# consistency with the other 3 models' splitfix reruns, but do not expect
# this one to pass; if it fails again under the fix, that is NOT itself
# new evidence the fix broke something (it was already failing before the
# fix touched anything).
#
# Matches job 68's exact invocation (--n 20000, --skip-sae, --fit-frac
# 0.5, --seed 0, same HF_HUB_* robustness env vars). Only the output dir
# changes.

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
OUT="$REPO/results/e0_qwen32b_layer24_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model qwen3-32b \
    --layer 24 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (qwen3-32b, layer 24, SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Check $OUT/E0.json's own gate_open field -- this layer FAILED even"
    echo "under the OLD leaky split (agreement=0.296), so passing now would"
    echo "itself be a notable reversal, not just a routine confirmation."
else
    echo "E0 (qwen3-32b, layer 24, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
    echo "This layer already failed under the leaky split too -- check whether" >&2
    echo "the numbers moved at all (worth knowing) before treating this as a" >&2
    echo "new, fix-caused problem." >&2
fi
exit "$exit_code"

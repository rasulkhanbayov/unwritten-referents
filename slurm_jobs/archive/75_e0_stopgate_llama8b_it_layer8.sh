#!/bin/bash -l
#SBATCH -J skc_e0_llama8bit_l8
#SBATCH -o e0_llama8bit_l8_%j.out
#SBATCH -e e0_llama8bit_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# GPU item 3 of 9: E0 on Llama 3.1 8B INSTRUCT (llama-3.1-8b-it), layer
# 8 -- prerequisite for extending E6 (chain-of-thought gate restoration)
# to a second model, per the paper's own disclosed gap ("Extend to the
# other models" in Section~\ref{sec:res-cot}). A gate direction is fit
# per CHECKPOINT, not per architecture family -- the base-model
# llama-3.1-8b gate already validated in this project cannot be reused
# on the instruction-tuned checkpoint, mirroring exactly why
# gemma-2-9b-it needed its own separate E0 (job 382551) rather than
# reusing gemma-2-9b's own base-model gate.
#
# REAL BUG FOUND AND FIXED (job 390469, first attempt): this script
# originally omitted --skip-sae. skc_run_e0.py's own hardcoded
# enforcement (raise if --skip-sae is missing) only covers
# llama-3.1-70b/qwen3-32b explicitly -- it does NOT also cover
# llama-3.1-8b-it, even though NO Llama checkpoint (base or -it) has a
# public SAE dictionary in this pipeline's SAE_RELEASE_IDS registry.
# Without the flag, the run got all the way through fit-split labeling
# (10015/10015 items) before crashing on "no public SAE dictionary for
# 'llama-3.1-8b-it'" at the SAE-fitting step. Fixed by adding
# --skip-sae explicitly below, the same fix every other Llama E0 run in
# this project already needed.
#
# LAYER CHOICE: 8, the SAME starting layer as every other new-checkpoint
# E0 attempt this project has made (base llama-3.1-8b passed at layer
# 8; gemma-2-9b-it also passed at layer 8 on its second attempt, after
# layer 16 came close but missed the agreement bar) -- a reasonable
# first bet given both close relatives' own outcomes, not a guarantee.
# If this fails, try layer 16 next (the layer gemma-2-9b-it's own
# attempt got closest at, 0.162 agreement) before any other layer.
#
# --mem 60000MB: matches gemma-2-9b-it's own E0 allocation (an 8B
# instruction-tuned model needs meaningfully more than a 2B one for the
# same batch sizes).

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see other scripts' identical comment.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e0_llama8b_it_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model llama-3.1-8b-it \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (llama-3.1-8b-it, layer 8) PASSED. Check the actual agreement"
    echo "value in e0_llama8bit_l8_*.out before proceeding -- 'exit 0' means"
    echo "the run completed, gate_open may still be false. If gate_open is"
    echo "true, submit 76_e6_cot_llama8b_it.sh next."
else
    echo "E0 (llama-3.1-8b-it, layer 8) FAILED (exit $exit_code)." >&2
    echo "Try layer 16 next if this is a genuine agreement-bar failure," >&2
    echo "not an OOM or gated-repo error." >&2
fi
exit "$exit_code"

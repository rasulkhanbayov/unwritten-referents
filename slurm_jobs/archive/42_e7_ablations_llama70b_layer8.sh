#!/bin/bash -l
#SBATCH -J skc_e7_llama70b_l8
#SBATCH -o e7_llama70b_l8_%j.out
#SBATCH -e e7_llama70b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 140000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# --mem higher than E4/E1/E3/E5's own 120000MB for this model, same
# reasoning as the Gemma layer-8 E7 script: the (k) label-validation
# subcommand loads TWO models at once on the SAME GPU -- llama-3.1-70b
# (4-bit, under test) PLUS gemma-2-9b-it (bf16, judge). Sized up rather
# than reusing another job's allocation blindly.
#
# STEP 19: E7 controls on Llama 3.1 70B (4-bit), layer 8. Same three
# subcommands as every other E7 layer-8 script this session: (a)
# named-bridge, (j) random-direction, (k) label-validation. (b)
# popularity and (e) attention are not in this script for the same
# reasons as every other E7 script (popularity needs its own
# preprocessing pass, not done for this model; attention needs new
# capture code that does not exist).
#
# $GATE points at e0_llama70b_layer8/frozen_gate_diffmeans.json (job
# 384352, PASSED on the first genuine attempt for this model).
#
# --layer 8 EVERYWHERE below, matching $GATE -- checked directly, not
# assumed, per the standing caution in every other E7 script this
# session about layer/gate mismatches.
#
# (k)'s judge model is gemma-2-9b-it, SAME choice as every other E7(k)
# run this session, for consistency of the judge across all reported
# label-validation numbers -- not because it is a natural pairing with
# a 70B model under test, but because judge INDEPENDENCE from the model
# under test is what this control needs, and a fixed, already-used judge
# lets E7(k) results be compared across models on equal footing.

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
GATE="$REPO/results/e0_llama70b_layer8/frozen_gate_diffmeans.json"
OUT="$REPO/results/e7_llama70b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"

echo "=== (a) named-bridge control ==="
python3 skc_run_e7_ablations.py named-bridge \
    --queries "$REPO/data/twohop_queries.json" \
    --model llama-3.1-70b --layer 8 --gate-path "$GATE" --seed 0 --out "$OUT"

echo "=== (j) random-direction negative control ==="
python3 skc_run_e7_ablations.py random-direction \
    --queries "$REPO/data/twohop_queries.json" \
    --model llama-3.1-70b --layer 8 --gate-path "$GATE" --coefficient 4.0 --seed 0 --out "$OUT"

echo "=== (k) label validation against an LLM judge ==="
# --layer 8 is a no-op for this subcommand (cmd_judge_validation() never
# reads args.layer) but argparse still requires it on the shared subparser.
python3 skc_run_e7_ablations.py label-validation \
    --queries "$REPO/data/twohop_queries.json" \
    --model llama-3.1-70b --layer 8 --judge-model gemma-2-9b-it --n 100 --seed 0 --out "$OUT"

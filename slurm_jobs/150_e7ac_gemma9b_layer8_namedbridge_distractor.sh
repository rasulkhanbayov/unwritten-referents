#!/bin/bash -l
#SBATCH -J skc_e7ac_gemma9b
#SBATCH -o e7ac_gemma9b_l8_%j.out
#SBATCH -e e7ac_gemma9b_l8_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 08:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos -- see jobs 145-149's own headers
# for why (A100 nodes reported full at submission time; valid alternative
# account/qos confirmed by the user).

# VERSION3 era: named-bridge (a) and length-matched distractor (c)
# controls for Gemma-2-9B, needed for the abstract's own headline claim
# ("writing the referent in recovers X; a length-matched distractor
# recovers nothing") -- this was never run on either in-scope model
# under the fixed split; only the OLD gemma-2-2b numbers exist, and that
# model just left scope after C3's own multi-seed instability finding.
#
# Matches jobs 31/62's own invocation exactly (same subcommands, same
# --queries conventions per subcommand: named-bridge uses the FULL
# twohop_queries.json per its own historical convention; distractor
# uses the E1-filtered composed set), pointed at this model's own
# confirmed-stable gate (e0_gemma9b_layer8_splitfix, agreement 0.50-0.52
# across all 5 seeds) instead of the old gemma-2-2b gate.

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
GATE="$REPO/results/e0_gemma9b_layer8_splitfix/frozen_gate_diffmeans.json"
OUT="$REPO/results/e7ac_gemma9b_layer8_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"

echo "=== (a) named-bridge control ==="
python3 skc_run_e7_ablations.py named-bridge \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-9b --layer 8 --gate-path "$GATE" --seed 0 --out "$OUT"

echo "=== (c) length-matched distractor control ==="
python3 skc_run_e7_ablations.py distractor \
    --queries "$REPO/results/e1_gemma9b_layer8_skipbridge_splitfix/queries_filtered.json" \
    --model gemma-2-9b \
    --layer 8 \
    --gate-path "$GATE" \
    --seed 0 \
    --out "$OUT"

echo "Results in $OUT/e7a_named_bridge.json and $OUT/e7c_distractor.json"

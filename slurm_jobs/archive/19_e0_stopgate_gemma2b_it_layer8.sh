#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_it_l8
#SBATCH -o e0_gemma2b_it_l8_%j.out
#SBATCH -e e0_gemma2b_it_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 9-PREREQUISITE, ROUND 2: E0 on gemma-2-2b-it at LAYER 8, not 16.
#
# WHY: 18_e0_stopgate_gemma2b_it.sh (layer 16) FAILED the stop gate --
# diffmeans passed (AUROC 0.743) but probe failed (0.600 < 0.65), and the
# two estimators actively DISAGREED (Spearman -0.072). This is the EXACT
# same failure signature seen earlier this session on the BASE model at
# layer 16, before layer 8 was found to pass there cleanly (agreement
# 0.620 -- see results/e0_gemma2b_layer8/E0.json). Layer 16 was only ever
# chosen for the base-model pipeline because Patchscopes (bridge
# decoding) needed a deeper layer than layer 8 supports -- confirmed via
# the round-3 diagnostic sweep earlier this session (layers 16/20/24
# decode cleanly, layer 8/12 do not).
#
# E6 (the only consumer of this -it gate) does NOT use Patchscopes at
# all -- its own bridge-resolution mechanism is find_self_written_bridge(),
# a text search over the model's own chain-of-thought trace, not a
# Patchscopes decode (see skc_run_e6.py's module docstring). E6 therefore
# has NO ACTUAL REQUIREMENT to use layer 16 -- that constraint only
# existed because the base-model pipeline happened to standardize on
# layer 16 for E1/E3/E5's sake. This run tests whether gemma-2-2b-it's
# gate, like the base model's, simply prefers a shallower layer, rather
# than concluding the -it model is fundamentally incompatible with this
# gate-fitting procedure at ANY layer.
#
# Same --n/--fit-frac/--seed/--skip-sae as both prior E0 runs (base model,
# and the -it/layer-16 attempt) for a like-for-like comparison. Output to
# a NEW directory (e0_gemma2b_it_layer8) -- the layer-16 -it attempt's
# own (failed, but real) result is not overwritten; both are legitimate
# records of what was tried.

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
OUT="$REPO/results/e0_gemma2b_it_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b-it \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b-it, layer 8) PASSED. frozen_gate_*.json files are in $OUT."
    echo "Point 10_e6_cot_gemma2b_it.sh's --gate-dir and --layer at $OUT / 8 before running E6."
else
    echo "E0 (gemma-2-2b-it, layer 8) ALSO FAILED (exit $exit_code)." >&2
    echo "Both layer 16 and layer 8 have now failed for gemma-2-2b-it -- this starts to look" >&2
    echo "like a real property of the instruction-tuned model, not a layer-choice artifact." >&2
    echo "Flag back per skc_runbook.txt FLAG-BACK TRIGGERS rather than sweeping further layers" >&2
    echo "without discussing it first." >&2
fi
exit "$exit_code"

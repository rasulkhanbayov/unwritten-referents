#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_l8_v2
#SBATCH -o e0_gemma2b_l8_v2_%j.out
#SBATCH -e e0_gemma2b_l8_v2_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 3, RE-RUN AFTER THE PADDING-SIDE FIX: E0 at layer 8, base model.
#
# WHY: E0 at layer 16, re-run WITH the capture_activations() padding-side
# fix applied (job 382870), FAILED -- agreement -0.227, even though BOTH
# individual estimators pass with even better AUROCs than before
# (diffmeans 0.863, probe 0.712). Traced this to a real, structural
# property of layer 16's activations, not a bug: diffmeans's score
# distribution has a heavy right tail (~12% of held-out points score
# >50, some >100) that INVERTS rank order relative to probe for those
# same points -- caused by a small number of "massive activation"
# dimensions (e.g. dim 334, values 22-210, essentially never near zero)
# that diffmeans's std-only normalization does not meaningfully dampen,
# while probe's L2 regularization does. Verified (both mathematically and
# empirically) that mean-centering fit_diffmeans CANNOT fix this -- it
# does not change the direction at all (a mean shift cancels in
# mu_k - mu_u), and Spearman correlation is invariant to bias/threshold
# shifts regardless. See EXECUTION_PLAN.md for the full trail.
#
# Rather than design a new dimension-exclusion/clipping scheme for
# diffmeans (real work, not yet validated), trying layer 8 again instead
# -- it showed a MUCH milder version of the same massive-activation
# phenomenon earlier this session (7% outlier fraction at layer 8 vs 14%
# at layer 12, per the original diagnostic), and passed cleanly
# (agreement 0.620) before the padding-side bug was even found. This is
# the FIRST layer-8 run with the padding fix actually applied -- treat
# it as the real, trustworthy result, not a re-confirmation of the old
# (pre-fix) e0_gemma2b_layer8 result.
#
# If this passes: the tradeoff of losing layer 16's Patchscopes support
# (E1/E3/E5 needed layer 16+ for Patchscopes to decode cleanly) needs a
# separate decision -- not resolved by this run alone.

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
OUT="$REPO/results/e0_gemma2b_layer8_v2"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b, layer 8, padding-fix applied) PASSED. frozen_gate_*.json files are in $OUT."
    echo "This is the first TRUSTWORTHY E0 result this session -- decide whether to"
    echo "continue the E1/E3/E5 chain at layer 8 (loses Patchscopes support) or keep"
    echo "chasing layer 16's agreement problem before proceeding."
else
    echo "E0 (gemma-2-2b, layer 8, padding-fix applied) FAILED (exit $exit_code)." >&2
    echo "Both layer 8 and layer 16 now fail post-fix -- this is a different, more" >&2
    echo "serious situation than before. Flag back per skc_runbook.txt FLAG-BACK" >&2
    echo "TRIGGERS and discuss with the user rather than trying another layer" >&2
    echo "unprompted." >&2
fi
exit "$exit_code"

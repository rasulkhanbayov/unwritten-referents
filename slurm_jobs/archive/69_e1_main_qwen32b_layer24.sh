#!/bin/bash -l
#SBATCH -J skc_e1_qwen32b_l24
#SBATCH -o e1_qwen32b_l24_%j.out
#SBATCH -e e1_qwen32b_l24_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# *** DISCLOSED LIMITATION, READ BEFORE TRUSTING ANY NUMBER FROM THIS
# JOB: E0 (jobs 389099 at layer 8, 389192 at layer 24) FAILED its own
# stop-gate at BOTH layers tried. diffmeans and probe BOTH individually
# separate known/unknown entities strongly (layer 24: AUROC 0.9070 and
# 0.8816 respectively, both with tight CIs well above the 0.65
# threshold and chance) -- but their AGREEMENT with each other is far
# below the required 0.5 (min pairwise Spearman = 0.2963 at layer 24,
# 0.2966 at layer 8 -- essentially IDENTICAL despite a large depth
# change, 12.5% to 37.5%, which is itself evidence this is a genuine
# property of the model rather than a layer-selection artifact). E0
# exists specifically to prevent exactly this situation from producing
# an uninterpretable E1 result: "the model is blind" and "my probe does
# not work" produce the same NULL E1 output, and a low-agreement gate
# means "which entities diffmeans calls known" and "which entities
# probe calls known" are substantially DIFFERENT sets, so whichever one
# this run uses as --primary-estimator is picking out a real but
# NOT-CROSS-VALIDATED notion of "the model knows this."
#
# USER'S OWN EXPLICIT CHOICE (asked directly, not decided unilaterally):
# proceed to E1/E3/E5 anyway using the layer-24 frozen gates, with
# PROMINENT DISCLOSURE throughout (this comment, and repeated in
# EXECUTION_PLAN.md and eventually skc_main.tex) rather than either (a)
# silently treating this the same as every OTHER model's clean E0 pass,
# or (b) refusing to run E1/E3/E5 for this model at all. This is the
# SAME class of explicit, disclosed exception as HaloScope's own
# "best-faith reconstruction, not verified" flag elsewhere in this
# project -- a real result under a real, stated limitation, not
# something to quietly firm up later or hide.
#
# --primary-estimator diffmeans: kept for consistency with every OTHER
# model in this project (all use diffmeans as primary throughout
# E1/E3/E4/E5); diffmeans also had the (marginally) higher of the two
# individually-passing AUROCs at both layers tried.
#
# Uses the LAYER 24 gates (results/e0_qwen32b_layer24/), not layer 8 --
# layer 24 is this model's best-attempted layer (both estimators
# stronger there), even though the DISAGREEMENT itself did not improve.
#
# --skip-bridge-resolution: same reasoning as every other non-Gemma
# model this project has run -- Patchscopes has never been verified on
# Qwen (or Llama) at any layer, only Gemma at layer 16+.

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
OUT="$REPO/results/e1_qwen32b_layer24"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model qwen3-32b \
    --layer 24 \
    --gate-dir "$REPO/results/e0_qwen32b_layer24" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --out "$OUT"

echo ""
echo "REMINDER: this model's E0 did NOT pass its own stop-gate (both"
echo "estimators individually strong but disagree with each other,"
echo "min pairwise Spearman ~0.296 at both layers tried). Report every"
echo "number from this run with that caveat attached, do not present it"
echo "as an ordinary clean-E0 result."

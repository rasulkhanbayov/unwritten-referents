#!/bin/bash -l
#SBATCH -J skc_e4_ptrue_oc_gemma2b
#SBATCH -o e4_ptrue_oc_gemma2b_l8_%j.out
#SBATCH -e e4_ptrue_oc_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00

# Fills the LAST remaining \RES{} cells in Table tab:bgr-full and
# tab:bgr: P(True), Operadic consistency, and Oracle observing M. All
# three were previously either unimplemented (P(True)/Operadic, left as
# ">>> TO RUN, do not fabricate from a title alone") or not computed at
# all (Oracle, never wired into run_E4()'s baseline_scores dict).
#
# THIS JOB REQUIRED THREE REAL FIXES TO driver_scripts/skc_run_e4.py,
# EACH VERIFIED AGAINST ITS SOURCE BEFORE BEING TRUSTED (see
# EXECUTION_PLAN.md for the full verification trail; summarized here):
#
# 1. P(True) (baseline_p_true): the PRE-EXISTING implementation had TWO
#    real bugs, found only by actually reading Kadavath et al. 2022
#    (arXiv:2207.05221, Section 4.1) rather than trusting the function's
#    own "TO RUN" note that it was an unverified paraphrase --
#    (a) wrong prompt entirely (used a free-text "yes/no" question
#    instead of the paper's own multiple-choice "(A) True / (B) False"
#    template), and (b) was scored by greedily decoding a token instead
#    of reading the model's own logit probability mass on the True
#    option (the paper's actual method -- a continuous, calibrated
#    score, not a hard 0/1 label). ALSO fixed a THIRD bug at the call
#    site: it was passing the query's GROUND-TRUTH answer for the model
#    to judge, rather than the model's OWN greedily-generated answer --
#    judging the correct answer every time makes a truthfulness
#    detector trivially uninformative about the model's actual
#    hallucinations.
#
# 2. Operadic consistency (baseline_operadic_consistency, NEW): the
#    citation (Bottman, Liu & Richardson 2026, arXiv:2606.13649) was
#    flagged in skc_refs.bib with "VERIFY: author list unresolved" --
#    checked the live arXiv page directly before writing a single line
#    of implementation; title/authors/year/abstract all confirmed real.
#    Implements the paper's actual method (compare the model's direct
#    answer to the composed question against an answer obtained by
#    decomposing into sub-questions and composing the sub-answers;
#    score their disagreement -- label-free), reusing the same two-hop
#    decomposition machinery as Self-Ask below rather than a second,
#    independently-diverging copy of that logic.
#
# 3. Self-Ask itself got a related fix while verifying Operadic
#    consistency's shared decomposition path: Press et al. (2023,
#    arXiv:2210.03350, Section 3.1) states self-ask is NEVER zero-shot
#    (always 2-6 few-shot exemplars) -- the pre-existing implementation
#    ran it zero-shot, understating this MANDATORY baseline (the wrong
#    direction to be wrong in). Fixed with two in-domain few-shot
#    exemplars matching this paper's own two-hop template shape. This
#    means self-ask's own AUROC/AURC/Risk@50%/rho in this run will NOT
#    exactly match the earlier e4_gemma2b_layer8_se / e4_gemma2b_layer8_rho
#    runs -- that is EXPECTED and correct, not a reproducibility failure;
#    the earlier runs used the unverified zero-shot version.
#
# 4. Oracle observing M: added directly to run_E4()'s own baseline_scores
#    dict (scored as m.astype(float), where m is the same M label used
#    throughout this file, already computed from the model's own greedy
#    generations by label_queries_batched() earlier in main()) -- a
#    genuine upper-bound ceiling, not assumed or hand-set to a round
#    number.
#
# All four changes verified via skc_pipeline_test.py (51/51 still pass)
# before this job was written.

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
OUT="$REPO/results/e4_gemma2b_layer8_ptrue_oc"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-p-true \
    --run-operadic-consistency \
    --run-self-ask \
    --run-semantic-entropy \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "P(True), Operadic consistency, and Oracle observing M scores (plus"
echo "self-ask/semantic-entropy/BGR re-scored under the corrected code) are"
echo "in $OUT/E4.json's own \"selectors\" and \"rho\" dicts -- fill Table"
echo "tab:bgr-full and tab:bgr's remaining \\RES{} cells from this file."
echo "Note: self-ask's own numbers will differ from the earlier"
echo "e4_gemma2b_layer8_rho run (that used an unverified zero-shot"
echo "template; this run uses the verified few-shot one) -- use THIS"
echo "run's self-ask numbers, not the older ones, going forward."

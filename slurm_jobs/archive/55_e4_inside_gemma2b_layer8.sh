#!/bin/bash -l
#SBATCH -J skc_e4_inside_gemma2b
#SBATCH -o e4_inside_gemma2b_l8_%j.out
#SBATCH -e e4_inside_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 32 (Phase 2, item 9 of 12 E4 baselines): INSIDE / EigenScore
# (Chen et al., ICLR 2024), Gemma 2 2B, layer 8.
#
# Fully UNSUPERVISED, no fit split at all -- unlike every prior E4
# baseline this session, INSIDE has no fit_*/score_* pair and needs no
# --saplma-train-queries. The score is computed directly per E4 test
# query: sample k=10 generations, form the (k x k) Gram matrix of their
# UNIT-NORMALIZED hidden states (last token of prompt+sample), take the
# regularized log-determinant. High EigenScore = dispersed/hallucinated,
# low = consistent/truthful (negated in score_inside() to match this
# script's own "higher = more truthful" convention for every selector).
#
# REAL BUG FOUND AND FIXED BEFORE THIS RUN (caught offline, never
# reached a GPU): the first version of eigenscore() mean-centered the k
# representations across samples before normalizing. Verified directly
# with synthetic data that this is WRONG: centering k vectors and then
# unit-normalizing forces their pairwise cosine similarities toward
# EXACTLY -1/(k-1) regardless of the ORIGINAL data's structure --  a
# tightly-clustered "consistent" set and a fully-random "dispersed" set
# produced nearly IDENTICAL EigenScores under centering (-5.977 vs
# -5.969), versus a correct, massive separation once centering was
# removed (-59.0 vs -0.01). Centering was destroying exactly the
# "common direction" signal EigenScore is built to detect. Fixed by
# removing the centering step entirely -- verified again offline with a
# fresh random seed after the fix, same large, correctly-ordered
# separation.
#
# -t 14:00:00: INSIDE needs k=10 sampled generations PER TEST QUERY
# (239 queries), plus a capture pass over each of the 10 samples per
# query -- costlier per query than any prior E4 baseline's eval-time
# cost, though with NO fit-time cost at all (nothing to train).

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
OUT="$REPO/results/e4_gemma2b_layer8_inside"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-saplma \
    --run-entity-token-probe \
    --run-inside \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma2b_layer8_v2.json" \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

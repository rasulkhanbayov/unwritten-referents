#!/bin/bash -l
#SBATCH -J skc_e4_hs_gemma2b
#SBATCH -o e4_hs_gemma2b_l8_%j.out
#SBATCH -e e4_hs_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# STEP 31 (Phase 2, item 8 of 12 E4 baselines): HaloScope (Du, Xiao & Li,
# NeurIPS 2024), Gemma 2 2B, layer 8.
#
# *** BEST-FAITH RECONSTRUCTION, NOT VERIFIED AGAINST THE PUBLISHED
# ALGORITHM *** -- per the user's own explicit choice, given genuine
# uncertainty recalling HaloScope's precise method from memory alone
# (see fit_haloscope()'s and _self_consistency_membership_score()'s own
# docstrings in skc_run_e4.py for exactly which parts are confident vs.
# reconstructed): the GENERAL framing (an unsupervised direction
# estimated from a soft membership split of UNLABELED hidden states,
# with NO ground-truth labels used to fit the direction itself) is
# faithful to HaloScope's actual contribution; the SPECIFIC membership
# proxy (self-consistency across k=10 sampled generations, via
# generate_sampled()) and the specific direction-estimation step
# (difference-of-means between top/bottom membership terciles, not
# their own SVD/subspace procedure) are disclosed simplifications, not
# a line-by-line reproduction. Verify against the actual paper before
# treating this baseline's numbers as a faithful HaloScope result in
# any submitted table -- this run is a real, honest data point under a
# disclosed uncertain reconstruction, not a placeholder to silently
# firm up later.
#
# -t 20:00:00 (up from every other E4 job this session): HaloScope's
# own fit step needs k=10 SAMPLED generations PER FIT QUERY (via
# generate_sampled(), on top of the usual one greedy generation), on
# the full ~10k-query fit split -- a real, substantial cost increase
# over every prior E4 baseline, which sampled at most once per fit
# query.
#
# K1 is computed via label_queries_batched() as a DIAGNOSTIC ONLY
# (printed in the saved gate's own notes field), never used to fit the
# direction -- the whole point of testing an "unsupervised" baseline is
# that it does not require labels for its core estimation step.

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
OUT="$REPO/results/e4_gemma2b_layer8_haloscope"
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
    --run-haloscope \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma2b_layer8_v2.json" \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

#!/bin/bash -l
#SBATCH -J skc_e4_etp_gemma2b
#SBATCH -o e4_etp_gemma2b_l8_%j.out
#SBATCH -e e4_etp_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 29 (Phase 2, item 6 of 12 E4 baselines): entity-level token probe
# (Obeso et al. 2025), Gemma 2 2B, layer 8. First real GPU run.
#
# ADAPTATION DISCLOSED (see fit_entity_token_probe()'s own docstring in
# skc_run_e4.py): Obeso et al.'s actual setting is long-form generation
# with MULTIPLE entity mentions per response, each probed independently.
# This pipeline's queries produce one short answer per query, so this
# implementation probes the SINGLE entity-mention position where
# generation begins (checked directly to be where the answer entity is
# actually named for this pipeline's templates, via
# _strip_generation_filler()'s own established finding that real
# generations name the answer immediately before any descriptive
# filler) -- the same underlying mechanism (linear probe at an ENTITY
# MENTION position, not the prompt's own entity or the response's final
# token, which is what distinguishes it from the surface gate and from
# SAPLMA respectively), applied to this pipeline's own single-answer
# setup rather than a literal reproduction of the long-form multi-entity
# setting.
#
# Includes a runtime guard (_check_no_seam_remerge) on every single
# query, checked against the exact bug class already found twice this
# session (BOS-offset and decode/re-encode token-boundary bugs in
# skc_harness.py's own bridge-finding logic) -- raises loudly rather
# than silently mis-positioning if the tokenizer ever re-merges tokens
# across the prompt/answer seam for a specific prompt, rather than
# trusting the general argument (verified once, offline, on gpt2) to
# hold for every real query in this run.
#
# Runs together with --run-saplma (already run once in job 51, but
# cheap enough to re-run alongside a new baseline and keeps this job
# self-contained) and --run-self-ask, all sharing the SAME loaded model
# and the SAME fit-split file. -t raised further (14h vs. job 51's 12h)
# since this adds a full second generation+capture pass over the ~10k
# fit queries.

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
OUT="$REPO/results/e4_gemma2b_layer8_entitytoken"
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
    --saplma-train-queries "$REPO/data/onehop_fit_gemma2b_layer8_v2.json" \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

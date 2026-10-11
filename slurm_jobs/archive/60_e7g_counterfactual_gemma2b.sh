#!/bin/bash -l
#SBATCH -J skc_e7g_cf_gemma2b
#SBATCH -o e7g_cf_gemma2b_%j.out
#SBATCH -e e7g_cf_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 4:00:00

# Phase 3, item 1 of 2 LOAD-BEARING controls: (g) Counterfactual chains,
# Gemma 2 2B. Paper's own words (skc_main.tex, Appendix~\ref{app:controls}):
# "Counterfactual chains, so memorized composites cannot drive the result."
# Explicitly flagged in the paper's own text as one of THREE load-bearing
# controls (with (f) random-direction and (j) cross-type generalization) --
# "without which memorized composites can produce the effect."
#
# PROTOCOL (both explicit choices, confirmed directly with the user rather
# than assumed): (1) IN-CONTEXT counterfactual, NOT model editing -- states
# a false bridge premise directly in the prompt text ("Suppose the place of
# birth of {e1} is actually {b_fake}."), no ROME/MEMIT-style weight edit at
# all. (2) BEHAVIORAL measurement only -- does the model's final answer
# track the STATED (fake) premise, or fall back to a memorized TRUE chain?
# No gate/probe reading at the bridge position under this prompt (a fuller
# variant would also capture activations here; explicitly scoped out in
# favor of the simpler behavioral test for this first pass).
#
# DATASET: data/counterfactual_twohop.json, built by
# skc_build_dataset_counterfactual.py from THIS SAME model/layer's own
# standard 239-query E1-filtered test set
# (results/e1_gemma2b_layer8/queries_filtered.json), NOT the full raw
# 3503-query pool -- explicit choice, for direct comparability with every
# other E4/E7 result in this pipeline, all of which use this same 239-query
# test set. Every counterfactual bridge (b_fake) is a REAL city drawn from
# this SAME 239-query pool's own 55 unique cities (spanning 38 countries),
# with a REAL, different country than the query's own true bridge -- never
# a fabricated place. Confirmed zero (a_true == a_fake) collisions and zero
# (b_true == b_fake) identical picks across all 239 built items before this
# run.
#
# COST: cheap relative to Phase 2's E4 baselines -- ONE greedy generation
# per query (max_new_tokens=8, short country-name answers), no sampling, no
# fit split, no activation capture at all. -t 4:00:00 is generous, not tight.
#
# LETTERING NOTE: skc_run_e7_ablations.py's own module docstring previously
# used a DIFFERENT, stale lettering scheme for this and several other
# controls -- corrected this same session after checking skc_main.tex's own
# Appendix~\ref{app:controls} text directly (the manuscript is the ground
# truth). This control is (g) there, not (i) as an earlier internal note
# had it.

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
OUT="$REPO/results/e7g_counterfactual_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e7_ablations.py counterfactual \
    --counterfactual-queries "$REPO/data/counterfactual_twohop.json" \
    --model gemma-2-2b \
    --seed 0 \
    --out "$OUT"

echo "Result (p_composed, p_memorized, p_neither_or_ambiguous) is in"
echo "$OUT/e7g_counterfactual.json -- per-item detail in e7g_counterfactual_per_item.json."

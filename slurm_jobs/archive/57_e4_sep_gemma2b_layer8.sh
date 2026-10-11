#!/bin/bash -l
#SBATCH -J skc_e4_sep_gemma2b
#SBATCH -o e4_sep_gemma2b_l8_%j.out
#SBATCH -e e4_sep_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 36:00:00

# STEP 34 (Phase 2, item 11 of 12 E4 baselines): Semantic Entropy Probe
# (Kossen, Han, Razzak, Schut, Malik & Gal, arXiv:2406.15927,
# "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in
# LLMs"), Gemma 2 2B, layer 8.
#
# NOTE on venue: the paper itself says "Preprint. Under review." -- could
# NOT verify a final peer-reviewed venue beyond the arXiv preprint (a web
# search suggested an ICML 2024 workshop, but this was not confirmed
# inside the paper text itself). Cite as arXiv:2406.15927, not a venue
# that hasn't been directly confirmed.
#
# VERIFIED directly against the arXiv PDF (not a memory reconstruction):
# real semantic entropy (Farquhar et al. 2024) needs k~10 sampled
# generations PER QUERY at INFERENCE time; SEPs instead train a probe to
# PREDICT semantic entropy from a SINGLE hidden state (one greedy
# generation, no sampling) -- eval-time cost is cheap by construction,
# unlike every other Class-2 baseline in this file (INSIDE, HaloScope)
# which resample at eval time too. The expensive part is FIT time only:
# real semantic entropy labels the fit split (k=10 samples + Kuhn et
# al. 2023 bidirectional-NLI-entailment clustering + Shannon entropy
# over clusters -- this run's own generate_sampled +
# bidirectional_entailment_clusters + semantic_entropy, first real use
# of the NLI-clustering path in this file), the paper's own linear
# logistic-regression probe (not an MLP) is then fit on TBG ("token
# before generating", the prompt's own last token) hidden states to
# predict a MEDIAN split of that continuous entropy into low/high-SE
# groups (reuses skc_gates.fit_probe exactly as E0's estimators do).
#
# DISCLOSED SIMPLIFICATIONS (see fit_semantic_entropy_probe()'s own
# docstring): (1) single fixed layer (this pipeline's own --layer),
# not the paper's own per-model 5-adjacent-top-layer concatenation from
# a dedicated layer sweep; (2) median split, not the paper's own
# MSE-minimizing threshold search; (3) TBG only, not also SLT (second-
# last response token) -- SLT needs locating an end-of-response token
# this pipeline's short-answer generations do not reliably produce.
#
# COST WARNING, THE REASON FOR -t 36:00:00 (longest of any E4 job this
# session): the NLI-clustering fit step is genuinely expensive -- up to
# k*(k-1) = 90 pairwise NLI calls (bidirectional, so 2x the C(10,2)=45
# unordered pairs) PER FIT QUERY, over the full ~10k-item fit split (no
# subsampling used, per explicit choice to keep full statistical power
# for this baseline). CORRECTED cost estimate (a first offline timing
# attempt gave a wildly wrong ~30s/NLI-call figure because that test
# accidentally ran OUTSIDE the skc_iclr27 conda env, under a different
# torch/transformers build with some CPU inference pathology -- caught
# and redone correctly): inside skc_iclr27, a real NLI forward pass on
# CPU is ~0.24s; worst-case (no union-find early-exit savings, which in
# practice will reduce this somewhat whenever samples cluster together
# before every pair is checked) that is ~60 CPU-hours for the whole fit
# split's clustering step alone, before counting generation or the base
# model's own forward passes. GPU should be faster than this CPU figure
# but the exact speedup for a single-example NLI forward pass (batch
# size 1, not batched across pairs) was NOT verified before this
# submission -- -t is set generous specifically because of this
# unresolved uncertainty, and the REAL wall-clock number from this run
# is itself useful information for sizing any later semantic-entropy-
# family baseline (Semantic Entropy itself, next up, reuses this exact
# same expensive clustering machinery).
#
# Offline-verified before this run (skc_iclr27 conda env, correct
# transformers==4.57.6/torch==2.4.1): bidirectional_entailment_clusters
# + semantic_entropy produce CORRECT real results on hand-built test
# cases -- 3 paraphrases of "Paris" cluster together apart from
# "Berlin" (entropy 0.562 for a 4-item 3/1 split); all-identical inputs
# give entropy exactly 0; all-mutually-contradicting inputs give the
# correct max entropy ln(3)=1.0986 for 3 singleton clusters. Also
# verified fit_probe's sign convention directly: acts_known=low-SE
# (truthful) activations score HIGHER than acts_unknown=high-SE
# (confabulating) activations on synthetic separable data, matching
# every other selector's "higher score = more truthful" convention in
# this file.

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
OUT="$REPO/results/e4_gemma2b_layer8_sep"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-semantic-entropy-probe \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma2b_layer8_v2.json" \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

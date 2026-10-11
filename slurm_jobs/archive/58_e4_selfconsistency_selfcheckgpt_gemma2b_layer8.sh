#!/bin/bash -l
#SBATCH -J skc_e4_scsc_gemma2b
#SBATCH -o e4_scsc_gemma2b_l8_%j.out
#SBATCH -e e4_scsc_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 35-36 (Phase 2, items 9 and 12 of 12 E4 baselines -- bundled in
# ONE job since both are FULLY UNSUPERVISED, eval-time-only, no fit
# split, and no eager attention needed, unlike Lookback Lens/ICR Probe):
# k=10 self-consistency sampling baseline AND SelfCheckGPT (Manakul,
# Liusie & Gales, EMNLP 2023, arXiv:2303.08896, NLI variant), Gemma 2
# 2B, layer 8. --layer is a no-op for both -- neither reads a residual-
# stream layer at all, kept only because argparse requires --layer for
# the shared subparser.
#
# INDEPENDENT of job 57's own outcome (Semantic Entropy Probe) -- does
# NOT wait on it. Submitted in parallel per explicit instruction to
# avoid wasting idle time between jobs whenever two pieces of work do
# not actually depend on each other.
#
# self-consistency: pairwise agreement across k=10 samples, reusing
# _self_consistency_membership_score() (built for HaloScope's fit step)
# rather than reimplementing the same pairwise-comparison logic inline
# -- avoids yet another instance of the "two copies of the same logic
# diverging" bug class already hit several times this project.
#
# SelfCheckGPT: VERIFIED directly against the arXiv PDF (not a memory
# reconstruction). Paper's own N=20 samples (NOT this file's usual
# k=10 -- kept faithful to the paper rather than matching this file's
# other sampling baselines). NLI variant: each sampled passage is the
# PREMISE, the model's own greedy response is the HYPOTHESIS, only the
# entailment/contradiction logits are used (neutral dropped), mean
# contradiction probability over the N samples, negated (1 - mean
# contradiction) to match this file's own "higher = more truthful"
# convention. ADAPTATION DISCLOSED: paper's own design assumes multi-
# sentence passages; this pipeline's short single-phrase answers are
# treated as ONE "sentence" scored against each full sampled response,
# a mechanical, disclosed adaptation to a regime the paper itself does
# not test (confirmed: paper is silent on short-response applicability).
#
# COST: both are eval-time-only (no ~10k-item fit split unlike SAPLMA/
# entity-token-probe/Lookback Lens/HaloScope/ICR Probe/Semantic Entropy
# Probe), bounded by the E4 test set size (239 queries) -- self-
# consistency: 1 greedy + k=10 sampled generations per query;
# SelfCheckGPT: 1 greedy + N=20 sampled generations + up to N=20 NLI
# calls per query. Estimated NLI-call cost alone: ~8 minutes at a
# conservative 0.1s/call GPU estimate (verified CPU number inside the
# correct skc_iclr27 conda env was 0.24s/call; GPU should not be
# slower) -- small relative to generation cost, well within -t.
#
# Offline-verified before this run (skc_iclr27 conda env, gpt2/CPU):
# baseline_selfcheckgpt() runs end-to-end, correct shape, all scores
# finite and in [0,1], real variance across prompts (0.968 vs 0.260 on
# a 2-prompt test). baseline_self_consistency() also runs end-to-end;
# both test scores came back exactly 0.0 on gpt2 -- investigated
# directly (not assumed a bug): gpt2's own k=4 samples at temperature=1
# for a vague prompt were four completely different, non-overlapping
# continuations with no real entity agreement, so 0.0 is the correct,
# honest score for a weak base model's genuine inconsistency, not a
# degenerate function. _entailment_contradiction_probs() verified
# directly: a clear entailment pair scores P(entail)=0.9999, a clear
# contradiction pair scores P(contradict)=0.9997, both correctly sum to
# 1.0 (confirming the neutral-excluded softmax works as intended).

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
OUT="$REPO/results/e4_gemma2b_layer8_scsc"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-self-consistency \
    --run-selfcheckgpt \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

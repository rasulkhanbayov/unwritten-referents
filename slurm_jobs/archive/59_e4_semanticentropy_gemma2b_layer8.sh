#!/bin/bash -l
#SBATCH -J skc_e4_se_gemma2b
#SBATCH -o e4_se_gemma2b_l8_%j.out
#SBATCH -e e4_se_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 37 (Phase 2, item 12 of 12 E4 baselines -- THE LAST ONE): Semantic
# Entropy (Farquhar, Kossen, Kuhn & Gal, Nature 2024, "Detecting
# hallucinations in large language models using semantic entropy"),
# Gemma 2 2B. --layer is a no-op -- this method reads no hidden state at
# all, only sampled TEXT + NLI clustering, kept only because argparse
# requires --layer for the shared subparser.
#
# This is the DIRECT method Semantic Entropy Probe (job 388791, --run-
# semantic-entropy-probe, AUROC=0.4137) exists to approximate cheaply --
# pays the k=10-samples-plus-NLI-clustering cost at EVAL time, per test
# query, rather than once at fit time. Fully unsupervised, no fit split,
# no --saplma-train-queries needed.
#
# REUSES the SAME NLI-clustering machinery (generate_sampled,
# bidirectional_entailment_clusters, semantic_entropy) already built and
# GPU-verified this phase for HaloScope/INSIDE/Semantic Entropy Probe --
# including _truncate_for_nli(), REQUIRED here for the exact same reason
# it was required in fit_semantic_entropy_probe()/baseline_selfcheckgpt()
# (raw generate_sampled() output at temperature=1.0 reliably continues
# past the real answer into a fabricated new Q&A turn; see
# _truncate_for_nli()'s own docstring for the real GPU bugs this fixed,
# jobs 388434/388439). This baseline does NOT need the ceiling-split fix
# that Semantic Entropy Probe's own FIT step needed (job 388675/388791)
# -- that fix was specific to binarizing a skewed fit-split distribution
# for probe TRAINING; Semantic Entropy itself is a continuous score
# reported directly, no split of any kind.
#
# COST: bounded by the 239-query E4 TEST set, not the ~10k-item fit
# split Semantic Entropy Probe's own fit step paid its NLI-clustering
# cost on (239/9985 = 2.4% the size) -- should be dramatically cheaper
# than that job, well within -t 14:00:00.
#
# Offline-verified before this run (skc_iclr27 conda env, real GPU,
# gemma-2-2b): _semantic_entropy_for_prompt() gives entropy=0.0 for
# "capital of France" (all 10 samples correctly agree on "Paris") and
# entropy=1.748 for the real fit-split query "place of birth of Jeff
# Lebo" (an obscure entity the model does not know -- matches the exact
# same value independently measured for this identical query while
# investigating Semantic Entropy Probe's own fit-split bug, confirming
# consistency across both baselines' shared code paths).

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
OUT="$REPO/results/e4_gemma2b_layer8_se"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-semantic-entropy \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

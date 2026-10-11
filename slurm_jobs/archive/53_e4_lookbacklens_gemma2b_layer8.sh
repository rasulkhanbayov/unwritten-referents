#!/bin/bash -l
#SBATCH -J skc_e4_ll_gemma2b
#SBATCH -o e4_ll_gemma2b_l8_%j.out
#SBATCH -e e4_ll_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00

# --mem 60000MB (up from 40000MB): Lookback Lens now loads a SECOND,
# separate model instance under eager attention (see below for why),
# so peak memory briefly holds two loaded model copies at once, not one.
#
# STEP 30 (Phase 2, item 7 of 12 E4 baselines): Lookback Lens (Chuang et
# al. 2024), Gemma 2 2B, layer 8 (--layer here is a no-op for this
# baseline specifically -- Lookback Lens's own feature spans ALL layers
# by construction, unlike SAPLMA/entity-token-probe which read one
# chosen layer -- kept only because argparse requires --layer for the
# shared subparser).
#
# REAL INFRASTRUCTURE GAP FOUND AND FIXED: load_model() never supported
# attn_implementation="eager". Confirmed directly (gpt2, offline) that
# the DEFAULT attention backend (SDPA) does not materialize per-head
# attention weights at all -- model(..., output_attentions=True)
# .attentions[0] comes back None, not a smaller tensor, under the
# default. load_model() now takes an explicit need_attentions=True
# flag (kwargs["attn_implementation"]="eager" only when set), defaulting
# to False so every OTHER model load in this pipeline keeps its prior
# (faster) behavior unchanged. This job is the first real use of that
# flag -- lookback_ratio_features() itself raises loudly if attentions
# ever come back None, rather than silently scoring on a garbage
# feature vector.
#
# SECOND REAL BUG FOUND AND FIXED, on the FIRST successful run of this
# job (387696): eager attention was originally loaded onto the SAME
# model instance every other selector (surface_gate, BGR-min/lr, SAPLMA,
# entity-token-probe, self-ask) also used. Confirmed directly that
# attn_implementation="eager" is NOT numerically equivalent to the
# default SDPA backend on bfloat16 -- mean absolute logit difference
# 0.168 on a real forward pass (checked directly, gpt2, bfloat16), from
# the two implementations summing attention scores in a different order
# and bfloat16's low precision not cancelling the difference out. This
# silently changed surface_gate/BGR-min/BGR-lr's own numbers relative to
# every other E4 run this session (confirmed: those three scores were
# byte-identical across the SAPLMA-only and entity-token-probe runs,
# then changed once --run-lookback-lens was added to the same model
# load). Fixed by giving Lookback Lens its OWN, separate model instance
# (loaded under eager attention), so every OTHER selector in this run
# now uses the exact same (default-backend) model instance as every
# other E4 run this session -- this job's own surface_gate/BGR-min/BGR-lr
# numbers should now match jobs 51/52's exactly again.
#
# ADAPTATION DISCLOSED (see lookback_ratio_features()'s own docstring):
# Chuang et al.'s own setup is token/span-level in long-form generation;
# this pipeline's single-short-answer setting averages the lookback
# ratio over the WHOLE generated answer span per (layer, head), the same
# collapse-to-one-unit pattern already used for SAPLMA and the
# entity-token probe.
#
# Verified offline (gpt2, CPU) before this run: feature vector shape
# (144,) = 12 layers x 12 heads on that model; all values in the valid
# [0,1] ratio range with real variance across heads, not a degenerate
# constant vector.
#
# -t 16:00:00 (up from the other E4 jobs' 12-14h): eager attention is
# slower than the default backend, and this run also carries self-ask +
# SAPLMA + entity-token-probe alongside Lookback Lens in one job.

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
OUT="$REPO/results/e4_gemma2b_layer8_lookbacklens"
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
    --run-lookback-lens \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma2b_layer8_v2.json" \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

#!/bin/bash -l
#SBATCH -J skc_e4_icr_gemma2b
#SBATCH -o e4_icr_gemma2b_l8_%j.out
#SBATCH -e e4_icr_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# --mem 60000MB: ICR Probe loads its OWN second eager-attention model
# instance (same reason as Lookback Lens -- eager attention is NOT
# numerically identical to the default SDPA backend on bfloat16,
# confirmed directly, mean abs logit diff 0.168), so peak memory briefly
# holds two loaded model copies at once, not one.

# STEP 33 (Phase 2, item 10 of 12 E4 baselines): ICR Probe (Zhang, Hu,
# Zhang, Zhang & Wan, ACL 2025, "ICR Probe: Tracking Hidden State
# Dynamics for Reliable Hallucination Detection in LLMs",
# aclanthology.org/2025.acl-long.880, arXiv:2507.16488), Gemma 2 2B,
# layer 8 (--layer here is a no-op for this baseline specifically --
# ICR's own feature spans ALL layers by construction, same situation as
# Lookback Lens; kept only because argparse requires --layer for the
# shared subparser).
#
# VERIFIED, NOT RECONSTRUCTED: unlike HaloScope, the exact algorithm was
# read directly from the ACL Anthology PDF (equations + algorithm boxes),
# not recalled from memory. ICR = "Information Contribution to Residual
# stream". At every layer l and answer token i: Delta_x_i^l = x_i^l -
# x_i^{l-1} (the residual-stream update at that layer/position). Two
# distributions over earlier tokens 0..i are compared: Proj_i^l (softmax
# of Delta_x_i^l projected onto each earlier token's PRE-update hidden
# state) and Attn_i^l (the model's own mean-over-heads attention
# distribution at that layer/position). ICR score = their
# Jensen-Shannon divergence, restricted to the top-k=20 tokens by
# attention weight (the paper's own ablation found k=20 optimal). Mean-
# pooled over every answer token, per layer, into a 1xL feature vector.
# The "ICR Probe" itself is the paper's supervised MLP
# (L -> 128 -> 64 -> 32 -> 1, LeakyReLU, sigmoid), trained here on this
# pipeline's own one-hop fit split with the same K1-correctness label
# every other trained E4 baseline uses (SIMPLIFIED relative to the
# paper's full training-loop spec -- no dropout/batch-norm/Kaiming
# init/LR schedule -- same "core architecture faithful, training loop
# simplified" pattern already used for SAPLMA).
#
# Offline-verified before this run (gpt2, CPU): _icr_scores_for_text()
# produces a (n_layers,) vector with correct shape (12 for gpt2), all
# values finite, non-negative, and bounded above by ln(2) (JSD's
# theoretical max under natural log) -- confirmed both bounds hold
# exactly, real variance across layers (not a degenerate constant
# vector). ICRProbeClassifier verified to reach AUROC=1.0 on clearly-
# separable synthetic data (a sanity check that training actually
# learns a real decision boundary, not that it will generalize here).
#
# COST WARNING: ICR's fit step does a full attention+hidden-state
# forward pass over EVERY fit-split item (~10k), with an O(seq^2)
# Python-level inner loop per layer per answer token (computing the
# projection and top-k JSD by hand, not a single batched matrix op) --
# this is a heavier fit-time cost than any prior E4 baseline except
# HaloScope's k=10 sampling. Scoped to ONLY self-ask + ICR Probe in this
# job (no SAPLMA/entity-token-probe/other baselines bundled in) to keep
# this job's own failure mode isolated and its walltime budget legible
# if it needs a second attempt.

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
OUT="$REPO/results/e4_gemma2b_layer8_icrprobe"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-icr-probe \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma2b_layer8_v2.json" \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Cost report (forward passes / tokens / wall-clock per selector) is in"
echo "$OUT/e4_cost_report.json -- this is a REQUIRED table in the paper, not optional."

#!/bin/bash -l
#SBATCH -J skc_e0_sweep_gemma2b
#SBATCH -o e0_sweep_gemma2b_%j.out
#SBATCH -e e0_sweep_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# GPU item: REAL per-layer E0 refit sweep, Gemma 2 2B, all 26 layers --
# resolves the E3 "sweep every layer" TORUN properly this time, per the
# user's own explicit choice (build the real per-layer E0 refit
# pipeline, all three estimators, same 20k-query protocol as the
# existing single-layer E0), after job 390468's own naive
# --layers 0..25 attempt on skc_run_e3.py silently only tested layer 8
# (that tool has a hard guard refusing to score any layer against a
# gate fit at a DIFFERENT layer -- see EXECUTION_PLAN.md).
#
# NEWLY BUILT this session: skc_run_e0_layersweep.py (the real per-
# layer refit driver) and skc_gates.capture_activations_all_layers()
# (captures ALL requested layers in ONE forward pass per batch, instead
# of one pass per layer -- VERIFIED to produce byte-identical
# activations to 26 separate single-layer hook passes on a synthetic
# multi-layer module before ever touching a real model, since forward
# hooks do not affect the forward computation itself). This is what
# makes 26 layers' worth of gate-fitting tractable in one job: the
# model load and the K1/K2 labeling passes (the genuinely expensive
# parts) happen ONCE, shared across every layer; only the per-layer
# gate-fitting (diffmeans/probe: cheap numpy; SAE: one extra sae.encode
# call per layer) and scoring repeat 26 times.
#
# ALL THREE ESTIMATORS at every layer: confirmed directly (not assumed)
# that Gemma Scope's "gemma-scope-2b-pt-res-canonical" release has a
# canonical width-16k SAE dictionary at EVERY ONE of Gemma-2-2b's 26
# layers (checked via sae_lens's own
# get_pretrained_saes_directory()), so no --skip-sae fallback is needed
# at any layer for this specific model.
#
# SAME PROTOCOL as the existing single-layer E0 (job 382897/etc.): same
# --n 20000 subsample, same --fit-frac 0.5 entity-level split, same
# 0.65 AUROC / 0.50 agreement bars, same named-bridge K2 check -- not a
# lighter-weight sweep-specific protocol.
#
# --mem 60000MB (up from the single-layer E0's 40000MB): holding 26
# layers' worth of captured activations in memory simultaneously (for
# the fit split, held-out split, AND named-bridge queries, each ~20k/
# 10k/10k items x 2304-dim x 26 layers) is a real memory increase over
# a single layer's own capture.
#
# -t 20:00:00: dominated by the SAME K1/K2 labeling cost as a normal E0
# run (unchanged, done once) plus 26x the SAE-encode cost (cheap per
# layer) and 3x the forward-pass cost of a single-layer capture (one
# pass captures all 26 layers, but the fit/held-out/named-bridge splits
# each still need their OWN pass) -- generous budget for the first real
# attempt at this scale.

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
OUT="$REPO/results/e0_layersweep_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0_layersweep.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --fit-frac 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Full per-layer E0 results are in $OUT/e0_layersweep_full.json -- report"
echo "which layers (if any) clear the full stop-gate (n_estimators_passing>=2"
echo "AND agreement>=0.50), and whether any of them ALSO have a working"
echo "Patchscopes decode (cross-reference against the disclosed-limitation"
echo "note in Section~\ref{sec:res-blindspot} of the paper, which tried layers"
echo "8/12/16 for this specific combination)."

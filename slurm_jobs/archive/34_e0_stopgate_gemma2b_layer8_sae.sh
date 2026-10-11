#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_l8_sae
#SBATCH -o e0_gemma2b_l8_sae_%j.out
#SBATCH -e e0_gemma2b_l8_sae_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 11: E0 at layer 8, base model, WITH the SAE estimator enabled for the
# first time this session.
#
# WHY THIS COULD NOT RUN BEFORE: skc_run_e0.py's SAE branch was a stub
# (`raise NotImplementedError`) -- fit_sae_gate() itself was fully
# implemented and correct, but nothing ever called sae.encode() to turn
# raw residual-stream activations into per-latent SAE activations, and
# load_sae()'s own docstring flagged the loading API as UNVERIFIED,
# never run. Two real bugs were found and fixed while wiring this up
# (see skc_harness.py's load_sae() docstring and EXECUTION_PLAN.md):
#   1. SAE_IDS held each model's HUGGING FACE REPO id
#      ("google/gemma-scope-2b-pt-res"), but sae_lens's `release` argument
#      wants a DIFFERENT string -- its own release-registry key, which for
#      the canonical (single recommended L0) release has a "-canonical"
#      suffix. Confirmed directly by reading the installed sae_lens
#      package's own pretrained_saes.yaml, not guessed.
#   2. SAE.from_pretrained(...) returns the SAE object DIRECTLY in the
#      installed sae_lens==6.5.3, not a (sae, cfg, _) tuple as the
#      original code assumed -- confirmed by reading sae_lens's own
#      internal usage in sae_lens/evals.py.
# Verified end-to-end offline (no GPU) before this run: load_sae()
# loads the real Gemma Scope layer-8 width-16k canonical SAE
# (d_in=2304 matching Gemma-2-2b's hidden size, d_sae=16384,
# W_dec shape (16384, 2304)), and sae.encode() on a random probe tensor
# returns the expected (n, 16384) shape.
#
# This run drops --skip-sae (present in every earlier E0 run this
# session) so all three estimators are fit and their agreement checked.
# Output to a NEW directory (e0_gemma2b_layer8_sae) -- the existing
# e0_gemma2b_layer8_v2 (diffmeans+probe only, agreement 0.547, PASSED,
# already feeding E1/E3/E4/E5/E7/E2) is untouched and remains the primary
# gate for those downstream experiments; this run is specifically to
# report the SAE estimator's own agreement and to fill in Table tab:gap's
# "all three estimators" precondition, not to replace the existing chain.

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
OUT="$REPO/results/e0_gemma2b_layer8_sae"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b, layer 8, all three estimators including SAE) PASSED."
    echo "frozen_gate_sae.json is in $OUT alongside diffmeans/probe -- check"
    echo "the 3-way agreement number specifically, since SAE was never in the"
    echo "loop for any earlier passing E0 run this session."
else
    echo "E0 (gemma-2-2b, layer 8, all three estimators) FAILED (exit $exit_code)." >&2
    echo "Check whether SAE specifically is the estimator dragging agreement" >&2
    echo "down -- diffmeans+probe alone already passed at this layer" >&2
    echo "(e0_gemma2b_layer8_v2, agreement 0.547)." >&2
fi
exit "$exit_code"

#!/bin/bash -l
#SBATCH -J skc_e0_qwen32b_l24
#SBATCH -o e0_qwen32b_l24_%j.out
#SBATCH -e e0_qwen32b_l24_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# RETRY of E0 on Qwen3-32B at a DIFFERENT LAYER, after job 389099
# (layer 8) FAILED the stop-gate with a REAL, interesting finding: BOTH
# diffmeans (AUROC=0.887) and probe (AUROC=0.848) individually cleared
# the 0.65 threshold comfortably (tight, clearly-above-chance CIs), but
# their AGREEMENT was too low (min pairwise Spearman=0.297, required
# >=0.5) -- they separate known/unknown entities well on their OWN, but
# disagree substantially about WHICH specific entities are known/
# unknown. This is the SAME disagreement class as Gemma-2-9B/
# Llama-3.1-8B's earlier SAE-vs-others split, but here it is between
# diffmeans and probe THEMSELVES (no SAE involved at all, since
# --skip-sae was already required for this 4-bit model) -- --skip-sae
# does not help here.
#
# LAYER CHOICE: 24 (out of 64 total layers, confirmed directly via
# AutoConfig.from_pretrained('Qwen/Qwen3-32B').num_hidden_layers = 64,
# not assumed), 37.5% depth. Checked the actual layer counts of every
# model that succeeded before picking this, not guessed blind:
# Gemma-2-2b succeeded at layer 8/26 (30.8% depth); Llama-3.1-70B
# succeeded at layer 8/80 (10% depth, notably even SHALLOWER than
# Qwen3-32B's own failing 12.5% at layer 8/64) -- so depth fraction
# alone does not cleanly explain the disagreement (Llama's own success
# at a shallower fraction argues against "layer 8 is just too early"
# being the whole story). Chosen as a substantial, informative jump
# from layer 8 rather than the smallest possible step, on the
# reasoning (common in interpretability work, not verified specific to
# Qwen3) that residual-stream "knowledge"-type signals in larger models
# often concentrate somewhat past the first quarter of depth -- a
# genuinely different bet, report the actual numbers here before
# choosing a third layer if this one also fails, rather than guessing
# further blind.
#
# Everything else mirrors job 64 (the layer-8 attempt) exactly --
# same --n 20000 subsample, --skip-sae, --fit-frac 0.5, seed 0, same
# HF_HUB_* download-robustness env vars (this run should hit a cache
# hit from job 64/389099's own download, not a fresh one, unless the
# cache was cleared since).

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see other scripts' identical comment.

export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=1

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e0_qwen32b_layer24"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model qwen3-32b \
    --layer 24 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (qwen3-32b, 4-bit, layer 24) PASSED. Continue the E1/E3/E5"
    echo "chain at THIS layer (update --layer/--gate-dir in"
    echo "65_e1_main_qwen32b_layer8.sh etc. from layer8 to layer24 before"
    echo "submitting them, and rename the scripts/output dirs for clarity)."
else
    echo "E0 (qwen3-32b, 4-bit, layer 24) FAILED (exit $exit_code)." >&2
    echo "Check whether this is the SAME disagreement pattern as layer 8" >&2
    echo "(both estimators individually pass but disagree with each other)" >&2
    echo "or a DIFFERENT failure mode (e.g. one or both estimators now" >&2
    echo "failing AUROC outright) -- these mean different things about" >&2
    echo "whether a third layer is worth trying versus accepting the" >&2
    echo "disagreement as a genuine finding." >&2
fi
exit "$exit_code"

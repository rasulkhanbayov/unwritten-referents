#!/bin/bash -l
#SBATCH -J skc_e0_qwen32b_l8
#SBATCH -o e0_qwen32b_l8_%j.out
#SBATCH -e e0_qwen32b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# Phase 3, item 5 (final item of the original Phase 3 scope): E0 on
# Qwen3-32B, 4-bit (bitsandbytes), layer 8, --skip-sae (no public SAE
# dictionary exists for Qwen3 -- same reasoning as Llama-3.1-70B, which
# ALSO has no public SAE and used --skip-sae successfully, see
# results/e0_llama70b_layer8/ -- job 384352, PASSED on its first
# genuine attempt: diffmeans AUROC=0.771, probe AUROC=0.752,
# agreement=0.514). Mirrors that model's own already-proven E0 script
# (37_e0_stopgate_llama70b_layer8.sh) almost exactly -- same --mem/-c
# allocation, same --n 20000 subsample, same HF_HUB_* download-
# robustness env vars (a REAL prior failure on Llama-3.1-70B, job
# 384036, spent nearly its whole budget stuck on repeated HEAD-request
# timeouts before the model ever finished loading -- fixed there by
# raising HF_HUB_ETAG_TIMEOUT/DOWNLOAD_TIMEOUT from their 10s defaults
# to 120s and enabling the hf_transfer Rust backend; carried forward
# here defensively, since Qwen3-32B is a different but still large
# (32B, 4-bit) download from a potentially congested repo).
#
# ARCHITECTURE VERIFIED BEFORE THIS RUN (not assumed to match Llama):
# load_model()'s own post-load assertion checks `model.model.layers`
# exists -- confirmed directly against the installed transformers
# source (v5.16.1, modeling_qwen3.py): Qwen3ForCausalLM.__init__ sets
# self.model = Qwen3Model(config), and Qwen3Model.__init__ sets
# self.layers = nn.ModuleList([Qwen3DecoderLayer(...) for ...]) --
# Qwen3PreTrainedModel.base_model_prefix = "model", so model.model.layers
# resolves exactly as it does for Llama/Gemma-2, not GPT-2's own
# different model.transformer.h layout. This assertion will still fire
# loudly if this verification turns out wrong for the actual installed
# version, rather than silently misbehaving deep inside a hook.
#
# LAYER CHOICE: 8, same starting point as every other new-model E0
# attempt this project has made (mirrors Llama-3.1-70B's own job 37
# comment) -- a genuinely different bet for a 32B/64-layer(ish) model
# than it was for Gemma-2-2b's 26 layers, not a proportional match. If
# this fails, a deeper, proportionally-matched layer is the natural next
# attempt -- report the actual numbers before resubmitting rather than
# guessing a new layer blind.
#
# Qwen/Qwen3-32B may or may not be a gated HF repo (unlike
# meta-llama/Llama-3.1-70B, which IS gated) -- if this run fails with a
# 401/GatedRepoError, that is the first thing to check, distinct from an
# OOM or a real AUROC/agreement failure.

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
OUT="$REPO/results/e0_qwen32b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model qwen3-32b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (qwen3-32b, 4-bit, layer 8) PASSED. Continue the E1/E3/E5"
    echo "chain at this layer, difference-of-means/probe only (no SAE row"
    echo "for this model in Table tab:gap, marked with its own dagger footnote,"
    echo "same convention as Llama-3.1-70B)."
else
    echo "E0 (qwen3-32b, 4-bit, layer 8) FAILED (exit $exit_code)." >&2
    echo "Check whether the failure is OOM, a gated-repo/401 error, a" >&2
    echo "download timeout, or a real AUROC/agreement failure -- these need" >&2
    echo "different fixes and look different in the log." >&2
fi
exit "$exit_code"

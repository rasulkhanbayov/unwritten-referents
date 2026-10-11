#!/bin/bash -l
#SBATCH -J skc_diag_act_collision
#SBATCH -o diag_act_collision_%j.out
#SBATCH -e diag_act_collision_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 20000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 00:15:00

# DIAGNOSTIC ONLY, not part of the main E0-E7 pipeline.
#
# WHY: investigating E1's silence-check KS statistic (large, unexpected
# given AUROC(gate->M) is at chance) found -- via offline inspection of
# the ALREADY-SAVED acts_surface_filtered.npy, no GPU needed for that
# part -- that three pairs of genuinely different queries in E1's 158-item
# filtered set have BYTE-IDENTICAL captured activation vectors (confirmed
# via np.array_equal on the full 2304-dim vectors, not just a similar
# magnitude). All six affected queries fall in the SAME batch (global
# indices 32-47, batch_size=16), at alternating (not adjacent) positions.
# This looks like a real indexing/padding bug in
# capture_activations()/entity_spans_for_prompts(), used by EVERY
# experiment this session (E0/E1/E3/E4/E5 all route through
# capture_for_queries -> capture_activations) -- worth pinning down before
# trusting any of those numbers further.
#
# This run re-captures JUST that one 16-item batch with verbose per-item
# logging (real token count, computed entity span, actual index used,
# padded batch length, tokenizer padding_side) to find exactly where the
# indexing goes wrong. Cheap: 16 prompts, 1 layer, 1 forward pass.

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

cd "$REPO/src/driver_scripts"
python3 skc_diag_activation_collision.py \
    --queries "$REPO/results/e1_gemma2b_layer16/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 16 \
    --batch-start 32 \
    --batch-size 16

echo
echo "Look for rows sharing the same act_hash (a short fingerprint of the"
echo "captured vector) -- those are the (near-)collisions. Cross-reference"
echo "against real_tokens/span/idx to see whether the collision traces to"
echo "identical spans, identical indices, or something else entirely."

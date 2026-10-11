#!/bin/bash -l
#SBATCH -J skc_diag_act_collision_repro
#SBATCH -o diag_act_collision_repro_%j.out
#SBATCH -e diag_act_collision_repro_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 20000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 00:15:00

# DIAGNOSTIC ONLY, not part of the main E0-E7 pipeline.
#
# ROUND 2, CONFIRMING THE MECHANISM: round 1 (24_diag_activation_collision.sh)
# forced padding_side="right" before capturing and found ZERO collisions
# -- but the REAL E1 run never forces padding_side before
# capture_for_queries -> capture_activations (confirmed: skc_gates.py has
# no reference to padding_side at all; capture_for_queries never calls
# set_padding_side either). Round 1 also happened to print
# "tokenizer.padding_side at load time: 'left'" -- so the real run's
# padding_side at capture time was very plausibly left over as "left"
# from label_queries_batched()'s own generate_greedy() call (which forces
# "left" internally, then restores the PRIOR side -- itself the
# tokenizer's own untouched default, i.e. "left", since load_model()
# never sets a side explicitly). This run passes --reproduce-bug to skip
# forcing "right", confirming the collision reproduces under the
# tokenizer's own real, unforced state.

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
    --batch-size 16 \
    --reproduce-bug

echo
echo "If collisions now appear (matching round 1's already-confirmed offline"
echo "finding: Miloserdov/Buryakina/Stanic/Kleiza and Popescu/Sobral sharing"
echo "act_hash), that confirms padding_side was the real cause -- fix"
echo "capture_for_queries() to call set_padding_side(tok, 'right') explicitly"
echo "before capturing, the same discipline generate_greedy() already uses"
echo "for its own required side."

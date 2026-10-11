#!/bin/bash -l
#SBATCH -J skc_e3_gemma2b_full
#SBATCH -o e3_gemma2b_fullsweep_%j.out
#SBATCH -e e3_gemma2b_fullsweep_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# GPU item 2 of 9: E3 frozen-direction sweep across EVERY layer of
# Gemma 2 2B (26 layers total, confirmed directly via
# AutoConfig.from_pretrained('google/gemma-2-2b').num_hidden_layers),
# not only layer 8 -- closes the paper's own disclosed limitation
# ("the runs below sweep three position offsets at the single layer
# where each model's own gate is currently trustworthy").
#
# NOTE: this sweeps the FROZEN gate direction fit at EACH layer
# independently is NOT what this does -- skc_run_e3.py's own --layers
# argument captures activations at every listed layer and scores them
# against the SAME primary-estimator gate from --gate-dir (fit only at
# the layer E0 passed, layer 8). Sweeping "every layer" here means:
# does the layer-8 gate direction, read at OTHER layers' own residual
# stream, still show a K2 signal at the bridge position? This is a
# different, cheaper question than re-fitting a whole new E0 gate at
# every layer (which would need re-running E0 26 times, not done here
# -- if the paper's own intent is a fully independent per-layer gate
# fit, that is a substantially bigger follow-up, flag this distinction
# when writing up the result rather than presenting one as the other).
#
# -t 20:00:00 (up from the single-layer sweep's usual 12h): 26 layers x
# 3 position offsets = 78 total (layer, position) activation captures,
# roughly 26x a single-layer run's own capture cost, though the
# permutation null (n-perm=1000) is computed once per (layer, position)
# pair, not once per layer -- still a real, substantial cost increase.

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
OUT="$REPO/results/e3_gemma2b_fullsweep"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --position-offsets -1 0 1 \
    --n-perm 1000 \
    --seed 0 \
    --out "$OUT"

echo "Full per-layer AUROC grid is in $OUT/e3_full_map.json -- report the peak"
echo "(layer, position) pair and whether it differs from the single-layer"
echo "(layer 8) result already in the paper."

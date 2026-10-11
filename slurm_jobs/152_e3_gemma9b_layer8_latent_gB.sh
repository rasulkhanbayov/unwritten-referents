#!/bin/bash -l
#SBATCH -J skc_e3_gemma9b_latent
#SBATCH -o e3_gemma9b_l8_latent_%j.out
#SBATCH -e e3_gemma9b_l8_latent_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 10:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos -- see jobs 145-151's own headers.

# CRITICAL FIX: this session's entire "bridge score" (g_B) -- used in every
# E0/E1/E4 run, the gs-vs-gb finding, and the C4 bridge-only-vs-BGR-min
# ablation -- was NEVER a latent read. skc_harness.capture_for_queries()'s
# own docstring says outright: its third return value ("acts_hop2", what
# every g_B in this project traces back to) is captured from an ISOLATED
# hop2 prompt that NAMES THE GOLD BRIDGE ("The country of the city
# {gold bridge} is"), not from inside the composed prompt. Verified directly
# by reading skc_harness.py:646-692 and tracing skc_run_e1.py:269's own call
# site. AUROC(g_B->K2)=0.853 on Llama-3.1-70b confirms this: a gate reading
# a prompt that already names the bridge naturally predicts K2 (whether the
# model knows THAT bridge's own country) very well -- expected once you see
# what is actually being read, not evidence of latent bridge resolution.
#
# skc_run_e3.py already implements the ACTUAL latent read this paper needs:
# the frozen gate scored at an offset from the SURFACE ENTITY'S OWN SPAN,
# inside the COMPOSED prompt itself (never naming the bridge) --
# see that script's own extensive header for why offsets are anchored to
# the entity span's end, not the prompt's end. It was previously only ever
# used to plot the AUROC(gate->K2) sweep curve (Figure fig:offsets), never
# to produce a corrected g_B for the BGR/bridge-only pipeline.
#
# This job sweeps offsets -1,0,1 at layer 8 (this model's own confirmed-
# stable positive-control layer, agreement 0.50-0.52 across 5 seeds) and,
# via the NEW --save-peak-activations flag (added this session), saves the
# raw per-item activations at the sweep's own peak (layer, offset) -- the
# corrected, genuinely latent g_B -- to peak_activations.npy/peak_labels.json.
#
# --queries points at this model's own K1=1-filtered eval set
# (queries_filtered.json from job 106/e1_gemma9b_layer8_skipbridge_splitfix)
# -- the SAME population every other real number for this model uses, so
# the corrected g_B is directly comparable to the existing (contaminated)
# one.

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
OUT="$REPO/results/e3_gemma9b_layer8_latent_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/results/e1_gemma9b_layer8_skipbridge_splitfix/queries_filtered.json" \
    --model gemma-2-9b \
    --gate-dir "$REPO/results/e0_gemma9b_layer8_splitfix" \
    --primary-estimator diffmeans \
    --layers 8 \
    --position-offsets -1 0 1 \
    --seed 0 \
    --save-peak-activations \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E3 latent-read sweep (gemma-2-9b, layer 8) completed (exit 0)."
    echo "Check $OUT/e3_full_map.json's own peak and permutation_null -- report"
    echo "whether the peak clears the null before trusting peak_activations.npy"
    echo "as the corrected g_B."
else
    echo "E3 latent-read sweep (gemma-2-9b, layer 8) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

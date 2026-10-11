#!/bin/bash -l
#SBATCH -J skc_e3_llama70b_latent
#SBATCH -o e3_llama70b_l25_latent_%j.out
#SBATCH -e e3_llama70b_l25_latent_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 20:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos -- see jobs 145-151's own headers.

# CRITICAL FIX companion to job 152 (Gemma-2-9B) -- see that job's own
# header for the full account of the named-gold-bridge-read bug this
# closes. This job sweeps offsets -1,0,1 at layer 25 (this model's own
# NEW, confirmed-stable positive-control layer, agreement 0.60-0.63
# across 5 seeds, the widest passing band of any model in the project),
# NOT the old layer 8 the model's own e3_llama70b_layer8 result used.
#
# --queries points at this model's own K1=1-filtered eval set
# (queries_filtered.json from job 107/e1_llama70b_layer25_skipbridge_splitfix).
#
# HF_HUB_* robustness env vars: same as every other 70B job this project.

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
OUT="$REPO/results/e3_llama70b_layer25_latent_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --layers 25 \
    --position-offsets -1 0 1 \
    --seed 0 \
    --save-peak-activations \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E3 latent-read sweep (llama-3.1-70b, layer 25) completed (exit 0)."
    echo "Check $OUT/e3_full_map.json's own peak and permutation_null -- report"
    echo "whether the peak clears the null before trusting peak_activations.npy"
    echo "as the corrected g_B."
else
    echo "E3 latent-read sweep (llama-3.1-70b, layer 25) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

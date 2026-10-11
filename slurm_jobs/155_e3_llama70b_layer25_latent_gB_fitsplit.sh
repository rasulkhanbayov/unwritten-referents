#!/bin/bash -l
#SBATCH -J skc_e3_llama70b_latent_fit
#SBATCH -o e3_llama70b_l25_latent_fit_%j.out
#SBATCH -e e3_llama70b_l25_latent_fit_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 14:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos -- see jobs 145-153's own headers.

# Companion to job 153 (Llama-3.1-70B's own eval-set latent g_B capture)
# and job 154 (Gemma-2-9B's own fit-split companion) -- see job 154's own
# header for the full account of why a fresh fit-split capture is needed
# for the corrected C4 ablation's Standardizer.
#
# ONLY sweeps offset -1 (this model's own CONFIRMED peak from job 153,
# AUROC(gate->K2)=0.682, p=0.0 against the permutation null).
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
OUT="$REPO/results/e3_llama70b_layer25_latent_fitsplit"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/data/c4_fit_queries.json" \
    --model llama-3.1-70b \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --layers 25 \
    --position-offsets -1 \
    --seed 0 \
    --save-peak-activations \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E3 latent-read fit-split capture (llama-3.1-70b, layer 25, offset -1) completed (exit 0)."
    echo "peak_activations.npy/peak_labels.json here are the FIT-SPLIT g_B for the"
    echo "corrected C4 ablation's Standardizer."
else
    echo "E3 latent-read fit-split capture (llama-3.1-70b) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

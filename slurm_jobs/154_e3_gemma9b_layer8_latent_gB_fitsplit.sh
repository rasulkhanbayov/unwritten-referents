#!/bin/bash -l
#SBATCH -J skc_e3_gemma9b_latent_fit
#SBATCH -o e3_gemma9b_l8_latent_fit_%j.out
#SBATCH -e e3_gemma9b_l8_latent_fit_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 08:00:00

# Using gpu-H200/h200_hbku/h200_hbku_qos -- see jobs 145-153's own headers.

# Companion to job 152 (Gemma-2-9B's own eval-set latent g_B capture) --
# this job captures the SAME corrected, genuinely latent g_B (via
# skc_run_e3.py's own composed-prompt offset read, --save-peak-activations)
# on the DISJOINT FIT-SPLIT population (data/c4_fit_queries.json, built
# earlier this project for the original C4 ablation's Standardizer --
# 1253 real, entity-disjoint two-hop chains) instead of the eval set.
#
# WHY THIS IS NEEDED: rebuilding the C4 ablation honestly (bridge-only vs
# BGR-min vs BGR-mean) on the corrected latent g_B needs a genuine
# fit-split Standardizer (per selectors.Standardizer's own docstring: eval-
# set standardization leaks). The ORIGINAL fit-split export
# (f1_twohop_fitsplit.jsonl, job 137) used the CONTAMINATED g_B (via
# skc_run_e1.py's own capture_for_queries() hop2 read) -- that export is
# now stale for this purpose and cannot be reused; a fresh fit-split
# capture with the corrected read is needed.
#
# ONLY sweeps offset -1 (this model's own CONFIRMED peak from job 152,
# AUROC(gate->K2)=0.657, p=0.0 against the permutation null) -- not the
# full -1/0/1 range, since the peak is already known from the eval-set
# sweep and offset selection itself must not be re-done on this
# (different) population to avoid a second, new form of leakage.

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
OUT="$REPO/results/e3_gemma9b_layer8_latent_fitsplit"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/data/c4_fit_queries.json" \
    --model gemma-2-9b \
    --gate-dir "$REPO/results/e0_gemma9b_layer8_splitfix" \
    --primary-estimator diffmeans \
    --layers 8 \
    --position-offsets -1 \
    --seed 0 \
    --save-peak-activations \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E3 latent-read fit-split capture (gemma-2-9b, layer 8, offset -1) completed (exit 0)."
    echo "peak_activations.npy/peak_labels.json here are the FIT-SPLIT g_B for the"
    echo "corrected C4 ablation's Standardizer."
else
    echo "E3 latent-read fit-split capture (gemma-2-9b) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

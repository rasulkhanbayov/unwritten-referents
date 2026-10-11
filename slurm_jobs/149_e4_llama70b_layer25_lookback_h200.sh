#!/bin/bash -l
#SBATCH -J skc_e4_llama70b_ll_h200
#SBATCH -o e4_llama70b_l25_ll_h200_%j.out
#SBATCH -e e4_llama70b_l25_ll_h200_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 120000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 20:00:00

# Phase 5 / M2/M3: REBUILD of job 142 on H200, for hardware consistency
# with jobs 145-148 (the other 9 baselines for this model, all run on
# H200 after A100 nodes were reported full). Job 142 itself SUCCEEDED
# on A100 (lookback_lens=0.666, icr_probe=0.583) -- this is not a
# failure rebuild, it is a consistency rebuild: comparing job 142's own
# surface_gate/BGR-min/BGR-lr against 145-148's identical values found a
# small but real discrepancy (surface_gate 0.4949 on A100 vs 0.4839 on
# H200, BGR-min 0.5387 vs 0.5375, BGR-lr 0.6664 vs 0.6692 -- differences
# of 0.001-0.011 AUROC), most plausibly the same class of hardware/
# kernel floating-point difference already documented elsewhere in this
# project (e.g. the eager-vs-SDPA-attention bfloat16 numerics finding).
# Small and within this model's own measured seed-to-seed noise (5-seed
# E0 agreement spread was 0.599-0.633, a much larger range), but per the
# user's own choice: re-run on H200 so ALL 11 baselines + BGR variants
# for this model come from one consistent hardware run, rather than
# disclosing a mixed-hardware caveat in the paper.
#
# Output to a NEW directory (e4_llama70b_layer25_lookback_h200) -- the
# old A100 result (e4_llama70b_layer25_lookback) stays on disk as a
# historical record / for the hardware-difference measurement above to
# remain checkable.
#
# Otherwise IDENTICAL to job 142 (same flags, same --sequential-model-load
# fix, same queries/gate-dir/saplma-train-queries).

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
OUT="$REPO/results/e4_llama70b_layer25_lookback_h200"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix/queries_filtered.json" \
    --model llama-3.1-70b \
    --layer 25 \
    --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-lookback-lens \
    --run-icr-probe \
    --sequential-model-load \
    --saplma-train-queries "$REPO/data/onehop_fit_llama70b_layer25_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/llama-3.1-70b/f1_twohop_baselines_lookback_h200.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (llama-3.1-70b, layer 25, Lookback Lens + ICR-probe, H200) completed (exit 0)."
    echo "Verify surface_gate/BGR-min/BGR-lr numbers here match jobs 145-148's own"
    echo "EXACTLY (0.4839487814952499 / 0.537546468401487 / 0.6692275919041718) --"
    echo "if so, this confirms the earlier discrepancy against job 142 was purely"
    echo "hardware, and this H200 result replaces job 142's own A100 numbers in"
    echo "the final Table tab:bgr-full."
else
    echo "E4 (llama-3.1-70b, layer 25, Lookback Lens + ICR-probe, H200) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

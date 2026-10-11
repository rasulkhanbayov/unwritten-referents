#!/bin/bash -l
#SBATCH -J skc_e4_rho_gemma2b
#SBATCH -o e4_rho_gemma2b_l8_%j.out
#SBATCH -e e4_rho_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# Rerun of job 59 (e4_gemma2b_layer8_se), same queries/gate/seed, needed
# ONLY because rho ($\hat\rho$, the share of the blind term a selector
# closes) was never computed for any selector except BGR-min --
# skc_experiments.py::run_E4() hardcoded a single recovery_fraction() call
# for "BGR-min" specifically, and raw per-item scores were never saved to
# disk, so self-ask/semantic-entropy/BGR-lr's own rho could not be
# recovered from the existing e4_gemma2b_layer8_se/E4.json after the fact
# (confirmed via two research-only subagent passes before deciding to
# rerun rather than guess or leave blank -- see EXECUTION_PLAN.md).
#
# FIX (already applied, not part of this job): run_E4() now computes
# recovery_fraction() for EVERY selector in `scores`, not just BGR-min --
# rho is defined for any candidate repair score (recovery_fraction()
# takes score_repair as a plain array; read/ref/blind/r_read_star/
# r_ref_star are identical across selectors, only r_repair and hence rho
# varies). Verified this change is backward compatible: skc_pipeline_test.py
# still passes all 51 checks (including "rho is reported" for BGR-min)
# after the change.
#
# Everything else about this job is IDENTICAL to job 59
# (e4_gemma2b_layer8_se) -- same queries, gate, layer, seed, coverage,
# --run-self-ask/--run-semantic-entropy flags -- so AUROC/AURC/Risk@50%
# for every selector should reproduce exactly (deterministic given the
# same seed and cached model), and the only NEW information in this run's
# output is the `rho` dict now attached to every selector.

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
OUT="$REPO/results/e4_gemma2b_layer8_rho"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-semantic-entropy \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "rho for every selector (self_ask, semantic_entropy, BGR-lr, BGR-min) is in"
echo "$OUT/E4.json's own top-level \"rho\" dict -- fill Table tab:bgr/tab:bgr-full's"
echo "remaining \\RES{} rho cells from this file, cross-checked against the"
echo "AUROC/AURC/Risk@50% numbers already in the paper (should match"
echo "e4_gemma2b_layer8_se/E4.json's own selectors block exactly, since this"
echo "is the same run with the same seed)."

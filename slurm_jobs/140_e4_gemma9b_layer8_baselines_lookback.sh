#!/bin/bash -l
#SBATCH -J skc_e4_gemma9b_ll
#SBATCH -o e4_gemma9b_l8_ll_%j.out
#SBATCH -e e4_gemma9b_l8_ll_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00

# Phase 5 / M2/M3: Lookback Lens + ICR-probe on Gemma-2-9B -- SEPARATE
# job from 139's own "main" batch because both need a SECOND, eager-
# attention model instance (need_attentions=True), which was confirmed
# NOT numerically equivalent to the default SDPA backend on bfloat16
# (mean abs logit diff 0.168) -- combining these with any OTHER selector
# sharing the DEFAULT model instance would silently corrupt that other
# selector's own score, the exact real bug found and fixed earlier this
# project (see job 53's own header for the original gemma-2-2b incident).
# skc_run_e4.py already isolates them onto their OWN model load
# internally (ll_model/ll_tok, confirmed by reading the script directly)
# -- this job still keeps them in their own invocation, matching the
# historical pattern, rather than trusting that isolation for the first
# time on a new model without the same discipline.
#
# --run-self-ask included too (mandatory per skc_REDFLAGS.md, cheap, uses
# the DEFAULT model instance -- safe alongside Lookback Lens/ICR-probe
# since self-ask's own scoring happens on `model`/`tok`, not `ll_model`).
#
# Same --queries/--gate-dir/--saplma-train-queries as job 139.

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
OUT="$REPO/results/e4_gemma9b_layer8_lookback"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma9b_layer8_skipbridge_splitfix/queries_filtered.json" \
    --model gemma-2-9b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma9b_layer8_splitfix" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-lookback-lens \
    --run-icr-probe \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma9b_layer8_splitfix.json" \
    --coverage 0.5 \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/gemma-2-9b/f1_twohop_baselines_lookback.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (gemma-2-9b, layer 8, Lookback Lens + ICR-probe) completed (exit 0)."
    echo "Verify surface_gate/BGR-min/BGR-lr numbers here match job 139's own"
    echo "EXACTLY (same seed, same gate -- should be byte-identical) before"
    echo "trusting the isolation held on this model too."
else
    echo "E4 (gemma-2-9b, layer 8, Lookback Lens + ICR-probe) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

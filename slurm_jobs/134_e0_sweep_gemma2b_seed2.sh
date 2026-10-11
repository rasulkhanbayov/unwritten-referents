#!/bin/bash -l
#SBATCH -J skc_e0_sweep_gemma2b_seed2
#SBATCH -o e0_sweep_gemma2b_seed2_%j.out
#SBATCH -e e0_sweep_gemma2b_seed2_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# VERSION3 era, Phase 4 / C3: multi-seed layer search for Gemma-2-2b, after
# jobs 108-114 (E0 seed 1-4 at layer 16, the model's PREVIOUS headline
# layer) revealed layer 16's own seed-0 pass (agreement=0.706) was a lucky
# outlier -- all 4 other seeds FAIL, with wildly unstable, sign-flipping
# agreement (+0.336, -0.237, -0.226, -0.015). Per the user's explicit
# choice: attempt a genuine multi-seed layer search rather than dropping
# this model or accepting the single-seed result.
#
# WHY A FRESH FULL SWEEP, NOT JUST A FEW CANDIDATE LAYERS: the original
# 26-layer sweep (job 84, results/e0_layersweep_gemma2b/) that identified
# layer 16 as the best candidate was itself BOTH (a) leaky-split (fixed
# this session in skc_run_e0.py, but NOT YET in skc_run_e0_layersweep.py
# at the time job 84 ran) AND (b) single-seed -- exactly the two things
# that turned out to matter. A second single-seed sweep, even under the
# fix, would repeat the same mistake with different noise. This job is
# ONE of 5 (seeds 0-4) full re-sweeps under `skc_run_e0_layersweep.py` AS
# IT NOW STANDS (both its own split-leakage fix AND its relaxed --model
# choices, both applied earlier this session) -- run all 5, then rank
# layers by AGREEMENT STABILITY ACROSS SEEDS (e.g. minimum or mean
# agreement across all 5 seeds), not by any single seed's own best value,
# before picking a new headline layer.
#
# NOTE: even THIS job's own seed=0 run is not a reuse of job 84's old
# output -- skc_run_e0_layersweep.py's own split-leakage bug (a SEPARATE,
# independently-found bug from skc_run_e0.py's) was fixed AFTER job 84
# ran, so job 84's numbers (and the per-layer ranking jobs 92/93 used
# earlier this session) are not trustworthy at all, not just at layer 16.

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
OUT="$REPO/results/e0_layersweep_gemma2b_seed2"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0_layersweep.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --fit-frac 0.5 \
    --seed 2 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "Gemma-2-2b full layer sweep, seed 2 (SPLIT-LEAKAGE FIX applied to"
    echo "the SWEEP SCRIPT ITSELF, not just skc_run_e0.py) completed (exit 0)."
    echo "Once all 5 seeds (0-4) are done, compare each layer's agreement ACROSS"
    echo "seeds -- do not pick a headline layer from any single seed's own"
    echo "ranking again."
else
    echo "Gemma-2-2b full layer sweep, seed 2 FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

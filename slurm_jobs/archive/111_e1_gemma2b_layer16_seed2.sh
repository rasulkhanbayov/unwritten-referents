#!/bin/bash -l
#SBATCH -J skc_e1_gemma2b_l16_seed2
#SBATCH -o e1_gemma2b_l16_seed2_%j.out
#SBATCH -e e1_gemma2b_l16_seed2_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# VERSION3 era, Phase 4 / C3: 5-seed reruns, E1 half.
#
# RUN THIS ONLY AFTER THE MATCHING E0 SEED JOB (e0_gemma2b_l16_seed2)
# HAS COMPLETED AND PASSED -- this job's --gate-dir depends on that job's own
# frozen_gate_{estimator}.json existing. If that seed's E0 did NOT pass
# (gate_open=false), DO NOT run this job for that seed -- report the failed
# seed instead, the same "flag back, don't force it" discipline as every
# other E0 stop-gate in this project.
#
# Also exports real per-item records for VERSION3 via --export-records,
# split=eval (this pipeline's held-out entities), seed=2 -- so this run
# both produces the E1 result AND feeds Phase 4/C3's own seed requirement
# for VERSION3's re-analysis scripts directly (reconcile.py's own
# io.require_seeds(recs, 5) check).
#
# Bridge-resolution flag matches this model/layer's own established
# convention from seed 0 (see e1_gemma2b_layer16_splitfix's own build rationale) --
# not re-decided per seed, since Patchscopes decodability is a property of
# model/layer depth, not of which entities landed in which split.

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
OUT="$REPO/results/e1_gemma2b_layer16_seed2"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/twohop_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-2b \
    --layer 16 \
    --gate-dir "$REPO/results/e0_gemma2b_layer16_seed2" \
    --primary-estimator diffmeans \
    --seed 2 \
    --export-records "$REPO/VERSION3/data/records/gemma-2-2b/f1_twohop_seed2.jsonl" \
    --export-split eval \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (gemma-2-2b, layer 16, seed 2) completed (exit 0)."
    echo "Check $OUT/E1.json. Records exported to"
    echo "VERSION3/data/records/gemma-2-2b/f1_twohop_seed2.jsonl -- NOTE:"
    echo "these need to be MERGED with the other 4 seeds' own files into one"
    echo "f1_twohop.jsonl (VERSION3's io.py loads one file per model/family, not"
    echo "per seed) before reconcile.py can see all 5 seeds at once."
else
    echo "E1 (gemma-2-2b, layer 16, seed 2) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

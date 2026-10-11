#!/bin/bash -l
#SBATCH -J skc_e0_sweep_llama8b
#SBATCH -o e0_sweep_llama8b_%j.out
#SBATCH -e e0_sweep_llama8b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 20:00:00

# VERSION3 era: full per-layer E0 sweep for Llama-3.1-8B, under the
# SPLIT-LEAKAGE FIX, per the user's explicit choice (full sweeps for all
# 3 models still failing their headline layer, matching the rigor
# Gemma-2-2b's own 26-layer sweep already got -- not blind layer guesses).
#
# WHY THIS JOB EXISTS: job 96 (layer 8, splitfix) FAILED
# (agreement=0.453, down from the leaky split's own 0.613) -- same
# pattern as Gemma-2-2b/layer-8's own reversal. No historical per-layer
# sweep exists for this model (unlike Gemma-2-2b), so this sweep is the
# first one.
#
# TWO FIXES BAKED INTO skc_run_e0_layersweep.py BEFORE THIS JOB, BOTH
# NEEDED FOR THIS RUN TO BE TRUSTWORTHY:
#  1. skc_run_e0_layersweep.py PREVIOUSLY called split_entities() WITHOUT
#     exclude_from_fit, despite already requiring --named-bridge-queries
#     as an argument -- it loaded that file later (for K2 labeling) but
#     never consulted it for the split. This means the ORIGINAL
#     Gemma-2-2b 26-layer sweep (e0_layersweep_gemma2b/) has the SAME
#     leak as skc_run_e0.py's own pre-fix single-layer results, and its
#     own per-layer rankings (used to pick jobs 92/93's candidate layers)
#     were themselves computed on a leaky split. This is now fixed at the
#     source (skc_run_e0_layersweep.py now loads --named-bridge-queries
#     early and passes exclude_from_fit into split_entities(), mirroring
#     skc_run_e0.py's own fix) -- confirmed via skc_pipeline_test.py
#     (51/51 still passing) before this job was built.
#  2. --model was previously hard-restricted to choices=["gemma-2-2b"]
#     because full per-layer SAE coverage was confirmed only for that
#     model. Relaxed to include this model: the per-layer SAE fit is
#     already wrapped in its own try/except (see the script's "SAE
#     fitting failed" fallback) and degrades to diffmeans+probe only for
#     whichever layers lack coverage, rather than silently skipping the
#     whole run or assuming full coverage. Llama Scope IS listed in
#     skc_harness.SAE_RELEASE_IDS for this model, but per-layer coverage
#     completeness was never separately confirmed the way Gemma Scope's
#     was -- this sweep's own per-layer output will show which layers
#     got a working SAE gate and which fell back, rather than assuming.
#
# LAYER COUNT: meta-llama/Llama-3.1-8B is publicly documented as 32
# transformer layers (0-31), but per this project's own "confirm
# directly, don't assume" discipline (see e.g. Qwen3-32B's own
# AutoConfig.num_hidden_layers check), --layers below is generated from
# a live AutoConfig read at the START of this job, not hardcoded from
# outside knowledge -- if the live count differs from 32, the printed
# value below will show it BEFORE the sweep argument is built, so a
# mismatch is caught immediately rather than silently sweeping the wrong
# range.
#
# --mem 100000MB (up from job 96's single-layer 60000MB): holding 32
# layers' worth of captured activations simultaneously (fit + held-out +
# named-bridge splits) is proportionally larger than Gemma-2-2b's 26-layer
# sweep needed (60000MB, up from single-layer 40000MB) -- scaled up
# further here since 8B's per-layer hidden dim and total layer count are
# both larger than 2B's.
#
# Reuses job 96's own --n 20000 / --fit-frac 0.5 / --seed 0 protocol,
# same as the Gemma-2-2b sweep matched its own single-layer protocol.

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
OUT="$REPO/results/e0_layersweep_llama8b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"

echo "Confirming meta-llama/Llama-3.1-8B's actual layer count via AutoConfig ..."
N_LAYERS=$(python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('meta-llama/Llama-3.1-8B')
print(c.num_hidden_layers)
")
echo "AutoConfig reports num_hidden_layers=$N_LAYERS"
LAYERS=$(seq 0 $((N_LAYERS - 1)))

python3 skc_run_e0_layersweep.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model llama-3.1-8b \
    --layers $LAYERS \
    --fit-frac 0.5 \
    --seed 0 \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "Llama-3.1-8B full layer sweep (SPLIT-LEAKAGE FIX applied) completed (exit 0)."
    echo "Report which layers (if any) clear gate_open (n_estimators_passing>=2"
    echo "AND agreement>=0.50) from $OUT/e0_layersweep_full.json, ranked by"
    echo "cross-estimator agreement, the same way Gemma-2-2b's own sweep was analyzed."
else
    echo "Llama-3.1-8B full layer sweep FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

#!/bin/bash -l
#SBATCH -J skc_e7_gemma2b_l8
#SBATCH -o e7_gemma2b_l8_%j.out
#SBATCH -e e7_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 06:00:00

# --mem is higher than other single-model jobs in this directory (60GB vs
# 40GB) because the (k) label-validation subcommand below loads TWO models
# at once (gemma-2-2b under test + gemma-2-9b-it as judge) on the SAME GPU.
# This is a shared cluster -- size this to what the job actually needs
# rather than copying the 40GB default from other scripts; an OOM kill here
# wastes a queue slot other users could have used.
#
# STEP 10, SECONDARY TRACK, RE-RUN AFTER THE PADDING-SIDE FIX AND THE
# LAYER-8-VS-16 TRADEOFF DECISION: E7 at layer 8.
#
# WHY: $GATE now points at e0_gemma2b_layer8_v2/frozen_gate_diffmeans.json
# -- the first genuinely trustworthy E0 result this session (agreement
# 0.547, WITH the capture_activations() padding-side fix applied). The
# original 11_e7_ablations_gemma2b.sh's own history is a cautionary
# tale about exactly this class of layer/gate mismatch (see that
# script's own comment: it once silently scored layer-8 activations
# against a layer-16-fit gate direction) -- --layer below is 8
# EVERYWHERE it appears, matching $GATE, checked directly rather than
# assumed.
#
# Runs the named-bridge control (a), random-direction negative control
# (j), and label validation (k) subcommands in sequence, same set as
# the layer-16 script. popularity (b) still needs a --pageviews file
# (not fetched anywhere in this bundle) and attention (h) is still
# unimplemented -- neither is in this script, same as the layer-16
# version.
#
# Output to a NEW directory (e7_gemma2b_layer8) -- the old
# e7_gemma2b result (layer 16) stays on disk as a historical record.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here. torch==2.4.1+cu124's wheel
# bundles its own complete CUDA 12.4 runtime (cusparse, nvJitLink, etc.) and
# needs nothing from the system toolkit -- nothing in this pipeline compiles
# a custom CUDA extension. `module load cuda12.1/toolkit/12.1.1` was tried
# here originally and caused a real failure: it prepends an OLDER cusparse/
# nvJitLink onto LD_LIBRARY_PATH ahead of torch's own bundled cu124 libs,
# producing `undefined symbol: __nvJitLinkComplete_12_4` on the first real
# GPU job. Do not re-add a module load line to "fix" a CUDA-looking error
# without checking whether the module itself is the cause first.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }

# google/gemma-2-2b (and every other Gemma checkpoint) is a GATED model on
# HuggingFace -- Google requires an authenticated, license-accepted account
# to download it, or transformers fails with GatedRepoError/401 even though
# the network path itself works. .env holds HF_TOKEN=... (never echo this
# file's contents or commit/zip it into the submission -- see EXECUTION_PLAN.md).
set -a; source "$REPO/.env"; set +a
GATE="$REPO/results/e0_gemma2b_layer8_v2/frozen_gate_diffmeans.json"
OUT="$REPO/results/e7_gemma2b_layer8"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"

echo "=== (a) named-bridge control ==="
python3 skc_run_e7_ablations.py named-bridge \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b --layer 8 --gate-path "$GATE" --seed 0 --out "$OUT"

echo "=== (j) random-direction negative control ==="
python3 skc_run_e7_ablations.py random-direction \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b --layer 8 --gate-path "$GATE" --coefficient 4.0 --seed 0 --out "$OUT"

echo "=== (k) label validation against an LLM judge ==="
# gemma-2-9b-it used as judge, same choice/caveat as the layer-16 script.
# --layer 8 is a no-op for this subcommand (cmd_judge_validation() never
# reads args.layer) but argparse still requires it on the shared subparser.
python3 skc_run_e7_ablations.py label-validation \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b --layer 8 --judge-model gemma-2-9b-it --n 100 --seed 0 --out "$OUT"

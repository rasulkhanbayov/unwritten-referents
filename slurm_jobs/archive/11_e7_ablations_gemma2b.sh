#!/bin/bash -l
#SBATCH -J skc_e7_gemma2b
#SBATCH -o e7_gemma2b_%j.out
#SBATCH -e e7_gemma2b_%j.err
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
# STEP 10: E7 ablations that are ready to run against gemma-2-2b once E0
# has passed. Runs the named-bridge control (a), random-direction negative
# control (j), and label validation (k) subcommands in sequence, since each
# is comparatively cheap and this keeps GPU idle time down between them.
# popularity (b) needs a --pageviews file (Wikipedia pageview data, not
# fetched anywhere in this bundle) and attention (h) is unimplemented -- see
# skc_run_e7_ablations.py's module docstring for both. Neither is in this
# script; run popularity separately once pageview data exists, and treat
# attention as a follow-up implementation task, not something to run yet.
#
# --layer FIXED to 16 (was 8): $GATE below points at
# e0_gemma2b_layer16/frozen_gate_diffmeans.json -- a direction fit at
# LAYER 16 -- but both subcommands were still passing --layer 8, a
# leftover from before this session's layer-8 -> layer-16 migration
# (Patchscopes needed layer 16; the whole E0/E1/E3/E5 chain moved there,
# but this script's own --layer flags were never updated to match). Scoring
# layer-8 activations against a layer-16-fit gate direction would silently
# produce a meaningless number rather than an error -- caught by re-reading
# this script's own arguments against $GATE's actual path before running it,
# not by any runtime check.

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
GATE="$REPO/results/e0_gemma2b_layer16/frozen_gate_diffmeans.json"
OUT="$REPO/results/e7_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"

echo "=== (a) named-bridge control ==="
python3 skc_run_e7_ablations.py named-bridge \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b --layer 16 --gate-path "$GATE" --seed 0 --out "$OUT"

echo "=== (j) random-direction negative control ==="
python3 skc_run_e7_ablations.py random-direction \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b --layer 16 --gate-path "$GATE" --coefficient 4.0 --seed 0 --out "$OUT"

echo "=== (k) label validation against an LLM judge ==="
# >>> TO RUN: pick a genuine judge model distinct from the model under test.
# gemma-2-9b-it is used here as a placeholder judge; state this choice (and
# that it differs from the model actually being evaluated for E0-E6) in
# Appendix B.
#
# --layer 16 added: cmd_judge_validation() never actually reads args.layer
# (confirmed: grep shows no reference to it in that function -- it's a
# required argument on the shared subparser, not something this specific
# subcommand needs), but argparse still requires it be passed. Missing on
# the previous run ("error: the following arguments are required:
# --layer"), which is why (k) never got to run at all. Value itself is a
# no-op for this subcommand; 16 matches every other script in this
# pipeline for consistency, not because this command uses it.
python3 skc_run_e7_ablations.py label-validation \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b --layer 16 --judge-model gemma-2-9b-it --n 100 --seed 0 --out "$OUT"

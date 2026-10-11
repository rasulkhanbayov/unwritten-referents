#!/bin/bash -l
#SBATCH -J skc_e6_llama8b_it
#SBATCH -o e6_llama8bit_%j.out
#SBATCH -e e6_llama8bit_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# GPU item 3, part 2 of 9: E6 chain-of-thought gate restoration on
# Llama 3.1 8B Instruct -- the SECOND model for this experiment,
# closing the paper's own disclosed gap ("Extend to the other models;
# abstention quality under this split is not yet reported" in
# Section~\ref{sec:res-cot}). ONLY SUBMIT once
# 75_e0_stopgate_llama8b_it_layer8.sh has PASSED with gate_open=true --
# update --layer/--gate-dir below to match if E0 needed a different
# layer than 8.
#
# Mirrors 10_e6_cot_gemma9b_it.sh's own structure exactly (same
# --cot-max-new-tokens=200, same primary-estimator).
#
# NOTE: this closes the MODEL-EXTENSION half of the paper's own TORUN
# ("extend to the other models"), but NOT the "abstention quality under
# this split" half -- that needs run_E6() in skc_experiments.py to
# compute and save a selective-risk/AURC-style curve split by
# bridge-correct vs.\ bridge-wrong, which it currently does not (only 3
# AUROC scalars are computed and saved -- confirmed directly by reading
# both skc_run_e6.py's main() and skc_experiments.py's run_E6(), see
# EXECUTION_PLAN.md's own investigation). That is a small CODE change
# plus a re-run of BOTH this model and gemma-2-9b-it, not just a script
# to submit -- out of scope for this job, flagged here so it is not
# silently forgotten once this script's own, narrower gap closes.

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
OUT="$REPO/results/e6_llama8b_it"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e6.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model llama-3.1-8b-it \
    --layer 8 \
    --gate-dir "$REPO/results/e0_llama8b_it_layer8" \
    --primary-estimator diffmeans \
    --cot-max-new-tokens 200 \
    --seed 0 \
    --out "$OUT"

echo "MANUALLY AUDIT $OUT/e6_cot_traces_sample.json before trusting the self-written-"
echo "bridge span heuristic (see skc_run_e6.py's module docstring)."

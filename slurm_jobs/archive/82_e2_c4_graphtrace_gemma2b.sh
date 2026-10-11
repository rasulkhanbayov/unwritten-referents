#!/bin/bash -l
#SBATCH -J skc_c4_graphs_gemma2b
#SBATCH -o c4_graphs_gemma2b_%j.out
#SBATCH -e c4_graphs_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 8:00:00

# GPU item 6 of 9: C4 graph trace, Gemma 2 2B -- builds REAL attribution
# graphs (circuit-tracer, Ameisen et al. 2025) on matched composed and
# named-bridge prompts. NEWLY BUILT this session
# (driver_scripts/skc_run_e2_c4_graphs.py); circuit-tracer was confirmed
# genuinely NOT installed anywhere in this environment before this job
# (checked directly: `python3 -c "import circuit_tracer"` ->
# ModuleNotFoundError, `pip show circuit-tracer` -> not found).
#
# *** REAL INCIDENT, FIXED HERE -- READ BEFORE RESUBMITTING: the FIRST
# version of this script ran `pip install -e circuit-tracer/` directly
# into the SHARED skc_iclr27 conda environment. This UPGRADED torch
# from the project's pinned 2.4.1+cu124 to 2.14.0+cu130 (and
# transformers, transformer-lens, safetensors, triton, multiple
# nvidia-cu13 packages), silently breaking the pinned environment EVERY
# OTHER script in this project depends on (confirmed directly: job
# 390472's own install log shows "Uninstalling torch-2.4.1+cu124" /
# "Successfully installed ... torch-2.14.0 ... transformers-4.57.3" in
# the shared env, and a job running concurrently in that same env,
# 390470, crashed on a missing .so file as a direct side effect). This
# was caught, and the shared environment was restored and thoroughly
# re-verified (torch==2.4.1+cu124, transformers==4.57.6, both offline
# suites clean, a real GPU op, and a real end-to-end model load+
# generate all confirmed working again) -- see EXECUTION_PLAN.md for
# the full incident and verification record.
#
# FIXED HERE: circuit-tracer is now installed into its OWN, ISOLATED
# conda environment (skc_circuit_tracer), created fresh by this script
# if it does not already exist, and NEVER installed into skc_iclr27
# again. This script activates skc_circuit_tracer instead of
# skc_iclr27 for its own Python invocation -- if skc_circuit_tracer
# needs any OTHER package this pipeline's own driver scripts import
# (numpy, etc.), install those into skc_circuit_tracer specifically,
# never by touching skc_iclr27's own package set again.
#
# *** REAL, DISCLOSED SCOPE LIMIT, NOT A BUG IN THIS SCRIPT: this job
# builds and saves RAW attribution graphs only. It does NOT identify
# "the refusal features" or compute a summed-edge-weight score --
# researched directly before writing this script: circuit-tracer's own
# released dictionaries carry NO pre-labeled refusal feature (feature
# labels only exist where a human has manually annotated a specific
# published graph, e.g. via Neuronpedia); identifying candidate refusal
# features is separate, original interpretability research the library
# does not provide. This is made WORSE, not just theoretically absent,
# by an already-confirmed finding from this same project: Gemma-2-2b's
# refusal rate on this exact composed-prompt template is EXACTLY 0.000
# at every steering coefficient tried (C1/C2/C3 all agree, job 383331)
# -- there is no observed refusal behavior on this model/template to
# correlate candidate features against in the first place. Control (m)
# (graph-trace faithfulness) needs graph trace itself to produce a real
# refusal-feature-edge-weight claim FIRST, so it remains open after this
# job too -- this job is necessary but not sufficient for (m).
#
# Builds graphs for a SUBSAMPLE of 20 queries (composed + named-bridge
# each, 40 graphs total), not the full test set -- confirmed via
# research that graph construction is Colab-feasible/interactive-scale
# per query, not multi-hour, but a full-dataset pass was not benchmarked
# and this is a reasonable first real batch to inspect before committing
# to a larger run.
#
# CONFIRMED FIRST-CLASS SUPPORT for gemma-2-2b specifically (researched
# directly, not assumed): "gemma" transcoder shortcut, GemmaScope/CLT
# dictionaries on HuggingFace (mntss/gemma-scope-transcoders,
# clt-gemma-2-2b-426k), works within ~15GB VRAM with CPU/disk offload
# per the library's own demo notebooks.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
# DELIBERATELY NOT activating skc_iclr27 here -- this job creates/uses its
# OWN isolated env (skc_circuit_tracer, below) instead, precisely to avoid
# repeating the environment-corruption incident this script's own comment
# above describes. DELIBERATELY NOT loading a cuda module either -- see
# other scripts' identical comment on torch's own bundled CUDA runtime.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a

# *** ISOLATED ENVIRONMENT, NOT skc_iclr27 -- this is the fix for the
# real incident described above. circuit-tracer's own dependency
# resolution wants a much newer torch/transformers than this project's
# pinned versions; installing it into skc_iclr27 directly corrupted
# that shared environment once already (job 390472) and broke a
# concurrently-running, unrelated job (390470) as a direct side effect.
# NEVER run `pip install -e circuit-tracer/` (or anything else new)
# into skc_iclr27 again -- create/reuse a SEPARATE conda env instead. ***
CT_ENV="skc_circuit_tracer"
if ! conda env list | grep -q "^${CT_ENV} "; then
    echo "Creating isolated conda env '${CT_ENV}' for circuit-tracer ..."
    conda create -y -n "$CT_ENV" python=3.11
fi
conda activate "$CT_ENV"

# Install circuit-tracer from source into the ISOLATED env only --
# idempotent (skips cleanly if already installed from a prior run of
# this same script). numpy<2.0 pinned explicitly BEFORE circuit-tracer's
# own install, matching this project's own requirements.txt reasoning
# (numpy>=2.0 conflicts with sae-lens-family dependency trees) -- this
# isolated env has its own independent numpy, but the same conflict
# class could recur here too if left unpinned.
pip install "numpy>=1.26,<2.0" 2>&1 | tail -5
if ! python3 -c "import circuit_tracer" 2>/dev/null; then
    echo "circuit-tracer not found in ${CT_ENV}, installing from source ..."
    CT_DIR="$REPO/circuit-tracer"
    if [ ! -d "$CT_DIR" ]; then
        git clone https://github.com/safety-research/circuit-tracer "$CT_DIR"
    fi
    pip install -e "$CT_DIR"
else
    echo "circuit-tracer already importable in ${CT_ENV}, skipping install."
fi

OUT="$REPO/results/c4_graphs_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e2_c4_graphs.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --n 20 \
    --offload cpu \
    --seed 0 \
    --out "$OUT"

echo ""
echo "Raw graphs + manifest are in $OUT. REMINDER: refusal-feature"
echo "identification and summed-edge-weight scoring are NOT done by this job --"
echo "see c4_graph_manifest.json's own NOT_DONE field. Control (m)"
echo "(graph-trace faithfulness) remains open until that analysis exists."

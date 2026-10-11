#!/bin/bash -l
#SBATCH -J skc_pilot_gemma2b
#SBATCH -o pilot_gemma2b_%j.out
#SBATCH -e pilot_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00

# STEP 2, and the FIRST JOB THAT SHOULD EVER TOUCH A GPU IN THIS PROJECT.
# skc_runbook.txt: "do this first, it takes an afternoon." 300 queries,
# Gemma-2-2B, one layer, diffmeans gate only (no SAE fitting -- that is
# exercised properly in E0). This is a same-afternoon aliveness check, not
# a paper result.
#
# PARTITION CHOICE: gpu-A100, not gpu-H200. Gemma-2-2B in bf16 needs a few
# GB; skc_runbook.txt says "one 40-80GB GPU is enough for 2B and 8-9B."
# It would ALSO fit on H200, but there is no reason to spend H200 allocation
# (max 4 concurrent jobs per skc_REDFLAGS/Panther guide) on a model this
# small when the larger runs later (E4's 70B/32B scale check, and any
# multi-GPU work) actually need the extra headroom. Conserve H200 for those.
#
# --account/--qos: CONFIRMED via `sacctmgr show assoc user=$USER
# format=account,qos` on panther-login -- a100_hbku/a100_hbku_qos is this
# account's real A100 grant.
#
# --queries: data/twohop_queries.json, built by 02_build_pilot_dataset.sh
# from Ferrando et al. (2025)'s released entity data (player->city->country
# chains, 3503 total) -- NOT a live Wikidata SPARQL pull. See that script's
# header for why the SPARQL path was abandoned (it ran without error but
# returned a semantically broken chain on unverified placeholder QIDs/PIDs).

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

OUT="$REPO/results/pilot"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_pilot.py \
    --queries "$REPO/data/twohop_queries.json" \
    --n 300 \
    --layer 8 \
    --seed 0 \
    --out "$OUT"

echo "Pilot verdict is in $OUT/pilot_result.json -- read it before submitting anything else."
echo "ALIVE -> proceed to 04_offline_e0 prep. DEAD -> back to instrument debugging."
echo "INVERTED -> STOP, flag back to the team, do not proceed to E0-E7 as currently scoped."

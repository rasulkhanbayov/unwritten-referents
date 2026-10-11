#!/bin/bash -l
#SBATCH -J skc_e5_gemma2b
#SBATCH -o e5_gemma2b_%j.out
#SBATCH -e e5_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# STEP 8: E5, compounding across hops (Proposition 4). Requires E0 passed
# and a THREE-HOP (k=3) query set.
#
# DATASET, RESOLVED: data/threehop_queries.json now exists (3961 real,
# verified chains: player --place_of_birth--> city --country--> country
# --continent--> continent). Ferrando et al.'s released data was checked
# directly and does NOT support a third hop (see
# skc_build_dataset_threehop.py's own docstring for the full check) --
# built instead via a NEW, verified Wikidata SPARQL path
# (skc_data.fetch_three_hop, skc_build_dataset_threehop.py). Not a
# placeholder/guessed pull: every QID/PID used was confirmed against the
# live endpoint, and the resulting chains are real and sensible (e.g.
# "Jason Richardson" -> Saginaw -> United States -> North America). See
# EXECUTION_PLAN.md for the full build/verification trail, and
# 23_build_threehop_dataset.sh if the dataset ever needs rebuilding.
#
# skc_run_e5.py filters on bridge resolution (Patchscopes) just like E1,
# via the same bridge_resolved_patchscope() -- implemented and validated
# (see EXECUTION_PLAN.md's "round 4" entry), restricted to K1=1 chains
# just like 05_e1_main_gemma2b.sh. This is E5's first real run.

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
OUT="$REPO/results/e5_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e5.py \
    --queries "$REPO/data/threehop_queries.json" \
    --model gemma-2-2b \
    --layer 16 \
    --gate-dir "$REPO/results/e0_gemma2b_layer16" \
    --primary-estimator diffmeans \
    --seed 0 \
    --out "$OUT"

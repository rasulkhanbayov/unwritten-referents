#!/bin/bash -l
#SBATCH -J skc_e3_gemma2b
#SBATCH -o e3_gemma2b_%j.out
#SBATCH -e e3_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 10:00:00

# STEP 5. E3 sweeps layer x position to find where the frozen gate reads K2
# above chance on the STATIC composed prompt. E3's own result stands on its
# own merits as an experiment ("where does the frozen gate read K2 above
# chance") and is NOT a prerequisite for E1/E5's Patchscopes filter anymore:
# an earlier design fed E3's peak (e3_full_map.json) into
# skc_run_e1.py's --e3-map to supply bridge_resolved_patchscope()'s decode
# site, but an exhaustive follow-up sweep found that NO static-prompt
# position, at any layer or offset, ever exposes the bridge -- the composed
# prompt's word order jumps straight from the entity mention to the FINAL
# answer once it has enough context, never stopping at the intermediate
# bridge. bridge_resolved_patchscope() was rebuilt (see EXECUTION_PLAN.md's
# "round 4" entry) to instead capture DURING the model's own hop1-only
# generation and locate the bridge per query in that generation's own text --
# validated to decode correctly on K1=1 queries. skc_run_e1.py and
# skc_run_e5.py no longer take --e3-map. E2's C2/C3 remain a separate,
# not-yet-wired consumer of the NEW during-generation site (see
# skc_run_e2.py's own updated comment), also no longer via --e3-map.
#
# --layers here must match a layer E0 was ALSO run at (skc_run_e3.py warns
# and skips any layer that does not match the loaded frozen gate's own
# fitting layer). If you want a genuine multi-layer sweep, you must first
# run skc_run_e0.py once per candidate layer and point --gate-dir at each
# in turn -- this script only sweeps POSITIONS at the single layer E0 used.
#
# --position-offsets CHANGED from "0 -1 -2 -3" (offsets from the PROMPT'S
# END) to "-1 0 1" (offsets from the SURFACE ENTITY'S OWN LAST TOKEN). The
# first E3 run (peak L8_pos-2) fed a bridge-resolution position into
# skc_run_e1.py's Patchscopes decode that turned out to land INSIDE the
# surface entity's own multi-token surname for most real queries -- e.g.
# "Domingos" tokenizes as "Dom"+"ingos", and offset -2 from the prompt's end
# landed on "Dom", one token before "ingos". Patchscopes decoded that
# position to the literal next word-piece of the SAME name ("ingos"),
# regardless of the true bridge entity, for every multi-token name tested --
# confirmed via slurm_jobs/13_diag_patchscope_layers.sh across 5 different
# layers, ruling out "wrong decode layer" as the explanation. The prompt-
# end-relative offset was the actual bug: entity names tokenize to a
# variable number of pieces, so "N tokens before the end" is not a
# consistent position across queries. Offsets from the entity's own span
# end are consistent (composed_2's fixed " is" tail after the entity is the
# same length regardless of the entity's name), and are what
# skc_run_e3.py now computes.

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
OUT="$REPO/results/e3_gemma2b_layer16"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e3.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b \
    --gate-dir "$REPO/results/e0_gemma2b_layer16" \
    --primary-estimator diffmeans \
    --layers 16 \
    --position-offsets -1 0 1 \
    --n-perm 1000 \
    --seed 0 \
    --out "$OUT"

echo "Peak (layer, position) is in $OUT/e3_full_map.json -- this is E3's own result"
echo "(where the frozen gate reads K2 above chance), no longer an input to"
echo "05_e1_main_gemma2b.sh or 09_e5_khop_gemma2b.sh (bridge_resolved_patchscope() now"
echo "sources its site from the model's own hop1 generation instead). Still relevant to"
echo "07_e2_causal_gemma2b.sh's C1 coefficient sweep and as its own reportable result."

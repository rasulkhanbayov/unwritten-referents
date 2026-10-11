#!/bin/bash -l
#SBATCH -J skc_diag_patchscope
#SBATCH -o diag_patchscope_%j.out
#SBATCH -e diag_patchscope_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 20000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 00:30:00

# DIAGNOSTIC ONLY, not part of the main E0-E7 pipeline.
#
# ROUND 1 (resolved): tested whether a LATER layer (12/16/20/24) decodes
# more cleanly than layer 8, using the OLD, BUGGY prompt-length-relative
# offsets. Every layer decoded to the same template noise; the noise was
# traced to the offset landing inside the surface entity's own multi-token
# name. Fixed to be entity-span-relative in skc_run_e3.py and
# skc_harness.py::bridge_resolved_patchscope(). Because the offset itself
# was wrong, ROUND 1 NEVER ACTUALLY TESTED "does a later layer decode
# cleanly at a well-defined position" -- that question was confounded by
# the same bug, at every layer.
#
# ROUND 2 (resolved): with the offset fixed, ran layer 8 only at offset -1
# (one token INTO the entity's own name) with unpatched/random-vector
# controls. Result: the injection MECHANISM is confirmed working (real
# patch differs from both controls, and is entity-sensitive) -- there is
# no code bug left in patchscope_decode() or its hook. But offset -1 (and
# E3's own AUROC sweep at offsets -1/0) still reads the SURFACE ENTITY'S
# OWN name, not the bridge; offset +1 (one token PAST the entity, the
# fixed " is" tail) scored BELOW CHANCE in E3's own sweep at layer 8.
#
# ROUND 3 (this run): now that the mechanism is confirmed working and the
# position convention is fixed, actually test whether a LATER layer decodes
# more cleanly AT OFFSET +1 (past the entity -- the position with the only
# principled claim to being "after the bridge would be resolved, right
# before generation") -- something round 1 never validly tested. If layer 8
# at offset +1 is genuinely below chance for K2 (per E3) AND does not
# decode cleanly here, a later layer might still carry the bridge
# resolution that an early layer has not yet computed.
#
# Cheap: 5 queries x 5 layers x (1 unpatched + 2 patched, unpatched only
# once per query) = ~55 generate() calls, still well under 30 minutes.

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

cd "$REPO/src/driver_scripts"
python3 skc_diag_patchscope_layers.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b \
    --offset 1 \
    --layers 8 12 16 20 24 \
    --n 5

echo
echo "Read the decodes per layer. If some layer produces a plausible entity-like"
echo "completion at offset +1 (past the entity, before generation) where layer 8"
echo "does not, redo skc_run_e0.py's gate fit at THAT layer and redo E3/E1 from"
echo "there -- the decode layer does not have to match the AUROC-peak layer used"
echo "for the gate itself."
echo
echo "If NO layer decodes cleanly at offset +1 either, the honest conclusion is:"
echo "this template/model does not expose a position where Patchscopes can"
echo "independently confirm bridge resolution, separate from surface-entity"
echo "identity. That is itself a real, reportable finding about the method's"
echo "limits -- not something to keep chasing indefinitely. See EXECUTION_PLAN.md"
echo "for the decision this would force (alternative verification, or a stated"
echo "limitation on E1's bridge-resolved filter)."

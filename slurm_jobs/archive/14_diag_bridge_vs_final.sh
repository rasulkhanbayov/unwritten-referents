#!/bin/bash -l
#SBATCH -J skc_diag_bridge_vs_final
#SBATCH -o diag_bridge_vs_final_%j.out
#SBATCH -e diag_bridge_vs_final_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 20000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 01:00:00

# DIAGNOSTIC ONLY, not part of the main E0-E7 pipeline.
#
# ROUND 1 (resolved, 10 queries x 1 layer x 3 offsets x 2 forms = 60 decode
# calls): the first E1 run at layer 16 (E3's peak, L16_pos1) resolved
# 0/3503 queries against the bridge-resolved filter, but its own saved
# patchscope_sample.json showed clean, mostly-correct decodes of the FINAL
# composed answer (country, e.g. "Jake Bornheimer" -> " United States")
# rather than the BRIDGE (city, "New Brunswick") the filter checks against.
# Ran this script at layer 16, offsets -1/0/+1, on BOTH the composed
# prompt and the unambiguous hop1-only prompt (single-hop template,
# relation=r1 -- no composition to race ahead of). Result: the city NEVER
# appeared at any tested offset, on EITHER prompt form. offset -1/0 gave
# surname-fragment noise (the same surface-entity-name confound found
# earlier this session); offset +1 gave the COUNTRY even on the hop1-ONLY
# prompt (e.g. "Mike Butler" -> "the United States of America" from a
# prompt that only asked about place of birth, no country relation
# anywhere in it). So this is not a composed-prompt artifact: at layer 16,
# offset +1, the model appears to already represent the country-level
# answer even when only asked the city-level question -- or layer 16 is
# simply past wherever the city was ever locally decodable, having already
# resolved into the country.
#
# ROUND 2 (this run): widen BOTH axes at once per the user's choice --
# more offsets (further before the entity's own name, to skip the surname
# zone entirely, and further after +1, in case the city surfaces briefly
# before being overwritten) AND earlier layers (8/10/12/14, below E3's
# AUROC peak, on the theory that if the city is represented at all before
# being resolved into the country, an earlier layer closer to where hop1's
# relation lookup happens is the more principled place to find it).
#
# 5 layers x 8 offsets x 2 forms x 8 queries = 640 decode calls. Trimmed
# --n from 10 to 8 and widened the time limit to 1 hour to keep this
# comfortably inside budget; each call is a single short generate()
# (max_new_tokens=8) plus one forward pass, cheap individually but this
# adds up over 640 of them.

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
python3 skc_diag_bridge_vs_final.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b \
    --layers 8 10 12 14 16 \
    --offsets -3 -2 -1 0 1 2 3 4 \
    --n 8

echo
echo "Read the decodes per query and layer. If HOP1-ONLY names the city"
echo "cleanly at ANY (layer, offset) combination, that tells you the bridge"
echo "IS representable via Patchscopes -- the fix is to change what E1"
echo "scores against or where/at-what-layer it looks, not to give up on"
echo "Patchscopes for this filter."
echo
echo "If HOP1-ONLY still never names the city at any layer/offset tried"
echo "here, the honest conclusion is that this model, at least for this"
echo "relation and these positions, does not expose a place where"
echo "Patchscopes can independently confirm bridge resolution BEFORE the"
echo "final answer is already computed -- a real, reportable limitation on"
echo "E1's bridge-resolved filter as specified, not something to keep"
echo "chasing indefinitely. See EXECUTION_PLAN.md for the decision this"
echo "would force."

#!/bin/bash -l
#SBATCH -J skc_diag_bridge_during_gen
#SBATCH -o diag_bridge_during_gen_%j.out
#SBATCH -e diag_bridge_during_gen_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 20000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 00:30:00

# DIAGNOSTIC ONLY, not part of the main E0-E7 pipeline.
#
# CONTEXT: an exhaustive prior sweep (14_diag_bridge_vs_final.sh, 5 layers x
# 8 offsets x hop1-only vs composed prompts) found the bridge entity (a
# player's birth CITY) never decodable via Patchscopes at ANY static prompt
# position -- every position either read the surface entity's own name or
# had already jumped to the composed query's FINAL answer (the country),
# even on the hop1-only prompt that never mentions country anywhere. That
# entire sweep used capture_activations(), which only ever sees a single
# forward pass over a prompt AS GIVEN -- it cannot address a position that
# exists only once the model starts writing its own answer.
#
# THIS RUN tests the one site that sweep structurally could not reach:
# capture the residual stream WHILE the model autoregressively generates
# its own answer to the hop1-only prompt, at EVERY generated-token position
# (not just the prompt's last token), and Patchscopes-decode each one. Uses
# skc_harness.capture_during_generation(), added specifically for this --
# skc_harness.py never previously combined generation with hook-based
# capture (capture_activations() and generate_greedy() were separate,
# single-purpose functions).
#
# Cheap: 8 queries x 1 layer x (1 prefill + 6 generated positions) = 56
# Patchscopes decode calls, well under 30 minutes.

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
python3 skc_diag_bridge_during_gen.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b \
    --layers 16 \
    --max-new-tokens 6 \
    --n 8

echo
echo "Read the decodes per query. Compare 'model's own greedy completion'"
echo "against 'true bridge (city)' to confirm the model DOES know the city"
echo "when actually asked (K1-style check on this same sample). Then check"
echo "whether ANY generated-token-position decode names that city -- if the"
echo "PROMPT'S LAST TOKEN (prefill) site still fails but a GENERATED TOKEN"
echo "site succeeds, that pins down a genuinely new capture site for E1's"
echo "bridge-resolved filter to use. If no site at any position decodes the"
echo "city here either, that closes out this investigation: report 0%"
echo "bridge-resolved via Patchscopes as a real finding (see"
echo "EXECUTION_PLAN.md's round-2 entry for the three options this would"
echo "leave), since capturing during generation was believed to be the"
echo "one remaining site the prior sweep could not test."

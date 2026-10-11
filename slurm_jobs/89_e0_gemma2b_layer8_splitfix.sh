#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b_l8_splitfix
#SBATCH -o e0_gemma2b_l8_splitfix_%j.out
#SBATCH -e e0_gemma2b_l8_splitfix_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# VERSION3 era. Re-run of job 26 (e0_gemma2b_layer8_v2, the source of the
# frozen gate every headline BGR/availability number in the old paper --
# and every number VERSION3's own manuscript reports for Gemma 2 2B --
# depends on), needed because a confirmed, real leakage bug was found and
# fixed in split_entities() (driver_scripts/skc_run_e0.py) while grounding
# VERSION3's own Phase 0 export-layer design in the real pipeline code.
#
# THE BUG (measured directly against the real data, not assumed): the old
# split_entities() call partitioned fit/held-out ONLY over
# data/onehop_queries.json's own head entities, with ZERO cross-check
# against data/twohop_queries.json's own bridge/target entities -- the
# exact entities E1 reads the frozen gate direction back at, at their
# latent position. Direct measurement: 402 of 847 (47.5%) of twohop's own
# bridge entities were landing in the OLD fit set. For each of those 402,
# the gate was fit using that entity's own known/unknown label, then
# evaluated by reading the SAME direction at that SAME entity's own
# bridge position -- precisely the leakage vector VERSION3's own
# REDFLAGS.md item C7 names as "the one leakage vector that would explain
# the headline finding," now confirmed real rather than hypothetical.
#
# THE FIX: split_entities() now takes an exclude_from_fit set. main() now
# loads --named-bridge-queries (already a required argument, reused here
# for a second purpose) BEFORE the split and collects every head/bridge/
# target entity across it, so none of them can ever land in the fit set.
# Verified directly against the real files with a pure-Python dry run (no
# GPU): post-fix, ZERO overlap between twohop's own heads/bridges/targets
# and the corrected fit set (down from 402/847 bridge entities before the
# fix). Fit set size dropped from 17,031 to 14,895 entities (2,136 fewer,
# correctly reserved for evaluation only, not a bug).
#
# EVERYTHING ELSE about this invocation is IDENTICAL to job 26 (same
# --queries/--named-bridge-queries/--n/--layer/--fit-frac/--seed/
# --skip-sae) -- only the internal split logic changed. If E0 still
# passes its own stop-gate (>=2/3 estimators agree) under the corrected
# split, that is real evidence the original result was not an artifact of
# the leak; if it now fails, that is itself an important, disclosable
# finding, not a bug in this job.
#
# Output to a NEW directory (e0_gemma2b_layer8_splitfix) so the
# leaky-split result (e0_gemma2b_layer8_v2) stays on disk as a labeled,
# comparable historical record, not silently overwritten.

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
OUT="$REPO/results/e0_gemma2b_layer8_splitfix"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layer 8 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 (gemma-2-2b, layer 8, SPLIT-LEAKAGE FIX applied) PASSED. frozen_gate_*.json"
    echo "files are in $OUT. Compare diffmeans/probe/agreement against"
    echo "e0_gemma2b_layer8_v2's own E0.json to see whether the corrected split changed"
    echo "the positive control's own outcome. Re-run E1 against THIS gate directory next,"
    echo "not the old e0_gemma2b_layer8_v2 one -- every downstream number needs the"
    echo "clean gate before it can be trusted."
else
    echo "E0 (gemma-2-2b, layer 8, SPLIT-LEAKAGE FIX applied) FAILED (exit $exit_code)." >&2
    echo "Do NOT proceed to E1 against this gate. If the corrected split is why this" >&2
    echo "now fails where the leaky one passed, that is itself an important, disclosable" >&2
    echo "finding about the original result -- report it, do not quietly revert the fix." >&2
fi
exit "$exit_code"

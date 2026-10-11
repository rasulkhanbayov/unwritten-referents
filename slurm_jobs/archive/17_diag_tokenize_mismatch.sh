#!/bin/bash -l
#SBATCH -J skc_diag_tokenize_mismatch
#SBATCH -o diag_tokenize_mismatch_%j.out
#SBATCH -e diag_tokenize_mismatch_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 20000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 00:30:00

# DIAGNOSTIC ONLY, not part of the main E0-E7 pipeline.
#
# WHY: two consecutive fixes to skc_harness.find_entity_span_in_generated_tokens()
# (formerly find_entity_span_in_generated_text) -- first removing a spurious
# BOS token from a re-tokenization, then abandoning re-tokenization
# entirely in favor of searching generated_tokens directly for a candidate
# answer's OWN standalone tokenization -- have each changed the CODE but
# produced BYTE-IDENTICAL patchscope_sample.json output on the real E1
# run (jobs 380606 and 380609). That is suspicious enough to stop guessing
# at a third code fix blind and instead print exactly what the real
# tokenizer computes at each step of the search, for a handful of real
# K1=1 queries -- no transformers/tokenizers package is available outside
# this Slurm/conda environment to inspect this offline.
#
# Cheap: labels a 200-query pool for K1 (same cost as prior K1-filtered
# diagnostics), then just tokenizes a few short strings per K1=1 query --
# no Patchscopes decode calls at all this time.

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
python3 skc_diag_tokenize_mismatch.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b \
    --pool-size 200 \
    --n 8 \
    --max-new-tokens 12

echo
echo "Read each query's probe_tokens vs generated_tokens comparison. If"
echo "match_at is None for both prefixed=True and prefixed=False, that IS"
echo "the bug: the standalone tokenization of the bridge answer does not"
echo "reproduce how it is tokenized when embedded in context -- decide a"
echo "different search strategy (e.g. use offset_mapping from re-tokenizing"
echo "generated_text with add_special_tokens=False and accept a possible"
echo "few-token slop, or match on DECODED substrings of generated_tokens"
echo "one at a time rather than a full probe match) based on what the"
echo "actual mismatch looks like here."
echo
echo "Also check 'full re-tokenization == generated_tokens': if this is"
echo "False even for the FULL generated text, that confirms decode-then-"
echo "re-encode is unreliable on this tokenizer in general, not just for"
echo "short probes -- useful context for picking the next approach."

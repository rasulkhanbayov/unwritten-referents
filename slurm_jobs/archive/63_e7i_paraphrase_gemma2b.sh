#!/bin/bash -l
#SBATCH -J skc_e7i_para_gemma2b
#SBATCH -o e7i_para_gemma2b_%j.out
#SBATCH -e e7i_para_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 4:00:00

# Phase 3, item 4: (i) Template and paraphrase robustness, Gemma 2 2B,
# layer 8. Paper's own words (skc_main.tex, Appendix~\ref{app:controls}):
# "Template and paraphrase robustness, following the formatting
# sensitivity reported by Tamoyan et al. (2025)." No further protocol
# detail exists anywhere in the repo/manuscript beyond this sentence --
# the exact design below is an EXPLICIT CHOICE confirmed directly with
# the user.
#
# PROTOCOL: 2 genuine paraphrases of TEMPLATES["composed_2"]
# (skc_data.py), built to satisfy the SAME design constraint the paper's
# own template follows (Appendix~\ref{app:prompts}): entity placed LATE,
# prompt ends on a FIXED token ("is") -- following Tamoyan et al.'s own
# finding that this form avoids spurious correlations from a variable-
# token ending. This constraint genuinely LIMITS how much a paraphrase
# can restructure the sentence -- several candidates were drafted and
# REJECTED (e.g. "Regarding the place of birth of the player {e1}, the
# country is") because moving the relation-2 word to just before "is"
# pulls e1's own mention EARLIER, violating "entity late". The two kept
# (PARAPHRASE_TEMPLATES in skc_run_e7_ablations.py) are lexical/
# syntactic variants within the SAME clause structure.
#
# Reports AUROC(gate->K2) for the ORIGINAL template and BOTH
# paraphrases -- stability (a small spread) is evidence the paper's own
# result is not an artifact of this one specific phrasing; a large
# spread would mean the opposite.
#
# REUSES the query's own EXISTING K2 labels (computed once, from the
# ISOLATED hop2 prompt) rather than relabeling under each paraphrase --
# confirmed directly from skc_data.compute_labels()'s own code that K_j
# is computed ENTIRELY from generations["hop{j}"], never from the
# composed prompt's own phrasing, so this is the exact same quantity by
# construction, not an approximation.
#
# Offline-verified before this run: all 239 (query, paraphrase) pairs in
# the real test set render correctly (end on "is", entity-late
# satisfied) -- checked directly, not assumed. All 239 test-set queries
# have MULTI-WORD e1 (player full names) -- confirmed each locates
# correctly as one intact substring under both paraphrase templates (no
# repeat of the exact multi-word-entity-splitting bug found in control
# (c)'s own first attempt, job 389064 -- this template substitutes e1 as
# one whole string via .format(), which cannot split it the way that
# control's separate word-count-based filler-insertion logic did).
#
# COST: mirrors control (a)/(c)'s own cheap cost profile -- one
# generation (K2 labeling) + THREE capture passes (original + 2
# paraphrases) per query, no sampling, same 239-query test set.
# -t 4:00:00 is generous, not tight.

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
OUT="$REPO/results/e7i_paraphrase_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e7_ablations.py paraphrase \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-path "$REPO/results/e0_gemma2b_layer8_v2/frozen_gate_diffmeans.json" \
    --seed 0 \
    --out "$OUT"

echo "Result (AUROC per template, max-minus-min spread) is in"
echo "$OUT/e7i_paraphrase.json"

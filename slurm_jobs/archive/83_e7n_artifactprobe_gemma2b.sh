#!/bin/bash
# GPU item 7 of 9 (actually CPU-only, no GPU needed): (n) Artifact probe
# (TxTemb, Hussain & Kantarcioglu 2026, arXiv:2605.17028), Gemma 2 2B --
# NEWLY IMPLEMENTED this session (cmd_artifact_probe() in
# skc_run_e7_ablations.py did not exist before now).
#
# NOT A SLURM SCRIPT -- deliberately a plain bash script, no #SBATCH
# directives. TxTemb is TF-IDF cosine similarity, verified directly
# against the real paper: no trained classifier, no neural embedding
# model, NO model internals at all -- there is nothing here that needs
# a GPU or even a loaded language model. Run this directly on a login
# node (or any node with the skc_iclr27 conda env), not via sbatch.
#
# ADAPTATION, an EXPLICIT CHOICE confirmed directly with the user: the
# original paper compares a GENERATED response against a reference
# answer; that would trivially correlate with Y by construction for
# SKC-Bench (a correct generation IS textually similar to the target by
# definition), so it would not test for a benchmark-construction
# artifact the way it does in the original's own setting. Here, TxTemb
# compares the COMPOSED PROMPT TEXT ITSELF against the target answer
# text -- the genuine concern control (n) exists to check: does the
# prompt leak the answer via surface overlap, since the bridge entity
# is never written in a composed query.
#
# REAL RESULT ALREADY OBSERVED on this exact data (offline-verified
# before this script was written, not guessed): AUROC=0.159, notably
# BELOW chance rather AT it. Investigated directly rather than reported
# blind: 211/239 items have EXACTLY ZERO vocabulary overlap between
# prompt and target (the expected, "no leak" case), and the AUROC is
# driven entirely by the remaining 28 nonzero-overlap items, ALL of
# which happen to fall in the Y=0 (incorrect) group -- a real, but
# NARROWER finding than a benchmark-wide answer leak (which would need
# a HIGH, not low, AUROC driven by a large fraction of items). The
# script's own output JSON now includes frac_zero_overlap and the
# per-class mean score specifically so this nuance is never silently
# lost behind one headline AUROC number in any future run.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
OUT="$REPO/results/e7n_artifactprobe_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e7_ablations.py artifact-probe \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --labels "$REPO/results/e1_gemma2b_layer8/e1_labels_filtered.json" \
    --seed 0 \
    --out "$OUT"

echo "Result (AUROC, frac_zero_overlap, per-class mean score) is in"
echo "$OUT/e7n_artifact_probe.json"

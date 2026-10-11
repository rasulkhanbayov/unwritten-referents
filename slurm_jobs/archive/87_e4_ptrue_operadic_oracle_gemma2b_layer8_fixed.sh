#!/bin/bash -l
#SBATCH -J skc_e4_ptrue_oc_fix_gemma2b
#SBATCH -o e4_ptrue_oc_fix_gemma2b_l8_%j.out
#SBATCH -e e4_ptrue_oc_fix_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00

# RERUN of job 86 (392222) -- that run completed cleanly (no errors) but
# its output caught a REAL bug before anything was written into the
# paper: "oracle_M", "self_ask", and "operadic_consistency" all reported
# BYTE-IDENTICAL auroc=0.5/aurc/risk_at_coverage. Investigated rather
# than trusted -- self_ask/operadic_consistency's match turned out to be
# a real, already-known finding (this model has exactly 0.000 refusal
# rate on this template, documented throughout skc_main.tex, so both
# selectors' scores are genuinely constant, not buggy -- confirmed by
# checking self_ask's own numbers were IDENTICAL to the earlier
# e4_gemma2b_layer8_se/_rho runs' self_ask, before and after the
# few-shot fix). But oracle_M's match was a REAL bug: this paper's own
# Section~\ref{sec:res-blindspot} already reports P(M=0)=0.439 for this
# EXACT query set, so `m` cannot legitimately be constant -- and a
# mathematical check confirmed exactly auroc=0.5 (to 16 significant
# figures) is only possible from a genuinely constant score array, not
# a real-but-uncorrelated one (which gives ~0.49, not exactly 0.5).
#
# ROOT CAUSE, FOUND AND FIXED: main()'s own labeling call
# (`label_queries_batched(model, tok, queries)`, no hop_require_prefix
# override) was the ONE E4 labeling site in this file missing the
# `hop_require_prefix={2: False}` override that skc_run_e0.py, e1.py,
# e2.py, e3.py, e5.py, e6.py, e7_ablations.py, and skc_run_pilot.py all
# already carry (with their own comments explaining why: real Gemma-2-2b
# hop2 generations reliably name the correct answer inside a longer
# descriptive sentence rather than starting with it verbatim, so the
# strict require_prefix=True default undercounts K2, and therefore M
# --- which for these k=2 queries IS K2 --- almost to zero). This call
# had always been missing that override (it predates this session's own
# changes), but it only became VISIBLE as a bug once Oracle observing M
# started reading `m` from it -- nothing upstream of this session ever
# consumed `m` from this exact call, so the bug had no way to surface
# before now. Confirmed Y itself is UNAFFECTED (compute_labels() hard-
# codes Y's own require_prefix with no override available at all, per
# its own docstring), so every previously-reported AUROC that depends
# only on y (surface_gate, BGR-min/lr, p_true, semantic_entropy) is
# unaffected and does not need to be re-verified because of this fix.
#
# Everything else about this run is IDENTICAL to job 86 -- see that
# job's own script (86_e4_ptrue_operadic_oracle_gemma2b_layer8.sh) for
# the full account of the P(True)/Operadic-consistency/Self-Ask fixes
# themselves, unchanged here. Re-verified skc_pipeline_test.py (51/51)
# after the hop_require_prefix fix, same as after every other change.

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
OUT="$REPO/results/e4_gemma2b_layer8_ptrue_oc_fixed"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-p-true \
    --run-operadic-consistency \
    --run-self-ask \
    --run-semantic-entropy \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "P(True), Operadic consistency, and (this time, correctly non-degenerate)"
echo "Oracle observing M scores are in $OUT/E4.json. BEFORE trusting oracle_M,"
echo "confirm its own auroc/aurc/risk_at_coverage are NOT byte-identical to"
echo "self_ask/operadic_consistency again -- if they are, the bug is not"
echo "fixed and this needs another investigation pass, not a copy-paste of"
echo "these numbers into the paper."

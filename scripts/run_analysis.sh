#!/usr/bin/env bash
# CPU-only analysis chain: turns the saved outputs of the GPU jobs into the paper's
# numbers. No model is loaded. Run after the GPU jobs (slurm_jobs/) finish, or after
# scripts/download_from_hf.sh restores their saved activations.
#
# Steps (exact commands used to produce the current results):
#   1. export seed-0 per-item records for VERSION3 from the E1 runs (jobs 106/107)
#   2. merge seed 0 + seeds 1-4 (jobs 117-131) + fit split (jobs 137/138) into one file
#   3. VERSION3 reconcile.py  -> VERSION3/code/results/reconcile/
#   4. VERSION3 ablations.py  -> VERSION3/code/results/ablations/      (LEGACY, see below)
#   5. corrected C4 rebuild   -> results/c4_latent_rebuild/            (CURRENT)
#
# !! Steps 3-4 read the `g_B` field of VERSION3/data/records, which was captured by
# !! skc_harness.capture_for_queries()'s hop-2 prompt that NAMES THE GOLD BRIDGE -- a
# !! known bug (docs/SESSION_HISTORY.md, "named-gold-bridge g_B"). reconcile.py's
# !! g_S-based numbers are unaffected; ablations.py's bridge-only/BGR numbers are not
# !! the latent read the paper describes. Step 5 is the corrected version and is what
# !! Section 4.3 / Tables tab:bgr, tab:bgr-full now report.
#
# Usage: bash scripts/run_analysis.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DS="$REPO/src/driver_scripts"
REC="$REPO/VERSION3/data/records"

echo "== 1. seed-0 record export (posthoc, from saved E1 activations) =="
python3 "$DS/skc_export_records_posthoc.py" \
  --e1-dir "$REPO/results/e1_gemma9b_layer8_skipbridge_splitfix" \
  --gate-dir "$REPO/results/e0_gemma9b_layer8_splitfix" \
  --estimator diffmeans --split eval --seed 0 \
  --out "$REC/gemma-2-9b/f1_twohop.jsonl"
python3 "$DS/skc_export_records_posthoc.py" \
  --e1-dir "$REPO/results/e1_llama70b_layer25_skipbridge_splitfix" \
  --gate-dir "$REPO/results/e0_llama70b_layer25_splitfix" \
  --estimator diffmeans --split eval --seed 0 \
  --out "$REC/llama-3.1-70b/f1_twohop.jsonl"

echo "== 2. merge seeds 0-4 + fit split =="
for m in gemma-2-9b llama-3.1-70b; do
  python3 "$DS/skc_merge_seed_records.py" --model "$m" --family f1_twohop --overwrite
done

echo "== 3. VERSION3 reconcile.py =="
(cd "$REPO/VERSION3/code" && python3 scripts/reconcile.py --config configs/experiments.yaml)

echo "== 4. VERSION3 ablations.py (LEGACY: uses the named-gold-bridge g_B) =="
# --mde 0.038 is Gemma 2 9B's minimum detectable AUROC difference from step 3; it was
# applied to both models when these results were produced, and is kept for reproducibility.
(cd "$REPO/VERSION3/code" && python3 scripts/ablations.py --config configs/experiments.yaml --mde 0.038)

echo "== 5. corrected C4 rebuild (latent g_B from skc_run_e3.py, fit-split standardizer) =="
python3 "$DS/skc_rebuild_c4_latent.py"

echo "== done =="

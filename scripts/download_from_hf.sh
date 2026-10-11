#!/usr/bin/env bash
# Restore the large artifacts uploaded by scripts/upload_to_hf.sh into this repo,
# at their original paths (results/..., logs/slurm/...).
#
# You only need this to re-run the CPU analysis steps without re-running GPU jobs
# (e.g. src/driver_scripts/skc_rebuild_c4_latent.py and the posthoc record export
# read the saved .npy activations). Every GPU job regenerates its own artifacts.
#
# Usage:
#   export HF_DATASET_REPO=<user-or-org>/<dataset-name>
#   bash scripts/download_from_hf.sh                 # core (~7.6 GB)
#   bash scripts/download_from_hf.sh --all           # core + archived C4 graphs (~17 GB)
#   bash scripts/download_from_hf.sh --only "results/e3_*latent*/*"   # one subset
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${HF_DATASET_REPO:?set HF_DATASET_REPO=<user-or-org>/<dataset-name>}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

case "${1:-}" in
  --all)  INCLUDE=("results/*.npy" "results/*_raw_generations.json" "logs/slurm/*" "results/*.pt") ;;
  --only) INCLUDE=("${2:?--only needs a glob pattern}") ;;
  *)      INCLUDE=("results/*.npy" "results/*_raw_generations.json" "logs/slurm/*") ;;
esac

hf download "$HF_DATASET_REPO" --repo-type dataset --local-dir "$REPO" --include "${INCLUDE[@]}"

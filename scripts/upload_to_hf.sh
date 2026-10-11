#!/usr/bin/env bash
# Upload the large, gitignored artifacts to a Hugging Face DATASET repo.
# Paths are preserved, so scripts/download_from_hf.sh restores them in place.
#
# Groups (see HF_MANIFEST.md for per-directory sizes):
#   core    ~7.6 GB  results/**/*.npy (captured activations), results/**/*_raw_generations.json
#                    (raw model generations behind every K1/K2/Y label), logs/slurm/* (job logs)
#   graphs  ~9.5 GB  results/**/*.pt -- attribution graphs from the ARCHIVED Gemma 2 2B C4
#                    experiment (results/c4_graphs_gemma2b). No current paper number uses them.
#
# The include patterns below were checked against the real tree with
# huggingface_hub's own filter_repo_objects(): they match exactly the files .gitignore
# excludes from git, and nothing outside results/ and logs/.
#
# Usage:
#   export HF_DATASET_REPO=<user-or-org>/<dataset-name>
#   hf auth login            # or put HF_TOKEN in .env
#   bash scripts/upload_to_hf.sh            # core only
#   bash scripts/upload_to_hf.sh --all      # core + graphs
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${HF_DATASET_REPO:?set HF_DATASET_REPO=<user-or-org>/<dataset-name>}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

INCLUDE=("results/*.npy" "results/*_raw_generations.json" "logs/slurm/*")
if [ "${1:-}" = "--all" ]; then
  INCLUDE+=("results/*.pt")
fi

echo "Uploading to dataset repo $HF_DATASET_REPO (private if it does not exist yet):"
printf '   %s\n' "${INCLUDE[@]}"
# upload-large-folder is resumable: re-run the same command after any interruption.
# It writes a .cache/huggingface/ folder in $REPO to track progress (gitignored).
hf upload-large-folder "$HF_DATASET_REPO" "$REPO" \
  --repo-type dataset --private \
  --include "${INCLUDE[@]}" \
  --exclude "archive/*" ".cache/*"

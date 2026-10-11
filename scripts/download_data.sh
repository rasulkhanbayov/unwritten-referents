#!/usr/bin/env bash
# Recreate the public third-party inputs this repo does not ship.
#
#   data/ferrando_entities/{player,city,movie,song}.json  (+ LICENSE.txt, README.md)
#       from Ferrando et al. (2025), github.com/javiferran/sae_entities, Apache-2.0,
#       pinned to the exact commit these files were taken from. MD5s below were
#       measured on the files actually used to build data/twohop_queries.json.
#
# Optional (--with-circuit-tracer): install circuit-tracer at the pinned commit.
#   Only needed by src/driver_scripts/skc_run_e2_c4_graphs.py (archived C4
#   attribution-graph experiment on Gemma 2 2B). requirements-freeze.txt and
#   environment.yml already pin it, so a full env install covers this too.
#
# Not downloaded here (fetched on demand by the pipeline itself):
#   - model weights: from the Hugging Face Hub at run time (needs HF_TOKEN in .env;
#     Gemma and Llama are gated -- accept each licence on huggingface.co first)
#   - SAE dictionaries (Gemma Scope / Llama Scope): via sae_lens at run time
#   - Wikidata three-hop chains: already shipped as data/threehop_queries.json;
#     to re-pull, see slurm_jobs/23_build_threehop_dataset.sh (live SPARQL, so
#     results can drift from the 2026-08-26 snapshot)
#   - Wikipedia pageviews: shipped as data/pageviews_twohop.json; re-pull with
#     src/driver_scripts/skc_fetch_pageviews.py
#
# Usage:  bash scripts/download_data.sh [--with-circuit-tracer]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SAE_ENTITIES_URL="https://github.com/javiferran/sae_entities.git"
SAE_ENTITIES_COMMIT="5ec36961623e093f4397d0120dd0d482b92343a2"
CIRCUIT_TRACER_URL="https://github.com/safety-research/circuit-tracer"
CIRCUIT_TRACER_COMMIT="6018ed8d35e40f2c50062822e8dde422b8e52e2d"

declare -A MD5=(
  [player]=45187513b2011c6e69287a8f05c906d9
  [city]=3b78a21d6a9bf53b2831504407dc89e3
  [movie]=5a34b1d02b96334359c5588efdd3fdb3
  [song]=3de25bb0e221c96973c2a1af95aa555e
)

DEST="$REPO/data/ferrando_entities"
mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== cloning sae_entities @ ${SAE_ENTITIES_COMMIT:0:12} =="
git clone --quiet "$SAE_ENTITIES_URL" "$TMP/sae_entities"
git -C "$TMP/sae_entities" checkout --quiet "$SAE_ENTITIES_COMMIT"

for t in player city movie song; do
  cp "$TMP/sae_entities/dataset/processed/entities/$t.json" "$DEST/$t.json"
  got="$(md5sum "$DEST/$t.json" | cut -d' ' -f1)"
  if [ "$got" != "${MD5[$t]}" ]; then
    echo "MD5 MISMATCH for $t.json: got $got, expected ${MD5[$t]}" >&2
    exit 1
  fi
  echo "   $t.json  OK"
done
cp "$TMP/sae_entities/LICENSE.txt" "$DEST/LICENSE.txt"
cp "$TMP/sae_entities/README.md" "$DEST/README.md"
echo "== data/ferrando_entities ready (byte-identical to the files used in the paper) =="

if [ "${1:-}" = "--with-circuit-tracer" ]; then
  echo "== installing circuit-tracer @ ${CIRCUIT_TRACER_COMMIT:0:12} =="
  pip install "git+${CIRCUIT_TRACER_URL}@${CIRCUIT_TRACER_COMMIT}"
fi

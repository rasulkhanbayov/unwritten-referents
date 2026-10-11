#!/bin/bash -l
#SBATCH -J skc_build_dataset
#SBATCH -o build_dataset_%j.out
#SBATCH -e build_dataset_%j.err
#SBATCH -p cpu-all
#SBATCH -c 2
#SBATCH --mem 4000MB
#SBATCH -t 00:10:00

# STEP 1: build the two-hop query set. CPU-only, no network needed (this is
# now a pure offline JSON transform, not a live Wikidata SPARQL call).
#
# HISTORY: the original version of this script called
# skc_build_dataset.py --source wikidata with placeholder QIDs/PIDs
# (Q5/P106/P22). That call executed without error but returned garbage: an
# unresolved entity LABEL (a raw QID string instead of a name) and a
# semantically broken chain (person--occupation-->Muse--father-->Zeus).
# "Ran without crashing" was not "worked" -- confirmed by actually reading
# the sample row it printed.
#
# FIX: this now uses skc_build_dataset_from_ferrando.py against the ACTUAL
# entity data Ferrando et al. (2025) released
# (github.com/javiferran/sae_entities, Apache 2.0) -- already downloaded
# into data/ferrando_entities/{player,city}.json and confirmed against
# Table 3 of their paper (counts match: 7487 players, 7904 cities). It
# builds real two-hop chains via the one bridge structure that exists in
# this released data: player.place_birth values that are themselves full
# city entities (3503 of 7487 players) -- e.g. "The country of the place
# of birth of the player Jake Bornheimer is" (bridge: New Brunswick, never
# written; target: United States of America). See that script's module
# docstring for why only player->city->country is buildable from this
# release (movie/song attribute values are not themselves top-level
# entities in the released data).
#
# This produces data/twohop_queries.json directly -- the same file every
# other Slurm script in this directory already points --queries at. No
# separate "pilot_queries.json" is needed; 03_pilot_gemma2b.sh subsamples
# 300 of these itself via --n 300.

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
OUT="$REPO/data"
mkdir -p "$OUT"
cd "$REPO/src/driver_scripts"

python3 skc_build_dataset_from_ferrando.py \
    --player-json "$REPO/data/ferrando_entities/player.json" \
    --city-json "$REPO/data/ferrando_entities/city.json" \
    --out "$OUT/twohop_queries.json" \
    --wikidata-date 2025-01-01

echo "Wrote $OUT/twohop_queries.json -- 3503 real two-hop (player->city->country)"
echo "queries confirmed against Ferrando et al. (2025) Table 3. Proceed to"
echo "03_pilot_gemma2b.sh."

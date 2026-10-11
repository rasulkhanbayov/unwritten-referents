#!/usr/bin/env python3
"""
skc_build_dataset_from_ferrando.py
Builds real two-hop ComposedQuery chains from Ferrando et al. (2025)'s
released single-hop entity data (github.com/javiferran/sae_entities,
Apache 2.0), instead of skc_data.py's live Wikidata SPARQL path
(skc_build_dataset.py --source wikidata), which needs unverified QIDs/PIDs
and network access this account may not have from every node.

WHY THIS EXISTS
  skc_build_dataset.py's wikidata path was tried with placeholder QIDs/PIDs
  (Q5/P106/P22) and returned garbage: an unresolved entity label and a
  semantically nonsensical chain (person--occupation-->Muse--father-->Zeus).
  Rather than guess at more QIDs/PIDs, this script uses the ACTUAL dataset
  Ferrando et al. built and released -- confirmed against Table 3 of their
  paper (Player 7487, Movie 10895, City 7904, Song 8448 entities; counts
  match exactly).

THE BRIDGE STRUCTURE THIS SCRIPT USES
  Of the four entity types, only ONE pair has a usable two-hop bridge in
  this released data: player.place_birth values that are THEMSELVES full
  city entities in city.json (3503 of 7487 players, confirmed by direct
  set intersection). No other attribute value in player/movie/song is
  itself a top-level entity in another file, so this is the only chain
  buildable from this data without further scraping:

    hop1: player {e1} -- place_birth --> city {bridge}
    hop2: city {bridge} -- {country|population|elevation} --> {target}

  This exactly matches the paper's own two-hop pattern (surface entity
  e1 written, bridge entity never written, target is what K2 measures) --
  e.g. "The country of the birthplace of the player {e1} is".

INPUT FILES (not included in this repo -- see instructions below)
  player.json and city.json from
  sae_entities-main/dataset/processed/entities/{player,city}.json in the
  Ferrando et al. release. Extract them with:
    unzip -j sae_entities.zip \\
      "sae_entities-main/dataset/processed/entities/player.json" \\
      "sae_entities-main/dataset/processed/entities/city.json" \\
      -d <output_dir>

LIMITATION, STATED EXPLICITLY
  This gives ONE entity-type pair (player/city), not all four Ferrando et
  al. types (player, movie, city, song) that skc_runbook.txt asks E0 to
  cover. It is enough for the PILOT (300 queries) and for a first E0 pass.
  Building two-hop chains for movie/song still needs either the live
  Wikidata SPARQL path (with verified QIDs/PIDs) or additional scraped
  bridge data this released repo does not include.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skc_data import ComposedQuery, Hop, save_queries, snapshot_id  # noqa: E402

CITY_ATTR_TO_RELATION = {
    "country": "country",
    "population": "population",
    "elevation": "elevation",
    "location": "coordinates",
}

# population/coordinates are numeric/geo and are explicitly EXCLUDED by
# skc_data.py's own EXCLUDED_RELATIONS set ("population", "coordinates",
# "coordinate location", ...) -- ComposedQuery.validate() will raise on
# these relation names, so only "country" (a textual answer) is usable as
# hop2 without editing skc_data.py's exclusion list. This is intentional,
# not an oversight: population/elevation answers are numeric and alias_match
# is a TEXT match, so scoring them the same way as "country" would silently
# miscount correctness. Restrict hop2 to country here; leave the numeric
# attributes for a follow-up that adds numeric-tolerant scoring.
USABLE_CITY_RELATIONS = {"country"}


def build_queries(players: list[dict], cities: list[dict]) -> list[ComposedQuery]:
    city_by_name = {c["entity"]: c for c in cities}
    queries = []
    for p in players:
        place_birth = next((a["attribute_value"] for a in p["attributes"]
                            if a["attribute_type"] == "place_birth"), None)
        if place_birth is None or place_birth not in city_by_name:
            continue
        city = city_by_name[place_birth]
        for city_attr in city["attributes"]:
            attr_type = city_attr["attribute_type"]
            if CITY_ATTR_TO_RELATION.get(attr_type) not in USABLE_CITY_RELATIONS:
                continue
            q = ComposedQuery(
                qid=f"ferrando_player_{p['entity']}_{attr_type}".replace(" ", "_"),
                entity_type="player",
                hops=[
                    Hop(subject=p["entity"], subject_type="player",
                        relation="place of birth", answers=[place_birth]),
                    Hop(subject=place_birth, subject_type="city",
                        relation=CITY_ATTR_TO_RELATION[attr_type],
                        answers=[city_attr["attribute_value"]]),
                ],
            )
            try:
                q.validate()
            except ValueError as e:
                print(f"  skipping {q.qid}: {e}", file=sys.stderr)
                continue
            queries.append(q)
    return queries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player-json", required=True,
                    help="path to player.json extracted from the sae_entities release")
    ap.add_argument("--city-json", required=True,
                    help="path to city.json extracted from the sae_entities release")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wikidata-date", required=True,
                    help="YYYY-MM-DD -- record when the Ferrando et al. release's "
                         "underlying Wikidata snapshot was taken (check their repo/paper; "
                         "if genuinely unknown, use the date you downloaded the release, "
                         "and note in Appendix B that the ORIGINAL scrape date is unconfirmed)")
    args = ap.parse_args()

    with open(args.player_json) as f:
        players = json.load(f)
    with open(args.city_json) as f:
        cities = json.load(f)

    if len(players) != 7487:
        print(f"WARNING: expected 7487 players (Table 3, Ferrando et al. 2025), "
             f"got {len(players)}. Confirm this is the same release.", file=sys.stderr)
    if len(cities) != 7904:
        print(f"WARNING: expected 7904 cities (Table 3), got {len(cities)}. "
             "Confirm this is the same release.", file=sys.stderr)

    queries = build_queries(players, cities)
    if not queries:
        raise SystemExit("0 valid two-hop chains built; check the input files.")

    snap = snapshot_id(queries, args.wikidata_date)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_queries(queries, args.out)
    with open(args.out + ".snapshot.json", "w") as f:
        json.dump({"snapshot_id": snap, "n_queries": len(queries),
                   "wikidata_date": args.wikidata_date,
                   "source": "ferrando_et_al_2025_release",
                   "source_url": "https://github.com/javiferran/sae_entities",
                   "bridge_relation": "player.place_birth -> city.country"}, f, indent=1)

    print(f"Saved {len(queries)} two-hop (player -> city -> country) queries to {args.out}")
    print(f"snapshot_id: {snap}")
    print("Record this snapshot_id AND the source_url/bridge_relation in Appendix B.")


if __name__ == "__main__":
    main()

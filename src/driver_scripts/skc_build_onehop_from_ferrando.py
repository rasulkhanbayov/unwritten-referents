#!/usr/bin/env python3
"""
skc_build_onehop_from_ferrando.py
Builds single-hop (k=1) ComposedQuery objects directly from Ferrando et al.
(2025)'s released entity data, for skc_run_e0.py's --queries argument.

E0 needs one-hop prompts over the four Ferrando et al. entity types
(player, movie, city, song), split into a FIT set (used to fit the gate
directions) and a HELD-OUT set (used to check they generalize) --
skc_run_e0.py does that split itself, at the entity level, from whatever
single query file this script produces. This script's only job is turning
each entity's own attributes into single-hop ComposedQuery objects; it does
not do any fit/held-out splitting itself.

Every attribute in player.json/movie.json/city.json/song.json becomes one
one-hop query PER (entity, attribute) pair, using the exact question
phrasing entity-type -> attribute-type -> relation mapping below (matched to
skc_data.py's TEMPLATES["single_hop"]: "The {relation} of the {etype} {entity}
is"). Numeric/geographic attributes (population, elevation, coordinates,
weight, height, dates/years as bare numbers) are EXCLUDED here for the same
reason skc_build_dataset_from_ferrando.py excludes them from hop2: alias_match
is a text match, and scoring "1962" against a generation of "born in 1962"
needs numeric-tolerant matching this bundle's skc_data.py does not implement.
Only textual attributes are included.

MULTI-VALUE ANSWERS ARE ALSO EXCLUDED, and this needed a real E0 run to find.
skc_data.compute_labels/alias_match check whether the generation's PREFIX
matches the ENTIRE gold answer string. player.teams_list and movie.cast
store their gold answer as a single comma-joined string of every known
team/cast member ("Golden State Warriors, Philadelphia Warriors" or a list
of 15+ actor names) -- alias_match against the WHOLE joined string will
essentially never match even when the model correctly names one true team
or actor, because the model's generation contains one name, not the whole
list in the same order and phrasing. This was confirmed directly: on a 4000-
query E0 fit split, per-entity-type K1 rates came back 0.0%-3.1%, and
inspecting the raw generations showed genuinely correct single-name answers
being scored as wrong for exactly this reason. TEXTUAL_ATTRIBUTES below
therefore excludes player.teams_list, movie.cast, and movie.genres/song.genres
(also frequently multi-valued, e.g. "drama film, erotic thriller, thriller
film, LGBT-related film") entirely. For the remaining attributes, which are
mostly-but-not-always single-valued (movie.directors, movie.screenwriters,
song.album, song.performers each have some multi-value entities), individual
queries whose gold answer contains a comma are dropped per-item in
build_onehop_queries() rather than trusting the attribute-type judgment call
alone -- see _looks_multivalued().
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skc_data import ComposedQuery, Hop, save_queries, snapshot_id, EXCLUDED_RELATIONS  # noqa: E402

# entity_type -> {attribute_type: relation phrase for the single_hop template}
# Textual, PREDOMINANTLY single-value attributes only. player.teams_list,
# movie.cast, movie.genres, and song.genres are excluded entirely -- their
# gold answers are comma-joined lists most of the time (76%/95%/93%/rare-
# but-nonzero respectively, measured directly against the released data),
# and alias_match's prefix check against the WHOLE joined string cannot
# credit a correct single answer. See module docstring.
TEXTUAL_ATTRIBUTES = {
    "player": {
        "place_birth": "place of birth",
    },
    "movie": {
        "directors": "director",
        "screenwriters": "screenwriter",
    },
    "city": {
        "country": "country",
    },
    "song": {
        "album": "album",
        "performers": "performer",
    },
}


def _looks_multivalued(answer: str) -> bool:
    """
    Even within the mostly-single-value attributes kept above, some
    individual entities still have a comma-joined multi-value answer (e.g.
    a movie with two co-directors). Drop those PER ITEM rather than
    trusting the attribute-type-level judgment call alone -- a single
    global attribute/entity_type decision cannot capture this residual
    per-entity variation.
    """
    return "," in answer


def build_onehop_queries(entity_type: str, entities: list[dict]) -> list[ComposedQuery]:
    relation_map = TEXTUAL_ATTRIBUTES.get(entity_type, {})
    if not relation_map:
        raise ValueError(f"no textual attribute mapping defined for entity_type={entity_type!r}")

    n_dropped_multivalued = 0
    queries = []
    for e in entities:
        for attr in e["attributes"]:
            attr_type = attr["attribute_type"]
            relation = relation_map.get(attr_type)
            if relation is None:
                continue
            if _looks_multivalued(attr["attribute_value"]):
                n_dropped_multivalued += 1
                continue
            # ComposedQuery.validate() hard-requires k>=2 (it checks chain
            # integrity between consecutive hops), so it does not apply to
            # single-hop objects -- there is no "previous hop" to check
            # against. Do the ONE part of validate() that still makes sense
            # for k=1 (the relation-exclusion check) directly, rather than
            # calling validate() and having every single-hop query rejected
            # with "need at least two hops", which is what happened on the
            # first real run of this script.
            if relation.lower() in EXCLUDED_RELATIONS:
                print(f"  skipping {entity_type}/{e['entity']}/{attr_type}: "
                     f"relation {relation!r} is excluded", file=sys.stderr)
                continue
            q = ComposedQuery(
                qid=f"onehop_{entity_type}_{e['entity']}_{attr_type}".replace(" ", "_"),
                entity_type=entity_type,
                hops=[Hop(subject=e["entity"], subject_type=entity_type,
                         relation=relation, answers=[attr["attribute_value"]])],
            )
            queries.append(q)
    if n_dropped_multivalued:
        print(f"  {entity_type}: dropped {n_dropped_multivalued} multi-value answers "
             "(comma-joined gold string)", file=sys.stderr)
    return queries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entities-dir", required=True,
                    help="directory containing player.json, movie.json, city.json, "
                         "song.json from the Ferrando et al. release")
    ap.add_argument("--entity-types", nargs="+",
                    default=["player", "movie", "city", "song"],
                    choices=["player", "movie", "city", "song"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--wikidata-date", required=True)
    args = ap.parse_args()

    all_queries = []
    for etype in args.entity_types:
        path = os.path.join(args.entities_dir, f"{etype}.json")
        with open(path) as f:
            entities = json.load(f)
        qs = build_onehop_queries(etype, entities)
        print(f"{etype}: {len(entities)} entities -> {len(qs)} single-hop queries")
        all_queries.extend(qs)

    if not all_queries:
        raise SystemExit("0 single-hop queries built; check --entities-dir and --entity-types.")

    snap = snapshot_id(all_queries, args.wikidata_date)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_queries(all_queries, args.out)
    with open(args.out + ".snapshot.json", "w") as f:
        json.dump({"snapshot_id": snap, "n_queries": len(all_queries),
                   "wikidata_date": args.wikidata_date,
                   "source": "ferrando_et_al_2025_release",
                   "source_url": "https://github.com/javiferran/sae_entities",
                   "entity_types": args.entity_types}, f, indent=1)

    print(f"\nSaved {len(all_queries)} single-hop queries to {args.out}")
    print(f"snapshot_id: {snap}")


if __name__ == "__main__":
    main()

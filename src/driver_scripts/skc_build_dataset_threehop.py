#!/usr/bin/env python3
"""
skc_build_dataset_threehop.py
Builds real THREE-hop ComposedQuery chains from live Wikidata, for E5
("compounding across hop depth" -- skc_runbook.txt, Proposition 4), which
needs k=3 chains this bundle never had (data/twohop_queries.json, built by
skc_build_dataset_from_ferrando.py, only ever produces k=2).

WHY THIS EXISTS
  The Ferrando et al. (2025) released entity files (player/city/movie/song,
  already extracted locally under data/ferrando_entities/) were checked
  directly for a usable three-hop structure and do NOT have one: city's
  own attributes (country/population/elevation/location) have no further
  attribute file to chain a third hop from, and cross-referencing movie
  directors / song performers against player.json found only 9 coincidental
  name overlaps out of ~6600 unique names -- nowhere near enough for a real
  dataset (see EXECUTION_PLAN.md for this check). A genuine three-hop chain
  needs data this bundle did not already have, so this script queries
  Wikidata's live SPARQL endpoint directly instead.

THE CHAIN THIS SCRIPT BUILDS, AND WHY
  player --place_of_birth--> city --country--> country --continent--> continent

  This is the SAME player->city->country chain data/twohop_queries.json
  already uses (so E5's k=2 prefix, required by skc_runbook.txt's own
  nesting requirement, is directly comparable to the existing two-hop
  dataset), extended by one more real, well-populated Wikidata hop
  (country -> continent).

VERIFIED, NOT GUESSED
  skc_build_dataset.py's existing wikidata path (skc_data.fetch_two_hop)
  was tried earlier in this project with placeholder QIDs/PIDs and
  returned garbage. This script's own (etype_qid, r0-r3 PIDs) were each
  checked directly against the live endpoint before use:
    - Q3665646 = "basketball player" (confirmed via a direct label lookup)
    - P106 (occupation), NOT P31 (instance of) -- confirmed empirically:
      "?e1 wdt:P31 wd:Q3665646" returns 0 rows (people are "instance of:
      human" in Wikidata, with occupation as a SEPARATE property);
      "?e1 wdt:P106 wd:Q3665646" returns real players immediately. This
      is also a real, pre-existing bug in skc_data.SPARQL_TWO_HOP (which
      hard-codes P31), flagged there but not fixed, since the correct
      predicate depends on the caller's own etype_qid.
    - P19 (place of birth), P17 (country), P30 (continent) -- each
      confirmed correct by a direct SPARQL label lookup against the live
      endpoint.
    - A small (limit=10) and medium (limit=500) live pull both returned
      real, sensible chains (e.g. "Deon Thomas" -> Chicago -> United
      States [-> North America once continent is joined], "Álex Mumbrú"
      -> Barcelona -> Spain -> Europe).
  See EXECUTION_PLAN.md for the full verification trail, including two
  real bugs found and fixed in skc_data.fetch_three_hop() while building
  this (a 4-way-join timeout, fixed by splitting into two queries; a URL-
  length limit on the second query at real scale, fixed by batching +
  POST).

USAGE
  This is a CPU-only, network-only script (no GPU, no local model) --
  run it on the login/shared server or a cpu-all node, not gpu-A100/H200.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skc_data import ComposedQuery, Hop, fetch_three_hop, save_queries, snapshot_id  # noqa: E402


def build_queries(rows: list[dict], etype_label: str, r1_label: str,
                  r2_label: str, r3_label: str) -> list[ComposedQuery]:
    """
    Turn fetch_three_hop()'s flat rows into ComposedQuery chains.

    Ambiguous e1 labels (several distinct entities sharing a surface form,
    or the same e1 appearing with more than one bridge1) are dropped --
    same discipline skc_build_dataset.py's wikidata path already uses for
    its two-hop chains, for the same reason: the entity_type
    disambiguation the prompt templates rely on does not hold if "the
    player named X" could mean more than one real X.
    """
    seen_e1 = {}
    for r in rows:
        seen_e1.setdefault(r["e1Label"], []).append(r)
    ambiguous = {k for k, v in seen_e1.items()
                if len({row["bridge1Label"] for row in v}) > 1}
    if ambiguous:
        print(f"  WARNING: {len(ambiguous)} surface forms map to >1 distinct bridge1; "
             "dropping them as ambiguous.", file=sys.stderr)

    queries, seen_qids = [], set()
    for r in rows:
        if r["e1Label"] in ambiguous:
            continue
        qid = f"wd3_{r['e1Label']}_{r3_label}".replace(" ", "_")
        if qid in seen_qids:
            continue  # same e1 already used (can't happen given the ambiguity
                      # filter above removed multi-bridge e1s, but a
                      # duplicate row from the API itself is still possible)
        seen_qids.add(qid)
        q = ComposedQuery(
            qid=qid,
            entity_type=etype_label,
            hops=[
                Hop(subject=r["e1Label"], subject_type=etype_label,
                    relation=r1_label, answers=[r["bridge1Label"]]),
                Hop(subject=r["bridge1Label"], subject_type="city",
                    relation=r2_label, answers=[r["bridge2Label"]]),
                Hop(subject=r["bridge2Label"], subject_type="country",
                    relation=r3_label, answers=[r["targetLabel"]]),
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
    ap.add_argument("--etype-qid", default="Q3665646",
                    help="Wikidata QID for the surface entity's occupation/class "
                         "(default: Q3665646, 'basketball player', VERIFIED against "
                         "the live endpoint -- see this script's own docstring)")
    ap.add_argument("--etype-label", default="player")
    ap.add_argument("--r0-pid", default="P106",
                    help="property linking e1 to etype_qid -- P106 (occupation) for "
                         "a person/occupation entity, P31 (instance of) for a genuine "
                         "Wikidata class. Default P106 matches --etype-qid's default.")
    ap.add_argument("--r1-pid", default="P19", help="e1 -> bridge1 (default: place of birth)")
    ap.add_argument("--r1-label", default="place of birth")
    ap.add_argument("--r2-pid", default="P17", help="bridge1 -> bridge2 (default: country)")
    ap.add_argument("--r2-label", default="country")
    ap.add_argument("--r3-pid", default="P30", help="bridge2 -> target (default: continent)")
    ap.add_argument("--r3-label", default="continent")
    ap.add_argument("--limit", type=int, default=4000,
                    help="max (e1, bridge1) pairs to pull in step 1 -- NOT the final "
                         "query count, since ambiguous/invalid rows are dropped after")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wikidata-date", required=True,
                    help="YYYY-MM-DD -- the date this pull was run, recorded in the "
                         "snapshot for reproducibility (skc_runbook.txt section 0)")
    ap.add_argument("--confirm", action="store_true",
                    help="pause after fetching and print a sample for manual "
                         "verification before saving -- recommended for the FIRST run "
                         "of a new (etype_qid, r0-r3) combination; the default combo "
                         "here is already verified (see this script's own docstring), "
                         "so this is off by default for that case")
    args = ap.parse_args()

    print(f"Fetching three-hop chains from Wikidata: {args.etype_label} "
         f"--{args.r1_label}--> city --{args.r2_label}--> country "
         f"--{args.r3_label}--> continent ...", flush=True)
    rows = fetch_three_hop(args.etype_qid, args.r0_pid, args.r1_pid, args.r2_pid,
                           args.r3_pid, limit=args.limit)
    if not rows:
        raise SystemExit(
            f"Wikidata returned 0 rows for etype={args.etype_qid} r0={args.r0_pid} "
            f"r1={args.r1_pid} r2={args.r2_pid} r3={args.r3_pid}. Check the QIDs/PIDs "
            "are correct before assuming the query itself is fine."
        )
    print(f"Fetched {len(rows)} raw three-hop rows.")

    if args.confirm:
        print(f"\nSample response (first row of {len(rows)}):")
        print(json.dumps(rows[0], indent=1))
        ans = input("\nDoes this look like a correct three-hop chain? [y/N] ")
        if ans.strip().lower() != "y":
            raise SystemExit("Aborting: verify the SPARQL query before proceeding.")

    queries = build_queries(rows, args.etype_label, args.r1_label, args.r2_label, args.r3_label)
    if not queries:
        raise SystemExit("0 valid three-hop chains survived construction; nothing to save.")

    snap = snapshot_id(queries, args.wikidata_date)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_queries(queries, args.out)
    with open(args.out + ".snapshot.json", "w") as f:
        json.dump({"snapshot_id": snap, "n_queries": len(queries),
                   "wikidata_date": args.wikidata_date,
                   "source": "wikidata_sparql_live",
                   "chain": f"{args.etype_label} -> city -> country -> continent",
                   "pids": {"r0": args.r0_pid, "r1": args.r1_pid,
                           "r2": args.r2_pid, "r3": args.r3_pid}}, f, indent=1)

    print(f"\nSaved {len(queries)} three-hop ({args.etype_label} -> city -> country -> "
         f"continent) queries to {args.out}")
    print(f"snapshot_id: {snap}")
    print("Record this snapshot_id AND the PIDs used in Appendix B.")
    print(f"\nNext: pass {args.out} as --queries to 09_e5_khop_gemma2b.sh "
         "(skc_run_e5.py truncates each chain to its own two-hop PREFIX in Python, "
         "per skc_runbook.txt's nesting requirement -- do not build a separate "
         "two-hop file for the k=2 comparison).")


if __name__ == "__main__":
    main()

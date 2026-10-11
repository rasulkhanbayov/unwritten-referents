#!/usr/bin/env python3
"""
skc_build_dataset.py
CPU-only driver that turns raw two-hop chains into a saved list of
skc_data.ComposedQuery objects, ready for the pilot and E0-E7.

Two independent, mutually exclusive sources, selected by --source:

  wikidata   Calls skc_data.fetch_two_hop() (SPARQL against
             query.wikidata.org) for a given (entity_type, r1, r2) triple.
             This is the path skc_REDFLAGS.md/README flag UNVERIFIED --
             it has never been run against the live endpoint. Verify the
             FIRST response by hand (this script prints a sample and pauses)
             before trusting a full pull, per skc_data.py's own docstring.

  jsonl      Reads a pre-existing two-hop file already in TwoHopFact's
             (Yang et al. 2024) or Biran et al. (2024)'s released format
             and re-expresses each row as a ComposedQuery. THE FIELD NAMES
             BELOW ARE PLACEHOLDERS (marked >>> TO RUN) -- confirm the
             actual column names in the released TwoHopFact/Biran et al.
             files before pointing this at them; do not assume this script's
             guess is correct without checking one row by hand.

Output: skc_data.save_queries() JSON, plus a snapshot_id recorded next to it
(skc_data.snapshot_id -- required so the dataset is reproducible against a
Wikidata date, per skc_runbook.txt section 0).

This script does not run on a GPU and should be run once on a CPU node
(cpu-all partition) or the login/shared server, not gpu-A100/gpu-H200.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skc_data import ComposedQuery, Hop, fetch_two_hop, save_queries, snapshot_id  # noqa: E402


def build_from_wikidata(etype_qid: str, etype_label: str, r1_pid: str, r1_label: str,
                        r2_pid: str, r2_label: str, limit: int, confirm: bool) -> list[ComposedQuery]:
    rows = fetch_two_hop(etype_qid, r1_pid, r2_pid, limit=limit)
    if not rows:
        raise RuntimeError(
            f"Wikidata returned 0 rows for etype={etype_qid} r1={r1_pid} r2={r2_pid}. "
            "Check the QIDs/PIDs are correct before assuming the query is fine."
        )

    print(f"\nSample response (first row of {len(rows)}):")
    print(json.dumps(rows[0], indent=1))
    if confirm:
        ans = input("\nDoes this look like a correct two-hop chain? [y/N] ")
        if ans.strip().lower() != "y":
            raise SystemExit("Aborting: verify the SPARQL query before proceeding.")

    # Ambiguous labels (several entities sharing a surface form) must be
    # filtered, per skc_data.fetch_two_hop's own docstring, or the
    # entity-type disambiguation the templates rely on does not hold.
    seen_e1 = {}
    for r in rows:
        seen_e1.setdefault(r["e1Label"], []).append(r)
    ambiguous = {k: v for k, v in seen_e1.items() if len(v) > 1
                and len({row["bridgeLabel"] for row in v}) > 1}
    if ambiguous:
        print(f"WARNING: {len(ambiguous)} surface forms map to >1 distinct bridge; "
              "dropping them as ambiguous.", file=sys.stderr)
    rows = [r for r in rows if r["e1Label"] not in ambiguous]

    queries = []
    for i, r in enumerate(rows):
        q = ComposedQuery(
            qid=f"wd_{etype_qid}_{r1_pid}_{r2_pid}_{i}",
            entity_type=etype_label,
            hops=[
                Hop(subject=r["e1Label"], subject_type=etype_label,
                    relation=r1_label, answers=[r["bridgeLabel"]]),
                Hop(subject=r["bridgeLabel"], subject_type="entity",  # >>> TO RUN: fetch bridge's own type
                    relation=r2_label, answers=[r["targetLabel"]]),
            ],
        )
        try:
            q.validate()
        except ValueError as e:
            print(f"  skipping {q.qid}: {e}", file=sys.stderr)
            continue
        queries.append(q)
    return queries


def build_from_jsonl(path: str, k: int) -> list[ComposedQuery]:
    """
    >>> TO RUN: the field names e1/e1_type/r1/r2/bridge/target/aliases below
    are placeholders for what this script GUESSES the released TwoHopFact /
    Biran et al. (2024) format looks like. Open one line of your actual file
    and confirm the real keys before trusting this function's output --
    silently mis-mapping columns would corrupt every label downstream and
    the failure would not be visible until E0.
    """
    queries = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hops = []
            for j in range(1, k + 1):
                hops.append(Hop(
                    subject=row[f"e{j}"],
                    subject_type=row.get(f"e{j}_type", "entity"),
                    relation=row[f"r{j}"],
                    answers=row.get(f"e{j+1}_aliases", [row[f"e{j+1}"]]),
                ))
            q = ComposedQuery(qid=row.get("qid", f"row_{i}"),
                              entity_type=row.get("e1_type", "entity"),
                              hops=hops, meta={"source_row": i})
            try:
                q.validate()
            except ValueError as e:
                print(f"  skipping row {i}: {e}", file=sys.stderr)
                continue
            queries.append(q)
    return queries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["wikidata", "jsonl"], required=True)
    ap.add_argument("--out", required=True, help="output path for save_queries() JSON")
    ap.add_argument("--wikidata-date", required=True,
                    help="YYYY-MM-DD, the Wikidata snapshot date to record (required, "
                         "not optional -- skc_data.snapshot_id refuses a bad format)")
    # wikidata source args
    ap.add_argument("--etype-qid"); ap.add_argument("--etype-label")
    ap.add_argument("--r1-pid"); ap.add_argument("--r1-label")
    ap.add_argument("--r2-pid"); ap.add_argument("--r2-label")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip the interactive sample-check pause (only for reruns you already verified)")
    # jsonl source args
    ap.add_argument("--jsonl-path")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    if args.source == "wikidata":
        for req in ["etype_qid", "etype_label", "r1_pid", "r1_label", "r2_pid", "r2_label"]:
            if getattr(args, req) is None:
                ap.error(f"--source wikidata requires --{req.replace('_', '-')}")
        queries = build_from_wikidata(args.etype_qid, args.etype_label, args.r1_pid,
                                      args.r1_label, args.r2_pid, args.r2_label,
                                      args.limit, confirm=not args.no_confirm)
    else:
        if not args.jsonl_path:
            ap.error("--source jsonl requires --jsonl-path")
        queries = build_from_jsonl(args.jsonl_path, args.k)

    if not queries:
        raise SystemExit("0 valid queries survived construction; nothing to save.")

    snap = snapshot_id(queries, args.wikidata_date)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_queries(queries, args.out)
    with open(args.out + ".snapshot.json", "w") as f:
        json.dump({"snapshot_id": snap, "n_queries": len(queries),
                   "wikidata_date": args.wikidata_date, "source": args.source}, f, indent=1)

    print(f"\nSaved {len(queries)} queries to {args.out}")
    print(f"snapshot_id: {snap}")
    print("Record this snapshot_id in Appendix B before running anything downstream.")


if __name__ == "__main__":
    main()

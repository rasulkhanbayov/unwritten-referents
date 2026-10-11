#!/usr/bin/env python3
"""
Fetch English Wikipedia pageview counts for every bridge entity in a
queries JSON, via the public Wikimedia Pageviews REST API
(skc_data.fetch_pageviews). Writes {article_title: summed_views} JSON,
the exact input skc_run_e7_ablations.py's `popularity` subcommand
expects for --pageviews (which itself feeds skc_run_e2.py's C2 donor
matching).

This was the genuinely missing piece blocking E2 -- everything
downstream (assign_popularity_deciles, match_on_decile, C2's own
matching logic in skc_run_e2.py) was already implemented and unused,
waiting on a real pageviews file that nothing in this bundle had ever
fetched. No GPU needed; this is a network-only, CPU job.

Bridge titles are matched to Wikipedia article titles by exact string
equality (spaces -> underscores) -- see fetch_pageviews()'s own
docstring for why this is a real, disclosed limitation, not resolved
here (no redirect/disambiguation resolution). Report the drop count
this script prints in the paper if using this dataset for E2/E7(b).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skc_data import load_queries, fetch_pageviews  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True,
                    help="Two-hop (or deeper) queries JSON to pull bridge entities from.")
    ap.add_argument("--start", default="2024010100")
    ap.add_argument("--end", default="2024120100")
    ap.add_argument("--out", required=True, help="Output pageviews JSON path.")
    args = ap.parse_args()

    queries = load_queries(args.queries)
    bridges = sorted({q.bridge for q in queries if q.k >= 2})
    print(f"Fetching pageviews for {len(bridges)} unique bridge entities "
         f"({args.start} to {args.end}) ...", flush=True)

    pageviews = fetch_pageviews(bridges, start=args.start, end=args.end)

    missing = len(bridges) - len(pageviews)
    print(f"Resolved {len(pageviews)}/{len(bridges)} bridge titles to a real Wikipedia "
         f"article ({missing} dropped -- no exact-title match, see fetch_pageviews()'s "
         "docstring). Report this drop count alongside any result that uses this file.")

    with open(args.out, "w") as f:
        json.dump(pageviews, f, indent=1)
    print(f"Saved to {args.out} -- pass this as --pageviews to "
         "skc_run_e7_ablations.py's popularity subcommand.")


if __name__ == "__main__":
    main()

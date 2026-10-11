#!/usr/bin/env python3
"""
skc_build_singlehop_correct.py
Builds a K1=1-restricted single-hop queries file for skc_run_e1.py's
--singlehop-correct-queries, from generations ALREADY COLLECTED and saved
by a prior skc_run_e0.py run (fit_raw_generations.json / held_raw_generations.json).

WHY THIS EXISTS
  skc_run_e1.py's --singlehop-correct-queries needs a single-hop queries
  file already restricted to items the model answers correctly (K1=1) --
  used as the comparison distribution for E1's silence check (is the gate
  as confident on composed queries as on single-hop queries it gets right?).
  Building this requires knowing K1 per query, which requires a real model
  generation pass. Rather than spend NEW GPU time re-labeling queries the
  E0 run already labeled, this reads E0's own saved raw-generation dumps
  (produced by skc_run_e0.py's diagnostic logging) and filters offline,
  entirely on CPU, no GPU needed.

Uses hop_require_prefix's containment matching (require_prefix=False),
matching E0's own K1 scoring convention exactly -- using the strict default
here instead would silently disagree with what "K1=1" meant when E0 itself
decided the gate was informative about K1, which would make this file's
K1=1 label mean something different from E0's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skc_data import load_queries, save_queries, alias_match  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onehop-queries", required=True,
                    help="the same onehop_queries.json E0 was run against")
    ap.add_argument("--generations", nargs="+", required=True,
                    help="one or more raw-generation JSON dumps from a skc_run_e0.py run "
                         "(fit_raw_generations.json, held_raw_generations.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-n", type=int, default=None,
                    help="optional cap on the number of K1=1 items kept (uniform prefix "
                         "of whatever order they appear in --generations; not a random "
                         "sample, since this is meant to be cheap and deterministic)")
    args = ap.parse_args()

    qs = {q.qid: q for q in load_queries(args.onehop_queries)}

    kept = []
    for path in args.generations:
        with open(path) as f:
            gens = json.load(f)
        for qid, prompts in gens.items():
            q = qs.get(qid)
            if q is None:
                continue
            gen = prompts.get("hop1", {}).get("generation", "")
            if alias_match(gen, q.hops[0].answers, require_prefix=False):
                kept.append(q)
                if args.max_n is not None and len(kept) >= args.max_n:
                    break
        if args.max_n is not None and len(kept) >= args.max_n:
            break

    if not kept:
        raise SystemExit("0 K1=1 items found across the given generation dumps.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_queries(kept, args.out)
    print(f"Saved {len(kept)} K1=1 single-hop queries to {args.out}")


if __name__ == "__main__":
    main()

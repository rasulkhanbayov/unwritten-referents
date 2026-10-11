#!/usr/bin/env python3
"""
Split a queries JSON into two DISJOINT files (by qid) -- a validation
split for selecting C1/C3's steering coefficient, and a test split to
report from. skc_run_e2.py's own module docstring requires this split to
exist and be disjoint (skc_runbook.txt: select the coefficient on
validation, never on the test set reported from); it refuses to run
without two distinct --queries/--val-queries files and checks for a qid
overlap itself.

A plain random shuffle+split, not stratified -- there is nothing in
skc_runbook.txt requiring balance across entity_type or popularity_decile
here, and popularity_decile is assigned by a LATER step
(skc_run_e7_ablations.py popularity) that runs on each split's own file
independently.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, save_queries  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--out-test", required=True)
    args = ap.parse_args()

    queries = load_queries(args.queries)
    import random
    rng = random.Random(args.seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * args.val_fraction))
    val, test = shuffled[:n_val], shuffled[n_val:]

    assert not ({q.qid for q in val} & {q.qid for q in test}), \
        "split produced overlapping qids -- this should be impossible for a partition"

    save_queries(val, args.out_val)
    save_queries(test, args.out_test)
    print(f"{len(queries)} queries -> {len(val)} validation / {len(test)} test "
         f"(val_fraction={args.val_fraction}, seed={args.seed}). Disjoint by qid, verified.")
    print(f"Saved to {args.out_val} and {args.out_test}.")


if __name__ == "__main__":
    main()

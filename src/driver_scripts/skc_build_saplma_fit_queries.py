#!/usr/bin/env python3
"""
skc_build_saplma_fit_queries.py

Builds a per-model SAPLMA/probe-baseline fit-query file, matching the
exact approach used for data/onehop_fit_gemma2b_layer8_v2.json (see
EXECUTION_PLAN.md, "SAPLMA fit-split data built"): filter
data/onehop_queries.json down to exactly the qids recorded in a given
E0 run's own fit_raw_generations.json -- the SAME fit split that model's
diffmeans/probe estimators were themselves trained on, so SAPLMA's own
training data stays consistent with this pipeline's established
fit/held-out discipline rather than drawing an arbitrary new split.

Usage:
    python3 skc_build_saplma_fit_queries.py \\
        --fit-raw-generations ../results/e0_gemma9b_layer8_splitfix/fit_raw_generations.json \\
        --onehop-queries ../data/onehop_queries.json \\
        --out ../data/onehop_fit_gemma9b_layer8_splitfix.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skc_data import load_queries, save_queries  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-raw-generations", required=True)
    ap.add_argument("--onehop-queries", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.fit_raw_generations) as f:
        fit_gen = json.load(f)
    fit_qids = set(fit_gen.keys())
    print(f"fit_raw_generations.json has {len(fit_qids)} qids.")

    queries = load_queries(args.onehop_queries)
    matched = [q for q in queries if q.qid in fit_qids]
    print(f"Matched {len(matched)}/{len(fit_qids)} fit qids against "
         f"{args.onehop_queries}'s own {len(queries)} queries.")
    if len(matched) != len(fit_qids):
        missing = len(fit_qids) - len(matched)
        raise SystemExit(f"{missing} fit qids from {args.fit_raw_generations} were NOT found "
                         f"in {args.onehop_queries} -- these files may not correspond to the "
                         "same dataset snapshot. Do not proceed on a partial match.")

    save_queries(matched, args.out)
    print(f"Saved {len(matched)} matched fit-split queries to {args.out}")


if __name__ == "__main__":
    main()

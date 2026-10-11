#!/usr/bin/env python3
"""
skc_build_c4_fit_queries.py

Builds a genuinely disjoint "fit" split of two-hop composed queries for
VERSION3's C4 ablation (scripts/ablations.py needs a fit split to
standardize g_S/g_B before scoring the bridge-only/bgr_min/bgr_mean arms --
see selectors.Standardizer's own docstring on why eval-split statistics
would leak).

WHY THIS EXISTS, RATHER THAN SPLITTING THE EXISTING data/twohop_queries.json:
The C7 split-leakage fix (skc_run_e0.py's split_entities(), exclude_from_fit)
excludes EVERY entity (head/bridge/target) appearing in
data/twohop_queries.json from ever landing in E0's own fit_set -- by
design, so NONE of twohop_queries.json's own 3503 queries are available
for a disjoint fit split; the population is fully claimed by
eval/held-out. Checked directly (0/3503 have e1 in fit_set) before
building this script, rather than assuming a workaround was needed.

Checked whether MORE data exists to draw from instead of splitting the
existing 3503: Ferrando et al.'s own released player.json has exactly
3503 players with a usable city bridge (7487 total players, only 3503
have place_of_birth pointing to a full city entity) -- this population
is EXHAUSTED, confirmed via skc_build_dataset_from_ferrando.py's own
docstring. No more same-relation data available from that source.

SOLUTION: data/threehop_queries.json (3961 real, live-Wikidata-verified
player->city->country->continent chains, built earlier this project via
skc_build_dataset_threehop.py, entirely SEPARATE from the Ferrando
release) already contains, as each chain's own k=2 PREFIX
(player->city->country), the exact composed-query shape C4 needs. This
script truncates every threehop chain to its k=2 prefix
(skc_run_e5.py's own truncate_to_k2(), reused directly -- not
reimplemented), filters to the subset whose e1/bridge/target are ALL
disjoint from data/twohop_queries.json's own entities (same C7
discipline, checked explicitly rather than assumed), and saves the
result as a load_queries()-compatible file ready to feed into
skc_run_e1.py (with --export-split fit) the same way any other k=2
query file would be.

This needs NO new network/GPU access -- data/threehop_queries.json
already exists on disk from earlier in this project.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, save_queries, ComposedQuery  # noqa: E402
from skc_run_e5 import truncate_to_k2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threehop-queries", default=None,
                    help="default: <repo>/data/threehop_queries.json")
    ap.add_argument("--twohop-queries", default=None,
                    help="default: <repo>/data/twohop_queries.json (the existing "
                         "eval-side population to stay disjoint from)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # <repo>/src/driver_scripts/x.py -> <repo>
    threehop_path = args.threehop_queries or os.path.join(repo, "data", "threehop_queries.json")
    twohop_path = args.twohop_queries or os.path.join(repo, "data", "twohop_queries.json")

    threehop = load_queries(threehop_path)
    twohop = load_queries(twohop_path)

    exclude_entities: set[str] = set()
    for q in twohop:
        exclude_entities.add(q.e1)
        exclude_entities.add(q.bridge)
        exclude_entities.update(q.target_answers)
    print(f"Excluding {len(exclude_entities)} entities present in {twohop_path} "
         "(the existing eval-side population).")

    k2_prefixes: list[ComposedQuery] = []
    n_overlap = 0
    for q in threehop:
        k2 = truncate_to_k2(q)
        e1 = k2.e1
        bridge = k2.bridge  # the city
        target = k2.target_answers[0]  # the country (k2 prefix's own target)
        if e1 in exclude_entities or bridge in exclude_entities or target in exclude_entities:
            n_overlap += 1
            continue
        k2_prefixes.append(k2)

    print(f"threehop total: {len(threehop)}")
    print(f"dropped for entity overlap with the eval-side population: {n_overlap}")
    print(f"clean (disjoint) k=2 prefixes retained: {len(k2_prefixes)}")

    if not k2_prefixes:
        raise SystemExit("0 disjoint k=2 prefixes survived -- cannot build a fit split "
                         "this way; report back rather than falling back to something else.")

    save_queries(k2_prefixes, args.out)
    print(f"Saved {len(k2_prefixes)} disjoint k=2 fit-split queries to {args.out}")
    print("Next: run skc_run_e1.py with --queries THIS file, --export-split fit, "
         "for each in-scope model, to capture g_S/g_B on this fit-only population.")


if __name__ == "__main__":
    main()

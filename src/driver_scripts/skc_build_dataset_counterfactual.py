#!/usr/bin/env python3
"""
skc_build_dataset_counterfactual.py
Builds control (g)'s counterfactual-chain dataset: for each real two-hop
query (e1 -- place of birth --> B_true -- country --> A_true), constructs
an IN-CONTEXT counterfactual query that states a FALSE bridge premise
("Suppose the place of birth of {e1} is actually {B_fake}") and then asks
the same second-hop question.

WHY THIS EXISTS (paper's own words, skc_main.tex, Appendix control (g)):
"Counterfactual chains, so memorized composites cannot drive the result."
If the model's correct two-hop answers came from a MEMORIZED (e1, A_true)
composite -- having learned the whole chain as one pattern rather than
genuinely composing e1 -> bridge -> answer -- then stating a FALSE bridge
in the prompt should not change its answer: it would still say A_true,
ignoring the stated premise. If the model genuinely composes, it should
instead answer A_fake (the true country of the STATED, fake bridge
B_fake), tracking the premise rather than falling back to the memorized
real chain.

THIS IS NOT MODEL EDITING. No weights are touched (no ROME/MEMIT-style
parameter edit) -- the counterfactual is stated directly IN THE PROMPT
TEXT, and the measurement is purely BEHAVIORAL (does the final answer
track the stated premise or the memorized truth), per explicit choice
recorded in EXECUTION_PLAN.md. No new capture/gate-reading code is
needed for this control.

DATA SOURCE: reuses data/twohop_queries.json entirely -- no new Wikidata
queries needed. B_fake is drawn from this SAME dataset's own pool of 847
real (city, country) pairs, so every counterfactual premise states a
REAL fact about a REAL city (never a fabricated place), just not the
place actually associated with e1 in reality. This keeps the "false
premise" false only in its ATTACHMENT to e1, not false as a standalone
fact -- a plausible counterfactual, not a nonsense one.

SAMPLING: for each real query, B_fake is drawn uniformly at random
(seeded) from every OTHER city in the pool whose country differs from
A_true (guaranteeing a genuinely different, checkable target answer).
One counterfactual pair per original query.

OUTPUT SCHEMA (data/counterfactual_twohop.json), one dict per item:
  {
    "qid": str,
    "e1": str, "e1_type": str,
    "b_true": str, "a_true": str,
    "b_fake": str, "a_fake": str,
    "prompt": str  -- the full counterfactual prompt text
  }
Deliberately NOT a ComposedQuery: this control's prompt states a FALSE
relation as fact, which ComposedQuery/render_prompts() has no way to
represent (every existing template renders only TRUE relations) --
forcing this into that machinery would either silently mis-render or
require weakening ComposedQuery's own validation, which is used
elsewhere to catch real data-quality problems. A dedicated flat schema
is the correct shape for a fundamentally different query type, not a
missing feature of the existing one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skc_data import load_queries  # noqa: E402

COUNTERFACTUAL_TEMPLATE = (
    "Suppose the place of birth of the player {e1} is actually {b_fake}. "
    "Given that, the country of the place of birth of the player {e1} is"
)


def build_counterfactual_queries(twohop_path: str, seed: int = 0) -> list[dict]:
    import random

    queries = load_queries(twohop_path)
    # Pool of (city, country) pairs actually present in this dataset --
    # every candidate B_fake is a REAL bridge city with a REAL country,
    # just possibly not the one true for this particular e1.
    city_to_country: dict[str, str] = {}
    for q in queries:
        b, a = q.hops[0].answers[0], q.hops[1].answers[0]
        city_to_country[b] = a
    all_cities = list(city_to_country.keys())
    if len(all_cities) < 2:
        raise ValueError(
            f"only {len(all_cities)} distinct bridge cities in {twohop_path} -- "
            "cannot sample a counterfactual bridge with a different country."
        )

    rng = random.Random(seed)
    out = []
    for q in queries:
        e1 = q.hops[0].subject
        b_true = q.hops[0].answers[0]
        a_true = q.hops[1].answers[0]
        candidates = [c for c in all_cities if city_to_country[c] != a_true]
        if not candidates:
            print(f"  skipping {q.qid}: no candidate city has a country "
                 f"different from {a_true!r}", file=sys.stderr)
            continue
        b_fake = rng.choice(candidates)
        a_fake = city_to_country[b_fake]
        prompt = COUNTERFACTUAL_TEMPLATE.format(e1=e1, b_fake=b_fake)
        out.append({
            "qid": f"counterfactual_{q.qid}",
            "e1": e1, "e1_type": q.entity_type,
            "b_true": b_true, "a_true": a_true,
            "b_fake": b_fake, "a_fake": a_fake,
            "prompt": prompt,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twohop-queries", required=True,
                    help="path to the real two-hop dataset (e.g. data/twohop_queries.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = build_counterfactual_queries(args.twohop_queries, seed=args.seed)
    print(f"Built {len(items)} counterfactual queries from "
         f"{args.twohop_queries}.", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(items, f, indent=1)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

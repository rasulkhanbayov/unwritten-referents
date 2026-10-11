#!/usr/bin/env python3
"""
skc_diag_bridge_k1filtered.py
DIAGNOSTIC, not part of the main pipeline.

WHY THIS SCRIPT EXISTS: skc_diag_bridge_during_gen.py's first run (see
EXECUTION_PLAN.md) captured activations while the model generated its own
answer to the hop1-only prompt, on the first 8 queries in twohop_queries.json
-- and found the city never decodable at any position, at any point during
generation. But reading the printed "model's own greedy completion" for
those same 8 queries revealed the REAL explanation: the model's own hop1
answer was WRONG for nearly all of them (e.g. "Garcia Domingos" born Luanda
-> model said "the city of Porto, Portugal"; "Artem Zabelin" born Khabarovsk
-> model said "the city of Moscow"). Every diagnostic run so far --
skc_run_e3.py's own sweep, both rounds of skc_diag_bridge_vs_final.py, and
skc_diag_bridge_during_gen.py -- sampled queries by TAKING THE FIRST N IN
FILE ORDER, with no K1 filter. If the model never correctly resolved hop1
for most of that sample, there was no correct bridge representation for
Patchscopes to find in the first place -- the entire "bridge not decodable"
investigation may have been testing queries that fail skc_runbook.txt's own
precondition ("(2) CONDITION EVERY PRIMARY ANALYSIS ON K1 = 1") from the
start, rather than testing Patchscopes itself.

This script fixes the sampling: label a larger pool of queries for K1 first
(skc_harness.label_queries_batched, the same function skc_run_e1.py's main
filter uses), keep only queries where K1=1 (the model DOES answer hop1
correctly in isolation), and run the SAME during-generation Patchscopes
capture (skc_harness.capture_during_generation, added in the previous round)
on that correctly-filtered sample. If the bridge STILL never decodes on
queries the model demonstrably gets right, that is a real Patchscopes/
representation finding. If it now decodes cleanly, the entire earlier
"0% bridge-resolved" result was a sampling artifact, not evidence about
Patchscopes or the model's internal representations at all.

Read the printed decodes by eye. This does not decide anything or write a
filter.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts               # noqa: E402
from skc_harness import (                                          # noqa: E402
    load_model, label_queries_batched, patchscope_decode,
    capture_during_generation,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="two-hop (k=2) queries JSON")
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--layers", type=int, nargs="+", default=[16])
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--pool-size", type=int, default=200,
                    help="how many queries (in file order) to label for K1 before "
                         "filtering -- needs to be large enough to find --n K1=1 "
                         "queries; prior runs found K1=1 at roughly 6.5% of the "
                         "full 3503-query set, so 200 should comfortably yield "
                         "several K1=1 examples")
    ap.add_argument("--n", type=int, default=8, help="number of K1=1 queries to run the "
                    "Patchscopes-during-generation diagnostic on")
    args = ap.parse_args()

    pool = load_queries(args.queries)
    if any(q.k != 2 for q in pool):
        raise SystemExit("this diagnostic requires two-hop (k=2) queries.")
    pool = pool[:args.pool_size]

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    print(f"Labeling {len(pool)} queries for K1 ...", flush=True)
    # hop_require_prefix={1: False}: matches skc_run_e0.py's own K1
    # convention (see EXECUTION_PLAN.md) -- this diagnostic's K1=1 sample
    # should mean the same thing K1=1 means everywhere else in the
    # pipeline, not a locally stricter definition.
    labels = label_queries_batched(model, tok, pool, hop_require_prefix={1: False})
    k1_queries = [q for q, lab in zip(pool, labels) if lab.get("K1", 0) == 1]
    print(f"K1=1: {len(k1_queries)}/{len(pool)} "
         f"({100 * len(k1_queries) / len(pool):.1f}%)")
    if not k1_queries:
        raise SystemExit("no K1=1 queries found in this pool -- raise --pool-size.")
    k1_queries = k1_queries[:args.n]

    for q in k1_queries:
        hop1_prompt = render_prompts(q)["hop1"]
        print(f"\n=== {q.qid} (K1=1 confirmed) ===")
        print(f"  true bridge (city): {q.bridge!r}")
        print(f"  hop1 prompt:        {hop1_prompt!r}")

        for layer in args.layers:
            hidden_states, generated_text, generated_tokens = capture_during_generation(
                model, tok, hop1_prompt, layer, max_new_tokens=args.max_new_tokens)
            print(f"  --- layer {layer} ---")
            print(f"    model's own greedy completion: {generated_text!r}")
            print(f"    generated tokens: {generated_tokens}")
            for i, h in enumerate(hidden_states):
                site = "PROMPT'S LAST TOKEN (prefill)" if i == 0 else \
                       f"GENERATED TOKEN #{i} ({generated_tokens[i-1]!r})"
                decoded = patchscope_decode(model, tok, h, layer)
                print(f"    [{site}] Patchscopes decode: {decoded!r}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skc_diag_bridge_during_gen.py
DIAGNOSTIC, not part of the main pipeline.

WHY THIS SCRIPT EXISTS: an exhaustive prior sweep (skc_diag_bridge_vs_final.py,
5 layers x 8 offsets x hop1-only vs composed prompts, see EXECUTION_PLAN.md's
"round 2" entry) found the bridge entity (a player's birth CITY) never
decodable via Patchscopes at ANY prompt position tested -- every position
either read the surface entity's own name or had already jumped to the
composed query's FINAL answer (the country), even on the hop1-only prompt
that never mentions country. All of that used capture_activations(), which
only ever sees a single forward pass over a STATIC prompt -- it structurally
cannot address a position that exists only once the model starts writing
its own answer.

This script tests the one site that prior sweep could not reach: capture the
residual stream WHILE the model autoregressively generates its answer to the
hop1-only prompt ("The place of birth of the player {e1} is"), at every
generated-token position (not just the prompt's own last token), and
Patchscopes-decode each one. If the city is represented at all, either (a)
the position where the model is ACTUALLY WRITING the city's tokens should
decode close to an identity (patching that hidden state into a fresh prompt
should reproduce something city-like), or (b) some position shortly before
generation starts carries it in a way the frozen-prompt sweep missed because
it wasn't looking at a position generation itself produces.

Uses skc_harness.capture_during_generation(), a new function added
specifically for this (skc_harness.py never previously combined generation
with hook-based capture -- capture_activations() and generate_greedy() were
separate, single-purpose functions).

Read the printed decodes by eye. This does not decide anything or write a
filter.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts       # noqa: E402
from skc_harness import (                                 # noqa: E402
    load_model, patchscope_decode, capture_during_generation,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="two-hop (k=2) queries JSON")
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--layers", type=int, nargs="+", default=[16])
    ap.add_argument("--max-new-tokens", type=int, default=6,
                    help="hop1 answers (city names) are usually 1-4 tokens; a few "
                         "extra covers multi-word cities without wasting much compute")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    queries = load_queries(args.queries)
    if any(q.k != 2 for q in queries):
        raise SystemExit("this diagnostic requires two-hop (k=2) queries.")
    queries = queries[:args.n]

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    for q in queries:
        hop1_prompt = render_prompts(q)["hop1"]
        print(f"\n=== {q.qid} ===")
        print(f"  true bridge (city): {q.bridge!r}")
        print(f"  hop1 prompt:        {hop1_prompt!r}")

        for layer in args.layers:
            hidden_states, generated_text, generated_tokens = capture_during_generation(
                model, tok, hop1_prompt, layer, max_new_tokens=args.max_new_tokens)
            print(f"  --- layer {layer} ---")
            print(f"    model's own greedy completion: {generated_text!r}")
            print(f"    generated tokens: {generated_tokens}")
            # index 0 = prompt's own last token (prefill); index i>=1 = the
            # (i-1)-th newly generated token's own forward-pass hidden state.
            for i, h in enumerate(hidden_states):
                site = "PROMPT'S LAST TOKEN (prefill)" if i == 0 else \
                       f"GENERATED TOKEN #{i} ({generated_tokens[i-1]!r})"
                decoded = patchscope_decode(model, tok, h, layer)
                print(f"    [{site}] Patchscopes decode: {decoded!r}")


if __name__ == "__main__":
    main()

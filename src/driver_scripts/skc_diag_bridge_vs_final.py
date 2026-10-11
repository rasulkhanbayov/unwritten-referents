#!/usr/bin/env python3
"""
skc_diag_bridge_vs_final.py
DIAGNOSTIC, not part of the main pipeline.

WHY THIS SCRIPT EXISTS: the first real E1 run at layer 16 (E3's peak,
L16_pos1) resolved 0/3503 queries, but the saved patchscope_sample.json
showed clean, mostly-correct decodes -- of the FINAL composed answer
(country), not the BRIDGE (city) that E1's filter checks against. E.g.
"Jake Bornheimer" (born New Brunswick, USA) decoded to " United States"
-- a correct country, but q.bridge is "New Brunswick", and alias_match
never matches a country string against a city string.

For composed_2 ("The {r2} of the {r1} of the {etype} {e1} is"), the model
cannot know WHICH entity's chain is being asked about until it has seen
e1, which sits at the very end of the prompt (there is nothing entity-
specific to read out before that point). So there is no "earlier position"
in the composed prompt itself where the city could be resolved before the
country. This script instead asks the question the composed prompt cannot
answer on its own: can Patchscopes decode the CITY AT ALL, at this same
(layer, offset-from-entity-span) convention, on the UNAMBIGUOUS hop1-only
prompt ("The {r1} of the {etype} {e1} is", the single_hop template with
relation=r1), where there is only one hop and therefore no composition to
race ahead of? That isolates: is the bridge representable via Patchscopes
at all here, separate from whether the composed prompt happens to expose
it before jumping to the final answer.

Read the printed decodes by eye. This does not decide anything or write a
filter -- if hop1-only decodes the city cleanly at some offset, that
result (not this script) tells you what to try feeding back into
skc_run_e1.py's bridge-resolved filter (e.g. scoring bridge-resolution
against the hop1 prompt's own bridge position instead of a position inside
the composed prompt, or accepting a broader match as skc_run_e3.py's
docstring already flags as a live option).
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts   # noqa: E402
from skc_gates import capture_activations             # noqa: E402
from skc_harness import (                                              # noqa: E402
    load_model, patchscope_decode, find_entity_span,
)


def decode_at_offset(model, tok, prompt, entity, layer, offset, label):
    span_start, span_end = find_entity_span(tok, prompt, entity)
    enc_len = len(tok(prompt)["input_ids"])
    pos = (span_end - 1) + offset
    if pos < 0 or pos >= enc_len:
        print(f"    [{label}] offset {offset}: pos={pos} outside valid range "
             f"[0, {enc_len}) -- skipped")
        return None
    acts = capture_activations(model, tok, [prompt], layer, position=pos, batch_size=1)
    decoded = patchscope_decode(model, tok, acts[0], layer)
    print(f"    [{label}] offset {offset:+d} (pos {pos}/{enc_len}): {decoded!r}")
    return decoded


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="two-hop (k=2) queries JSON")
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--layers", type=int, nargs="+", default=[16])
    ap.add_argument("--offsets", type=int, nargs="+", default=[-1, 0, 1],
                    help="offsets from the entity span end, same convention as "
                         "skc_run_e3.py; applied identically to BOTH the hop1-only "
                         "prompt and the composed prompt so the two are comparable")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    queries = load_queries(args.queries)
    if any(q.k != 2 for q in queries):
        raise SystemExit("this diagnostic requires two-hop (k=2) queries.")
    queries = queries[:args.n]

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    for q in queries:
        prompts = render_prompts(q)
        hop1_prompt = prompts["hop1"]
        composed_prompt = prompts["composed"]
        print(f"\n=== {q.qid} ===")
        print(f"  true bridge (city):   {q.bridge!r}")
        print(f"  true final answer:    {q.target_answers!r}")
        print(f"  hop1 prompt:          {hop1_prompt!r}")
        print(f"  composed prompt:      {composed_prompt!r}")

        for layer in args.layers:
            print(f"  --- layer {layer} ---")
            for off in args.offsets:
                decode_at_offset(model, tok, hop1_prompt, q.e1, layer, off,
                                 label="HOP1-ONLY (should name the city if decodable at all)")
            for off in args.offsets:
                decode_at_offset(model, tok, composed_prompt, q.e1, layer, off,
                                 label="COMPOSED    (city, or already the final answer?)")


if __name__ == "__main__":
    main()

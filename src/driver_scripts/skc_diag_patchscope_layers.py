#!/usr/bin/env python3
"""
skc_diag_patchscope_layers.py
DIAGNOSTIC, not part of the main pipeline.

ROUND 1 (resolved): tested whether a LATER layer decodes more cleanly than
E3's AUROC peak. Result: REJECTED -- every layer 8-24 decoded to the same
kind of template noise, and the noise was traced to a real bug (offsets
counted from the prompt's END landed inside the surface entity's own
multi-token name; fixed in skc_run_e3.py/bridge_resolved_patchscope() to be
entity-span-relative instead).

ROUND 2 (this version): after the offset fix, Patchscopes decode became
IDENTICAL across all 3503 real E1 queries regardless of the true bridge
entity ("-> x\n135 -> " every time) -- which looks exactly like the model
completing the identity prompt's own few-shot pattern UNPATCHED, i.e. the
patch may be having ZERO effect. This version adds two controls per layer:
  - UNPATCHED: generate() on the identity prompt with no hook at all.
  - RANDOM PATCH: same hook, same layer, a random vector of matched norm
    instead of the real captured hidden state.
If REAL PATCH == UNPATCHED, the write has no effect at all (a real bug).
If REAL PATCH == RANDOM PATCH but != UNPATCHED, the hook DOES write
something, but generation is insensitive to WHICH value is written (a
different, still real, bug -- e.g. the position patched is not one later
generation actually attends to).

Run this manually and read the printed decodes by eye -- it does not decide
anything or write a filter; it is diagnosis only.
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
    load_model, patchscope_decode, find_entity_span, PATCHSCOPE_IDENTITY_PROMPT,
)
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--offset", type=int, default=1,
                    help="position offset from the SURFACE ENTITY'S OWN LAST TOKEN, "
                         "same corrected convention as skc_run_e3.py's sweep "
                         "(+1 lands on composed_2's fixed ' is' tail for every query)")
    ap.add_argument("--layers", type=int, nargs="+", default=[8])
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    queries = load_queries(args.queries)[:args.n]
    print(f"Loading {args.model} ...")
    model, tok = load_model(args.model)

    for q in queries:
        composed = render_prompts(q)["composed"]
        # entity-span-relative, matching skc_run_e3.py's corrected
        # convention -- NOT prompt-length-relative (the original version of
        # this script, and of skc_run_e3.py, used prompt-length-relative
        # offsets, which is exactly the bug this diagnostic helped find:
        # it lands inside the surface entity's own multi-token name).
        _span_start, span_end = find_entity_span(tok, composed, q.e1)
        enc_len = len(tok(composed)["input_ids"])
        pos = (span_end - 1) + args.offset
        if pos < 0 or pos >= enc_len:
            print(f"skipping {q.qid}: offset {args.offset} from entity span end "
                 f"gives pos={pos}, outside valid range [0, {enc_len})")
            continue
        print(f"\n=== {q.qid}  (true bridge: {q.bridge!r}) ===")
        print(f"  prompt: {composed!r}")

        # UNPATCHED CONTROL: run generate() on the identity prompt with NO
        # hook registered at all. If this matches the "patched" decode
        # below, the patch is having ZERO effect -- proof of a real bug,
        # not a Patchscopes/layer limitation. Only needs to run once (not
        # once per layer), since it does not touch any layer's hook.
        import torch
        enc = tok(PATCHSCOPE_IDENTITY_PROMPT, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        unpatched = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  UNPATCHED CONTROL (no hook at all): {unpatched!r}")

        for layer in args.layers:
            acts = capture_activations(model, tok, [composed], layer,
                                       position=pos, batch_size=1)
            decoded = patchscope_decode(model, tok, acts[0], layer)

            # RANDOM-VECTOR CONTROL: same hook mechanics, same layer, but the
            # patched value is random noise of MATCHED NORM instead of the
            # real captured hidden state. If this ALSO matches the real
            # decode, the hook's WRITE is not the problem -- but the specific
            # VALUE being written is not influencing generation, which is a
            # different (and equally real) bug to chase (e.g. the write
            # lands on a token position generation does not actually attend
            # to when producing the next token).
            rng = np.random.default_rng(0)
            random_vec = rng.normal(size=acts[0].shape)
            random_vec = random_vec / np.linalg.norm(random_vec) * np.linalg.norm(acts[0])
            random_decoded = patchscope_decode(model, tok, random_vec, layer)

            print(f"  layer {layer:2d} REAL PATCH:   {decoded!r}")
            print(f"  layer {layer:2d} RANDOM PATCH: {random_decoded!r}")
            if decoded == unpatched:
                print(f"    *** layer {layer}: REAL patch decode == UNPATCHED control. "
                     "Patch is having ZERO effect. ***")
            if decoded == random_decoded:
                print(f"    *** layer {layer}: REAL patch decode == RANDOM patch decode. "
                     "The hook writes SOMETHING, but generation does not depend on "
                     "the specific VALUE patched in. ***")


if __name__ == "__main__":
    main()

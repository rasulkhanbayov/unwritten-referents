#!/usr/bin/env python3
"""
skc_diag_activation_collision.py
DIAGNOSTIC, not part of the main pipeline.

WHY THIS SCRIPT EXISTS: investigating E1's silence-check KS statistic
(large, 0.611 -- unexpected given AUROC(gate->M) is at chance) led to
inspecting the ACTUAL saved gate scores for the 158-item filtered set
offline (results/e1_gemma2b_layer16/acts_surface_filtered.npy). Found a
handful of extreme outlier scores, and confirmed by direct comparison
(np.array_equal on the full 2304-dim vectors) that THREE PAIRS of
genuinely different queries share BYTE-IDENTICAL captured activation
vectors:
  - Aleksandr Miloserdov == Olga Buryakina == Maximiliano Stanic ==
    Linas Kleiza (4-way, not just a pair)
  - Bogdan Popescu == Marta Sobral

All six of these queries are in the SAME batch (global indices 32-47
under batch_size=16, i.e. batch #2), at positions 2/10/12/14 (the first
group) and 4/8 (the second) within that 16-item batch -- alternating,
not adjacent, and not the full batch. This is not plausible as a genuine
coincidence of different residual-stream states; it looks like a real
indexing or padding bug in capture_activations()
(skc_gates.py)/entity_spans_for_prompts() (skc_harness.py), which is used
by EVERY experiment this session (E0/E1/E3/E4/E5 all route activation
capture through capture_for_queries -> capture_activations).

This script re-captures JUST the one suspect batch (the same 16 composed
prompts from queries_filtered.json, indices 32-47) with VERBOSE per-item
logging: the prompt's own real (unpadded) token count, the entity span
entity_spans_for_prompts() computes, the actual index used to read the
hidden state, the batch's padded sequence length, and the tokenizer's
padding_side at capture time -- everything needed to see exactly where
the indexing goes wrong, since this shell has no tokenizer/GPU access to
trace it further offline.

Read the printed per-item table and look specifically at whether the
KNOWN-COLLIDING queries (Miloserdov/Buryakina/Stanic/Kleiza; Popescu/
Sobral) share the same computed idx, the same real token count, or
anything else that would explain reading the same hidden-state row.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts               # noqa: E402
from skc_harness import load_model, entity_spans_for_prompts, set_padding_side  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True,
                    help="results/e1_gemma2b_layer16/queries_filtered.json")
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--batch-start", type=int, default=32,
                    help="global index the suspect batch starts at (32 for batch_size=16, "
                         "batch #2, where the collision was found)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reproduce-bug", action="store_true",
                    help="skip forcing padding_side='right' -- use the tokenizer's own "
                         "untouched default, to confirm the collision reproduces under "
                         "the real (unforced) padding_side rather than only checking that "
                         "forcing 'right' avoids it")
    args = ap.parse_args()

    all_queries = load_queries(args.queries)
    batch_queries = all_queries[args.batch_start:args.batch_start + args.batch_size]
    print(f"Diagnosing batch [{args.batch_start}:{args.batch_start + args.batch_size}] "
         f"({len(batch_queries)} queries)")

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    print(f"tokenizer.padding_side at load time: {tok.padding_side!r}")
    # capture_activations() (skc_gates.py) uses tokenizer(batch, padding=True)
    # with WHATEVER padding_side is currently set -- capture_for_queries
    # (skc_harness.py) never sets it explicitly before calling
    # capture_activations, so the REAL E1 run's padding_side at this point
    # was whatever label_queries_batched()'s own generate_greedy() call
    # left it as (generate_greedy forces "left" internally, then restores
    # the PRIOR side afterward -- which, since load_model() never sets a
    # side explicitly, is the tokenizer's own untouched default). Round 1
    # of this diagnostic forced "right" explicitly and found ZERO
    # collisions -- --reproduce-bug skips that override, to confirm the
    # collision reproduces under the tokenizer's own real, unforced
    # padding_side (very plausibly "left", matching what round 1 printed
    # at load time).
    if args.reproduce_bug:
        print(f"--reproduce-bug: NOT forcing padding_side -- left at "
             f"{tok.padding_side!r} (whatever load_model() itself left it as).")
    else:
        set_padding_side(tok, "right")
        print(f"tokenizer.padding_side set to: {tok.padding_side!r} (right, matching "
             "capture_activations()'s documented convention)")

    composed_prompts, composed_entities = [], []
    for q in batch_queries:
        p = render_prompts(q)
        composed_prompts.append(p["composed"])
        composed_entities.append(q.e1)

    spans = entity_spans_for_prompts(tok, composed_prompts, composed_entities)

    import torch
    enc = tok(composed_prompts, return_tensors="pt", padding=True).to("cuda")
    print(f"\nBatch encoded shape: {tuple(enc['input_ids'].shape)} "
         f"(batch_size, padded_seq_len)")

    hidden = {}

    def hook(_m, _i, out):
        hidden["h"] = out[0] if isinstance(out, tuple) else out

    handle = model.model.layers[args.layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        handle.remove()
    h = hidden["h"]
    print(f"Hidden state shape: {tuple(h.shape)}\n")

    print(f"{'pos':>3} {'e1':<28} {'real_tokens':>11} {'span':>10} {'idx_used':>8} "
         f"{'attn_sum':>8} {'act_hash':>12}")
    for b in range(len(batch_queries)):
        e1 = composed_entities[b]
        real_tokens = int(enc["attention_mask"][b].sum())
        span_start, span_end = spans[b]
        idx = span_end - 1
        attn_sum = real_tokens
        row = h[b, idx].float().cpu().numpy()
        act_hash = hash(row.tobytes()) % 10**8  # short fingerprint for eyeballing collisions
        print(f"{b:>3} {e1:<28} {real_tokens:>11} {f'[{span_start},{span_end})':>10} "
             f"{idx:>8} {attn_sum:>8} {act_hash:>12}")

    print("\nAny two rows sharing the SAME act_hash have (near-)identical activation "
         "vectors -- cross-check those against real_tokens/span/idx to see where the "
         "indexing actually goes wrong. Known-colliding names to look for: Aleksandr "
         "Miloserdov, Olga Buryakina, Maximiliano Stanic, Linas Kleiza (one group); "
         "Bogdan Popescu, Marta Sobral (another).")


if __name__ == "__main__":
    main()

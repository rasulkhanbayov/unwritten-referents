#!/usr/bin/env python3
"""
skc_diag_tokenize_mismatch.py
DIAGNOSTIC, not part of the main pipeline.

WHY THIS SCRIPT EXISTS: two consecutive fixes to
skc_harness.find_entity_span_in_generated_tokens() (formerly
find_entity_span_in_generated_text) -- first removing a spurious BOS
token from a re-tokenization, then abandoning re-tokenization entirely in
favor of searching generated_tokens directly for a candidate answer's OWN
standalone tokenization -- have each changed the CODE but produced
BYTE-IDENTICAL patchscope_sample.json output on the real E1 run. That is
suspicious: either the fix genuinely made no difference for these
particular queries (needs an explanation), or the standalone
tokenization of a candidate answer (e.g. " Kaunas") does not reproduce
the SAME token sequence that appears when it is embedded in the model's
actual generated text (e.g. " the city of Kaunas, Lithuania..."), which
would be the SAME decode/re-encode mismatch class of bug, just one level
deeper (tokenizing a short PROBE string in isolation is not guaranteed
to match how a tokenizer's BPE/unigram merges split the identical
characters when they appear after different preceding context).

This script does not fix or filter anything -- it just prints, for a
handful of real K1=1 queries, EXACTLY what
find_entity_span_in_generated_tokens() computes at each step: the real
generated_tokens list, the probe token list for each candidate/prefix
combination, and whether/where a match was found -- so the actual
tokenizer's behavior (which nothing offline in this environment can
inspect -- no transformers/tokenizers package is available outside the
Slurm/conda environment) is visible directly, rather than inferred from
downstream Patchscopes decode text.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts, alias_match         # noqa: E402
from skc_harness import load_model, label_queries_batched, capture_during_generation  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--pool-size", type=int, default=200)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    args = ap.parse_args()

    pool = load_queries(args.queries)[:args.pool_size]
    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    print(f"Labeling {len(pool)} queries for K1 ...", flush=True)
    # hop_require_prefix={1: False}: matches skc_run_e0.py's own K1
    # convention (see EXECUTION_PLAN.md) -- this diagnostic's K1=1 sample
    # should mean the same thing K1=1 means everywhere else in the
    # pipeline, not a locally stricter definition.
    labels = label_queries_batched(model, tok, pool, hop_require_prefix={1: False})
    k1_queries = [q for q, lab in zip(pool, labels) if lab.get("K1", 0) == 1][:args.n]
    print(f"K1=1: {len(k1_queries)} queries selected for this diagnostic\n")

    for q in k1_queries:
        hop1_prompt = render_prompts(q)["hop1"]
        _hidden_states, generated_text, generated_tokens = capture_during_generation(
            model, tok, hop1_prompt, layer=16, max_new_tokens=args.max_new_tokens)

        print(f"=== {q.qid} ===")
        print(f"  true bridge: {q.bridge!r}")
        print(f"  generated_text:   {generated_text!r}")
        print(f"  generated_tokens: {generated_tokens}")

        # q.hops[0].answers is the BRIDGE's own answer list (hop 1's
        # answer IS the bridge); q.hops[1].answers is the FINAL target's
        # answer list (hop 2's answer). This diagnostic itself had the
        # same q.hops[1] mix-up found in skc_harness.bridge_resolved_
        # patchscope() -- see EXECUTION_PLAN.md. That bug is exactly what
        # this run's own printout exposed (probes for "Lithuania"/
        # "Hungary"/etc. instead of the true bridge cities), so it is
        # fixed here too even though this file's job (as a diagnostic) is
        # done.
        bridge_answers = q.hops[0].answers
        alias_hit = alias_match(generated_text, bridge_answers, require_prefix=False)
        print(f"  alias_match(generated_text, bridge_answers) = {alias_hit}")

        for form in bridge_answers[:1]:  # gold form only, keep output readable
            for prefixed in (True, False):
                probe = (" " + form) if prefixed else form
                probe_ids = tok(probe, add_special_tokens=False)["input_ids"]
                probe_tokens = tok.convert_ids_to_tokens(probe_ids)
                n = len(probe_tokens)
                match_at = None
                if n > 0:
                    for i in range(len(generated_tokens) - n + 1):
                        if generated_tokens[i:i + n] == probe_tokens:
                            match_at = i
                            break
                print(f"  probe={probe!r:30s} prefixed={prefixed!s:5s} "
                     f"-> probe_tokens={probe_tokens}  match_at={match_at}")

        # Also print what a FULL re-tokenization of generated_text looks
        # like, for direct comparison against generated_tokens -- this is
        # exactly the comparison the earlier (abandoned) re-tokenization
        # design implicitly relied on being aligned.
        full_retokenized = tok(generated_text, add_special_tokens=False)["input_ids"]
        full_retokenized = tok.convert_ids_to_tokens(full_retokenized)
        same_as_generated = full_retokenized == generated_tokens
        print(f"  full re-tokenization of generated_text: {full_retokenized}")
        print(f"  full re-tokenization == generated_tokens: {same_as_generated}")
        print()


if __name__ == "__main__":
    main()

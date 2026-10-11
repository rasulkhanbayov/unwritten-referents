#!/usr/bin/env python3
"""
skc_run_e6.py
E6: chain of thought -- does writing the bridge entity restore gating?
(skc_runbook.txt). Tests our mechanism against Cheng et al. (Findings
EMNLP 2025), who report CoT OBSCURES hallucination cues; our account
predicts the gate at the model's OWN written bridge mention becomes MORE
informative. Both are measurable on the same items.

METHOD, and the part that is genuinely new engineering here (not just a
capture_activations call): render the "cot" template (skc_data.TEMPLATES
already has one for k=2), generate with more tokens so the model's trace
can actually name the bridge, then LOCATE the model's own mention of the
bridge entity inside its GENERATED text (not the prompt) to get a span for
capture_activations. This differs from every other driver in this
directory, all of which locate entity spans inside a fixed, known PROMPT.
Here the span is inside model-generated, non-deterministic (well, greedy,
but content-dependent) text, and the model may write the WRONG bridge
entity or paraphrase it -- skc_runbook.txt requires splitting on whether
the self-written bridge is correct, and run_E6() itself refuses to pool
that split if either arm has fewer than 2 items.

Locating the self-written mention is done by trying, in order:
  1. exact/alias match of any of the true bridge's known aliases in the
     generated text (skc_data.alias_match against a length-1 answer list)
  2. if no alias matches, look for a capitalized span that appears in an
     ANSWER-INDICATING context (immediately after "is"/"was"/"in", or
     inside a "**Step N: ...**"/"**Therefore**"/"**Conclusion**"-style
     concluding line) AND is not the query's own surface entity (q.e1) or
     a name containing/contained by it -- see find_self_written_bridge()'s
     own docstring for the full validation trail. If no such span exists,
     the item is dropped rather than guessed.

VALIDATION HISTORY: a first version of step 2 used the FIRST capitalized
multi-word span ANYWHERE in the generation, with no context requirement
and no exclusion of the surface entity. A manual audit of 30 real CoT
traces (gemma-2-9b-it, "Answer step by step." elicitation) found this
matched the QUERY'S OWN SURFACE ENTITY (e.g. "Jake Bornheimer" for a
query about Jake Bornheimer) in the overwhelming majority of cases --
because most traces open with "1. **Identify the player:** X is a
professional athlete..." before ever attempting the actual question, and
the player's own name is the first capitalized span in that boilerplate.
Since the player is never a valid bridge (city/place) answer, this
mechanically classified nearly every non-exact-match item as "bridge
written WRONG" regardless of whether the model attempted an answer at
all -- inflating n_wrong (3105) far beyond what genuine wrong-bridge
attempts would produce, and making auroc_cot_bridge_wrong measure the
gate's behavior at an essentially arbitrary (usually self-referential)
text position rather than at a genuine incorrect claim. Fixed as
described above; re-audit the sample after any future change to this
heuristic, the same way this one was caught.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts, alias_match           # noqa: E402
from skc_gates import FrozenGate, capture_activations                     # noqa: E402
from skc_experiments import run_E6, RunManifest                            # noqa: E402
from skc_harness import (                                                  # noqa: E402
    load_model, generate_greedy, label_queries_batched, capture_for_queries,
)

_CAP_SPAN = r"[A-Z][a-zA-Z.'-]*(?:\s+[A-Z][a-zA-Z.'-]*)*"
# Answer-indicating contexts only -- NOT "any capitalized span anywhere",
# which was confirmed (see this module's own docstring) to overwhelmingly
# match the query's own surface entity in CoT scaffolding like
# "1. **Identify the player:** Jake Bornheimer is a professional..."
# before the model ever attempts the actual question. Each pattern anchors
# the capitalized span to a specific "the model is now stating a fact/
# conclusion" cue rather than matching boilerplate incidentally:
#   - a capitalized span at the very START of the generation (optionally
#     preceded only by whitespace/markdown emphasis markers), since the
#     "cot" template's own prompt already ends in "is" -- the model's
#     completion is frequently the direct continuation of that copula
#     ("...is" + " the United States.", "...is" + " **England**.") with
#     no "is"/"was" cue word actually appearing IN the generated text
#     itself. Confirmed as a real, not hypothetical, gap: an early version
#     of this fix (requiring an in-text cue) missed BOTH of the sample's
#     two immediate, correct, at-the-very-start answers ("Michael Scott"
#     -> "the United States", "Eddie Jones" -> "England") before this
#     pattern was added.
#   - "is/was/in <Span>" as a copula/locative continuation elsewhere in
#     the text (covers "was born in Brazil.", "in the United States")
#   - a markdown-bold "**Step N: ...**"-style header whose OWN body names
#     the span (covers "**Step 3: Provide the answer.** ... England.")
#   - "Therefore/Conclusion/Answer:" style concluding lines
_ANSWER_CONTEXT_PATTERNS = [
    # leading lowercase filler ("the", "a", "an") + markdown emphasis
    # markers, then the capitalized span -- covers "the United States."
    # as well as "**England**." at the very start of the generation.
    re.compile(rf"^[\s*:]*(?:(?:the|a|an)\s+)?({_CAP_SPAN})"),
    re.compile(rf"\b(?:is|was|in)\s+({_CAP_SPAN})"),
    re.compile(rf"(?:Therefore|Conclusion|Answer)[,:]?\s+.*?\b({_CAP_SPAN})", re.IGNORECASE),
]

# CoT scaffolding words that are guaranteed to appear as the FIRST
# capitalized token in a research-process trace regardless of whether the
# model ever attempts an answer (e.g. ":\n\n**Step 1:** Identify the
# player's name." -- "Step" is the first capitalized word, matched by the
# start-of-generation pattern above, but is never a real bridge). A short,
# explicit denylist of observed scaffolding headers, not a speculative
# guess -- same "short, explicit list grounded in observed generations"
# discipline skc_data._GENERATION_FILLER already uses for a different
# filler-stripping problem.
_SCAFFOLDING_WORDS = frozenset({
    "step", "identify", "research", "note", "important", "conclusion",
    "answer", "summary", "therefore",
})


def find_self_written_bridge(generation: str, true_bridge_answers: list[str],
                             exclude_entity: str | None = None):
    """
    Returns (char_start, char_end, correct: bool) for the model's own mention
    of the bridge entity in its CoT generation, or None if nothing usable is
    found at all (e.g. the model refused outright, or only described a
    research PROCESS without ever attempting the actual answer).

    exclude_entity: the query's own surface entity (q.e1) -- a candidate
    span matching, containing, or contained within this string is REJECTED,
    never treated as a claimed bridge. See this module's own docstring for
    why: the surface entity's own name is what the earlier, unvalidated
    version of this fallback was actually matching almost every time.
    """
    for alias in true_bridge_answers:
        idx = generation.find(alias)
        if idx >= 0:
            return idx, idx + len(alias), True

    def is_excluded(span: str) -> bool:
        # first word alone decides scaffolding exclusion (a span like
        # "Step 1" or bare "Step" should both be excluded on "step");
        # the FULL span decides entity exclusion (a partial-name overlap
        # with q.e1, e.g. "Michael" alone, should still be excluded).
        first_word = span.split()[0].lower() if span.split() else ""
        if first_word in _SCAFFOLDING_WORDS:
            return True
        if exclude_entity is None:
            return False
        return span in exclude_entity or exclude_entity in span

    for pattern in _ANSWER_CONTEXT_PATTERNS:
        for m in pattern.finditer(generation):
            span = m.group(1).strip()
            if not span or is_excluded(span):
                continue
            start = m.start(1)
            return start, start + len(span), False
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="two-hop (k=2) queries JSON, "
                    "instruction-tuned or reasoning-tuned model recommended (skc_runbook.txt "
                         "renders 'cot' only for k=2)")
    ap.add_argument("--model", required=True, help="use an -it (instruction-tuned) or "
                    "reasoning-tuned model key; CoT prompting on a base model is unreliable")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--primary-estimator", default="diffmeans")
    ap.add_argument("--cot-max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gate_path = os.path.join(args.gate_dir, f"frozen_gate_{args.primary_estimator}.json")
    if not os.path.exists(gate_path):
        raise SystemExit(f"{gate_path} not found; run skc_run_e0.py (PASSED) first.")
    gate = FrozenGate.load(gate_path)

    queries = load_queries(args.queries)
    if any(q.k != 2 for q in queries):
        raise SystemExit("E6 requires two-hop (k=2) queries (render_prompts only "
                         "templates 'cot' for k=2).")

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    print(f"Labeling {len(queries)} queries directly (no CoT) ...", flush=True)
    # hop_require_prefix={2: False}: same fix as skc_run_e1.py/skc_run_e3.py --
    # K2 is read right below and needs this to not collapse toward constant.
    labels_direct = label_queries_batched(model, tok, queries, hop_require_prefix={2: False})
    k2_direct = np.array([l["K2"] for l in labels_direct])

    print("Capturing surface-position activations for the DIRECT condition ...", flush=True)
    acts_direct, _, _ = capture_for_queries(model, tok, queries, args.layer)
    gate_direct = gate.score(acts_direct)

    print(f"Generating CoT traces ({args.cot_max_new_tokens} tokens) ...", flush=True)
    cot_prompts = [render_prompts(q)["cot"] for q in queries]
    cot_gens = generate_greedy(model, tok, cot_prompts, max_new_tokens=args.cot_max_new_tokens)

    print("Locating self-written bridge mentions ...", flush=True)
    spans, correct_flags, keep_idx = [], [], []
    for i, (q, gen) in enumerate(zip(queries, cot_gens)):
        # q.hops[0].answers, NOT q.hops[1].answers: hops[0] is (e1, r1,
        # answers=[bridge]) -- hop 1's answer IS the bridge (e.g. the
        # city). hops[1] is (bridge, r2, answers=[final target]) -- hop
        # 2's answer is the FINAL composed answer (e.g. the country), a
        # completely different entity. The same mix-up was found and
        # fixed in skc_harness.bridge_resolved_patchscope() after three
        # rounds of chasing what looked like tokenizer/indexing bugs (see
        # EXECUTION_PLAN.md) -- every symptom there was actually caused by
        # searching for the wrong string entirely, not a misplaced
        # position for the right one. E6 has not been run yet this
        # session; caught here before it could produce the same
        # confusing trail of symptoms.
        found = find_self_written_bridge(gen, q.hops[0].answers, exclude_entity=q.e1)
        if found is None:
            continue
        char_start, char_end, correct = found
        spans.append((char_start, char_end))
        correct_flags.append(correct)
        keep_idx.append(i)
    if len(keep_idx) < len(queries):
        print(f"WARNING: dropped {len(queries) - len(keep_idx)}/{len(queries)} items with "
             "no locatable entity mention in the CoT trace (likely an outright refusal "
             "with no candidate entity named).", file=sys.stderr)
    if not keep_idx:
        raise SystemExit("no CoT trace contained a locatable entity mention; cannot proceed.")

    # Need TOKEN spans (for capture_activations), not char spans, and they are
    # relative to the FULL sequence (prompt + generation), since the model
    # only just wrote this text -- re-tokenize prompt+generation together.
    full_texts, tok_spans = [], []
    for i, (cs, ce) in zip(keep_idx, spans):
        full_text = cot_prompts[i] + cot_gens[i]
        prompt_char_len = len(cot_prompts[i])
        enc = tok(full_text, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        target_start, target_end = prompt_char_len + cs, prompt_char_len + ce
        t_start = t_end = None
        for ti, (s, e) in enumerate(offsets):
            if s == e == 0 and ti > 0:
                continue
            if t_start is None and s <= target_start < e:
                t_start = ti
            if s < target_end <= e:
                t_end = ti + 1
        if t_start is None or t_end is None:
            t_start, t_end = None, None
        full_texts.append(full_text)
        tok_spans.append((t_start, t_end))

    good = [(t, s) for t, s in zip(full_texts, tok_spans) if s[0] is not None]
    if len(good) < len(full_texts):
        print(f"WARNING: {len(full_texts) - len(good)} spans failed to map onto token "
             "offsets and were dropped.", file=sys.stderr)
    if not good:
        raise SystemExit("no self-written bridge span could be mapped to a token index.")

    final_texts = [t for t, _ in good]
    final_spans = [s for _, s in good]
    final_keep_idx = [keep_idx[i] for i, s in enumerate(tok_spans) if s[0] is not None]
    final_correct = [correct_flags[i] for i, s in enumerate(tok_spans) if s[0] is not None]

    print("Capturing activations at the self-written bridge mention ...", flush=True)
    acts_cot_bridge = capture_activations(model, tok, final_texts, args.layer,
                                          position="last_entity_token",
                                          entity_spans=final_spans)
    gate_cot_selfwritten = gate.score(acts_cot_bridge)
    k2_cot = k2_direct[final_keep_idx]
    bridge_written_correct = np.array(final_correct)

    man = RunManifest(outdir=args.out, snapshot_id="see_dataset_snapshot_json",
                      model_name=args.model, seed=args.seed)
    print("Running E6 ...", flush=True)
    e6 = run_E6(k2_direct[final_keep_idx], gate_direct[final_keep_idx],
               k2_cot, gate_cot_selfwritten, bridge_written_correct, man)
    print(json.dumps(e6, indent=1, default=str))

    with open(os.path.join(args.out, "e6_cot_traces_sample.json"), "w") as f:
        json.dump([{"prompt": cot_prompts[i], "generation": cot_gens[i]}
                  for i in final_keep_idx[:30]], f, indent=1)
    print(f"\nWrote a 30-item sample of CoT traces to {args.out}/e6_cot_traces_sample.json "
         "-- MANUALLY AUDIT these before trusting the self-written-bridge span heuristic "
         "(the capitalized-span fallback for incorrect bridges is unvalidated).")


if __name__ == "__main__":
    main()

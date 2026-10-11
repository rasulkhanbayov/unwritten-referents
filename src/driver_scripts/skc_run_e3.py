#!/usr/bin/env python3
"""
skc_run_e3.py
E3: is the bridge state present and unused? (skc_runbook.txt, skc_README.md
section 6.4). This is what makes the paper non-trivial: it shows the frozen
surface-entity gate direction, evaluated (never refit) at candidate
bridge-resolution positions, is above-chance for K2 near the layer Biran et
al. (2024) identify, and at chance elsewhere.

UPDATE (see EXECUTION_PLAN.md's "round 4" entry for the full trail): this
script's own peak (layer, position) is NO LONGER the site
skc_harness.bridge_resolved_patchscope() decodes from. That was the
original design here -- capture a single position on the STATIC composed
prompt, at "the last token of the entity-type + relation phrase just
before the copula," on the theory that the model must have resolved the
bridge by the time it reaches that point. It does not work: an exhaustive
follow-up sweep (5 layers x 8 offsets x hop1-only vs composed prompts)
found the bridge undecoded at EVERY static-prompt position tested -- the
composed prompt's word order means the model has enough context, by the
time it reaches e1, to jump straight to the FINAL composed answer rather
than exposing the intermediate bridge. bridge_resolved_patchscope() was
rebuilt to instead capture DURING the model's own generation of the
hop1-only answer, and decode the position immediately before the bridge
entity's own span in THAT generation -- validated to decode the bridge
correctly on K1=1 queries (5/7 exact matches; see
skc_harness.bridge_resolved_patchscope()'s own docstring).

E3 remains a valid, separate experiment: it still shows whether the
frozen surface-entity gate direction reads K2 above chance at some
position on the composed prompt near the layer Biran et al. (2024)
identify, which is a real finding in its own right (skc_README.md section
6.4) even though it is no longer the source of E1/E5's bridge-resolved
filter. skc_run_e1.py and skc_run_e5.py no longer take this script's
output as an argument.

POSITION OFFSETS ARE ANCHORED TO THE SURFACE ENTITY'S OWN SPAN, not the
prompt's end. An earlier version of this script swept offsets from the
PROMPT'S END, which is confounded by entity name length: multi-word-piece
names (most real names) push a fixed "-2 tokens from the end" INTO the
entity's own surname, at a position that varies with how that specific name
happens to tokenize. Confirmed directly, downstream, in skc_harness.py's
Patchscopes decode: an offset chosen this way decoded to the literal next
word-piece of the surface entity's own name, for every multi-token name,
regardless of the true bridge entity -- i.e. it was reading "continue this
player's surname," not anything about the bridge. Anchoring offsets to the
entity span's end instead means every candidate position has a fixed,
principled relationship to the entity mention across all queries, since
composed_2 prompts differ after the entity only by a short, fixed
"is"-copula tail, not by the entity's own token count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts                    # noqa: E402
from skc_gates import FrozenGate, capture_activations                 # noqa: E402
from skc_experiments import run_E3, RunManifest                       # noqa: E402
from skc_harness import (                                              # noqa: E402
    load_model, label_queries_batched, capture_at_positions, entity_spans_for_prompts,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="two-hop (k=2) queries JSON")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--primary-estimator", default="diffmeans")
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="candidate layers to sweep, e.g. --layers 6 8 10 12 14 16 18")
    ap.add_argument("--position-offsets", type=int, nargs="+", default=[-1, 0, 1],
                    help="candidate positions as offsets from the SURFACE ENTITY'S OWN "
                         "LAST TOKEN (0 = last token of the entity mention itself; "
                         "-1 = one token INTO the entity's own name, as a same-length "
                         "negative control; +1 = one token past the entity, which for "
                         "the composed_2 template's fixed \" is\" tail lands on the "
                         "PROMPT'S OWN LAST TOKEN for every query regardless of entity "
                         "name length -- confirmed directly, e.g. both 'Jake Bornheimer' "
                         "and 'Garcia Domingos' put +1 at their prompt's final token "
                         "despite tokenizing to different lengths overall). Do not push "
                         "offsets past +1 for composed_2 prompts: there is no token there "
                         "yet to capture (the model has not generated anything), and "
                         "capture_activations()/capture_at_positions() will raise an "
                         "IndexError rather than silently reading garbage -- this is the "
                         "correct behavior, not a bug to work around. NOT offsets from "
                         "the prompt's end -- that was the original, WRONG design; see "
                         "the comment at entity_end_positions below for why.")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-peak-activations", action="store_true",
                    help="also save the raw per-item activations (and matching qids/Y/K1/"
                         "K2/M labels) at the SWEEP'S OWN PEAK (layer, offset) to "
                         "<out>/peak_activations.npy + peak_labels.json. Added after a real "
                         "bug was found this project: every prior 'bridge score' (g_B) in "
                         "E0/E1/E4/the whole BGR pipeline was actually captured from an "
                         "isolated single-hop prompt that NAMES the gold bridge "
                         "(skc_harness.capture_for_queries's own hop2 return value), not "
                         "from this script's own genuine LATENT read inside the composed "
                         "prompt. This flag lets a corrected g_B be exported from the "
                         "at-least-verified-latent read this script already implements, "
                         "instead of re-deriving another capture path.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gate_path = os.path.join(args.gate_dir, f"frozen_gate_{args.primary_estimator}.json")
    if not os.path.exists(gate_path):
        raise SystemExit(f"{gate_path} not found; run skc_run_e0.py (PASSED) first.")
    gate_by_layer = {}  # E0's frozen gate is layer-specific; if sweeping layers
                        # other than E0's fitting layer, technically you are
                        # evaluating a direction fit at layer L0 against
                        # activations from layer L != L0, which is meaningless
                        # (dimensions may even mismatch for different layers of
                        # the same residual width, but the DIRECTION itself is
                        # layer-specific). See the check below.
    base_gate = FrozenGate.load(gate_path)

    queries = load_queries(args.queries)
    if any(q.k != 2 for q in queries):
        raise SystemExit("E3 requires two-hop (k=2) queries.")

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    print(f"Labeling {len(queries)} queries for K2 ...", flush=True)
    # hop_require_prefix={2: False}: K2 comes from hop2-style prompts ("The
    # {r2} of the {btype} {bridge} is"), which on real Gemma-2-2b
    # generations reliably name the correct answer inside a descriptive
    # lead-in ("located in the state of X...") rather than as the
    # generation's first token(s) -- confirmed repeatedly on this bundle's
    # own pilot and E0 runs; using the strict default here produces a
    # constant (all-0) K2, which is exactly what happened on the first real
    # E3 run before this fix. See skc_data.compute_labels's docstring.
    labels = label_queries_batched(model, tok, queries, hop_require_prefix={2: False})
    k2 = np.array([l["K2"] for l in labels])
    if len(set(k2.tolist())) < 2:
        raise SystemExit("K2 is constant; cannot compute AUROC. Resample.")

    composed_prompts = [render_prompts(q)["composed"] for q in queries]
    # OFFSETS ARE RELATIVE TO THE SURFACE ENTITY'S OWN SPAN END, not the
    # prompt's total token length. Anchoring on prompt length was the
    # original design here and it is WRONG: "The {r2} of the {r1} of the
    # {etype} {e1} is" means the number of tokens between the entity mention
    # and the prompt's end is FIXED (just " is"), but the entity e1 ITSELF
    # tokenizes to a variable number of word-pieces (e.g. "Bornheimer" is 2
    # tokens, "Domingos" is 2 tokens split as "Dom"+"ingos", etc.). An offset
    # counted from the PROMPT'S END lands inside the entity's own name for
    # any multi-token name, at a position that varies with how that specific
    # name happens to tokenize -- not at a position with any principled
    # relationship to bridge resolution. Confirmed directly: patchscope_decode()
    # at prompt-end-relative offset -2 reliably decoded to the LITERAL NEXT
    # WORD-PIECE of the surface entity's own surname (e.g. "ingos", the
    # second half of "Domingos"), for every query with a multi-token name,
    # regardless of the true bridge entity -- i.e. the captured hidden state
    # was encoding "continue this player's surname," which has nothing to do
    # with the bridge. Anchoring on the entity span's END instead fixes this,
    # since every composed_2 prompt has EXACTLY " is" (one or two tokens,
    # itself worth confirming, but far more consistent than "however many
    # word-pieces this specific name happens to split into") between the
    # entity mention and the prompt's end.
    e1_names = [q.e1 for q in queries]
    entity_spans = entity_spans_for_prompts(tok, composed_prompts, e1_names)
    entity_end_positions = [end - 1 for (_start, end) in entity_spans]  # last token OF the entity

    acts_by_layer_pos = {}
    for layer in args.layers:
        if layer != base_gate.layer:
            print(f"WARNING: sweeping layer {layer} against a gate frozen at layer "
                 f"{base_gate.layer}. This is only meaningful if E0 was ALSO run at "
                 f"layer {layer} and you load that gate instead. Skipping layer {layer} "
                 "unless you pass --gate-dir pointing at a per-layer E0 run.", file=sys.stderr)
            continue
        print(f"Capturing layer {layer} at {len(args.position_offsets)} candidate positions ...",
             flush=True)
        for off in args.position_offsets:
            positions = [p + off for p in entity_end_positions]
            if any(p < 0 for p in positions):
                print(f"  skipping offset {off}: pushes at least one prompt before token 0",
                     file=sys.stderr)
                continue
            acts = capture_at_positions(model, tok, composed_prompts, layer, positions)
            acts_by_layer_pos[f"L{layer}_pos{off}"] = acts

    if not acts_by_layer_pos:
        raise SystemExit("no (layer, position) combinations captured -- check --layers "
                         "matches an E0-fitted layer.")

    man = RunManifest(outdir=args.out, snapshot_id="see_dataset_snapshot_json",
                      model_name=args.model, seed=args.seed)
    print("Running E3 sweep + permutation null ...", flush=True)
    e3 = run_E3(base_gate, acts_by_layer_pos, k2, man, n_perm=args.n_perm)
    print(json.dumps(e3, indent=1, default=str))

    with open(os.path.join(args.out, "e3_full_map.json"), "w") as f:
        json.dump(e3, f, indent=1, default=str)

    if args.save_peak_activations:
        peak_key = e3["peak"]["key"]
        peak_acts = acts_by_layer_pos[peak_key]
        np.save(os.path.join(args.out, "peak_activations.npy"), peak_acts)
        peak_labels = [{"qid": l["qid"], "Y": l["Y"], "K1": l["K1"], "K2": l["K2"], "M": l["M"]}
                       for l in labels]
        with open(os.path.join(args.out, "peak_labels.json"), "w") as f:
            json.dump({"peak_key": peak_key, "labels": peak_labels}, f, indent=1)
        print(f"\nSaved peak ({peak_key}) activations to {args.out}/peak_activations.npy "
             f"and matching qid/Y/K1/K2/M labels to {args.out}/peak_labels.json -- this is "
             "the CORRECTED, genuinely latent g_B for downstream use, replacing "
             "capture_for_queries()'s own named-gold-bridge hop2 read.")

    print(f"\nPeak: {e3['peak']}")
    print("If this peak clears the permutation null's 95th percentile, that "
         "(layer, position offset) shows where the FROZEN GATE reads K2 above "
         "chance on the composed prompt -- a real result in its own right "
         "(skc_README.md section 6.4).")
    print(
        "\nNOTE: this sweep locates where the FROZEN GATE reads K2 above chance. "
        "It is NOT the Patchscopes identity-prompt decode (which independently "
        "confirms WHAT ENTITY is represented there, not just that K2 is linearly "
        "readable), and as of the during-generation rebuild of "
        "skc_harness.bridge_resolved_patchscope() (see EXECUTION_PLAN.md's "
        "'round 4' entry), it is no longer an input to it either -- E1's and "
        "E5's bridge-resolved filter sources its capture site from the model's "
        "own hop1 generation, not from this peak. Do not treat this script's "
        "peak position as a stand-in for Patchscopes: 'the gate happens to read "
        "K2 here' and 'the model resolved the bridge here' were confirmed to be "
        "different claims -- an exhaustive sweep at exactly this kind of static "
        "position found the bridge undecoded everywhere."
    )


if __name__ == "__main__":
    main()

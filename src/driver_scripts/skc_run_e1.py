#!/usr/bin/env python3
"""
skc_run_e1.py
E1: THE MAIN RESULT (skc_runbook.txt). Decides between H1, H2, H3.

Prerequisite: E0 must have PASSED (skc_run_e0.py exited 0) and produced
frozen_gate_*.json files. This script refuses to run without them.

THE HEADLINE IS PART (d) -- the silence check -- NOT the AUROC table.
Everyone reports AUROCs; the reportable finding is that the gate is just as
confident on composed queries as on single-hop queries it answers correctly,
while being far more often wrong. Lead with _silence_check's output when
writing this up, per skc_runbook.txt.

CRITICAL DATA REQUIREMENT, called out repeatedly in skc_README.md,
skc_REDFLAGS.md, and skc_runbook.txt: this script filters to K1=1 AND
(by default) independently verified bridge resolution (Patchscopes)
before computing M, Y, or the decomposition. The FATAL flaw
skc_runbook.txt warns about is specifically skipping the K1=1 filter --
that turns a first-hop-failure measurement into a fake composition
result, and reviewers will catch it. K1=1 filtering is ALWAYS applied by
this script, unconditionally; only the ADDITIONAL Patchscopes-based
bridge-resolution evidence is possible to skip, via
--skip-bridge-resolution below, which is a real, disclosed methodological
weakening (skc_runbook.txt's own filter is "BOTH (a) K1=1 AND (b)
independent evidence the bridge was resolved" -- this flag drops (b)),
not the same failure mode as skipping (a).

This script calls skc_harness.bridge_resolved_patchscope(), which captures
DURING the model's own greedy generation of the hop1-only answer and
decodes the position immediately before the bridge entity's own span in
that generation -- see that function's own docstring and
EXECUTION_PLAN.md's "round 4" entry for the full empirical trail for why
this replaced an earlier design anchored on a single static-prompt
(layer, offset) sourced from skc_run_e3.py's own AUROC sweep.
Patchscopes only decodes cleanly at layer 16+ on this model (confirmed
via an exhaustive sweep) -- layers 8/12 decode to template noise at every
position tested. --skip-bridge-resolution exists for exactly this
situation: E0's gate agreement between diffmeans/probe passed cleanly at
layer 8 (0.547) but FAILED at layer 16 (-0.227, a real, unresolved
diffmeans outlier-sensitivity issue at that depth -- see
EXECUTION_PLAN.md), so running the full E0->E1 chain at a layer where
the gate itself is trustworthy currently means running it at a layer
where Patchscopes does not work. The user made this tradeoff explicitly,
aware of the cost -- --skip-bridge-resolution is OFF by default (opt-in,
not silent), and E1's own printed/saved output states plainly whenever
it is used that the bridge-resolution half of skc_runbook.txt's filter
was skipped, so this is never ambiguous in the output.

`bridge_resolved_patchscope()` is ONLY MEANINGFUL to call on queries the
model already answers hop1 correctly (K1=1) -- it raises if the bridge
cannot be located in the model's own generation, which is expected and
correct behavior for a K1=0 query (there is no correctly-resolved bridge
to find). This script calls it ONLY on the K1=1 subset for exactly that
reason; a K1=0 query is scored bridge_resolved=False without attempting
the call, both to avoid the raise and because skc_runbook.txt's own
precondition already requires K1=1 before this half of the filter matters
at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, save_queries, filter_bridge_resolved   # noqa: E402
from skc_gates import FrozenGate                                            # noqa: E402
from skc_experiments import run_E1, RunManifest                             # noqa: E402
from skc_harness import (                                                   # noqa: E402
    load_model, capture_for_queries, label_queries_batched,
    bridge_resolved_patchscope, save_activation_cache,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="two-hop (k=2) queries JSON")
    ap.add_argument("--singlehop-correct-queries", required=True,
                    help="single-hop queries JSON, restricted to items the model answers "
                         "correctly (K1=1), used for the silence-check comparison distribution")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--gate-dir", required=True,
                    help="directory containing frozen_gate_*.json from a PASSED skc_run_e0.py run")
    ap.add_argument("--primary-estimator", default="diffmeans",
                    choices=["diffmeans", "probe", "sae"])
    ap.add_argument("--skip-bridge-resolution", action="store_true",
                    help="Filter on K1=1 ONLY, skipping the Patchscopes-based independent "
                         "bridge-resolution check skc_runbook.txt also requires. OFF by "
                         "default -- this is a real, disclosed methodological weakening "
                         "(see this script's own module docstring), not a shortcut to use "
                         "casually. Intended for layers where Patchscopes does not decode "
                         "cleanly (confirmed: layers 8/12 on this model) but the gate's own "
                         "cross-estimator agreement DOES pass there (unlike layer 16, where "
                         "Patchscopes works but agreement currently does not -- see "
                         "EXECUTION_PLAN.md). K1=1 filtering itself is NEVER skipped, "
                         "regardless of this flag.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-records", default=None,
                    help="if given, also write per-item records for VERSION3's re-analysis "
                         "package to this path (VERSION3/code/skc/io.py's schema) -- "
                         "e.g. VERSION3/data/records/gemma-2-2b/f1_twohop.jsonl. Uses the "
                         "SAME filtered set (labels, acts_surface_f, acts_bridge_f) this "
                         "script already computed for E1 itself, so this is a read-only "
                         "addition: it cannot change E1's own numbers.")
    ap.add_argument("--export-split", default="eval", choices=["fit", "validation", "eval"],
                    help="split label to stamp on every exported record (see "
                         "skc_export_records.py's own module docstring for why 'eval' is "
                         "the default -- this pipeline's held-out entities play that role, "
                         "not a separate third partition).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gate_path = os.path.join(args.gate_dir, f"frozen_gate_{args.primary_estimator}.json")
    if not os.path.exists(gate_path):
        raise SystemExit(
            f"{gate_path} not found. Run skc_run_e0.py first and confirm it PASSED "
            "(exit code 0) before running E1 -- a null result here is uninterpretable "
            "without a working, frozen gate from E0."
        )
    gate = FrozenGate.load(gate_path)
    print(f"Patchscopes will decode during hop1's own generation at layer {args.layer} "
         "(see bridge_resolved_patchscope()'s docstring for the capture-site design).")

    queries = load_queries(args.queries)
    if any(q.k != 2 for q in queries):
        raise SystemExit("E1 requires two-hop (k=2) queries.")

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    print(f"Labeling {len(queries)} composed queries ...", flush=True)
    # hop_require_prefix={1: False, 2: False}: hop2's False matches
    # skc_run_e3.py's identical fix -- hop2-style prompts reliably name
    # the correct answer inside a descriptive lead-in on real Gemma-2-2b
    # generations, and the strict default collapses K2 toward constant.
    #
    # hop1's False matches skc_run_e0.py's OWN K1 convention (its own
    # comment there measures strict K1 = 19/1999 = 1.0% vs containment K1
    # = 163/1999 = 8.2% on the SAME generations, for exactly the same
    # descriptive-lead-in reason -- "located in the north-east of
    # Romania, in the county of..." rather than the city name as the
    # generation's first token). This was a real, found bug: this script
    # previously left K1 at the strict default while E0 and
    # data/onehop_correct.json (built by skc_build_singlehop_correct.py,
    # which explicitly uses require_prefix=False to match E0) both used
    # the loose convention -- the mismatch surfaced as this script's own
    # --singlehop-correct-queries re-check failing ("found 1540 items the
    # model gets wrong") on a file that most certainly was built
    # correctly, just to a DIFFERENT definition of K1=1 than this script
    # was checking it against. K1=1 alone will very likely rise well
    # above 6.5% (the strict-K1 number from every earlier run this
    # session) as a direct, expected consequence -- not a regression, a
    # correction to match the definition of K1 used everywhere else in
    # this pipeline. Y (the COMPOSED answer) keeps the strict default:
    # it measures whether the model answers the actual composed question
    # correctly, a different, already-validated prompt shape untouched by
    # this fix (compute_labels always scores Y strictly, with no override
    # available at all).
    labels_unfiltered = label_queries_batched(model, tok, queries,
                                              hop_require_prefix={1: False, 2: False})
    n_k1_1 = sum(l["K1"] for l in labels_unfiltered)
    print(f"K1=1 alone (before the Patchscopes half of the filter): "
         f"{n_k1_1}/{len(labels_unfiltered)} ({n_k1_1 / len(labels_unfiltered):.1%})")

    if args.skip_bridge_resolution:
        # --skip-bridge-resolution: filter on K1=1 ONLY, per the user's
        # own explicit, informed choice (see this script's module
        # docstring and EXECUTION_PLAN.md for the full reasoning --
        # Patchscopes only decodes cleanly at layer 16+, but layer 16's
        # own gate agreement currently fails; layer 8's gate agreement
        # passes cleanly but Patchscopes does not work there). This is a
        # REAL, DISCLOSED methodological weakening relative to
        # skc_runbook.txt's own filter spec (K1=1 AND independent bridge-
        # resolution evidence) -- printed loudly here, and recorded in
        # the saved E1 result below, so it is never silently missing from
        # the output.
        print("\n*** --skip-bridge-resolution is ON: filtering on K1=1 ONLY, skipping the "
             "Patchscopes-based independent bridge-resolution check skc_runbook.txt also "
             "requires. This is a disclosed methodological weakening, not the default "
             "behavior -- see this script's module docstring. ***\n", file=sys.stderr)
        labels = filter_bridge_resolved(labels_unfiltered, bridge_resolved=None)
    else:
        # bridge_resolved_patchscope() is only meaningful on K1=1 queries -- it
        # locates the bridge INSIDE the model's own hop1 generation, so a K1=0
        # query (where that generation names something else entirely) has no
        # correctly-resolved bridge for it to find and will raise. Restricting
        # the call to K1=1 here is not an optimization, it is the same
        # precondition skc_runbook.txt already requires of this whole filter --
        # see bridge_resolved_patchscope()'s own docstring and
        # EXECUTION_PLAN.md's "round 4" entry for the empirical trail that found
        # this (every earlier diagnostic that skipped this precondition
        # concluded, wrongly, that the bridge is never decodable at all).
        k1_1_queries = [q for q, lab in zip(queries, labels_unfiltered) if lab["K1"] == 1]
        print(f"Checking bridge resolution (Patchscopes) for the {len(k1_1_queries)} K1=1 "
             "queries ...", flush=True)
        bridge_resolved = {q.qid: False for q in queries}  # K1=0 default: bridge not resolved
        n_skipped_not_found = 0
        patchscope_sample = []
        for q in k1_1_queries:
            try:
                # return_debug=True: get the decoded text AND the generation
                # from the SAME call the real filter uses, rather than
                # re-deriving them with a second, independent call. An earlier
                # version of this block did exactly that re-derivation, with
                # its own copy of capture_during_generation() using a
                # DIFFERENT max_new_tokens default and no bounds check --
                # confirmed to crash with IndexError on the first full-dataset
                # run for exactly that reason (see EXECUTION_PLAN.md). There is
                # now only one copy of this logic, inside
                # bridge_resolved_patchscope() itself.
                want_sample = len(patchscope_sample) < 30
                resolved, debug = bridge_resolved_patchscope(
                    model, tok, q, args.layer, return_debug=True)
                bridge_resolved[q.qid] = resolved
                if want_sample:
                    patchscope_sample.append({
                        "qid": q.qid, "true_bridge": q.bridge,
                        "model_hop1_generation": debug["generated_text"],
                        "decoded": debug["decoded"], "resolved": resolved,
                    })
            except ValueError:
                # bridge not findable in the model's own generation (see
                # bridge_resolved_patchscope's own docstring) -- excluded from
                # the K1=1+bridge-resolved filter rather than silently
                # defaulted to True/False either way.
                bridge_resolved[q.qid] = False
                n_skipped_not_found += 1
        if n_skipped_not_found:
            print(f"WARNING: {n_skipped_not_found}/{len(k1_1_queries)} K1=1 queries had a bridge "
                 "that could not be located in the model's own generation and were excluded "
                 "(treated as bridge-not-resolved).", file=sys.stderr)
        n_resolved = sum(bridge_resolved.values())
        print(f"Bridge resolved (Patchscopes) alone: {n_resolved}/{len(bridge_resolved)} "
             f"({n_resolved / len(bridge_resolved):.1%})")
        with open(os.path.join(args.out, "patchscope_sample.json"), "w") as f:
            json.dump(patchscope_sample, f, indent=1)
        print(f"Saved {len(patchscope_sample)} sample Patchscopes decodes to "
             f"{args.out}/patchscope_sample.json -- read these before trusting the filter; "
             "this is the FIRST real run of Patchscopes against a real transformer.")

        labels = filter_bridge_resolved(labels_unfiltered, bridge_resolved)

    retained = len(labels) / len(labels_unfiltered) if labels_unfiltered else 0.0
    filter_desc = "K1=1 only (--skip-bridge-resolution)" if args.skip_bridge_resolution \
        else "K1=1 + bridge-resolved"
    print(f"Retained {len(labels)}/{len(labels_unfiltered)} queries "
         f"({retained:.1%}) after {filter_desc} filter.")
    if not labels:
        raise SystemExit(
            f"0 queries survived the {filter_desc} filter. Without it you would "
            "be measuring first-hop failure and reporting it as composition -- there is "
            "nothing safe to compute here. Check your dataset's difficulty mix."
        )

    keep_qids = {l["qid"] for l in labels}
    kept_queries = [q for q in queries if q.qid in keep_qids]

    print("Capturing surface + bridge activations for the RETAINED (filtered) set ...", flush=True)
    acts_surface_f, _, acts_bridge_f = capture_for_queries(model, tok, kept_queries, args.layer)

    print("Capturing surface activations for the UNFILTERED set (needed for AUROC(gate->K1)) ...",
         flush=True)
    acts_surface_u, _, _ = capture_for_queries(model, tok, queries, args.layer)
    k1_unfiltered = np.array([l["K1"] for l in labels_unfiltered])

    singlehop_q = load_queries(args.singlehop_correct_queries)
    if any(q.k != 1 for q in singlehop_q):
        raise SystemExit("--singlehop-correct-queries must be single-hop (k=1).")
    # hop_require_prefix={1: False}: this is a SEPARATE label_queries_batched
    # call from the composed-query one above, and had its own, independent
    # copy of the same bug -- fixing the composed-query call's K1
    # convention did not fix this one, since they are two different call
    # sites. data/onehop_correct.json contains all four Ferrando entity
    # types (movie, song, city, player -- confirmed by inspection), not
    # just player like twohop_queries.json, so this re-check needs E0's
    # OWN K1 convention (hop_require_prefix={1: False}, matching
    # skc_build_singlehop_correct.py's own require_prefix=False) to agree
    # with how onehop_correct.json was actually built -- leaving this call
    # at the strict default reproduced the exact "found 1540 items the
    # model gets wrong" failure even AFTER the composed-query call was
    # fixed, which is what caught this second instance of the bug.
    sh_labels = label_queries_batched(model, tok, singlehop_q, hop_require_prefix={1: False})
    sh_k1 = np.array([l["K1"] for l in sh_labels])
    if not np.all(sh_k1 == 1):
        # This was a hard SystemExit before -- traced to confirm it is NOT a
        # data-quality problem: re-checking data/onehop_correct.json offline
        # against the EXACT saved generations (fit_raw_generations.json /
        # held_raw_generations.json) it was built from gives ZERO mismatches
        # (verified directly, no GPU needed, by re-running
        # skc_build_singlehop_correct.py's own alias_match check against
        # those saved generations). The disagreement only appears when THIS
        # run re-generates from scratch on a live GPU -- greedy decoding
        # producing slightly different text for a small fraction of items
        # across separate runs, consistent with floating-point/kernel
        # non-determinism at near-tied logits, not a stale or wrongly-built
        # file. Rebuilding data/onehop_correct.json again would reproduce
        # the byte-identical file (confirmed: it did, on the first attempt
        # to "fix" this by rebuilding). A hard failure here would make E1
        # permanently unrunnable end to end for a cause that cannot be
        # eliminated by fixing the file -- filtering the disagreeing items
        # out of THIS run's own comparison set is the correct response,
        # not a workaround for a bug.
        n_bad = int((sh_k1 == 0).sum())
        print(f"WARNING: {n_bad}/{len(singlehop_q)} singlehop items no longer verify as "
             "K1=1 on this run's live regeneration (greedy-decode non-determinism at "
             "near-tied logits -- confirmed NOT a stale/wrong data file; see this "
             "script's own comment above) -- dropping them from the silence-check "
             "comparison set.", file=sys.stderr)
        keep_sh = [i for i, k in enumerate(sh_k1) if k == 1]
        singlehop_q = [singlehop_q[i] for i in keep_sh]
        sh_labels = [sh_labels[i] for i in keep_sh]
        sh_k1 = sh_k1[keep_sh]
        if not singlehop_q:
            raise SystemExit(
                "0 singlehop items survived the K1=1 re-check -- something is "
                "seriously wrong (this should never drop ALL items), do not proceed."
            )
    from skc_harness import entity_spans_for_prompts
    from skc_gates import capture_activations
    from skc_data import render_prompts
    sh_prompts = [render_prompts(q)["hop1"] for q in singlehop_q]
    sh_entities = [q.e1 for q in singlehop_q]
    sh_spans = entity_spans_for_prompts(tok, sh_prompts, sh_entities)
    acts_singlehop_correct = capture_activations(model, tok, sh_prompts, args.layer,
                                                  position="last_entity_token",
                                                  entity_spans=sh_spans)

    man = RunManifest(outdir=args.out, snapshot_id="see_dataset_snapshot_json",
                      model_name=args.model, seed=args.seed)

    print("Running E1 ...", flush=True)
    e1 = run_E1(gate, acts_surface_f, acts_bridge_f, labels, acts_singlehop_correct, man,
               acts_surface_unfiltered=acts_surface_u, k1_unfiltered=k1_unfiltered)
    e1["retained_fraction"] = retained
    e1["n_before_filter"] = len(labels_unfiltered)
    # Recorded explicitly in the saved result, not just printed to the
    # console, so a reader of e1.json alone (without the job log) can
    # never miss that the Patchscopes bridge-resolution half of
    # skc_runbook.txt's filter was skipped for this run.
    e1["filter"] = "K1=1 only (--skip-bridge-resolution)" if args.skip_bridge_resolution \
        else "K1=1 + bridge-resolved (Patchscopes)"
    print(json.dumps(e1, indent=1, default=str))

    if e1["auroc"]["gate_to_M"] >= 0.65:
        print("\n*** AUC(gate -> M) is clearly above chance. H1 IS FALSE if this holds "
             "up under bootstrap. This is a REAL FINDING, not a failure -- flag back "
             "immediately per skc_runbook.txt. Do not slice the data to make it go away. ***",
             file=sys.stderr)

    save_activation_cache(os.path.join(args.out, "acts_surface_filtered.npy"), acts_surface_f,
                          {"model": args.model, "layer": args.layer})
    save_activation_cache(os.path.join(args.out, "acts_bridge_filtered.npy"), acts_bridge_f,
                          {"model": args.model, "layer": args.layer})
    with open(os.path.join(args.out, "e1_labels_filtered.json"), "w") as f:
        json.dump(labels, f, indent=1, default=str)
    # kept_queries is the ComposedQuery-object form of the same filtered set,
    # in skc_data.save_queries()'s format -- E4 and other downstream drivers
    # take --queries as a load_queries()-loadable file and need to re-render
    # prompts / re-capture activations themselves, so the label dict above
    # (e1_labels_filtered.json) is NOT sufficient as their input on its own.
    save_queries(kept_queries, os.path.join(args.out, "queries_filtered.json"))
    print(f"\nSaved the K1=1 + bridge-resolved query set to "
         f"{args.out}/queries_filtered.json -- pass THIS (not e1_labels_filtered.json) "
         "as --queries to skc_run_e4.py and other downstream drivers.")

    if args.export_records:
        # Re-score the SAME filtered activations E1 itself already used
        # (gate/acts_surface_f/acts_bridge_f/labels are all already in scope
        # from the run_E1() call above) -- this cannot change any E1 number,
        # it only writes what was already computed into VERSION3's schema.
        from skc_export_records import write_jsonl_records
        gs_export = gate.score(acts_surface_f)
        gb_export = gate.score(acts_bridge_f)
        # skc_run_e1.py only ever handles k=2 (two-hop) queries -- compute_labels()
        # always sets K2 directly for k>=2 (skc_data.py), so this is never missing here.
        write_jsonl_records(
            args.export_records,
            query_ids=[l["qid"] for l in labels],
            split=args.export_split,
            seed=args.seed,
            y=[l["Y"] for l in labels],
            k1=[l["K1"] for l in labels],
            k2=[l["K2"] for l in labels],
            g_s=gs_export.tolist(),
            g_b=gb_export.tolist(),
            estimator=args.primary_estimator,
            offset=0,
        )
        print(f"\nExported {len(labels)} per-item records for VERSION3 to "
             f"{args.export_records} (split='{args.export_split}', "
             f"estimator='{args.primary_estimator}').")


if __name__ == "__main__":
    main()

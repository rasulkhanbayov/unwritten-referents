#!/usr/bin/env python3
"""
skc_run_e2.py
E2: causal tests, Table 1 design (skc_runbook.txt). Separates "no signal"
from "signal not routed". Runs C1 (surface steer), C3 (gate inject), and C2
(bridge patch -- THE DECISIVE test), then calls run_E2 to assemble the
dissociation ratio.

C4 (attribution graphs, circuit-tracer) is DELIBERATELY NOT in this script.
It is a separate toolchain (circuit-tracer, from source; Neuronpedia for
on-demand Gemma-2-2B graphs) with its own environment and its own
verification burden -- see skc_run_e2_c4_graphs.py and the plan document.
Bundling it here would make one script depend on two independently fragile,
never-verified pieces of infrastructure; keeping them separate means a C4
failure does not block C1-C3, which are the load-bearing causal evidence.

Prerequisites:
  - E0 passed (frozen gate available)
  - A held-out VALIDATION split to select the C1/C3 steering coefficient on.
    skc_runbook.txt is explicit: select the coefficient on validation, never
    on the test set you report from. This script enforces two disjoint
    query files for that reason (--val-queries vs --queries).
  - skc_data.match_on_decile() pairs for C2, built from a dataset with
    popularity_decile and meta["bridge_known"] already set (see
    skc_run_e7_ablations.py's popularity-matching step, which must run
    BEFORE this script, not after).
  - E3 run at the SAME layer as --layer/--gate-dir below, so --e3-map's
    peak (layer, offset) actually corresponds to this gate's own fitting
    layer. E3's map is required here (see --e3-map) -- C2/C3 both patch/
    inject at "the bridge-resolution position" per skc_runbook.txt's own
    C2/C3 spec, and that position is E3's own empirical peak on the
    COMPOSED prompt, not a position from the during-generation Patchscopes
    site skc_harness.bridge_resolved_patchscope() uses for E1's filter --
    those are two different, deliberately separate uses of "bridge
    position" in this bundle (E1 needs Patchscopes to CONFIRM the model
    resolved the bridge at all; C2/C3 need a well-defined RESIDUAL-STREAM
    POSITION to intervene on, which is what E3's sweep locates, independent
    of whether Patchscopes can decode a clean entity there). Confirmed with
    the user before building this: use E3's composed-prompt position, not
    the hop1-generation site, for C2/C3 specifically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import (                                                            # noqa: E402
    load_queries, render_prompts, is_refusal, match_on_decile, alias_match,
)
from skc_gates import FrozenGate, capture_activations                            # noqa: E402
from skc_experiments import (                                                     # noqa: E402
    steer_at_position, patch_bridge, dissociation_ratio, run_E2, RunManifest,
)
from skc_harness import (                                                         # noqa: E402
    load_model, entity_spans_for_prompts, find_entity_span, capture_at_positions,
)


def _parse_e3_peak_key(key: str):
    """
    Parse skc_run_e3.py's "L{layer}_pos{offset}" key format (e.g. "L16_pos1")
    into (layer: int, offset: int). offset is relative to the SURFACE
    ENTITY'S OWN LAST TOKEN in the composed prompt, matching
    skc_run_e3.py's own sweep convention exactly.

    A near-identical helper (skc_harness._parse_e3_peak_key) existed
    earlier this session but was removed once skc_run_e1.py/skc_run_e5.py
    stopped needing it (their Patchscopes filter was rebuilt to source its
    site from the model's own hop1 generation instead of E3's map -- see
    EXECUTION_PLAN.md). C2/C3 here are a GENUINELY DIFFERENT consumer of
    E3's map (confirmed with the user): they need E3's own composed-prompt
    position, not the during-generation site, so this is not resurrecting
    dead code -- it is a small, locally-scoped helper for the one caller
    that still needs this exact parse.
    """
    import re
    m = re.fullmatch(r"L(-?\d+)_pos(-?\d+)", key)
    if not m:
        raise ValueError(f"unrecognized E3 peak key format: {key!r}; expected 'L<layer>_pos<offset>'")
    return int(m.group(1)), int(m.group(2))


def bridge_positions_for_queries(tok, queries, layer, offset):
    """
    Per-query entity-span-relative bridge-resolution POSITION on the
    COMPOSED prompt, using E3's own (layer, offset) convention -- NOT
    skc_harness.bridge_resolved_patchscope()'s during-generation site (see
    this module's own docstring for why C2/C3 use E3's site specifically).

    Returns (prompts, positions, skipped_qids): skipped_qids lists queries
    whose entity name tokenizes too long for this offset to land inside
    the prompt (mirrors bridge_resolved_patchscope()'s own "skip rather
    than clip" discipline) -- excluded from C2/C3 rather than silently
    clamped, which would patch/inject at the wrong token.
    """
    prompts, positions, skipped = [], [], []
    kept_queries = []
    for q in queries:
        composed = render_prompts(q)["composed"]
        span_start, span_end = find_entity_span(tok, composed, q.e1)
        enc_len = len(tok(composed)["input_ids"])
        pos = (span_end - 1) + offset
        if pos < 0 or pos >= enc_len:
            skipped.append(q.qid)
            continue
        prompts.append(composed)
        positions.append(pos)
        kept_queries.append(q)
    return kept_queries, prompts, positions, skipped


def sweep_coefficient(model, tok, val_prompts, val_positions, direction, layer, coeffs):
    """
    C1/C3 coefficient sweep on the VALIDATION split only. Returns
    {coeff: refusal_rate}. The caller picks the coefficient from this dict
    and reports the WHOLE sweep in the paper, not just the chosen value.
    """
    out = {}
    for c in coeffs:
        gens = steer_at_position(model, tok, val_prompts, direction, c, layer, val_positions)
        out[c] = float(np.mean([is_refusal(g) for g in gens]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="two-hop TEST queries JSON, with "
                    "popularity_decile and meta.bridge_known already set")
    ap.add_argument("--val-queries", required=True,
                    help="DISJOINT two-hop queries JSON for coefficient selection only")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--e3-map", required=True,
                    help="path to e3_full_map.json from an E3 run at the SAME layer as "
                         "--layer/--gate-dir. Its peak.key ('L{layer}_pos{offset}') supplies "
                         "the bridge-resolution POSITION C2/C3 both patch/inject at -- this "
                         "is E3's own composed-prompt site, not "
                         "skc_harness.bridge_resolved_patchscope()'s during-generation site "
                         "(see this module's own docstring for why those are different, "
                         "deliberate choices for different consumers).")
    ap.add_argument("--patch-layer-window", type=int, nargs=2, default=None,
                    help="[start, end) contiguous layer window for C2 patching, per "
                         "skc_runbook.txt's own C2 spec ('a CONTIGUOUS layer window'). "
                         "Defaults to a small window centered on --e3-map's own peak layer "
                         "([peak_layer-1, peak_layer+2)) if not given -- pass explicitly to "
                         "override.")
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--primary-estimator", default="diffmeans")
    ap.add_argument("--coeffs", type=float, nargs="+",
                    default=[-4, -2, -1, 1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.e3_map):
        raise SystemExit(
            f"{args.e3_map} not found. Run skc_run_e3.py first at the SAME layer as "
            "--layer/--gate-dir -- C2/C3 need its empirical peak (layer, position) and "
            "must not guess one independently."
        )
    with open(args.e3_map) as f:
        e3_map = json.load(f)
    e3_layer, e3_offset = _parse_e3_peak_key(e3_map["peak"]["key"])
    if e3_layer != args.layer:
        raise SystemExit(
            f"--e3-map's peak layer ({e3_layer}) does not match --layer ({args.layer}). "
            "E3 must have been run at the SAME layer as the gate C2/C3 use here, or the "
            "position and the gate direction refer to different residual streams. "
            "Re-run E3 at --layer, or point --layer at E3's own layer."
        )
    if args.patch_layer_window is None:
        args.patch_layer_window = [e3_layer - 1, e3_layer + 2]
        print(f"--patch-layer-window not given; defaulting to {args.patch_layer_window} "
             f"(centered on E3's own peak layer {e3_layer}).")
    print(f"C2/C3 will patch/inject at E3's peak: layer={e3_layer}, offset={e3_offset} "
         f"(auroc={e3_map['peak']['auroc']:.3f})")

    gate_path = os.path.join(args.gate_dir, f"frozen_gate_{args.primary_estimator}.json")
    if not os.path.exists(gate_path):
        raise SystemExit(f"{gate_path} not found; run skc_run_e0.py (PASSED) first.")
    gate = FrozenGate.load(gate_path)

    queries = load_queries(args.queries)
    val_queries = load_queries(args.val_queries)
    test_ids = {q.qid for q in queries}
    if test_ids & {q.qid for q in val_queries}:
        raise SystemExit(
            "--queries and --val-queries share qids. The coefficient must be selected "
            "on data disjoint from what you report from, or the C1/C3 result is "
            "tuned on the test set, which skc_runbook.txt explicitly forbids."
        )

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    # --- C1: surface steer ---
    print("C1: sweeping steering coefficient on validation split ...", flush=True)
    val_prompts, val_entities = [], []
    for q in val_queries:
        p = render_prompts(q)
        val_prompts.append(p["composed"]); val_entities.append(q.e1)
    val_spans = entity_spans_for_prompts(tok, val_prompts, val_entities)
    val_positions = [s[1] - 1 for s in val_spans]
    sweep = sweep_coefficient(model, tok, val_prompts, val_positions, gate.direction,
                              args.layer, args.coeffs)
    print(json.dumps(sweep, indent=1))
    best_coeff = max(sweep, key=sweep.get)   # >>> TO RUN: pick by whatever criterion the
                                              # paper commits to (e.g. max refusal shift);
                                              # this default (max refusal rate) is a
                                              # placeholder decision rule, confirm before use
    print(f"Selected coefficient (validation): {best_coeff}")

    print("C1: applying selected coefficient on the TEST split, split by M ...", flush=True)
    from skc_harness import label_queries_batched
    # hop_require_prefix={1: False, 2: False}: the SAME K1/K2 convention
    # skc_run_e0.py/skc_run_e1.py/skc_run_e5.py/skc_run_e7_ablations.py's
    # popularity subcommand all use (measured there: strict K1/K2 = ~1%
    # vs containment = ~8% on the same generations, real Gemma-2-2b hop2
    # answers reliably name the correct entity inside a descriptive
    # lead-in rather than as the first token). This call site was left at
    # the strict default -- the exact mismatch class already found and
    # fixed at every other label_queries_batched call site this session --
    # and it produced a real, observed failure: M was 0 for every one of
    # 2802 test queries (job 383261), which made d_refusal degenerate
    # downstream in C2 and crashed dissociation_ratio's guardrail
    # ("too few finite bootstrap ratios; effects are degenerate"). Fixed
    # at the source rather than working around the crash.
    test_labels = label_queries_batched(model, tok, queries,
                                        hop_require_prefix={1: False, 2: False})
    test_prompts, test_entities = [], []
    for q in queries:
        p = render_prompts(q)
        test_prompts.append(p["composed"]); test_entities.append(q.e1)
    test_spans = entity_spans_for_prompts(tok, test_prompts, test_entities)
    test_positions = [s[1] - 1 for s in test_spans]
    c1_gens = steer_at_position(model, tok, test_prompts, gate.direction, best_coeff,
                               args.layer, test_positions)
    m = np.array([l["M"] for l in test_labels])
    c1_refusal = np.array([int(is_refusal(g)) for g in c1_gens])
    c1_refusal_by_M = {int(v): c1_refusal[m == v].tolist() for v in (0, 1) if (m == v).any()}

    print("\nC1 complete.")
    print(json.dumps({
        "C1_coefficient_sweep": sweep, "C1_selected_coefficient": best_coeff,
        "C1_refusal_rate_by_M": {k: float(np.mean(v)) for k, v in c1_refusal_by_M.items()},
    }, indent=1))
    with open(os.path.join(args.out, "e2_c1_only.json"), "w") as f:
        json.dump({"sweep": sweep, "selected_coefficient": best_coeff,
                  "refusal_by_M": c1_refusal_by_M}, f, indent=1, default=str)

    # --- C2 and C3: both patch/inject at E3's own bridge-resolution
    # position on the COMPOSED prompt (args.layer, e3_offset) -- see this
    # module's own docstring for why this site, not
    # bridge_resolved_patchscope()'s during-generation one, is what C2/C3
    # use. bridge_positions_for_queries() computes this per query
    # (entity-span-relative, since e1 tokenizes to a different length per
    # query) and skips any query whose e1 tokenizes too long for this
    # offset to land inside the prompt, rather than clipping.
    print("\nComputing bridge-resolution positions on the composed TEST prompts ...",
         flush=True)
    kept_test_q, bridge_test_prompts, bridge_test_positions, skipped_test = \
        bridge_positions_for_queries(tok, queries, args.layer, e3_offset)
    if skipped_test:
        print(f"WARNING: {len(skipped_test)}/{len(queries)} test queries skipped (entity "
             "name too long for E3's offset).", file=sys.stderr)

    # --- C3: gate inject at the bridge position ------------------------
    print("C3: injecting the unknown-entity direction at the bridge position ...",
         flush=True)
    c3_baseline_gens = steer_at_position(model, tok, bridge_test_prompts, gate.direction,
                                         0.0, args.layer, bridge_test_positions)
    c3_injected_gens = steer_at_position(model, tok, bridge_test_prompts, gate.direction,
                                         best_coeff, args.layer, bridge_test_positions)
    c3_baseline_refusal = np.array([int(is_refusal(g)) for g in c3_baseline_gens])
    c3_injected_refusal = np.array([int(is_refusal(g)) for g in c3_injected_gens])
    c3_refusal_delta = c3_injected_refusal - c3_baseline_refusal

    # --- C2: bridge patch, THE DECISIVE test ----------------------------
    # Donor pairs: (known_q, unknown_q) matched on popularity decile,
    # relation type (implicit: match_on_decile groups within the SAME
    # dataset, which is already single-relation-type per file), and
    # (approximately) prompt length via entity-span-relative positioning.
    # The TEST query in a pair is the known-bridge one (its own bridge
    # residual gets REPLACED with the unknown-bridge donor's) -- this
    # matches skc_runbook.txt's own C2 wording: "Replace the bridge-
    # position residual with that of a matched query whose bridge is
    # unknown."
    print("C2: building known/unknown-bridge donor pairs (match_on_decile) ...", flush=True)
    pairs = match_on_decile(queries, seed=args.seed)
    if len(pairs) < 2:
        raise SystemExit(
            f"only {len(pairs)} known/unknown-bridge pairs found -- need >=2 for "
            "dissociation_ratio's paired bootstrap. Check popularity_decile and "
            "meta.bridge_known are set on --queries (skc_run_e7_ablations.py's "
            "popularity subcommand must run first)."
        )
    known_qs = [p[0] for p in pairs]
    unknown_qs = [p[1] for p in pairs]

    kept_known_q, known_prompts, known_positions, skipped_known = \
        bridge_positions_for_queries(tok, known_qs, args.layer, e3_offset)
    # unknown (donor) queries need positions computed against THEIR OWN
    # composed prompts too -- the position is entity-span-relative to
    # EACH prompt's own e1, not shared across the pair.
    kept_unknown_q, unknown_prompts, unknown_positions, skipped_unknown = \
        bridge_positions_for_queries(tok, unknown_qs, args.layer, e3_offset)
    # keep only pairs where BOTH sides survived the position computation
    known_by_qid = {q.qid: (p, pos) for q, p, pos in zip(kept_known_q, known_prompts, known_positions)}
    unknown_by_qid = {q.qid: (p, pos) for q, p, pos in zip(kept_unknown_q, unknown_prompts, unknown_positions)}
    usable_pairs = [(k, u) for k, u in pairs if k.qid in known_by_qid and u.qid in unknown_by_qid]
    if skipped_known or skipped_unknown:
        print(f"WARNING: dropped {len(pairs) - len(usable_pairs)}/{len(pairs)} donor pairs "
             "(entity name too long for E3's offset on one side).", file=sys.stderr)
    if len(usable_pairs) < 2:
        raise SystemExit(
            f"only {len(usable_pairs)} usable donor pairs survived position computation -- "
            "need >=2 for dissociation_ratio's paired bootstrap."
        )

    test_q_c2 = [k for k, _u in usable_pairs]
    donor_q_c2 = [u for _k, u in usable_pairs]
    test_prompts_c2 = [known_by_qid[k.qid][0] for k in test_q_c2]
    test_positions_c2 = [known_by_qid[k.qid][1] for k in test_q_c2]
    donor_prompts_c2 = [unknown_by_qid[u.qid][0] for u in donor_q_c2]
    donor_positions_c2 = [unknown_by_qid[u.qid][1] for u in donor_q_c2]

    print(f"C2: capturing donor (unknown-bridge) activations across layers "
         f"{list(range(args.patch_layer_window[0], args.patch_layer_window[1]))} ...",
         flush=True)
    donor_states = {}
    for li in range(args.patch_layer_window[0], args.patch_layer_window[1]):
        donor_states[li] = capture_at_positions(model, tok, donor_prompts_c2, li,
                                                donor_positions_c2)

    print("C2: baseline (unpatched) generations on the TEST (known-bridge) prompts ...",
         flush=True)
    from skc_harness import generate_greedy
    c2_baseline_gens = generate_greedy(model, tok, test_prompts_c2)
    print("C2: patched generations (donor's bridge-position residual substituted in) ...",
         flush=True)
    c2_patched_gens = patch_bridge(model, tok, test_prompts_c2, donor_states,
                                   list(range(args.patch_layer_window[0], args.patch_layer_window[1])),
                                   test_positions_c2)

    c2_baseline_recall = np.array([int(alias_match(g, q.target_answers))
                                   for g, q in zip(c2_baseline_gens, test_q_c2)])
    c2_patched_recall = np.array([int(alias_match(g, q.target_answers))
                                  for g, q in zip(c2_patched_gens, test_q_c2)])
    c2_baseline_refusal = np.array([int(is_refusal(g)) for g in c2_baseline_gens])
    c2_patched_refusal = np.array([int(is_refusal(g)) for g in c2_patched_gens])
    d_recall = (c2_patched_recall - c2_baseline_recall).astype(float)
    d_refusal = (c2_patched_refusal - c2_baseline_refusal).astype(float)

    # Saved BEFORE run_E2() is called, not after: dissociation_ratio() can
    # legitimately raise (its own guardrail against a degenerate ratio,
    # not a bug -- see EXECUTION_PLAN.md's job-383261 entry), and the
    # first time it did, nothing about WHY survived the crash -- only
    # e2_c1_only.json existed afterward, forcing a second full GPU run
    # just to see the C2 numbers this run needed anyway. Save the raw
    # inputs to run_E2 unconditionally, so a second guardrail failure
    # (or the first, on a rerun) leaves real evidence on disk instead of
    # requiring a third GPU run to diagnose.
    with open(os.path.join(args.out, "e2_c2_c3_raw.json"), "w") as f:
        json.dump({
            "c2_baseline_recall": c2_baseline_recall.tolist(),
            "c2_patched_recall": c2_patched_recall.tolist(),
            "c2_baseline_refusal": c2_baseline_refusal.tolist(),
            "c2_patched_refusal": c2_patched_refusal.tolist(),
            "d_recall": d_recall.tolist(),
            "d_refusal": d_refusal.tolist(),
            "c3_refusal_delta": c3_refusal_delta.tolist(),
            "c2_baseline_gens_sample": c2_baseline_gens[:20],
            "c2_patched_gens_sample": c2_patched_gens[:20],
        }, f, indent=1, default=str)

    man = RunManifest(outdir=args.out, snapshot_id="see_dataset_snapshot_json",
                      model_name=args.model, seed=args.seed)
    print("\nRunning E2 ...", flush=True)
    e2 = run_E2(d_recall, d_refusal, c1_refusal_by_M, c3_refusal_delta, man)
    e2["C2_n_pairs"] = len(usable_pairs)
    e2["C3_n_queries"] = len(bridge_test_prompts)
    print(json.dumps(e2, indent=1, default=str))

    with open(os.path.join(args.out, "e2_full.json"), "w") as f:
        json.dump(e2, f, indent=1, default=str)
    print(f"\nSaved the full E2 result (C1+C2+C3) to {args.out}/e2_full.json.")


if __name__ == "__main__":
    main()

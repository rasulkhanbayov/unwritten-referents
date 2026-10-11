#!/usr/bin/env python3
"""
skc_run_e5.py
E5: compounding across hops (skc_runbook.txt, Proposition 4). Shows the
uncovered fraction P(M_k=0) rises with chain length k.

USE NESTED CHAINS. skc_runbook.txt is emphatic: each three-hop chain must be
compared against ITS OWN two-hop prefix (same e1, same r1, same r2), not a
separate two-hop query set. Theorem 1's monotonicity clause
(M_{k+1} = M_k * K_{k+1} <= M_k) is a POINTWISE statement that only holds
when the two indicators are computed on the SAME items. This script
enforces that by construction: --queries must be three-hop (k=3) chains,
and the two-hop labels/gates are derived by TRUNCATING each chain to its
first two hops in Python, never by loading a separate file.

CAUTION (also from skc_runbook.txt): Proposition 4 bounds the LEADING TERM
P(M_k=0), which is provably non-decreasing in k. It does NOT prove the
floor itself rises, because the slack terms delta_k, v_k could grow faster.
If the floor FALLS because slack grew faster, that is not a contradiction;
report it as measured. This script prints both floor_grows and
leading_term_grows separately for exactly that reason -- do not conflate them
when writing up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, ComposedQuery, filter_bridge_resolved   # noqa: E402
from skc_gates import FrozenGate                                           # noqa: E402
from skc_experiments import run_E5, RunManifest                            # noqa: E402
from skc_harness import (                                                             # noqa: E402
    load_model, label_queries_batched, bridge_resolved_patchscope,
)


def truncate_to_k2(q: ComposedQuery) -> ComposedQuery:
    """The two-hop PREFIX of a three-hop chain: same qid lineage, first two hops only."""
    if q.k != 3:
        raise ValueError(f"{q.qid}: expected a three-hop chain, got k={q.k}")
    return ComposedQuery(qid=q.qid + "_k2prefix", hops=q.hops[:2],
                         entity_type=q.entity_type,
                         popularity_decile=q.popularity_decile, meta=dict(q.meta))


def capture_surface_only(model, tok, queries_k3, layer):
    """
    For three-hop queries, capture the surface entity (e1) activation once,
    and the two "downstream" activations needed for gate_k2/gate_k3: the
    gate is always read at the SURFACE position; what changes between k=2
    and k=3 is which labels (M_2 vs M_3) it is scored against, not the
    activation itself. capture_for_queries assumes k=2 chains internally
    (it renders "hop1"/"hop2" and q.bridge), so for k=3 chains this function
    captures the surface position directly via the k=3 "composed" template
    instead of reusing that helper.
    """
    from skc_gates import capture_activations
    from skc_harness import entity_spans_for_prompts
    from skc_data import render_prompts
    prompts, entities = [], []
    for q in queries_k3:
        p = render_prompts(q)
        prompts.append(p["composed"]); entities.append(q.e1)
    spans = entity_spans_for_prompts(tok, prompts, entities)
    return capture_activations(model, tok, prompts, layer, position="last_entity_token",
                               entity_spans=spans)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True,
                    help="THREE-HOP (k=3) queries JSON, Wikidata Family F2 chains "
                         "(skc_runbook.txt E5). Two-hop labels are derived from these "
                         "by truncation, never loaded separately.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--primary-estimator", default="diffmeans")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-bridge-resolution", action="store_true", default=False,
                    help="Skip the Patchscopes-based independent bridge-resolution check "
                         "and filter on K1=1 alone. Patchscopes only decodes cleanly at "
                         "layer 16+ on gemma-2-2b (confirmed on this pipeline's own E1 "
                         "investigation); at lower layers it is unreliable, not merely "
                         "unverified. This is a disclosed methodological weakening, not the "
                         "default -- see skc_run_e1.py's identical flag for the full "
                         "rationale, which applies here unchanged.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gate_path = os.path.join(args.gate_dir, f"frozen_gate_{args.primary_estimator}.json")
    if not os.path.exists(gate_path):
        raise SystemExit(f"{gate_path} not found; run skc_run_e0.py (PASSED) first.")
    gate = FrozenGate.load(gate_path)

    queries_k3 = load_queries(args.queries)
    if any(q.k != 3 for q in queries_k3):
        raise SystemExit("--queries must contain only three-hop (k=3) chains.")
    queries_k2 = [truncate_to_k2(q) for q in queries_k3]

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    print(f"Labeling {len(queries_k3)} three-hop chains (gives K1,K2,K3,M for k=3) ...",
         flush=True)
    # hop_require_prefix={1: False, 2: False, 3: False}: hop2/hop3's False
    # is the same fix as skc_run_e1.py/skc_run_e3.py -- hop2/hop3-style
    # prompts reliably name the correct answer inside a descriptive
    # lead-in on real Gemma-2-2b generations. Applied to BOTH hop2 and
    # hop3 here since there is no reason hop3's phrasing ("The {r3} of the
    # {r2} of the {r1} of the {etype} {e1} is", rendered as its own
    # single-hop decomposition) would behave differently from hop2's --
    # confirmed pattern, not re-verified specifically for hop3 by
    # inspection, so treat the first real E5 run as a debugging run for
    # this specifically.
    #
    # hop1's False matches skc_run_e0.py's OWN K1 convention (measured
    # there: strict K1 = 1.0% vs containment K1 = 8.2% on the same
    # generations) -- an identical mismatch was found and fixed in
    # skc_run_e1.py after its K1=1 rate silently disagreed with
    # data/onehop_correct.json's own K1=1 definition (see
    # EXECUTION_PLAN.md). K1 here gates which K1=1 chains
    # bridge_resolved_patchscope() is even attempted on (below), so this
    # script needs the SAME K1 definition as everywhere else in the
    # pipeline, not a stricter local one.
    labels_k3_all = label_queries_batched(
        model, tok, queries_k3, hop_require_prefix={1: False, 2: False, 3: False})
    print(f"Labeling {len(queries_k2)} two-hop prefixes (gives K1,K2,M for k=2) ...",
         flush=True)
    labels_k2_all = label_queries_batched(
        model, tok, queries_k2, hop_require_prefix={1: False, 2: False})

    # bridge_resolved_patchscope() locates the bridge INSIDE the model's own
    # hop1 generation, so it is only meaningful (and only tested) on K1=1
    # chains -- see skc_harness.bridge_resolved_patchscope()'s own docstring
    # and skc_run_e1.py's identical restriction for the full rationale.
    # Calling it on a K1=0 chain would raise (there is no correctly-resolved
    # bridge to find), so those are scored bridge_resolved=False directly.
    k1_by_qid = {l["qid"]: l["K1"] for l in labels_k3_all}
    if args.skip_bridge_resolution:
        print("\n*** --skip-bridge-resolution is ON: filtering on K1=1 ONLY, skipping the "
             "Patchscopes-based independent bridge-resolution check skc_runbook.txt also "
             "requires. This is a disclosed methodological weakening, not the default "
             "behavior -- see this script's module docstring. ***\n", file=sys.stderr)
        labels_k3 = filter_bridge_resolved(labels_k3_all, bridge_resolved=None)
    else:
        print("Checking bridge resolution (Patchscopes) on the K1=1 three-hop chains ...",
             flush=True)
        bridge_resolved = {}
        for q in queries_k3:
            if k1_by_qid[q.qid] != 1:
                bridge_resolved[q.qid] = False
                continue
            try:
                bridge_resolved[q.qid] = bridge_resolved_patchscope(model, tok, q, args.layer)
            except ValueError:
                bridge_resolved[q.qid] = False
        labels_k3 = filter_bridge_resolved(labels_k3_all, bridge_resolved)
    keep_qids_k3 = {l["qid"] for l in labels_k3}
    # keep the SAME underlying chains for k2, matched via the "_k2prefix" qid suffix
    # truncate_to_k2() appends -- this is what "same items" (Theorem 1's pointwise
    # requirement) means operationally, not "a k2 set built the same way".
    labels_k2 = [l for l in labels_k2_all if l["qid"].replace("_k2prefix", "") in keep_qids_k3]

    if not labels_k3 or not labels_k2:
        raise SystemExit("0 chains survived the K1=1 + bridge-resolved filter; cannot proceed.")
    print(f"Retained {len(labels_k3)}/{len(labels_k3_all)} chains after filtering "
         f"(same items used for both k=2 and k=3, per the runbook's nesting requirement).")

    keep_k3 = [q for q in queries_k3 if q.qid in keep_qids_k3]
    keep_k2 = [q for q in queries_k2 if q.qid.replace("_k2prefix", "") in keep_qids_k3]

    print("Capturing surface activations for k=2 and k=3 (same underlying chains) ...",
         flush=True)
    acts_k2 = capture_surface_only(model, tok, keep_k2, args.layer)  # k=2 "composed" template
    acts_k3 = capture_surface_only(model, tok, keep_k3, args.layer)

    gate_k2 = gate.score(acts_k2)
    gate_k3 = gate.score(acts_k3)

    man = RunManifest(outdir=args.out, snapshot_id="see_dataset_snapshot_json",
                      model_name=args.model, seed=args.seed)
    print("Running E5 ...", flush=True)
    e5 = run_E5(labels_k2, labels_k3, gate_k2, gate_k3, man)
    e5["filter"] = "K1=1 only (--skip-bridge-resolution)" if args.skip_bridge_resolution \
        else "K1=1 + bridge-resolved (Patchscopes)"
    print(json.dumps(e5, indent=1, default=str))

    if not e5["leading_term_grows"]:
        print("\n*** P(M=0) did NOT rise with k. This contradicts Proposition 4's leading-"
             "term monotonicity, which is provable, not measured -- this points to a bug "
             "in the nesting (are k=2 and k=3 really the SAME underlying chains?) rather "
             "than a real finding. Investigate before reporting. ***", file=sys.stderr)
    if not e5["floor_grows"]:
        print("\nFloor did not rise even though the leading term did (or comparably) -- "
             "per skc_runbook.txt this is NOT a contradiction, it means slack (delta_k + "
             "v_k) grew faster than the leading term. Report this as measured.")


if __name__ == "__main__":
    main()

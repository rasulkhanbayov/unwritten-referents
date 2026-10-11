#!/usr/bin/env python3
"""
skc_run_e7j_crosstype.py
Control (j), ENTITY-TYPE half only (skc_main.tex, Appendix~\\ref{app:controls}):
"fit the gate on one entity type ... test on held-out types."

Paper's own words flag this as ONE OF THREE LOAD-BEARING controls ("without
which the readout may be relation-specific"). This script covers only the
ENTITY-TYPE side of that sentence, not the RELATION-TYPE side ("... and the
bridge readout on one relation type ... test on held-out types") -- the
two-hop bridge dataset (data/twohop_queries.json) has exactly ONE relation
pair (place_birth -> country) across its entire 3503 items (confirmed
directly: skc_build_dataset_from_ferrando.py's own docstring states
player/city place_birth->country is the ONLY buildable two-hop bridge from
the released Ferrando data), so there is nothing to hold out on the relation
side with current data. DISCLOSED GAP, not silently omitted -- confirmed
directly with the user rather than building a second relation pair first
or skipping this control outright.

PROTOCOL: leave-one-entity-type-out, cycled across all 4 Ferrando entity
types (player, movie, city, song). For each held-out type T:
  1. Fit diffmeans + probe gates on the pooled single-hop FIT split of the
     OTHER 3 types only (T's own single-hop data never touches fitting).
  2. Score AUROC(gate -> K1) on T's own single-hop data, held out entirely.
  3. For comparison, ALSO score the SAME held-out-type data against a gate
     fit the ORDINARY way (pooled across all 4 types, entity-level split,
     mirroring skc_run_e0.py's own protocol) -- the "in-distribution"
     reference point this leave-one-type-out AUROC should be compared
     against, not chance alone. A held-out-type AUROC close to the
     in-distribution one is the direct evidence needed: the gate is not a
     type-specific fit that happens to also generalize by luck.

Reuses skc_run_e0.py's OWN capture_one_hop() (NOT reimplemented -- imported
directly), skc_gates.py's fit_diffmeans/fit_probe, skc_harness.py's
label_queries_batched, skc_metrics.py's auroc -- avoids yet another
instance of "two copies of the same logic diverging", the bug class this
codebase has hit repeatedly (see EXECUTION_PLAN.md for prior examples).

Does NOT touch skc_run_e0.py itself or produce anything skc_experiments.py's
require_E0()/FrozenGate.assert_not_fit_on() would need to recognize -- this
is a STANDALONE diagnostic, its own gates are fit and scored here, never
saved as a FrozenGate consumed by any other experiment, so there is no risk
of an E1/E3/E5 script accidentally loading a gate fit on a type-restricted
split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skc_metrics as M                                                  # noqa: E402
from skc_data import load_queries                                        # noqa: E402
from skc_gates import fit_diffmeans, fit_probe                           # noqa: E402
from skc_harness import load_model, label_queries_batched                # noqa: E402
from skc_run_e0 import capture_one_hop, split_entities                   # noqa: E402


def _prompts_entities(qs, render_prompts_fn):
    prompts, entities = [], []
    for q in qs:
        p = render_prompts_fn(q)
        prompts.append(p["hop1"])
        entities.append(q.e1)
    return prompts, entities


def main():
    from skc_data import render_prompts

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True,
                    help="single-hop queries JSON spanning multiple entity_type values "
                         "(e.g. data/onehop_queries.json)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-per-type", type=int, default=None,
                    help="subsample each entity type to at most this many queries before "
                         "any split, sampled uniformly at random. The full one-hop pool "
                         "is ~92k queries across 4 types; running all of them for a "
                         "diagnostic control is not necessary.")
    ap.add_argument("--fit-frac", type=float, default=0.5,
                    help="within each entity type used for fitting, fraction of entities "
                         "assigned to the fit vs. held-out split (mirrors skc_run_e0.py's "
                         "own --fit-frac)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    queries = load_queries(args.queries)
    if any(q.k != 1 for q in queries):
        raise SystemExit("This control needs SINGLE-HOP (k=1) queries, same requirement "
                         "as skc_run_e0.py.")

    all_types = sorted(set(q.entity_type for q in queries))
    if len(all_types) < 2:
        raise SystemExit(
            f"Only {len(all_types)} entity type(s) present in {args.queries} -- cross-type "
            "generalization needs at least 2 to hold one out against the rest."
        )
    print(f"Entity types found: {all_types}", flush=True)

    by_type: dict[str, list] = {t: [q for q in queries if q.entity_type == t] for t in all_types}
    if args.n_per_type is not None:
        for t in all_types:
            if len(by_type[t]) > args.n_per_type:
                idx = rng.choice(len(by_type[t]), size=args.n_per_type, replace=False)
                by_type[t] = [by_type[t][i] for i in idx]
        print(f"Subsampled each type to at most {args.n_per_type} queries: "
             f"{[(t, len(by_type[t])) for t in all_types]}", flush=True)

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    # Label + capture activations ONCE for every type's full pool -- reused
    # across every leave-one-type-out iteration below, rather than
    # relabeling/recapturing the same queries once per held-out type.
    print("Labeling + capturing activations for every entity type's full pool "
         "(reused across every leave-one-type-out fold below) ...", flush=True)
    acts_by_type: dict[str, np.ndarray] = {}
    k1_by_type: dict[str, np.ndarray] = {}
    for t in all_types:
        qs = by_type[t]
        labels = label_queries_batched(model, tok, qs, hop_require_prefix={1: False})
        k1_by_type[t] = np.array([l["K1"] for l in labels])
        prompts, entities = _prompts_entities(qs, render_prompts)
        acts_by_type[t] = capture_one_hop(model, tok, prompts, entities, args.layer)
        print(f"  {t}: {len(qs)} queries, K1 rate {k1_by_type[t].mean():.1%}", flush=True)

    results = {}
    for held_type in all_types:
        other_types = [t for t in all_types if t != held_type]
        print(f"\n=== Held-out type: {held_type} (fit on {other_types}) ===", flush=True)

        held_acts, held_k1 = acts_by_type[held_type], k1_by_type[held_type]
        if len(set(held_k1.tolist())) < 2:
            print(f"  SKIPPING {held_type}: held-out type has constant K1 "
                 "(cannot compute AUROC there).", file=sys.stderr)
            continue

        # --- Leave-one-type-out fit: pool the OTHER types' activations ---
        pooled_acts = np.concatenate([acts_by_type[t] for t in other_types], axis=0)
        pooled_k1 = np.concatenate([k1_by_type[t] for t in other_types], axis=0)
        known, unknown = pooled_acts[pooled_k1 == 1], pooled_acts[pooled_k1 == 0]
        if len(known) < 10 or len(unknown) < 10:
            print(f"  SKIPPING {held_type}: too few known ({len(known)}) or unknown "
                 f"({len(unknown)}) examples pooled across {other_types} to fit a "
                 "stable direction.", file=sys.stderr)
            continue
        gate_diffmeans = fit_diffmeans(known, unknown, layer=args.layer, model_name=args.model)
        gate_probe = fit_probe(known, unknown, layer=args.layer, model_name=args.model,
                               seed=args.seed)
        auroc_diffmeans_heldout_type = M.auroc(held_k1, gate_diffmeans.score(held_acts))
        auroc_probe_heldout_type = M.auroc(held_k1, gate_probe.score(held_acts))

        # --- In-distribution reference: ordinary pooled fit (ALL 4 types,
        # entity-level split, mirrors skc_run_e0.py's own protocol exactly),
        # scored on a held-out ENTITY subset of held_type specifically, so
        # the comparison is apples-to-apples (same held_type's own data,
        # entity-level held-out either way) rather than comparing across
        # two different held-out populations. ---
        held_type_entities = sorted(set(q.e1 for q in by_type[held_type]))
        fit_ent, ref_held_ent = split_entities(held_type_entities, args.fit_frac, args.seed)
        idx_fit_in_type = [i for i, q in enumerate(by_type[held_type]) if q.e1 in fit_ent]
        idx_held_in_type = [i for i, q in enumerate(by_type[held_type]) if q.e1 in ref_held_ent]
        if len(idx_held_in_type) < 5:
            print(f"  WARNING {held_type}: in-distribution reference held-out subset only "
                 f"has {len(idx_held_in_type)} items -- reference AUROC will be noisy.",
                 file=sys.stderr)
        # in-distribution fit ALSO includes held_type's own fit-half PLUS
        # every other type's full pool, mirroring skc_run_e0.py's own
        # "pool everything in the fit split" protocol.
        indist_known_parts = [pooled_acts[pooled_k1 == 1]]
        indist_unknown_parts = [pooled_acts[pooled_k1 == 0]]
        if idx_fit_in_type:
            fit_in_type_acts = held_acts[idx_fit_in_type]
            fit_in_type_k1 = held_k1[idx_fit_in_type]
            indist_known_parts.append(fit_in_type_acts[fit_in_type_k1 == 1])
            indist_unknown_parts.append(fit_in_type_acts[fit_in_type_k1 == 0])
        indist_known = np.concatenate(indist_known_parts, axis=0)
        indist_unknown = np.concatenate(indist_unknown_parts, axis=0)
        auroc_diffmeans_indist, auroc_probe_indist = None, None
        if len(idx_held_in_type) >= 2 and len(set(held_k1[idx_held_in_type].tolist())) >= 2 \
                and len(indist_known) >= 10 and len(indist_unknown) >= 10:
            gate_diffmeans_indist = fit_diffmeans(indist_known, indist_unknown,
                                                  layer=args.layer, model_name=args.model)
            gate_probe_indist = fit_probe(indist_known, indist_unknown,
                                          layer=args.layer, model_name=args.model,
                                          seed=args.seed)
            ref_acts = held_acts[idx_held_in_type]
            ref_k1 = held_k1[idx_held_in_type]
            auroc_diffmeans_indist = M.auroc(ref_k1, gate_diffmeans_indist.score(ref_acts))
            auroc_probe_indist = M.auroc(ref_k1, gate_probe_indist.score(ref_acts))
        else:
            print(f"  WARNING {held_type}: cannot compute an in-distribution reference "
                 "AUROC (too few held-out-entity items or constant K1 there); "
                 "reporting the leave-one-type-out AUROC alone for this type.",
                 file=sys.stderr)

        results[held_type] = {
            "n_held_out_type": int(len(held_acts)),
            "k1_rate_held_out_type": float(held_k1.mean()),
            "auroc_diffmeans_leave_type_out": auroc_diffmeans_heldout_type,
            "auroc_probe_leave_type_out": auroc_probe_heldout_type,
            "auroc_diffmeans_in_distribution_reference": auroc_diffmeans_indist,
            "auroc_probe_in_distribution_reference": auroc_probe_indist,
        }
        print(json.dumps(results[held_type], indent=1), flush=True)

    out = {
        "control": "(j) cross-type generalization -- ENTITY-TYPE half only",
        "relation_type_half_status": "NOT RUN -- data limitation, disclosed: the two-hop "
                                     "bridge dataset (data/twohop_queries.json) has exactly "
                                     "ONE relation pair (place_birth->country) across all "
                                     "3503 items, so there is nothing to hold out on the "
                                     "relation side with current data.",
        "reading": "For each held-out type, compare auroc_*_leave_type_out against "
                  "auroc_*_in_distribution_reference (same held-out type's own data "
                  "either way). A leave-type-out AUROC close to the in-distribution "
                  "reference is evidence the gate generalizes across entity types rather "
                  "than being a type-specific fit; a large drop would mean the readout "
                  "may be entity-type-specific, exactly what this control checks for.",
        "per_held_out_type": results,
    }
    with open(os.path.join(args.out, "e7j_crosstype.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\n" + json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

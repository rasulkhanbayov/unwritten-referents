#!/usr/bin/env python3
"""
skc_run_e0_layersweep.py
Real per-layer E0 refit sweep -- fits FRESH diffmeans/probe/SAE gates
INDEPENDENTLY at every requested layer (not just layer 8), each from its
own captured activations, each scored for the SAME positive-control
criteria skc_run_e0.py checks at a single layer: held-out AUROC per
estimator, cross-estimator agreement, and named-bridge K2 gate-
visibility. This is the REAL version of "sweep every layer," closing
the gap found when skc_run_e3.py's own hard guard (refuses to score a
layer against a gate fit at a DIFFERENT layer) made a naive "just pass
--layers 0..25" attempt silently only test layer 8 (job 390468 -- see
EXECUTION_PLAN.md).

WHY A SEPARATE SCRIPT, NOT A --layers FLAG BOLTED ONTO skc_run_e0.py:
skc_run_e0.py's own per-layer flow (label -> capture -> fit -> score)
would need to reload the model and relabel the SAME fit/held-out split
26 times if simply looped externally -- the model load and the K1
labeling pass (which does not depend on layer at all) are the expensive
parts, not the per-layer gate-fitting arithmetic itself. This script
loads the model ONCE, labels ONCE, then captures ALL 26 layers'
activations in a SINGLE forward pass per batch
(skc_gates.capture_activations_all_layers(), new this session,
VERIFIED to produce byte-identical activations to 26 separate single-
layer passes on a synthetic multi-layer module before ever touching a
real model), then fits and scores each layer independently from that
one shared capture.

ESTIMATORS PER LAYER: all three (diffmeans, probe, SAE) -- confirmed
directly (not assumed) that Gemma Scope's "gemma-scope-2b-pt-res-canonical"
release has a canonical width-16k SAE dictionary at EVERY ONE of
Gemma-2-2b's 26 layers (checked via
sae_lens.loading.pretrained_saes_directory.get_pretrained_saes_directory()
directly), so there is no coverage gap requiring --skip-sae at any
layer for this specific model.

PROTOCOL: identical to skc_run_e0.py's own single-layer protocol at
every layer -- same --n subsample, same --fit-frac entity-level split,
same K1 labeling, same 0.65 AUROC / 0.50 agreement bars, same named-
bridge K2 check. This is deliberately NOT a lighter-weight or
subsampled version; the per-layer sweep needs the same rigor as any
single E0 run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts                                     # noqa: E402
from skc_gates import (                                                                # noqa: E402
    fit_diffmeans, fit_probe, fit_sae_gate, capture_activations_all_layers,
)
from skc_experiments import run_E0, RunManifest                                        # noqa: E402
from skc_harness import load_model, load_sae, entity_spans_for_prompts, label_queries_batched  # noqa: E402
from skc_run_e0 import split_entities                                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True,
                    help="single-hop queries JSON, same schema as skc_run_e0.py's own")
    ap.add_argument("--named-bridge-queries", required=True,
                    help="two-hop queries JSON, same role as skc_run_e0.py's own")
    ap.add_argument("--model", required=True, choices=[
        "gemma-2-2b", "gemma-2-9b", "llama-3.1-8b", "llama-3.1-70b", "qwen3-32b"],
                    help="Originally restricted to gemma-2-2b (the only model with "
                         "confirmed FULL per-layer SAE coverage). Extended to the other "
                         "4 models: this is safe WITHOUT a separate coverage check "
                         "because the per-layer SAE fit below is already wrapped in its "
                         "own try/except (see the 'SAE fitting failed' fallback) and "
                         "degrades to diffmeans+probe only for whichever layers lack "
                         "coverage -- it does not silently skip the whole run or assume "
                         "coverage transfers. llama-3.1-70b/qwen3-32b have NO public SAE "
                         "at any layer (skc_harness.load_sae raises ValueError, caught "
                         "the same way) -- every layer sweeps diffmeans+probe only for "
                         "those two, matching their own existing --skip-sae convention.")
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="layers to sweep, e.g. --layers 0 1 2 ... 25 for the full sweep")
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-named-bridge", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    queries = load_queries(args.queries)
    if any(q.k != 1 for q in queries):
        raise SystemExit("E0 layer sweep requires SINGLE-HOP (k=1) queries, same as "
                         "skc_run_e0.py.")
    if args.n is not None and len(queries) > args.n:
        idx = rng.choice(len(queries), size=args.n, replace=False)
        queries = [queries[i] for i in idx]
        print(f"Subsampled --queries to {len(queries)} of the original set.", flush=True)

    # Split-leakage fix (see skc_run_e0.py's split_entities() and its own
    # docstring for the measured 47.5% bridge-entity overlap this closes).
    # This script previously called split_entities() WITHOUT
    # exclude_from_fit despite already requiring --named-bridge-queries as
    # an argument -- it loaded the file later (for K2 labeling) but never
    # consulted it for the split itself, so every layer-sweep result
    # produced by this script before this fix has the same leak as
    # skc_run_e0.py's own pre-fix results. Loading --named-bridge-queries
    # here (earlier than its other use below) is a second, independent
    # load of the same file -- harmless, avoids restructuring the K2
    # labeling code further down.
    nb_queries_for_split = load_queries(args.named_bridge_queries)
    if any(q.k < 2 for q in nb_queries_for_split):
        raise SystemExit("--named-bridge-queries must be two-hop (k=2) chains.")
    exclude_entities: set[str] = set()
    for q in nb_queries_for_split:
        exclude_entities.add(q.e1)
        exclude_entities.add(q.bridge)
        exclude_entities.update(q.target_answers)
    print(f"Excluding {len(exclude_entities)} entities from the fit split (every head/bridge/"
         "target entity in --named-bridge-queries) to close the split-leakage gap.",
         flush=True)

    entity_names = [q.e1 for q in queries]
    fit_set, held_set = split_entities(entity_names, args.fit_frac, args.seed,
                                       exclude_from_fit=exclude_entities)
    fit_q = [q for q in queries if q.e1 in fit_set]
    held_q = [q for q in queries if q.e1 in held_set]
    print(f"fit entities: {len(fit_set)} ({len(fit_q)} queries), "
         f"held-out entities: {len(held_set)} ({len(held_q)} queries)", flush=True)

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    print("Labeling fit + held-out single-hop prompts for K1 (ONCE, shared across every "
         "layer) ...", flush=True)
    fit_labels = label_queries_batched(model, tok, fit_q, hop_require_prefix={1: False})
    held_labels = label_queries_batched(model, tok, held_q, hop_require_prefix={1: False})
    fit_k1 = np.array([l["K1"] for l in fit_labels])
    held_k1 = np.array([l["K1"] for l in held_labels])
    if len(set(fit_k1.tolist())) < 2:
        raise SystemExit("fit split has a constant K1 -- cannot fit known/unknown gates.")
    if len(set(held_k1.tolist())) < 2:
        raise SystemExit("held-out split has a constant K1 -- cannot evaluate AUROC there.")

    def prompts_entities(qs):
        prompts, entities = [], []
        for q in qs:
            p = render_prompts(q)
            prompts.append(p["hop1"]); entities.append(q.e1)
        return prompts, entities

    fit_prompts, fit_entities = prompts_entities(fit_q)
    held_prompts, held_entities = prompts_entities(held_q)
    fit_spans = entity_spans_for_prompts(tok, fit_prompts, fit_entities)
    held_spans = entity_spans_for_prompts(tok, held_prompts, held_entities)

    nb_queries = load_queries(args.named_bridge_queries)
    if any(q.k != 2 for q in nb_queries):
        raise SystemExit("--named-bridge-queries must be two-hop (k=2) chains.")
    if args.n_named_bridge is not None and len(nb_queries) > args.n_named_bridge:
        idx = rng.choice(len(nb_queries), size=args.n_named_bridge, replace=False)
        nb_queries = [nb_queries[i] for i in idx]
    print(f"Labeling {len(nb_queries)} named-bridge queries for K2 (ONCE, shared across "
         "every layer) ...", flush=True)
    nb_labels = label_queries_batched(model, tok, nb_queries, hop_require_prefix={2: False})
    k2_named = np.array([l["K2"] for l in nb_labels])
    nb_prompts = [render_prompts(q)["named_bridge"] for q in nb_queries]
    nb_entities = [q.bridge for q in nb_queries]
    nb_spans = entity_spans_for_prompts(tok, nb_prompts, nb_entities)

    print(f"Capturing ALL {len(args.layers)} layers in a SINGLE forward pass per batch "
         "(fit split) ...", flush=True)
    fit_acts_by_layer = capture_activations_all_layers(
        model, tok, fit_prompts, args.layers, position="last_entity_token",
        entity_spans=fit_spans)
    print("Capturing all layers (held-out split) ...", flush=True)
    held_acts_by_layer = capture_activations_all_layers(
        model, tok, held_prompts, args.layers, position="last_entity_token",
        entity_spans=held_spans)
    print("Capturing all layers (named-bridge queries) ...", flush=True)
    nb_acts_by_layer = capture_activations_all_layers(
        model, tok, nb_prompts, args.layers, position="last_entity_token",
        entity_spans=nb_spans)

    results = {}
    for layer in args.layers:
        print(f"\n=== Layer {layer} ===", flush=True)
        fit_acts = fit_acts_by_layer[layer]
        held_acts = held_acts_by_layer[layer]
        nb_acts = nb_acts_by_layer[layer]
        known_acts, unknown_acts = fit_acts[fit_k1 == 1], fit_acts[fit_k1 == 0]
        if len(known_acts) < 10 or len(unknown_acts) < 10:
            print(f"  SKIPPING layer {layer}: too few known ({len(known_acts)}) or "
                 f"unknown ({len(unknown_acts)}) fit examples.", file=sys.stderr)
            continue

        gates = {
            "diffmeans": fit_diffmeans(known_acts, unknown_acts, layer=layer,
                                       model_name=args.model),
            "probe": fit_probe(known_acts, unknown_acts, layer=layer,
                              model_name=args.model, seed=args.seed),
        }
        try:
            import torch
            sae, cfg = load_sae(args.model, layer)
            fit_acts_t = torch.tensor(np.asarray(fit_acts, dtype=np.float32),
                                      device=sae.device)
            with torch.no_grad():
                fit_sae_acts = sae.encode(fit_acts_t).float().cpu().numpy()
            sae_by_type: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for etype in sorted(set(q.entity_type for q in fit_q)):
                idx = np.array([i for i, q in enumerate(fit_q) if q.entity_type == etype])
                type_k1 = fit_k1[idx]
                if len(set(type_k1.tolist())) < 2:
                    continue
                type_acts = fit_sae_acts[idx]
                sae_by_type[etype] = (type_acts[type_k1 == 1], type_acts[type_k1 == 0])
            if sae_by_type:
                gates["sae"] = fit_sae_gate(
                    sae_by_type, sae.W_dec.float().detach().cpu().numpy(),
                    layer=layer, model_name=args.model)
            else:
                print(f"  layer {layer}: every entity type has constant K1 in the fit "
                     "split -- skipping SAE for this layer only.", file=sys.stderr)
            del sae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  layer {layer}: SAE fitting failed ({e}) -- proceeding with "
                 "diffmeans+probe only for this layer.", file=sys.stderr)

        e0 = run_E0(gates, held_acts, held_k1, nb_acts, k2_named,
                   RunManifest(outdir=args.out, snapshot_id="see_dataset_snapshot_json",
                               model_name=args.model, seed=args.seed))
        results[layer] = e0
        print(json.dumps(e0, indent=1, default=str), flush=True)

    with open(os.path.join(args.out, "e0_layersweep_full.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)

    passing = {l: r for l, r in results.items() if r["gate_open"]}
    print(f"\n{len(passing)}/{len(results)} layers passed the full stop-gate "
         f"(n_estimators_passing>=2 AND agreement>=0.50): {sorted(passing.keys())}")
    print(f"Full per-layer results are in {args.out}/e0_layersweep_full.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skc_run_pilot.py
The cheap pilot from skc_runbook.txt section 1. Run this FIRST, before
anything else that touches a GPU.

Scope (fixed by the runbook, do not silently expand it):
  Gemma-2-2B only. 300 two-hop queries. One SAE layer -- the layer Ferrando
  et al. (2025) report as the peak for entity latents.

Steps P1-P4 from the runbook:
  P1. Label K1, K2, Y on the 300 queries by alias-aware exact match.
  P2. Report the joint distribution of (K1, K2, Y).
  P3. Extract the gate at the surface entity position (diffmeans estimator --
      no SAE fitting needed for a same-afternoon pilot; the SAE estimator is
      exercised properly in E0). Compute AUC(S -> K1) and AUC(S -> K2).
  P4. Run floor_report() from skc_metrics.

Verdict, per the runbook:
  ALIVE     AUC(S->K1) clearly above chance AND AUC(S->K2) near chance.
            Proceed to the full protocol (E0 onward).
  DEAD      AUC(S->K1) also near chance. The gate is not working; go back
            to E0 diagnostics, do not proceed to the full protocol.
  INVERTED  AUC(S->K2) clearly above chance. H1 is false. This is a REAL
            OUTCOME, not a failure -- flag back immediately per
            skc_runbook.txt's flag-back triggers. Do not slice the data
            until it goes away.

THE PILOT IS NOT A RESULT. Nothing this script prints belongs in a paper
table. It exists only to decide whether to spend the following six weeks
on the full protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skc_metrics as M                                    # noqa: E402
from skc_data import load_queries, filter_bridge_resolved   # noqa: E402
from skc_gates import fit_diffmeans                          # noqa: E402
from skc_harness import (                                    # noqa: E402
    load_model, entity_spans_for_prompts, label_queries_batched,
    save_activation_cache, capture_for_queries,
)


# Ferrando et al. (2025) report layer 12 as the peak for Gemma-2-2B entity
# latents on the residual stream. >>> TO RUN: confirm this against the
# published paper's exact reported peak layer before trusting it; this
# script uses a named constant so that confirmation is a one-line change.
PEAK_LAYER_GEMMA2_2B = 12


def joint_distribution(labels: list[dict]) -> dict:
    k1 = np.array([l["K1"] for l in labels])
    k2 = np.array([l["K2"] for l in labels])
    y = np.array([l["Y"] for l in labels])
    table = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                table[f"K1={a},K2={b},Y={c}"] = int(((k1 == a) & (k2 == b) & (y == c)).sum())
    n_k1_1_k2_0 = int(((k1 == 1) & (k2 == 0)).sum())
    return {"table": table, "n_total": len(labels),
           "n_K1=1_K2=0": n_k1_1_k2_0,
           "warning": None if n_k1_1_k2_0 >= 20 else (
               f"only {n_k1_1_k2_0} items have K1=1,K2=0 -- runbook says this cell "
               "must not be nearly empty. Resample toward less popular bridge "
               "entities and repeat before trusting AUC(S->K2)."
           )}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True,
                    help="path to a skc_data.save_queries() JSON with >=300 two-hop queries")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--layer", type=int, default=PEAK_LAYER_GEMMA2_2B)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    queries = load_queries(args.queries)
    if len(queries) < args.n:
        raise SystemExit(f"only {len(queries)} queries available, need >= {args.n}")
    idx = rng.choice(len(queries), size=args.n, replace=False)
    queries = [queries[i] for i in idx]

    print(f"Loading google/gemma-2-2b ...", flush=True)
    model, tok = load_model("gemma-2-2b")

    print(f"Labeling {len(queries)} queries (P1) ...", flush=True)
    gens_path = os.path.join(args.out, "raw_generations.json")
    # hop_require_prefix={2: False}: confirmed on this pilot's own
    # raw_generations.json that hop2 ("The country of the city X is")
    # answers are substantively correct but named country appears embedded
    # in a descriptive sentence ("located in the state of Indiana...")
    # rather than as the generation's first token(s), in a messier variety
    # of lead-ins than hop1 shows and not reducible to a short filler-phrase
    # list. Loosening to containment is safe here specifically because
    # max_new_tokens=24 bounds how much text there is for a stray country
    # name to hide in; hop1 (K1) and the composed answer (Y) keep the strict
    # default. See skc_data.compute_labels's docstring for the full reasoning.
    labels = label_queries_batched(model, tok, queries, save_generations_path=gens_path,
                                   hop_require_prefix={2: False})
    print(f"Raw prompts+generations saved to {gens_path} -- inspect these directly if "
         "K1/K2 come back implausibly low; a binary label alone cannot tell you whether "
         "the model is wrong, the scorer rejected a correct-looking answer, or the "
         "template does not elicit an answer at all.", flush=True)

    print("Joint distribution (P2):", flush=True)
    jd = joint_distribution(labels)
    print(json.dumps(jd, indent=1))
    if jd["warning"]:
        print(f"\nWARNING: {jd['warning']}\n", file=sys.stderr)

    # P3: gate at the surface entity position via difference-of-means.
    # Fit the direction on a SEPARATE single-hop known/unknown split, not on
    # the 300 pilot queries themselves -- fitting and scoring on the same
    # data would make "the pilot passed" mean nothing. This pilot fits on
    # the K1 label taken from the pilot set's own hop1 activations, split by
    # the model's own hop1 correctness, which is the closest same-afternoon
    # approximation to E0's proper fit/held-out split. Do not reuse this
    # shortcut in E0 itself -- E0 requires a genuine fit/held-out entity split.
    print("Capturing surface + hop1 + hop2 activations ...", flush=True)
    acts_surface, acts_hop1, acts_hop2 = capture_for_queries(
        model, tok, queries, layer=args.layer)

    k1 = np.array([l["K1"] for l in labels])
    k2 = np.array([l["K2"] for l in labels])
    y = np.array([l["Y"] for l in labels])

    if len(set(k1.tolist())) < 2 or (k1 == 0).sum() < 10 or (k1 == 1).sum() < 10:
        raise SystemExit(
            f"K1 split is too small or degenerate for this pilot sample "
            f"(K1=1: {(k1 == 1).sum()}, K1=0: {(k1 == 0).sum()}) -- cannot fit or "
            "evaluate a gate. Resample toward a harder entity mix; this is a data "
            "problem, not a model problem."
        )
    # Fit on the FIRST HALF of the pilot set's hop1 activations, split by K1,
    # score on the FULL set's surface activations. This is a same-afternoon
    # approximation of a fit/held-out split, not the genuine single-hop
    # fit/held-out ENTITY split E0 requires -- do not reuse this shortcut in
    # E0 itself; fitting and scoring here still overlap in entities.
    half = len(queries) // 2
    fit_known = acts_hop1[:half][k1[:half] == 1]
    fit_unknown = acts_hop1[:half][k1[:half] == 0]
    if len(fit_known) < 5 or len(fit_unknown) < 5:
        raise SystemExit("too few known/unknown examples in the fit half; resample.")
    gate = fit_diffmeans(fit_known, fit_unknown, layer=args.layer, model_name="gemma-2-2b")

    auc_s_k1 = M.auroc(k1, gate.score(acts_surface))
    auc_s_k2 = M.auroc(k2, gate.score(acts_surface))
    print(f"\nAUC(S -> K1) = {auc_s_k1:.3f}")
    print(f"AUC(S -> K2) = {auc_s_k2:.3f}")

    if auc_s_k1 >= 0.65 and auc_s_k2 < 0.60:
        verdict = "ALIVE"
    elif auc_s_k1 < 0.60:
        verdict = "DEAD"
    elif auc_s_k2 >= 0.65:
        verdict = "INVERTED"
    else:
        verdict = "AMBIGUOUS"
    print(f"\nVERDICT: {verdict}")
    if verdict == "INVERTED":
        print("\n*** FLAG BACK IMMEDIATELY. H1 may be false. Do not proceed to tune "
              "this away. See skc_runbook.txt 'FLAG-BACK TRIGGERS'. ***", file=sys.stderr)
    elif verdict == "DEAD":
        print("\n*** Gate is not separating K1. Go back to E0 diagnostics; do not "
              "proceed to the full protocol. ***", file=sys.stderr)

    # P4
    floor = M.floor_report(y, k2, gate.score(acts_surface), seed=args.seed)
    print("\nfloor_report (P4):")
    print(json.dumps(floor, indent=1))

    save_activation_cache(os.path.join(args.out, "acts_surface.npy"), acts_surface,
                          {"model": "gemma-2-2b", "layer": args.layer, "n": len(queries)})

    result = {"verdict": verdict, "auc_S_K1": auc_s_k1, "auc_S_K2": auc_s_k2,
             "joint_distribution": jd, "floor_report": floor,
             "layer": args.layer, "n": len(queries), "seed": args.seed,
             "note": "PILOT ONLY. Nothing here belongs in a paper table."}
    with open(os.path.join(args.out, "pilot_result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"\nWrote {os.path.join(args.out, 'pilot_result.json')}")


if __name__ == "__main__":
    main()

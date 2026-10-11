#!/usr/bin/env python3
"""
skc_run_e0.py
E0: the positive control and STOP GATE (skc_runbook.txt).

Do not skip or rush this. If it fails, a null result in E1 is
uninterpretable: "the model is blind" and "my probe does not work" produce
identical E1 output. require_E0() raises if fewer than 2 of 3 estimators
clear AUROC 0.65 on held-out entities, or if the estimators disagree
(min pairwise Spearman < 0.50). If it raises, STOP. Do not tune the gate
on data you will test on later -- that destroys the E1 result before it is
run. Flag back per skc_runbook.txt's FLAG-BACK TRIGGERS instead.

Data (per skc_runbook.txt / skc_REDFLAGS.md item 8):
  One-hop prompts only, over the four Ferrando et al. entity types
  (player, movie, city, song) plus TwoHopFact one-hop counterparts.
  Entities are split fit / held-out and NEVER reused across the split --
  this script enforces that split at the entity-name level, not just the
  query level, because two different queries can share an entity.

Fits all THREE estimators (SAE, diffmeans, probe) on the FIT split, then
freezes them (FrozenGate objects, saved to disk). Everything downstream
(E1, E3, ...) loads these same frozen gates and must never refit them --
FrozenGate.assert_not_fit_on() is the mechanical guard for that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts                                   # noqa: E402
from skc_gates import fit_diffmeans, fit_probe, fit_sae_gate, capture_activations    # noqa: E402
from skc_experiments import run_E0, require_E0, RunManifest                          # noqa: E402
from skc_harness import (                                                            # noqa: E402
    load_model, load_sae, entity_spans_for_prompts, save_activation_cache,
)


def split_entities(entity_names: list[str], fit_frac: float, seed: int,
                   exclude_from_fit: set[str] | None = None):
    """
    Entity-level fit/held-out split. Splitting at the entity level (not the
    query level) is what stops the same entity's known/unknown activation
    from leaking across the split -- a query-level split would silently
    violate the "never reuse" rule in skc_runbook.txt whenever the same
    entity appears in more than one query.

    exclude_from_fit: entities that must NEVER land in the fit set, because
    they will be evaluated downstream (as a two-/three-hop chain's own
    BRIDGE or TARGET entity, read back at its latent position by the same
    frozen direction) rather than fit on. This closes a real, measured
    leakage gap: this function used to split ONLY the entities named in
    its own `entity_names` argument (historically just the one-hop file's
    head entities), with no cross-check against the separate two-/three-hop
    files E1/E5 actually evaluate on. Direct measurement against the real
    data/onehop_queries.json and data/twohop_queries.json found 402 of 847
    (47.5%) of the two-hop set's own bridge entities were landing in the
    fit set under the old split -- for every one of those, the frozen gate
    direction was fit using that exact entity's own known/unknown label,
    then evaluated by reading the same direction back at that entity's own
    bridge position. This is exactly the leakage vector that would explain
    the headline result rather than a genuine transfer effect. Passing
    every downstream file's own head/bridge/target entities here as
    exclude_from_fit prevents it structurally: those entities are removed
    from the candidate pool before the fit/held partition is drawn, so they
    can only ever land in held_set (or nowhere, if excluded outright).
    """
    excl = exclude_from_fit or set()
    uniq = sorted(set(entity_names) - excl)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_fit = max(1, int(round(len(uniq) * fit_frac)))
    fit_set, held_set = set(uniq[:n_fit]), set(uniq[n_fit:]) | (excl & set(entity_names))
    if not held_set:
        raise ValueError("held-out entity set is empty; increase --n or lower --fit-frac")
    return fit_set, held_set


def capture_one_hop(model, tok, prompts, entities, layer, batch_size=16):
    spans = entity_spans_for_prompts(tok, prompts, entities)
    return capture_activations(model, tok, prompts, layer, position="last_entity_token",
                               entity_spans=spans, batch_size=batch_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True,
                    help="single-hop queries JSON (k=1 chains: one Hop each, entity_type in "
                         "{player,movie,city,song} or a TwoHopFact one-hop counterpart)")
    ap.add_argument("--named-bridge-queries", required=True,
                    help="two-hop queries JSON, used to render the named_bridge template and "
                         "check K2 gate-visibility when the bridge IS written (E0 part c), AND "
                         "(as of the split-leakage fix) to exclude its own head/bridge/target "
                         "entities from the fit split -- these are the entities E1 evaluates the "
                         "frozen gate on, so none of them may be used to fit it. See "
                         "split_entities()'s own docstring for the measured leakage this closes.")
    ap.add_argument("--threehop-queries", default=None,
                    help="three-hop queries JSON (E5/compounding's own evaluation set). Optional, "
                         "but STRONGLY recommended whenever this gate will also be used for the "
                         "compounding experiment: its head/bridge/target entities are excluded "
                         "from the fit split for the same reason as --named-bridge-queries's own. "
                         "Omitting this when E5 will reuse this gate re-opens the same leakage "
                         "vector for three-hop chains specifically.")
    ap.add_argument("--model", required=True, choices=[
        "gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b", "gemma-2-9b-it",
        "llama-3.1-8b", "llama-3.1-8b-it", "llama-3.1-70b", "qwen3-32b",
    ])
    # -it (instruction-tuned) variants added for skc_run_e6.py's own need:
    # a gate fit on the BASE model is not automatically valid for scoring
    # the INSTRUCTION-TUNED model's activations, since instruction tuning
    # can shift residual-stream geometry -- 10_e6_cot_gemma2b_it.sh's own
    # comment flags this explicitly. Re-running E0 on the exact -it model
    # E6 scores is the correct fix, not reusing the base model's gate.
    #
    # llama-3.1-70b/qwen3-32b WERE excluded here entirely, on the reasoning
    # that they are "4-bit scale-check only, no SAE" -- but skc_runbook.txt's
    # own wording is "difference-of-means and probes only", not "skip E0
    # entirely". E0's stop-gate function (does the gate separate known
    # from unknown on held-out entities, do the estimators agree) is still
    # exactly as necessary for these two models as any other -- "no SAE"
    # only changes which estimators are available, not whether E0's own
    # precondition check should run. A real GPU run (job 384008) hit this
    # exact gap: the driver refused these two models outright via argparse,
    # even with --skip-sae passed. Fixed by adding them here; --skip-sae
    # remains mandatory for them (enforced below), which is the correct
    # reading of skc_runbook.txt's actual sentence.
    ap.add_argument("--layer", type=int, required=True,
                    help="residual-stream layer to probe; sweep this across a small grid "
                         "before committing, per skc_runbook.txt's SAE width/layer note")
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=None,
                    help="subsample --queries to at most this many single-hop items "
                         "BEFORE the fit/held-out split, sampled uniformly at random. "
                         "The full Ferrando-derived one-hop set is ~92k queries -- "
                         "running all of them is a multi-hour, expensive generation "
                         "job on a shared cluster and is not what a stop-gate check "
                         "needs. Omit only when you specifically want the full set.")
    ap.add_argument("--n-named-bridge", type=int, default=None,
                    help="same subsampling, applied to --named-bridge-queries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-sae", action="store_true",
                    help="fit only diffmeans+probe (2 of 3 estimators). Use ONLY for the "
                         "32B/70B scale check where no public SAE exists -- E0's own gate "
                         "requires >=2/3 passing, so this is still a valid (if weaker) run, "
                         "but say explicitly in the paper that no SAE estimator was available.")
    args = ap.parse_args()
    if args.model in ("llama-3.1-70b", "qwen3-32b") and not args.skip_sae:
        raise SystemExit(
            f"{args.model!r} has no public SAE dictionary (skc_runbook.txt's scale-check "
            "note) -- pass --skip-sae explicitly. This is enforced rather than silently "
            "defaulted, so the paper's own text about which models have SAE coverage "
            "is never contradicted by an accidental run."
        )
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    queries = load_queries(args.queries)
    if any(q.k != 1 for q in queries):
        raise SystemExit(
            "E0 requires SINGLE-HOP (k=1) queries. This script found a k>1 chain in "
            f"--queries {args.queries}; build a dedicated one-hop set (see skc_build_dataset.py) "
            "rather than reusing a two-hop file's hop1 sub-prompts, so that entity coverage "
            "matches the Ferrando et al. entity types directly."
        )
    if args.n is not None and len(queries) > args.n:
        idx = rng.choice(len(queries), size=args.n, replace=False)
        queries = [queries[i] for i in idx]
        print(f"Subsampled --queries to {len(queries)} of the original set.")

    # Split-leakage fix (see split_entities()'s own docstring for the
    # measured 47.5% bridge-entity overlap this closes). Every entity that
    # will be READ (as a bridge or target, at its own latent position) by
    # E1/E5 downstream must be excluded from the fit split -- collected
    # here from the SAME two files this script already loads for other
    # reasons (--named-bridge-queries) plus the optional --threehop-queries,
    # rather than assuming disjointness the way the previous version of
    # this script silently did.
    exclude_entities: set[str] = set()
    nb_queries_for_split = load_queries(args.named_bridge_queries)
    for q in nb_queries_for_split:
        exclude_entities.add(q.e1)
        if q.k >= 2:
            exclude_entities.add(q.bridge)
            exclude_entities.update(q.target_answers)
    if args.threehop_queries:
        th_queries_for_split = load_queries(args.threehop_queries)
        for q in th_queries_for_split:
            exclude_entities.add(q.e1)
            if q.k >= 2:
                exclude_entities.add(q.bridge)
            exclude_entities.update(q.target_answers)
    print(f"Excluding {len(exclude_entities)} entities from the fit split (every head/bridge/"
         "target entity in the two-/three-hop evaluation files) to close the split-leakage gap.",
         flush=True)

    entity_names = [q.e1 for q in queries]
    fit_set, held_set = split_entities(entity_names, args.fit_frac, args.seed,
                                       exclude_from_fit=exclude_entities)
    fit_q = [q for q in queries if q.e1 in fit_set]
    held_q = [q for q in queries if q.e1 in held_set]
    print(f"fit entities: {len(fit_set)} ({len(fit_q)} queries), "
         f"held-out entities: {len(held_set)} ({len(held_q)} queries)")

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model(args.model)

    # "Known" vs "unknown" per entity requires the model's OWN answer to the
    # single-hop prompt, i.e. we must generate first to get K1, then capture
    # activations at the SAME prompt. Label first.
    #
    # hop_require_prefix={1: False}: confirmed by inspecting this exact fit
    # split's raw generations (fit_raw_generations.json from an earlier E0
    # run) that require_prefix=True's strict lead-in matching does not
    # generalize across the four Ferrando entity types the way the pilot's
    # narrow filler-phrase fix did for player.place_birth alone --
    # city.country answers are overwhelmingly phrased "located in the
    # north-east of Romania, in the county of..." (no filler word this
    # driver's filler-strip regex covers), and player/movie/song show a mix
    # of the same pattern plus genuinely wrong answers. Measured directly:
    # strict K1 = 19/1999 (1.0%), containment K1 = 163/1999 (8.2%) on the
    # SAME generations -- city alone went from 0.0% to 33.0%. Still low
    # overall (most of the remaining "wrong" is real model error, confirmed
    # by inspection, not a further scoring gap), but no longer near-total
    # class collapse. max_new_tokens=24 (the default) bounds how much text
    # there is for a stray correct-looking substring to hide in, same
    # justification as hop2's require_prefix=False in the pilot.
    from skc_harness import label_queries_batched
    print("Labeling fit + held-out single-hop prompts for K1 ...", flush=True)
    fit_gens_path = os.path.join(args.out, "fit_raw_generations.json")
    held_gens_path = os.path.join(args.out, "held_raw_generations.json")
    fit_labels = label_queries_batched(model, tok, fit_q, save_generations_path=fit_gens_path,
                                       hop_require_prefix={1: False}) if fit_q else []
    held_labels = label_queries_batched(model, tok, held_q, save_generations_path=held_gens_path,
                                        hop_require_prefix={1: False}) if held_q else []

    def hop1_prompts_entities(qs):
        prompts, entities = [], []
        for q in qs:
            p = render_prompts(q)
            prompts.append(p["hop1"]); entities.append(q.e1)
        return prompts, entities

    fit_prompts, fit_entities = hop1_prompts_entities(fit_q)
    held_prompts, held_entities = hop1_prompts_entities(held_q)

    print("Capturing fit-split activations ...", flush=True)
    fit_acts = capture_one_hop(model, tok, fit_prompts, fit_entities, args.layer)
    print("Capturing held-out-split activations ...", flush=True)
    held_acts = capture_one_hop(model, tok, held_prompts, held_entities, args.layer)

    fit_k1 = np.array([l["K1"] for l in fit_labels])
    held_k1 = np.array([l["K1"] for l in held_labels])
    if len(set(fit_k1.tolist())) < 2:
        raise SystemExit("fit split has a constant K1 -- cannot fit known/unknown gates. "
                         "Your entity mix is too easy or too hard; resample.")
    if len(set(held_k1.tolist())) < 2:
        raise SystemExit("held-out split has a constant K1 -- cannot evaluate AUROC there.")

    # Per-entity-type K1 rate, printed for diagnosis. If one entity type
    # (e.g. movie, with much longer/harder attribute answers than player)
    # dominates the "unknown" bucket or vice versa, the fitted direction may
    # be partly a TYPE detector rather than a knowledge detector -- exactly
    # the confound skc_gates.py's own max-min SAE-latent-selection logic is
    # built to avoid for the SAE estimator, but diffmeans/probe here are
    # fit on the pooled mix with no such per-type control.
    print("Per-entity-type K1 rate (fit split):")
    for etype in sorted(set(q.entity_type for q in fit_q)):
        idx = [i for i, q in enumerate(fit_q) if q.entity_type == etype]
        rate = fit_k1[idx].mean() if idx else float("nan")
        print(f"  {etype}: {fit_k1[idx].sum()}/{len(idx)} ({rate:.1%})")

    known_acts, unknown_acts = fit_acts[fit_k1 == 1], fit_acts[fit_k1 == 0]
    if len(known_acts) < 10 or len(unknown_acts) < 10:
        raise SystemExit(
            f"too few known ({len(known_acts)}) or unknown ({len(unknown_acts)}) fit "
            "examples for a stable direction; enlarge --queries or rebalance the entity mix."
        )

    print("Fitting gate estimators on the FIT split, then freezing ...", flush=True)
    gates = {
        "diffmeans": fit_diffmeans(known_acts, unknown_acts, layer=args.layer, model_name=args.model),
        "probe": fit_probe(known_acts, unknown_acts, layer=args.layer, model_name=args.model, seed=args.seed),
    }
    if not args.skip_sae:
        import torch
        sae, cfg = load_sae(args.model, args.layer)
        # sae.encode(x) takes a plain tensor of shape (..., d_in) and returns
        # (..., d_sae) -- verified directly against the installed sae_lens
        # (6.5.3) source (SAE.encode's signature and sae_lens/evals.py's own
        # usage), not guessed. fit_acts is the SAME raw residual-stream
        # activations already captured above for diffmeans/probe -- the SAE
        # estimator reads the identical positions, just through the
        # dictionary's encoder, so no separate capture pass is needed.
        fit_acts_t = torch.tensor(np.asarray(fit_acts, dtype=np.float32), device=sae.device)
        with torch.no_grad():
            # .float() before .cpu().numpy(): sae.encode() internally casts
            # its input to the SAE's OWN dtype (sae_lens/saes/sae.py's
            # encode(): "sae_in = sae_in.to(self.dtype)") and does not cast
            # the output back -- confirmed as a real crash on a GPU run
            # (job 384007): Llama Scope's SAE is bfloat16 (Gemma Scope's
            # happened to return float32 regardless, which is why this
            # worked there and not here), and numpy has no bfloat16 dtype
            # at all, so `.cpu().numpy()` on a bfloat16 tensor raises
            # "TypeError: Got unsupported ScalarType BFloat16" unconditionally,
            # not something narrower to catch. `.float()` here is
            # correct for BOTH SAE architectures, not a Llama-specific patch.
            fit_sae_acts = sae.encode(fit_acts_t).float().cpu().numpy()
        sae_by_type: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for etype in sorted(set(q.entity_type for q in fit_q)):
            idx = np.array([i for i, q in enumerate(fit_q) if q.entity_type == etype])
            type_k1 = fit_k1[idx]
            if len(set(type_k1.tolist())) < 2:
                print(f"WARNING: entity type {etype!r} has constant K1 in the fit split "
                     "-- skipping it for SAE max-min latent selection (a single-class "
                     "type contributes no separation signal and would make max-min "
                     "select on the remaining types alone).", file=sys.stderr)
                continue
            type_acts = fit_sae_acts[idx]
            sae_by_type[etype] = (type_acts[type_k1 == 1], type_acts[type_k1 == 0])
        if len(sae_by_type) < 1:
            raise SystemExit("every entity type has constant K1 in the fit split; "
                             "cannot select an SAE latent by max-min separation.")
        # .float() before .cpu().numpy(): SAME bfloat16-to-numpy gap as
        # sae.encode()'s own output above, just a second, separate call
        # site -- W_dec is stored in the SAE's own native dtype
        # (bfloat16 for Llama Scope), and this line was missed when the
        # encode() call was fixed for job 384007's crash. Caught on the
        # very next real GPU run (job 384351) at the next line that
        # converts an SAE tensor to numpy -- the same root cause, not a
        # new bug, just not yet applied everywhere it needed to be.
        gates["sae"] = fit_sae_gate(sae_by_type, sae.W_dec.float().detach().cpu().numpy(),
                                    layer=args.layer, model_name=args.model)
        print(f"SAE gate: width={sae.cfg.d_sae}, selected latent={gates['sae'].sae_latent_id}, "
             f"{gates['sae'].notes}")

    # Named-bridge premise check (E0 part c): is K2 gate-visible when the
    # bridge IS named? render_prompts() always includes "named_bridge" for
    # k=2 queries, which names q.bridge (the hop2 entity) directly.
    nb_queries = load_queries(args.named_bridge_queries)
    if any(q.k != 2 for q in nb_queries):
        raise SystemExit("--named-bridge-queries must be two-hop (k=2) chains.")
    if args.n_named_bridge is not None and len(nb_queries) > args.n_named_bridge:
        idx = rng.choice(len(nb_queries), size=args.n_named_bridge, replace=False)
        nb_queries = [nb_queries[i] for i in idx]
        print(f"Subsampled --named-bridge-queries to {len(nb_queries)} of the original set.")
    print(f"Labeling {len(nb_queries)} named-bridge queries for K2 ...", flush=True)
    # hop_require_prefix={2: False}: K2 comes from hop2-style prompts ("The
    # {r2} of the {btype} {bridge} is"), which on real Gemma-2-2b generations
    # reliably name the correct answer inside a descriptive lead-in ("located
    # in the state of X...") rather than as the generation's first token(s).
    # Confirmed on skc_run_pilot.py's raw_generations.json; see
    # skc_data.compute_labels's docstring for the full reasoning. Without
    # this, K2 came back constant across all 3503 named-bridge queries on
    # the first real E0 run, which is itself informative that the failure
    # is scoring, not the model: K2 should be HIGH here almost by
    # construction (the bridge is explicitly named), so an all-0 result
    # pointed straight at the known hop2 scoring issue rather than a new one.
    nb_gens_path = os.path.join(args.out, "named_bridge_raw_generations.json")
    nb_labels = label_queries_batched(model, tok, nb_queries,
                                      hop_require_prefix={2: False},
                                      save_generations_path=nb_gens_path)
    nb_prompts, nb_entities = [], []
    for q in nb_queries:
        p = render_prompts(q)
        nb_prompts.append(p["named_bridge"]); nb_entities.append(q.bridge)
    print("Capturing named-bridge activations at the bridge position ...", flush=True)
    acts_named_bridge = capture_one_hop(model, tok, nb_prompts, nb_entities, args.layer)
    k2_named = np.array([l["K2"] for l in nb_labels])
    if len(set(k2_named.tolist())) < 2:
        raise SystemExit("K2 is constant on the named-bridge set; resample for a harder mix.")

    man = RunManifest(outdir=args.out, snapshot_id="see_dataset_snapshot_json",
                      model_name=args.model, seed=args.seed)

    print("Running E0 stop-gate check on held-out entities ...", flush=True)
    e0 = run_E0(gates, held_acts, held_k1, acts_named_bridge, k2_named, man)
    print(json.dumps(e0, indent=1, default=str))

    for name, gate in gates.items():
        gate.save(os.path.join(args.out, f"frozen_gate_{name}.json"))
    save_activation_cache(os.path.join(args.out, "held_acts.npy"), held_acts,
                          {"model": args.model, "layer": args.layer, "split": "held_out"})

    try:
        require_E0(e0)
    except Exception as e:
        print(f"\n*** E0 STOP GATE FAILED: {e} ***\n", file=sys.stderr)
        print("Do NOT proceed to E1. Flag back per skc_runbook.txt FLAG-BACK TRIGGERS.",
             file=sys.stderr)
        sys.exit(1)

    print("\nE0 PASSED. Frozen gates saved to", args.out)
    print("Proceed to E1 using these frozen gates -- do not refit them.")


if __name__ == "__main__":
    main()

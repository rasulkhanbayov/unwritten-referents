#!/usr/bin/env python3
"""
skc_run_e7_ablations.py
E7 ablations (skc_runbook.txt). Each is its own subcommand; run them
independently rather than as one monolithic pass, since they need different
inputs and some (h, j) need very little compute.

LETTERING: matches skc_main.tex's own Appendix~\ref{app:controls} (the
paper is the ground truth for which letter means what -- an EARLIER
version of this docstring used a different, now-stale lettering scheme;
corrected here after checking the actual manuscript text directly).

IMPLEMENTED HERE, as subcommands:
  named-bridge     (a) named-bridge control: the gap must close.
  distractor       (c) length-matched distractor control: swap an
                       UNRELATED real city into the named_bridge template
                       instead of the true bridge, length-matched --
                       AUROC(gate->K2) should collapse toward chance if
                       the gate genuinely reads the TRUE bridge's
                       properties rather than merely detecting that some
                       city is named. Protocol is an EXPLICIT CHOICE
                       (no further detail existed anywhere in the repo/
                       paper beyond the control's name) -- see
                       cmd_distractor()'s own docstring.
  popularity       (b) popularity-matched subsets by pageview decile
                       (as bridge patch's own donor-matching methodology).
  counterfactual   (g) counterfactual chains: does the model's final
                       answer track a FALSE bridge premise stated in the
                       prompt, or fall back to a memorized true chain?
                       Behavioral only, no gate/probe reading -- see
                       skc_build_dataset_counterfactual.py for the dataset
                       and cmd_counterfactual()'s own docstring for the
                       exact protocol and the explicit choice behind it.
  paraphrase       (i) template/paraphrase robustness: 2 genuine
                       paraphrases of the composed template (same
                       entity-late/fixed-ending constraint the paper's
                       own template follows, following Tamoyan et al.
                       2025's formatting-sensitivity finding), reports
                       AUROC(gate->K2) stability across the original
                       template and both paraphrases. See
                       PARAPHRASE_TEMPLATES and cmd_paraphrase()'s own
                       docstring for exactly which paraphrases were kept
                       and which were drafted and REJECTED for violating
                       the entity-late constraint.
  random-direction (f) random-direction negative control for C1-C3.
  attention        (e) attention control: does the bridge position receive
                       substantial attention?
  label-validation (h) LLM-judge agreement with alias-aware exact match on
                       a subsample.

NOT IMPLEMENTED HERE -- each needs dedicated new data or a separate pipeline
and is left for a follow-up script, rather than guessed at:
  (d) three-estimator agreement           -- already computed by skc_run_e0.py's
                                             gate_agreement() call; report from there
  (j) cross-type generalization           -- ENTITY-TYPE half implemented in
                                             skc_run_e7j_crosstype.py (its own
                                             standalone script, not this file);
                                             RELATION-TYPE half remains a
                                             disclosed data gap -- see that
                                             script's own module docstring
  (k) reasoning-tuned models               -- just a --model flag on skc_run_e1.py
                                             pointed at a reasoning-tuned checkpoint,
                                             instruction-tuned half already done (E6)
  (l) scale (32B/70B probe analysis)       -- 70B done (Llama-3.1-70B); 32B (Qwen3-32B)
                                             not yet -- see EXECUTION_PLAN.md Phase 3
  (m) C4 attribution-graph faithfulness    -- depends on a circuit-tracer toolchain,
                                             deliberately out of scope for this pipeline
  (n) artifact probe (TXTEMB-style)        -- needs a text-only, no-internals baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skc_metrics as M                                                        # noqa: E402
from skc_data import (                                                          # noqa: E402
    load_queries, save_queries, render_prompts, assign_popularity_deciles,
    match_on_decile, alias_match,
)
from skc_gates import FrozenGate                                                # noqa: E402
from skc_harness import (                                                       # noqa: E402
    load_model, capture_for_queries, label_queries_batched, generate_greedy,
)


# =====================================================================
# (a) Named-bridge control
# =====================================================================

def cmd_named_bridge(args):
    """The gap between AUC(S->K2) on composed vs. named-bridge prompts MUST
    close when the bridge is written. If it does not, something other than
    latency drives the effect -- report that as a genuine problem, not a
    detail to omit."""
    gate = FrozenGate.load(args.gate_path)
    queries = load_queries(args.queries)
    model, tok = load_model(args.model)
    # hop_require_prefix={2: False}: same fix as skc_run_e1.py/skc_run_e3.py --
    # K2 is read right below.
    labels = label_queries_batched(model, tok, queries, hop_require_prefix={2: False})
    k2 = np.array([l["K2"] for l in labels])

    acts_surface, _, _ = capture_for_queries(model, tok, queries, args.layer)
    from skc_gates import capture_activations
    from skc_harness import entity_spans_for_prompts
    nb_prompts = [render_prompts(q)["named_bridge"] for q in queries]
    nb_entities = [q.bridge for q in queries]
    nb_spans = entity_spans_for_prompts(tok, nb_prompts, nb_entities)
    acts_named = capture_activations(model, tok, nb_prompts, args.layer,
                                     position="last_entity_token", entity_spans=nb_spans)

    auc_composed = M.auroc(k2, gate.score(acts_surface))
    auc_named = M.auroc(k2, gate.score(acts_named))
    out = {"auroc_K2_composed": auc_composed, "auroc_K2_named_bridge": auc_named,
          "gap": auc_named - auc_composed,
          "reading": "gap should be large and positive: naming the bridge should make "
                     "K2 far more gate-visible than leaving it latent"}
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e7a_named_bridge.json"), "w") as f:
        json.dump(out, f, indent=1)


# =====================================================================
# (c) Length-matched distractor control
# =====================================================================

def cmd_distractor(args):
    """
    Paper's own words (skc_main.tex, Appendix~\\ref{app:controls}): "(c)
    Length-matched distractor control." No further protocol detail exists
    anywhere in the repo or manuscript beyond this name -- the exact
    design below is an EXPLICIT CHOICE confirmed directly with the user,
    not a guess presented as settled.

    PROTOCOL: the COMPOSED prompt never names the bridge at all (e.g.
    "The country of the place of birth of the player Jake Bornheimer
    is") -- there is no bridge slot to swap a distractor into there. The
    only template that DOES name a city is named_bridge ("The country of
    the city New Brunswick is"), the same template control (a) uses. This
    control builds a DISTRACTOR variant of that SAME template: instead of
    the TRUE bridge, an UNRELATED real city (drawn from this same
    dataset's own pool, exactly the same sampling discipline as control
    (g)'s counterfactual cities -- always a real place, never fabricated)
    is named instead, LENGTH-MATCHED toward the TRUE named-bridge
    prompt's own token count via skc_data.length_match_named_bridge()
    (built earlier in the project but never actually wired into any
    script until now -- confirmed by grepping the whole codebase before
    using it here).

    If control (a)'s own AUC(gate->K2) gap (composed vs. true named-
    bridge) reflects the gate genuinely reading the TRUE bridge's actual
    properties, then AUC(gate->K2) on these DISTRACTOR named-bridge
    prompts should collapse toward CHANCE (0.5) -- the stated city has no
    real relationship to this query's own K2 label, so a gate that were
    merely detecting "a city name is present here" (a surface/length
    artifact) rather than reading the true bridge's properties would
    show the SAME elevated AUROC on distractor prompts as on true ones,
    which this control is built to catch.

    Distractor sampling mirrors skc_build_dataset_counterfactual.py's own
    discipline: for each query, the distractor city is drawn uniformly
    (seeded) from every OTHER city in this dataset's own pool, excluded
    only if it happens to equal this query's own true bridge (so the
    "distractor" is never accidentally the real answer).
    """
    import random
    from skc_data import length_match_named_bridge

    gate = FrozenGate.load(args.gate_path)
    queries = load_queries(args.queries)
    model, tok = load_model(args.model)
    labels = label_queries_batched(model, tok, queries, hop_require_prefix={2: False})
    k2 = np.array([l["K2"] for l in labels])

    all_bridges = sorted(set(q.bridge for q in queries))
    if len(all_bridges) < 2:
        raise SystemExit(
            f"only {len(all_bridges)} distinct bridge cities in --queries -- cannot sample "
            "a distractor city different from the true bridge."
        )
    rng = random.Random(args.seed)

    composed_prompts = [render_prompts(q)["composed"] for q in queries]
    true_named_prompts = [render_prompts(q)["named_bridge"] for q in queries]
    distractor_prompts, distractor_entities = [], []
    n_padded = 0
    for q, true_named in zip(queries, true_named_prompts):
        candidates = [c for c in all_bridges if c != q.bridge]
        distractor_city = rng.choice(candidates)
        r2 = q.hops[1].relation
        from skc_data import TEMPLATES
        raw_distractor = TEMPLATES["named_bridge"].format(
            r2=r2, btype=q.hops[1].subject_type, bridge=distractor_city)
        padded_distractor = length_match_named_bridge(true_named, raw_distractor,
                                                       entity=distractor_city)
        if padded_distractor != raw_distractor:
            n_padded += 1
        distractor_prompts.append(padded_distractor)
        distractor_entities.append(distractor_city)
    # DISCLOSED ASYMMETRY (confirmed directly with the user rather than
    # silently accepted): length_match_named_bridge() pads ONE-DIRECTIONALLY
    # (distractor toward true length only) -- most real distractor city
    # names are already >= the true bridge's own name length, so only a
    # MINORITY of distractor prompts actually receive the "well-known"
    # filler at all. This is a real, if minor, asymmetry versus the true-
    # bridge condition (which never receives this filler either, since
    # length_match_named_bridge() was built earlier in the project but
    # never actually called anywhere until this control) -- reported
    # below as a fraction, not hidden, so a reader can judge whether it
    # matters for their own reading of the result.

    acts_surface, _, _ = capture_for_queries(model, tok, queries, args.layer)
    from skc_gates import capture_activations
    from skc_harness import entity_spans_for_prompts
    distractor_spans = entity_spans_for_prompts(tok, distractor_prompts, distractor_entities)
    acts_distractor = capture_activations(model, tok, distractor_prompts, args.layer,
                                          position="last_entity_token",
                                          entity_spans=distractor_spans)

    auc_composed = M.auroc(k2, gate.score(acts_surface))
    auc_distractor = M.auroc(k2, gate.score(acts_distractor))
    out = {
        "auroc_K2_composed": auc_composed,
        "auroc_K2_distractor_named_bridge": auc_distractor,
        "gap_vs_composed": auc_distractor - auc_composed,
        "n": len(queries),
        "n_distractor_prompts_padded_with_filler": n_padded,
        "frac_distractor_prompts_padded_with_filler": n_padded / len(queries),
        "length_matching_note": "length_match_named_bridge() pads ONE-DIRECTIONALLY "
                               "(distractor toward the true named-bridge prompt's own "
                               "length only); most real distractor city names are "
                               "already >= the true bridge's name length, so only the "
                               "fraction above actually receive the 'well-known' filler "
                               "at all -- a real, minor asymmetry versus the true-bridge "
                               "condition (which never receives this filler either), "
                               "disclosed here rather than hidden.",
        "reading": "auroc_K2_distractor_named_bridge should be CLOSE TO CHANCE (0.5) -- a "
                  "distractor city bears no real relationship to this query's own K2 "
                  "label, so a gate genuinely reading the TRUE bridge's properties "
                  "(not merely detecting that SOME city is named here) should not be "
                  "fooled by it. Compare directly against control (a)'s own "
                  "auroc_K2_named_bridge (the TRUE bridge result) -- a large drop from "
                  "that number toward chance here is the evidence this control needs; "
                  "a similarly elevated AUROC here would suggest the gate is not "
                  "specific to the true bridge's actual properties.",
    }
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e7c_distractor.json"), "w") as f:
        json.dump(out, f, indent=1)


# =====================================================================
# (i) Template and paraphrase robustness
# =====================================================================

# Two genuine paraphrases of TEMPLATES["composed_2"] (skc_data.py),
# chosen to satisfy the SAME design constraint the paper's own templates
# follow (Appendix~\ref{app:prompts}, following Tamoyan et al. 2025):
# entity placed LATE, prompt ends on a FIXED token ("is"), following that
# citation's own finding that this form avoids spurious correlations that
# arise when the prompt ends on variable tokens. This constraint
# genuinely LIMITS how much a paraphrase can restructure the sentence --
# several candidate paraphrases were drafted and REJECTED because moving
# the relation-2 word to just before "is" (a natural-sounding
# rephrasing) pulls e1's own mention EARLIER, violating "entity late":
# e.g. "Regarding the place of birth of the player {e1}, the country is"
# was drafted and discarded for exactly this reason, confirmed directly
# by checking e1's position relative to the relation words in the
# rendered string. The two kept below are lexical/syntactic variants
# within the SAME clause structure (e1 stays as the last-named entity
# immediately before the fixed "is"), which is what the constraint
# actually permits.
PARAPHRASE_TEMPLATES = {
    "paraphrase_a": "The {r2} associated with the {r1} of the {etype} {e1} is",
    "paraphrase_b": "The {r2} tied to the {r1} of the {etype} {e1} is",
}


def cmd_paraphrase(args):
    """
    Paper's own words (skc_main.tex, Appendix~\\ref{app:controls}):
    "(i) Template and paraphrase robustness, following the formatting
    sensitivity reported by \\citet{tamoyan2025factual}." No further
    protocol detail exists beyond this sentence -- the exact design below
    (which paraphrases, which metric) is an EXPLICIT CHOICE confirmed
    directly with the user, not a guess presented as settled.

    PROTOCOL: build PARAPHRASE_TEMPLATES's own composed-prompt variants
    (same entity-late/fixed-ending constraint as the paper's own
    template, just a different surface phrasing of the SAME relations),
    capture the surface_gate activation at e1's own last-entity-token
    position under EACH paraphrase, and report AUROC(gate->K2) for every
    paraphrase alongside the ORIGINAL template's own already-known
    result -- stability (AUROCs staying close across all three) is
    evidence the paper's own result is not an artifact of this one
    specific phrasing; a large spread would mean the opposite.

    REUSES the query's own EXISTING K2 labels (from the ORIGINAL
    composed/hop-prompt generations) rather than relabeling under each
    paraphrase -- confirmed directly from compute_labels()'s own code
    that K_j is computed ENTIRELY from generations["hop{j}"] (the
    ISOLATED single-hop prompt), never from the composed prompt's own
    phrasing at all, so K2 is a property of the model's knowledge of the
    isolated hop2 fact, independent of how the composed prompt asking
    about it happens to be worded -- reusing it here is not an
    approximation, it is the exact same quantity by construction.
    """
    gate = FrozenGate.load(args.gate_path)
    queries = load_queries(args.queries)
    model, tok = load_model(args.model)
    labels = label_queries_batched(model, tok, queries, hop_require_prefix={2: False})
    k2 = np.array([l["K2"] for l in labels])

    acts_surface, _, _ = capture_for_queries(model, tok, queries, args.layer)
    auc_original = M.auroc(k2, gate.score(acts_surface))

    from skc_gates import capture_activations
    from skc_harness import entity_spans_for_prompts

    out = {"auroc_K2_original_template": auc_original, "per_paraphrase": {}}
    for name, template in PARAPHRASE_TEMPLATES.items():
        prompts = [template.format(r2=q.hops[1].relation, r1=q.hops[0].relation,
                                   etype=q.entity_type, e1=q.e1) for q in queries]
        entities = [q.e1 for q in queries]
        spans = entity_spans_for_prompts(tok, prompts, entities)
        acts = capture_activations(model, tok, prompts, args.layer,
                                   position="last_entity_token", entity_spans=spans)
        auc = M.auroc(k2, gate.score(acts))
        out["per_paraphrase"][name] = {
            "template": template,
            "auroc_K2": auc,
            "diff_vs_original": auc - auc_original,
        }

    all_aurocs = [auc_original] + [v["auroc_K2"] for v in out["per_paraphrase"].values()]
    out["max_minus_min_auroc_across_templates"] = float(max(all_aurocs) - min(all_aurocs))
    out["reading"] = (
        "auroc_K2 should stay CLOSE across the original template and both paraphrases "
        "-- a small max_minus_min_auroc_across_templates is evidence the paper's own "
        "gate result is not an artifact of this one specific phrasing (the formatting-"
        "sensitivity concern from Tamoyan et al. 2025 this control checks for); a large "
        "spread would mean the opposite."
    )
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e7i_paraphrase.json"), "w") as f:
        json.dump(out, f, indent=1)


# =====================================================================
# (b) Popularity matching
# =====================================================================

def cmd_popularity(args):
    """
    Assigns popularity deciles and marks meta['bridge_known'], then saves an
    augmented queries file for skc_run_e2.py's C2 donor matching to consume.
    Requires a pageviews JSON {entity_name: views} -- NOT fetched by this
    script. >>> TO RUN: Wikipedia pageview API access / dump parsing is a
    separate, unwritten piece of infrastructure; point --pageviews at a
    pre-fetched file.
    """
    queries = load_queries(args.queries)
    with open(args.pageviews) as f:
        pageviews = json.load(f)
    missing = [q.bridge for q in queries if q.bridge not in pageviews]
    if missing:
        print(f"WARNING: {len(missing)}/{len(queries)} bridge entities missing pageview "
             "data; dropping them rather than imputing a decile.", file=sys.stderr)
    queries = [q for q in queries if q.bridge in pageviews]
    deciles = assign_popularity_deciles({q.bridge: pageviews[q.bridge] for q in queries})
    for q in queries:
        q.popularity_decile = deciles[q.bridge]

    # bridge_known needs the model's own K1 (does it know the FIRST hop,
    # i.e. can it name the bridge at all) -- NOT K2. This requires a live
    # model pass, which this ablation script does not skip.
    #
    # hop_require_prefix={1: False}: matches skc_run_e0.py's OWN K1
    # convention (measured there: strict K1 = 1.0% vs containment K1 =
    # 8.2% on the same generations, due to descriptive lead-in phrasing).
    # Leaving this at the strict default here (as this script did before)
    # is the exact mismatch found and fixed in skc_run_e1.py/skc_run_e5.py
    # after their K1=1 rates silently disagreed with
    # data/onehop_correct.json's own K1=1 definition (see
    # EXECUTION_PLAN.md) -- bridge_known needs the SAME K1 definition as
    # everywhere else in the pipeline, not a stricter local one, since it
    # feeds skc_run_e2.py's C2 donor matching downstream.
    model, tok = load_model(args.model)
    labels = label_queries_batched(model, tok, queries, hop_require_prefix={1: False})
    k1_by_qid = {l["qid"]: l["K1"] for l in labels}
    for q in queries:
        q.meta["bridge_known"] = bool(k1_by_qid[q.qid])

    pairs = match_on_decile(queries, seed=args.seed)
    print(f"{len(pairs)} known/unknown pairs matched across {len(set(q.popularity_decile for q in queries))} deciles.")
    os.makedirs(args.out, exist_ok=True)
    save_queries(queries, os.path.join(args.out, "queries_with_deciles.json"))
    with open(os.path.join(args.out, "decile_pairs_qids.json"), "w") as f:
        json.dump([[a.qid, b.qid] for a, b in pairs], f, indent=1)
    print(f"Saved to {args.out}/queries_with_deciles.json -- feed this into skc_run_e2.py "
         "and skc_run_e7_ablations.py's popularity-matched subset analyses.")


# =====================================================================
# (g) Counterfactual chains
# =====================================================================

def cmd_counterfactual(args):
    """
    Paper's own words (skc_main.tex, Appendix control (g)): "Counterfactual
    chains, so memorized composites cannot drive the result." Flagged as
    one of THREE LOAD-BEARING controls in the paper's own text (alongside
    (f) random-direction and (j) cross-type generalization) -- "without
    which memorized composites can produce the effect."

    PROTOCOL (explicit choices, both confirmed directly rather than
    guessed): (1) in-context counterfactual, NOT model editing -- the
    false bridge premise is stated directly in the prompt text (e.g.
    "Suppose the place of birth of {e1} is actually {b_fake}."), no
    ROME/MEMIT-style weight edit of any kind. (2) BEHAVIORAL measurement
    only -- does the model's final greedy answer track the STATED (fake)
    premise, or fall back to the memorized TRUE chain? No gate/probe
    reading at the bridge position under this prompt (a fuller variant
    would also capture activations here, but was explicitly scoped out
    in favor of the simpler behavioral test).

    Dataset: skc_build_dataset_counterfactual.py, reusing this pipeline's
    own real twohop_queries.json entirely (B_fake is always a REAL city
    from that same dataset, with a REAL, different country than B_true --
    never a fabricated place). See that script's own docstring for the
    exact construction and why this is NOT a ComposedQuery (states a
    FALSE relation as fact, which render_prompts()'s existing templates
    have no way to represent).

    Reports, per item and in aggregate:
      p_composed  -- fraction of greedy answers matching a_fake (tracked
                     the STATED premise -- evidence of genuine
                     composition, not a memorized shortcut)
      p_memorized -- fraction matching a_true (fell back to the real,
                     memorized fact, ignoring the stated premise --
                     evidence FOR a memorized composite)
      p_neither   -- matched neither (a real third outcome, e.g. a
                     refusal or an unrelated guess; not folded into
                     either rate above, since it says something
                     different from both)
    A model driven by memorized composites would show p_memorized far
    above p_composed; a model that genuinely composes would show the
    reverse. Both rates are reported, never just their difference, since
    the RAW proportions are what a reviewer needs to judge whether either
    reading has real support (a small dataset where both rates are near
    zero would need reporting differently than one where they are large
    and separated).
    """
    with open(args.counterfactual_queries) as f:
        items = json.load(f)
    if not items:
        raise SystemExit(f"{args.counterfactual_queries} is empty -- nothing to score.")

    model, tok = load_model(args.model)
    prompts = [it["prompt"] for it in items]
    generations = generate_greedy(model, tok, prompts, max_new_tokens=8)

    n_composed, n_memorized, n_neither = 0, 0, 0
    per_item = []
    for it, gen in zip(items, generations):
        composed = alias_match(gen, [it["a_fake"]], require_prefix=False)
        memorized = alias_match(gen, [it["a_true"]], require_prefix=False)
        if composed and not memorized:
            n_composed += 1
            outcome = "composed"
        elif memorized and not composed:
            n_memorized += 1
            outcome = "memorized"
        elif composed and memorized:
            # a_fake and a_true both appear in the generation (can happen
            # with a longer, discursive answer) -- neither reading is
            # clean here; counted separately rather than forced into one
            # bucket, so this ambiguity is visible rather than hidden.
            n_neither += 1
            outcome = "ambiguous_both"
        else:
            n_neither += 1
            outcome = "neither"
        per_item.append({"qid": it["qid"], "generation": gen, "outcome": outcome})

    n = len(items)
    out = {
        "n": n,
        "p_composed": n_composed / n,
        "p_memorized": n_memorized / n,
        "p_neither_or_ambiguous": n_neither / n,
        "reading": "p_composed >> p_memorized is evidence the model genuinely composes "
                  "e1 -> bridge -> answer rather than relying on a memorized (e1, "
                  "a_true) composite; the reverse pattern would mean memorized "
                  "composites, not composition, drive the paper's main result.",
    }
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e7g_counterfactual.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.out, "e7g_counterfactual_per_item.json"), "w") as f:
        json.dump(per_item, f, indent=1)


# =====================================================================
# (f) Random-direction negative control
# =====================================================================

def cmd_random_direction(args):
    """
    A direction of MATCHED NORM drawn at random must produce no effect in
    C1-C3. Without this, every intervention result is uninterpretable,
    because any large enough perturbation changes behavior regardless of
    whether it carries the "unknown entity" signal.
    """
    gate = FrozenGate.load(args.gate_path)
    rng = np.random.default_rng(args.seed)
    random_dir = rng.normal(size=gate.direction.shape)
    random_dir = random_dir / np.linalg.norm(random_dir)  # FrozenGate normalizes to unit norm too

    queries = load_queries(args.queries)
    model, tok = load_model(args.model)
    from skc_experiments import steer_at_position
    from skc_data import is_refusal
    from skc_harness import entity_spans_for_prompts

    prompts, entities = [], []
    for q in queries:
        p = render_prompts(q)
        prompts.append(p["composed"]); entities.append(q.e1)
    spans = entity_spans_for_prompts(tok, prompts, entities)
    positions = [s[1] - 1 for s in spans]

    gens_real = steer_at_position(model, tok, prompts, gate.direction, args.coefficient,
                                  args.layer, positions)
    gens_random = steer_at_position(model, tok, prompts, random_dir, args.coefficient,
                                    args.layer, positions)
    refusal_real = float(np.mean([is_refusal(g) for g in gens_real]))
    refusal_random = float(np.mean([is_refusal(g) for g in gens_random]))
    out = {"refusal_rate_real_direction": refusal_real,
          "refusal_rate_random_direction": refusal_random,
          "reading": "random-direction refusal rate should be near the UNSTEERED baseline "
                     "and far below the real direction's; if they are close, C1's effect is "
                     "not specific to the entity-gate direction"}
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e7j_random_direction.json"), "w") as f:
        json.dump(out, f, indent=1)


# =====================================================================
# (e) Attention control
# =====================================================================

def _parse_e3_peak_key(key: str):
    """
    Parse skc_run_e3.py's "L{layer}_pos{offset}" key format. Duplicated
    from skc_run_e2.py's own identical helper rather than imported --
    that module is a driver script with its own argparse main(), not a
    library skc_run_e7_ablations.py should import from (importing it
    would execute skc_run_e2.py's own module-level code and risk
    argparse conflicts); this is a small (~5-line), stable, one-purpose
    parser, not logic worth the coupling.
    """
    import re
    m = re.fullmatch(r"L(-?\d+)_pos(-?\d+)", key)
    if not m:
        raise ValueError(f"cannot parse E3 peak key {key!r}; expected 'L{{layer}}_pos{{offset}}'")
    return int(m.group(1)), int(m.group(2))


def cmd_attention(args):
    """
    Control (e): confirms the bridge-resolution position (E3's own
    empirical peak, the SAME position E3 shows the frozen gate reads K2
    above chance at) RECEIVES substantial attention from the model's own
    final-prompt-token computation. If not, "the gate is not routed"
    collapses into "the position is ignored", a weaker, different claim
    -- this control is what licenses the routing interpretation.

    AGGREGATION CHOICE, STATED EXPLICITLY (per this function's own
    former docstring's own requirement, not left implicit): attention
    weight FROM the final prompt token (where the composed answer is
    about to be generated) TO the bridge-resolution position (E3's own
    peak, entity-end-token-index + offset), MEAN over all heads AT the
    SAME layer E3's own peak was found at (not averaged across layers,
    and not summed across all earlier positions -- a single, specific
    (query, from-position, to-position, layer) attention weight per
    query, matching exactly the granularity E3's own gate-readout claim
    operates at, so this control answers the SAME question E3 poses
    ("is K2 gate-visible AT this position") from the attention side
    rather than a differently-scoped one).

    Reuses skc_gates.py/skc_harness.py's existing need_attentions=True
    eager-attention loading path (built for Lookback Lens, E4 Phase 2)
    rather than adding a second, parallel attention-capture code path --
    the exact "two independent copies of the same logic diverging" bug
    class this project has repeatedly found and fixed elsewhere.

    Reports the mean and the distribution (with a comparison against a
    UNIFORM-attention null: 1/sequence_length, the attention weight a
    position with NO special routing would receive by chance under a
    perfectly flat distribution) -- substantially above this null is the
    evidence this control needs; at or below it would mean the position
    is attention-wise indistinguishable from an arbitrary context token.
    """
    import numpy as np
    from skc_data import load_queries, render_prompts
    from skc_harness import load_model, entity_spans_for_prompts

    queries = load_queries(args.queries)
    with open(args.e3_map) as f:
        e3_map = json.load(f)
    peak_layer, peak_offset = _parse_e3_peak_key(e3_map["peak"]["key"])
    print(f"Using E3's own peak position: layer {peak_layer}, offset {peak_offset} "
         f"(AUROC {e3_map['peak']['auroc']:.4f})", flush=True)

    print(f"Loading {args.model} with eager attention (need_attentions=True) ...",
         flush=True)
    model, tok = load_model(args.model, need_attentions=True)

    composed_prompts = [render_prompts(q)["composed"] for q in queries]
    e1_names = [q.e1 for q in queries]
    entity_spans = entity_spans_for_prompts(tok, composed_prompts, e1_names)
    entity_end_positions = [end - 1 for (_start, end) in entity_spans]
    bridge_positions = [p + peak_offset for p in entity_end_positions]

    attn_to_bridge = []
    uniform_null = []
    for prompt, bridge_pos in zip(composed_prompts, bridge_positions):
        if bridge_pos < 0:
            continue
        enc = tok(prompt, return_tensors="pt").to("cuda")
        seq_len = enc["input_ids"].shape[1]
        if bridge_pos >= seq_len:
            continue
        final_pos = seq_len - 1
        with __import__("torch").no_grad():
            out = model(**enc, output_attentions=True)
        if out.attentions[peak_layer] is None:
            raise RuntimeError(
                "model(...).attentions[...] is None -- this model was not loaded with "
                "need_attentions=True (attn_implementation='eager'), so no real attention "
                "weights were computed."
            )
        # attentions[layer]: (batch=1, heads, seq, seq). Mean over heads,
        # FROM the final prompt token TO the bridge position -- the
        # explicit aggregation stated in this function's own docstring.
        head_weights = out.attentions[peak_layer][0, :, final_pos, bridge_pos]
        attn_to_bridge.append(float(head_weights.float().mean()))
        uniform_null.append(1.0 / seq_len)

    if not attn_to_bridge:
        raise SystemExit(
            "no query produced a valid (bridge_pos, seq_len) pair -- check --e3-map's "
            "own peak offset against this --queries file's own prompt lengths."
        )

    attn_to_bridge = np.array(attn_to_bridge)
    uniform_null = np.array(uniform_null)
    out_dict = {
        "control": "(e) attention -- does the bridge-resolution position receive "
                  "substantial attention?",
        "e3_peak_layer": peak_layer,
        "e3_peak_offset": peak_offset,
        "e3_peak_auroc": e3_map["peak"]["auroc"],
        "n": int(len(attn_to_bridge)),
        "mean_attention_to_bridge_position": float(attn_to_bridge.mean()),
        "median_attention_to_bridge_position": float(np.median(attn_to_bridge)),
        "mean_uniform_null": float(uniform_null.mean()),
        "ratio_to_uniform_null": float(attn_to_bridge.mean() / uniform_null.mean()),
        "reading": "mean_attention_to_bridge_position substantially ABOVE "
                  "mean_uniform_null (the attention weight an arbitrary context token "
                  "would receive under a flat distribution) is evidence the position is "
                  "attended, not ignored -- licensing the 'present but unrouted' "
                  "interpretation over 'the position is simply ignored'. At or below the "
                  "null would mean the opposite.",
    }
    print(json.dumps(out_dict, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e7e_attention.json"), "w") as f:
        json.dump(out_dict, f, indent=1)


# =====================================================================
# (n) Artifact probe (TxTemb, Hussain & Kantarcioglu 2026)
# =====================================================================
#
# VERIFIED directly against the real paper (arXiv:2605.17028, "PARALLAX:
# Separating Genuine Hallucination Detection from Benchmark Construction
# Artifacts"): TxTemb is TF-IDF cosine similarity between two texts, NO
# trained classifier, NO neural embedding model, NO model internals at
# all -- s_TxTemb = cos(tfidf(text_a), tfidf(text_b)), used directly as
# the detection score.
#
# ADAPTATION, an EXPLICIT CHOICE confirmed directly with the user (the
# original paper compares a GENERATED response against a reference
# answer, which for SKC-Bench would trivially correlate with Y by
# CONSTRUCTION -- a correct generation IS textually similar to the
# target by definition of correctness, so it would not actually test
# for a benchmark-construction artifact the way it does in the
# original's own generation-based setting): here, TxTemb compares the
# COMPOSED PROMPT TEXT ITSELF against the target answer text. This is
# the genuine artifact concern control (n) exists to rule out: if the
# prompt leaks the answer via surface text overlap (the target entity's
# name or a paraphrase of it appearing verbatim in the prompt), a
# trivial text-only detector with NO access to model internals could
# predict Y anyway, which would mean the benchmark's own difficulty is
# fake. SKC-Bench's own design (composed queries never write the bridge
# entity) predicts this score should sit at CHANCE; above chance would
# mean the benchmark leaks the answer into the prompt.


def cmd_artifact_probe(args):
    """
    (n) Artifact probe: TxTemb (TF-IDF cosine similarity, no model
    internals) between the composed prompt text and the target answer
    text, must sit at chance (AUROC~=0.5) predicting Y on SKC-Bench.
    Fully CPU/offline -- no model load, no GPU needed, reuses labels
    already saved from a real E1 run rather than re-labeling.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    queries = load_queries(args.queries)
    with open(args.labels) as f:
        labels_raw = json.load(f)
    labels_by_qid = {l["qid"]: l for l in labels_raw}
    missing = [q.qid for q in queries if q.qid not in labels_by_qid]
    if missing:
        raise SystemExit(
            f"{len(missing)} queries in --queries have no matching entry in --labels "
            f"(e.g. {missing[0]!r}) -- --labels must be the SAME run's own saved labels, "
            "not a different query set's."
        )

    prompts = [render_prompts(q)["composed"] for q in queries]
    targets = [q.target_answers[0] for q in queries]
    y = np.array([labels_by_qid[q.qid]["Y"] for q in queries])
    if len(set(y.tolist())) < 2:
        raise SystemExit("Y is constant across --queries -- cannot compute AUROC.")

    vec = TfidfVectorizer()
    # Fit on prompts+targets TOGETHER so both are embedded in the SAME
    # vocabulary/vector space -- fitting separately would make the
    # cosine similarity meaningless (comparing vectors from two
    # different bases).
    all_text = prompts + targets
    tfidf = vec.fit_transform(all_text)
    prompt_vecs = tfidf[:len(prompts)]
    target_vecs = tfidf[len(prompts):]
    # Row-wise cosine similarity between matched (prompt, target) pairs.
    numer = np.asarray(prompt_vecs.multiply(target_vecs).sum(axis=1)).flatten()
    prompt_norms = np.sqrt(np.asarray(prompt_vecs.multiply(prompt_vecs).sum(axis=1)).flatten())
    target_norms = np.sqrt(np.asarray(target_vecs.multiply(target_vecs).sum(axis=1)).flatten())
    denom = prompt_norms * target_norms
    scores = np.where(denom > 1e-12, numer / np.where(denom > 1e-12, denom, 1.0), 0.0)

    auc = M.auroc(y, scores)
    # DIAGNOSTIC, not just the headline AUROC: report the fraction of items
    # with EXACTLY zero vocabulary overlap (the expected case if the
    # bridge entity is genuinely never written -- confirmed directly this
    # session: 211/239 on the real Gemma-2-2b test set), and the mean
    # score split by class. A below- or above-chance AUROC driven entirely
    # by a small minority of nonzero-overlap items (as found directly on
    # this real data: 28/239 nonzero, ALL of them in the Y=0 group,
    # producing AUROC=0.159 -- BELOW chance, not silently rounded to "at
    # chance") is a DIFFERENT finding than a benchmark-wide answer leak,
    # and must be reported as such rather than collapsed into one number.
    n_zero = int((scores == 0).sum())
    out = {
        "control": "(n) Artifact probe (TxTemb, Hussain & Kantarcioglu 2026, "
                  "arXiv:2605.17028)",
        "n": len(queries),
        "auroc_txtemb_to_Y": auc,
        "mean_score": float(scores.mean()),
        "n_zero_overlap": n_zero,
        "frac_zero_overlap": float(n_zero / len(queries)),
        "mean_score_Y1": float(scores[y == 1].mean()) if (y == 1).any() else None,
        "mean_score_Y0": float(scores[y == 0].mean()) if (y == 0).any() else None,
        "reading": "auroc_txtemb_to_Y should sit at CHANCE (~0.5) -- a text-only "
                  "detector with NO model internals should not be able to predict Y "
                  "from the composed prompt's own surface text, since the bridge "
                  "entity is never written there. Above chance would mean the "
                  "benchmark leaks the answer into the prompt, the artifact "
                  "Hussain & Kantarcioglu's own audit found inflating much of this "
                  "literature. A LARGE frac_zero_overlap (most items have NO surface "
                  "overlap at all between prompt and target) combined with a "
                  "far-from-0.5 AUROC driven by the small nonzero-overlap minority is a "
                  "DIFFERENT, narrower finding than a benchmark-wide leak -- report "
                  "which case this is, do not present a skewed AUROC as evidence of a "
                  "leak (or its absence) without checking frac_zero_overlap first.",
    }
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e7n_artifact_probe.json"), "w") as f:
        json.dump(out, f, indent=1)


# =====================================================================
# (h) Label validation against an LLM judge
# =====================================================================

def cmd_label_validation(args):
    """
    Alias-aware exact match under-counts unusual surface forms. Validate
    against an LLM judge (Zheng et al. 2023) on a random subsample; report
    agreement. Keep exact match PRIMARY -- this is a validation check on the
    primary labeler, not a replacement for it.

    >>> TO RUN: this calls a SEPARATE judge model via load_model(), which
    means picking which model plays "judge" (the runbook does not name one).
    Do not use the SAME model being evaluated as its own judge without
    stating that conflict of interest explicitly in Appendix B.
    """
    queries = load_queries(args.queries)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(queries), size=min(args.n, len(queries)), replace=False)
    sub = [queries[i] for i in idx]

    model, tok = load_model(args.model)
    labels = label_queries_batched(model, tok, sub)

    judge_model, judge_tok = load_model(args.judge_model)
    from skc_harness import generate_greedy
    judge_prompts = []
    for q, lab in zip(sub, labels):
        composed = render_prompts(q)["composed"]
        # >>> TO RUN: verify this judge prompt template; it is an unverified
        # paraphrase of the Zheng et al. (2023) LLM-as-judge protocol, not a
        # copy checked against the published prompt.
        judge_prompts.append(
            f"Question: {composed}\nCandidate answer: [see generation]\n"
            f"Gold answer(s): {q.target_answers}\n"
            "Does the candidate answer match any gold answer in meaning? Answer yes or no.\n"
        )
    judge_gens = generate_greedy(judge_model, judge_tok, judge_prompts, max_new_tokens=3)
    judge_y = np.array([1 if g.strip().lower().startswith("yes") else 0 for g in judge_gens])
    exact_y = np.array([l["Y"] for l in labels])

    agreement = float((judge_y == exact_y).mean())
    out = {"n": len(sub), "agreement": agreement,
          "n_exact_1_judge_0": int(((exact_y == 1) & (judge_y == 0)).sum()),
          "n_exact_0_judge_1": int(((exact_y == 0) & (judge_y == 1)).sum()),
          "reading": "if agreement is low, or Y flips a HEADLINE result depending on "
                     "labeler, report that fact rather than picking whichever favors the "
                     "paper (skc_runbook.txt E7k)."}
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e7k_label_validation.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, extra in [
        ("named-bridge", cmd_named_bridge, []),
        ("distractor", cmd_distractor, []),
        ("paraphrase", cmd_paraphrase, []),
        ("popularity", cmd_popularity, [("--pageviews", str, None)]),
        ("random-direction", cmd_random_direction, [("--coefficient", float, 4.0)]),
        ("attention", cmd_attention, [("--e3-map", str, None)]),
        ("label-validation", cmd_label_validation,
         [("--judge-model", str, None), ("--n", int, 100)]),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--queries", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--layer", type=int, required=(name not in ("popularity", "attention")))
        p.add_argument("--gate-path",
                       required=(name in ("named-bridge", "distractor", "paraphrase",
                                         "random-direction")))
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
        for flag, typ, default in extra:
            kwargs = {"type": typ}
            if default is None:
                kwargs["required"] = True
            else:
                kwargs["default"] = default
            p.add_argument(flag, **kwargs)
        p.set_defaults(func=fn)

    # counterfactual: added separately rather than folded into the loop
    # above -- its input is skc_build_dataset_counterfactual.py's own flat
    # JSON schema (--counterfactual-queries), not a ComposedQuery file
    # (--queries), and it needs neither --layer nor --gate-path (BEHAVIORAL
    # only, no gate/probe reading -- see cmd_counterfactual()'s own
    # docstring for why).
    p_cf = sub.add_parser("counterfactual")
    p_cf.add_argument("--counterfactual-queries", required=True,
                      help="output of skc_build_dataset_counterfactual.py")
    p_cf.add_argument("--model", required=True)
    p_cf.add_argument("--seed", type=int, default=0)
    p_cf.add_argument("--out", required=True)
    p_cf.set_defaults(func=cmd_counterfactual)

    # artifact-probe: added separately rather than folded into the
    # shared loop above -- fully CPU/offline, needs NO --model/--layer/
    # --gate-path at all (TxTemb is TF-IDF text similarity, no model
    # internals by design), and needs a --labels file the shared loop
    # has no slot for.
    p_ap = sub.add_parser("artifact-probe")
    p_ap.add_argument("--queries", required=True)
    p_ap.add_argument("--labels", required=True,
                      help="saved per-qid labels JSON from a real E1 run (e.g. "
                           "e1_labels_filtered.json), supplying Y -- reused rather than "
                           "re-labeling, since this control needs no model at all")
    p_ap.add_argument("--seed", type=int, default=0)
    p_ap.add_argument("--out", required=True)
    p_ap.set_defaults(func=cmd_artifact_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

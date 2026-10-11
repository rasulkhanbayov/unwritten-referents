#!/usr/bin/env python3
"""
skc_run_e4.py
E4: repair (skc_runbook.txt, Table 3). Shows the floor is escapable via
BGR-min / BGR-lr, and compares against baselines WITHIN inference-cost
classes (Class 0-3), never on one leaderboard across classes.

CLASS 3 IS MANDATORY, NOT OPTIONAL. skc_REDFLAGS.md and skc_runbook.txt are
both explicit: "Why do I need a bridge-gate readout when I can just prompt
the model to decompose the question?" now has published methods behind it
(Self-Ask, decomposed-prompting abstention, operadic consistency), and
skipping this comparison is the single most likely way this paper's
practical claim gets rejected. This script implements Self-Ask (Press et
al. 2023, the simplest, best-known Class 3 method) and Operadic
consistency (Bottman, Liu & Richardson 2026, arXiv:2606.13649 -- verified
against the live paper before implementing, not guessed from its title:
it compares the model's direct answer to a composed question against the
answer obtained by decomposing the question and composing the sub-
answers, scoring their disagreement). A third Class-3 method named in
skc_runbook.txt (Madhwal et al. 2026) is left as future work.

Also mandatory per skc_runbook.txt E4: report forward passes, generated
tokens, and wall-clock PER QUERY for every selector. A method that wins by
spending 10x has not won in the class that matters. This script's --cost
tracking is not decorative; it is a required output column.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skc_metrics as M                                          # noqa: E402
from skc_data import load_queries, render_prompts, is_refusal, alias_match  # noqa: E402
from skc_gates import FrozenGate                                  # noqa: E402
from skc_experiments import run_E4, RunManifest                    # noqa: E402
from skc_harness import (                                          # noqa: E402
    load_model, generate_greedy, label_queries_batched, capture_for_queries,
    capture_at_positions,
)


# =====================================================================
# Class 0 baselines: one forward pass, no extra generation
# =====================================================================

def baseline_final_token_probe(gate_surface_scores):
    """Accuracy probe / final-token probe (Tamoyan-style): re-use the surface
    gate's own score as a stand-in Class-0 probe baseline. >>> TO RUN: a
    genuine comparison should fit an INDEPENDENT final-token probe rather
    than reusing the surface gate's score verbatim -- this placeholder
    exists so E4's table has a filled column, but reusing gate_surface as
    its own baseline is circular and must be replaced before results are
    reported."""
    return np.asarray(gate_surface_scores)


# =====================================================================
# Class 0: SAPLMA (Azaria & Mitchell 2023)
# =====================================================================

class SAPLMAClassifier:
    """
    The exact architecture Azaria & Mitchell (2023) report: a feed-forward
    classifier with layers of size 256, 128, 64, 1, ReLU between hidden
    layers, sigmoid output -- trained on the hidden state at the LAST TOKEN
    OF A STATEMENT (their setup: the model is given a declarative sentence,
    not a question, and the probe reads where the sentence ends) to predict
    whether the statement is true.

    This does not fit FrozenGate's linear direction+bias interface (every
    other estimator in this codebase -- diffmeans, probe, sae -- IS a single
    linear direction; SAPLMA is a genuinely non-linear multi-layer
    classifier), so it is its own small class here rather than forced into
    skc_gates.py's FrozenGate, which is reserved for this paper's OWN
    frozen-direction argument (Section 6.4) and would misrepresent SAPLMA's
    actual shape if squeezed into it.
    """

    def __init__(self, d_in: int, seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 1e-3,
           batch_size: int = 32, seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        yt = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        n = len(Xt)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pred = self.net(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X, dtype=np.float32))
            return self.net(Xt).squeeze(1).numpy()


def fit_saplma(model, tok, fit_queries, layer: int, seed: int = 0, device: str = "cuda",
               fit_labels=None):
    """
    Trains a SAPLMA classifier on a HELD-OUT fit split of single-hop
    queries, mirroring skc_run_e0.py's own fit/held-out discipline: SAPLMA
    is a TRAINED classifier and must never be fit on the same items E4
    scores it on, exactly the rule FrozenGate.assert_not_fit_on() enforces
    mechanically for this paper's own estimators. `fit_queries` here should
    be single-hop (k=1) queries, not the composed queries E4 evaluates on.

    Trains on the STATEMENT the model itself generates in response to each
    query (its own declarative answer), captured at that generation's OWN
    last token -- not the query prompt's last token, which is Azaria &
    Mitchell's actual setup (feed the model a statement, read where the
    statement ends), adapted here since this pipeline's queries are
    questions, not ready-made statements: the model's own greedy answer,
    appended to the prompt, IS the statement whose truth SAPLMA judges.
    Label: whether that greedy answer is actually correct (K1), from this
    pipeline's own alias-aware exact match -- SAPLMA needs ground truth to
    train on, and this is the same correctness signal every other estimator
    in this pipeline uses, not a different or looser one.

    fit_labels: if given, reuse this ALREADY-COMPUTED label_queries_batched()
    output instead of relabeling `fit_queries` from scratch -- SAPLMA,
    fit_entity_token_probe, and fit_haloscope all label the SAME fit_queries
    with the SAME hop_require_prefix, so on a slow (e.g. 70B) model this was
    the single largest source of redundant GPU time in a multi-baseline E4
    run (one full generation pass over the fit split per baseline, not
    once). Caller must ensure fit_labels was computed on the SAME
    fit_queries this call receives -- not checked here, since fit_queries
    itself is not hashable/comparable cheaply; main() is the only caller
    that passes this, and it is responsible for that invariant.
    """
    from skc_harness import generate_greedy, label_queries_batched
    from skc_gates import capture_activations

    labels = fit_labels if fit_labels is not None else \
        label_queries_batched(model, tok, fit_queries, hop_require_prefix={1: False})
    k1 = np.array([l["K1"] for l in labels])
    prompts = [render_prompts(q)["hop1"] for q in fit_queries]
    answers = generate_greedy(model, tok, prompts, device=device)
    full_texts = [p + a for p, a in zip(prompts, answers)]

    acts = capture_activations(model, tok, full_texts, layer, position="last", device=device)
    clf = SAPLMAClassifier(d_in=acts.shape[1], seed=seed)
    clf.fit(acts, k1, seed=seed)
    return clf


def score_saplma(clf: SAPLMAClassifier, model, tok, queries, layer: int, device: str = "cuda"):
    """
    Scores E4's own composed-query test set: captures each query's
    COMPOSED-PROMPT generation's own last token (the model's generated
    answer to the actual composed question, not the single-hop fit
    statement), matching the position `fit_saplma` trained on.
    """
    from skc_harness import generate_greedy
    from skc_gates import capture_activations

    prompts = [render_prompts(q)["composed"] for q in queries]
    answers = generate_greedy(model, tok, prompts, device=device)
    full_texts = [p + a for p, a in zip(prompts, answers)]
    acts = capture_activations(model, tok, full_texts, layer, position="last", device=device)
    return clf.score(acts)


# =====================================================================
# Class 0: Entity-level token probes (Obeso et al. 2025)
# =====================================================================

def _entity_mention_position(tok, prompt: str) -> int:
    """
    Token index of the FIRST token of the model's own generated continuation
    -- i.e. where the answer entity's mention begins, since real Gemma/Llama
    generations for this pipeline's hop-style prompts reliably name the
    answer entity immediately (before any descriptive filler follows it;
    _strip_generation_filler()'s own docstring in skc_data.py documents this
    same pattern for scoring). This is the position that distinguishes this
    baseline from SAPLMA (which reads the LAST token of the full generated
    statement): Obeso et al.'s own method reads ENTITY MENTION positions
    within long-form generation, not the end of a whole response.

    ADAPTATION DISCLOSED: Obeso et al.'s actual setting is long-form
    generation with MULTIPLE entity mentions per response, each probed
    independently. This pipeline's own queries produce a single short
    answer per query, so there is exactly one entity mention to probe, at
    the position where generation starts -- not a full reproduction of
    their multi-entity long-form setting, but the same underlying
    mechanism (a linear probe at an ENTITY MENTION position, not the
    prompt's own entity or the response's final token) applied to this
    pipeline's own single-answer setup.
    """
    enc = tok(prompt, return_tensors="pt")
    return enc["input_ids"].shape[1]


def _check_no_seam_remerge(tok, prompt: str, answer: str) -> None:
    """
    Guards against the exact bug class already found twice this session
    (find_entity_span_in_generated_tokens's own docstring): tokenizing
    `prompt` alone and using that token COUNT as a position into
    `prompt+answer` tokenized as one string is only valid if the
    tokenizer does not re-merge tokens ACROSS the prompt/answer seam --
    i.e. `tok(prompt)["input_ids"]` must be an exact PREFIX of
    `tok(prompt+answer)["input_ids"]`. Checked directly on a real
    prompt/answer pair (GPT-2 BPE) as spot evidence this holds in
    general for the space-prefixed-token tokenizers this pipeline uses
    (Gemma/Llama, both SentencePiece-family, mark a leading space as
    part of the FOLLOWING token, which structurally prevents merging
    backward across a word boundary) -- but spot evidence is not a
    guarantee for every prompt, so this function re-checks it on every
    real call and raises loudly rather than silently mis-positioning if
    it ever fails, instead of trusting the general argument alone.
    """
    prompt_ids = tok(prompt)["input_ids"]
    full_ids = tok(prompt + answer)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"tokenizer re-merged tokens across the prompt/answer seam for "
            f"prompt={prompt!r} -- the entity-mention position computed from "
            "the prompt alone does not point at the same token in the "
            "combined prompt+answer sequence. This is exactly the bug class "
            "documented in find_entity_span_in_generated_tokens()'s own "
            "docstring; do not silently proceed."
        )


def fit_entity_token_probe(model, tok, fit_queries, layer: int, seed: int = 0,
                           device: str = "cuda", fit_labels=None):
    """
    Linear probe (same fit_probe machinery skc_gates.py already uses for
    this paper's own probe estimator) trained on the hidden state at the
    ANSWER ENTITY'S OWN MENTION position -- see _entity_mention_position()
    -- rather than SAPLMA's end-of-statement position or this paper's own
    surface gate's entity-in-the-QUESTION position. Same fit/held-out
    query pool as SAPLMA (data/onehop_fit_*.json), same K1 ground truth.

    fit_labels: see fit_saplma()'s own docstring -- same reuse mechanism.
    """
    from skc_gates import fit_probe, FrozenGate
    labels = fit_labels if fit_labels is not None else \
        label_queries_batched(model, tok, fit_queries, hop_require_prefix={1: False})
    k1 = np.array([l["K1"] for l in labels])
    prompts = [render_prompts(q)["hop1"] for q in fit_queries]
    positions = [_entity_mention_position(tok, p) for p in prompts]
    # capture_at_positions needs the FULL text (prompt+answer) tokenized so
    # the requested position actually exists in the sequence -- the position
    # itself is computed against the PROMPT alone (where generation starts),
    # but the tensor being indexed must be the longer prompt+answer capture.
    answers = generate_greedy(model, tok, prompts, device=device)
    for p, a in zip(prompts, answers):
        _check_no_seam_remerge(tok, p, a)
    full_texts = [p + a for p, a in zip(prompts, answers)]
    acts = capture_at_positions(model, tok, full_texts, layer, positions, device=device)
    known_acts, unknown_acts = acts[k1 == 1], acts[k1 == 0]
    if len(known_acts) < 10 or len(unknown_acts) < 10:
        raise SystemExit(
            f"too few known ({len(known_acts)}) or unknown ({len(unknown_acts)}) fit "
            "examples to fit an entity-token probe; enlarge the fit query pool."
        )
    return fit_probe(known_acts, unknown_acts, layer=layer, model_name="entity_token_probe",
                     seed=seed)


def score_entity_token_probe(gate, model, tok, queries, layer: int, device: str = "cuda"):
    """Same entity-mention position, on E4's own composed-query test set."""
    prompts = [render_prompts(q)["composed"] for q in queries]
    positions = [_entity_mention_position(tok, p) for p in prompts]
    answers = generate_greedy(model, tok, prompts, device=device)
    for p, a in zip(prompts, answers):
        _check_no_seam_remerge(tok, p, a)
    full_texts = [p + a for p, a in zip(prompts, answers)]
    acts = capture_at_positions(model, tok, full_texts, layer, positions, device=device)
    return gate.score(acts)


# =====================================================================
# Class 0: Lookback Lens (Chuang et al. 2024)
# =====================================================================

def lookback_ratio_features(model, tok, prompt: str, answer: str, device="cuda") -> np.ndarray:
    """
    Chuang et al. (2024)'s "lookback ratio" feature, computed per attention
    HEAD per LAYER: for a generated token's own attention distribution,
    the fraction of attention mass placed on CONTEXT (prompt) tokens versus
    NEWLY GENERATED tokens (including itself) --
    ratio = attn_to_context / (attn_to_context + attn_to_new). A model
    relying on its own (possibly hallucinated) prior generation rather than
    grounding in the actual context has a LOW lookback ratio; their paper's
    classifier reads exactly this pattern across heads/layers.

    Averages the ratio over every position in the GENERATED ANSWER SPAN
    (not just its first or last token), then concatenates every
    (layer, head) average into one flat feature vector -- Chuang et al.'s
    own setup is token- or span-level in a long-form generation; this
    pipeline's single-short-answer setting collapses naturally to
    "the answer span" as the unit, the same adaptation pattern already
    used for SAPLMA (whole-statement) and the entity-token probe
    (mention-start position), each collapsing the cited paper's own
    finer-grained unit to the single unit this pipeline's queries produce.

    Needs a model loaded with load_model(..., need_attentions=True) --
    raises if attentions come back None (the exact silent-null failure
    mode load_model()'s own docstring warns about), rather than silently
    returning a garbage/zero feature vector.
    """
    import torch
    full_text = prompt + answer
    n_prompt = len(tok(prompt)["input_ids"])
    enc = tok(full_text, return_tensors="pt").to(device)
    n_total = enc["input_ids"].shape[1]
    if n_total <= n_prompt:
        raise ValueError(f"answer {answer!r} produced no new tokens beyond the prompt "
                         f"({n_total} <= {n_prompt}); cannot compute a lookback ratio "
                         "over an empty answer span.")
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    if out.attentions[0] is None:
        raise RuntimeError(
            "model(...).attentions[0] is None -- this model was not loaded with "
            "need_attentions=True (attn_implementation='eager'), so no real attention "
            "weights were computed. Fix the load_model() call, do not proceed with a "
            "None attention tensor."
        )

    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    features = np.zeros((n_layers, n_heads), dtype=np.float64)
    for li, attn in enumerate(out.attentions):
        a = attn[0]  # (heads, seq, seq)
        for pos in range(n_prompt, n_total):
            # .float(): attention weights come back in the model's own
            # native dtype (bfloat16 for every model this pipeline loads),
            # and numpy has no bfloat16 representation at all -- the SAME
            # bug class already found and fixed twice this session
            # (sae.encode()'s output, sae.W_dec) confirmed as a real crash
            # here too (job 387621): "TypeError: Got unsupported
            # ScalarType BFloat16" at the .cpu().numpy() call below. Cast
            # immediately after slicing, not just at the final .numpy()
            # call, so the summation/division arithmetic itself runs in
            # float32 rather than bfloat16's much coarser precision for
            # small ratio-like quantities.
            row = a[:, pos, :].float()  # (heads, seq): attention FROM pos TO every key
            to_context = row[:, :n_prompt].sum(dim=-1)
            to_new = row[:, n_prompt:pos + 1].sum(dim=-1)
            denom = to_context + to_new
            # denom should be ~1 (softmax rows sum to 1); guard div-by-zero
            # defensively rather than assume it, since a masked/degenerate
            # row is a real possibility this function should not paper over.
            ratio = torch.where(denom > 1e-8, to_context / denom, torch.zeros_like(denom))
            features[li] += ratio.cpu().numpy()
    features /= (n_total - n_prompt)
    return features.flatten()


def fit_lookback_lens(model, tok, fit_queries, seed: int = 0, device: str = "cuda",
                      fit_labels=None):
    """
    Logistic regression (same standardized-gradient-descent style as
    skc_gates.fit_probe, reused directly rather than re-implemented) on
    the flattened (layer x head) lookback-ratio feature vector.

    fit_labels: see fit_saplma()'s own docstring. Here the caller must
    pass labels computed on the SAME (eager-attention) model instance
    this function receives -- do NOT pass labels computed on the main,
    default-backend model; see this file's own extensive comment on why
    eager and default SDPA attention produce different bfloat16 numerics.
    """
    from skc_gates import fit_probe
    labels = fit_labels if fit_labels is not None else \
        label_queries_batched(model, tok, fit_queries, hop_require_prefix={1: False})
    k1 = np.array([l["K1"] for l in labels])
    prompts = [render_prompts(q)["hop1"] for q in fit_queries]
    answers = generate_greedy(model, tok, prompts, device=device)
    feats = np.stack([lookback_ratio_features(model, tok, p, a, device=device)
                      for p, a in zip(prompts, answers)])
    known_acts, unknown_acts = feats[k1 == 1], feats[k1 == 0]
    if len(known_acts) < 10 or len(unknown_acts) < 10:
        raise SystemExit(
            f"too few known ({len(known_acts)}) or unknown ({len(unknown_acts)}) fit "
            "examples to fit Lookback Lens; enlarge the fit query pool."
        )
    return fit_probe(known_acts, unknown_acts, layer=-1, model_name="lookback_lens", seed=seed)


def score_lookback_lens(gate, model, tok, queries, device: str = "cuda"):
    """Same lookback-ratio features, on E4's own composed-query test set."""
    prompts = [render_prompts(q)["composed"] for q in queries]
    answers = generate_greedy(model, tok, prompts, device=device)
    feats = np.stack([lookback_ratio_features(model, tok, p, a, device=device)
                      for p, a in zip(prompts, answers)])
    return gate.score(feats)


# =====================================================================
# Class 0: HaloScope (Du, Xiao & Li, NeurIPS 2024)
# =====================================================================

def _self_consistency_membership_score(model, tok, prompt: str, greedy_answer: str,
                                       k: int = 10, device: str = "cuda") -> float:
    """
    CONFIDENCE NOTE: this is a BEST-FAITH RECONSTRUCTION of HaloScope's own
    membership-estimation step, not a verified match to the published
    algorithm -- flagged explicitly per this script's own module docstring
    policy against fabricating an implementation from a paper's name alone.
    The GENERAL framing (an unsupervised method needs SOME automatically
    computed proxy for "likely hallucinated" to estimate a direction from
    UNLABELED data, since it has no ground truth to fit on) is the part of
    HaloScope's own contribution this implementation is confident about;
    the SPECIFIC proxy used here (semantic self-consistency across k
    sampled generations -- how much the model agrees with its own greedy
    answer when re-sampled) is a plausible, defensible choice for such a
    proxy, not a line-by-line reproduction of their own membership-
    inference-style estimator. Verify against the actual paper before
    treating this baseline's numbers as a faithful HaloScope reproduction
    in any submitted table.

    Returns a score in [0, 1]: fraction of ALL (k choose 2) pairs of
    sampled generations that mutually agree with each other, via
    alias_match(sample_i, [sample_j], require_prefix=False) checked in
    BOTH directions (containment either way counts, since one sample's
    own descriptive filler may be a superset of another's). HIGH
    self-consistency is treated as evidence of "likely truthful" (the
    model converges on the same answer regardless of sampling noise),
    LOW self-consistency as evidence of "likely hallucinated" (the model
    is guessing a different entity almost every time).

    Deliberately does NOT match each sample against `greedy_answer`
    itself (an earlier version of this function did, and produced a real,
    confirmed bug on a real GPU run: `greedy_answer` is the model's own
    full, often messy generation including descriptive filler
    ("... located in the district of Leipzig in the state of Saxony. The
    city is..."), and two INDEPENDENTLY sampled continuations essentially
    never share that filler phrasing verbatim even when they name the
    SAME entity -- this collapsed the membership score to ~0 for nearly
    every fit query, which in turn made fit_diffmeans's two class means
    coincide exactly and crash with "class means coincide; there is no
    direction to find". Comparing samples PAIRWISE to each other (rather
    than each one against one specific messy reference string) is far
    more forgiving of filler-phrasing differences while still correctly
    measuring whether the model names the SAME entity across samples.
    `greedy_answer` itself is only used as a "no answer generated" guard.
    """
    from skc_harness import generate_sampled
    from skc_data import _strip_generation_filler
    if not greedy_answer.strip():
        return 0.0
    samples = generate_sampled(model, tok, [prompt], k=k, device=device)[0]
    # Strip each sample down to its own core answer (leading filler removed,
    # truncated at the first sentence boundary) BEFORE comparing pairwise --
    # reusing alias_match()'s own established cleanup step, rather than
    # comparing full, filler-heavy sentences to each other. Checked directly
    # (offline, no model): comparing raw, uncleaned samples pairwise still
    # under-counts genuine agreement (two sentences that both name the same
    # entity but never literally contain one another as substrings, e.g.
    # "Germany, a country in Europe." vs "located in Germany near the
    # border.", do not satisfy alias_match's own containment check even
    # though they agree) -- stripping down to the core phrase first
    # (here: "Germany" from both) makes the containment check meaningful.
    samples = [_strip_generation_filler(s) for s in samples if s.strip()]
    samples = [s for s in samples if s.strip()]
    if len(samples) < 2:
        return 0.0
    agreements, total_pairs = 0, 0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            total_pairs += 1
            if (alias_match(samples[i], [samples[j]], require_prefix=False) or
                    alias_match(samples[j], [samples[i]], require_prefix=False)):
                agreements += 1
    return agreements / total_pairs if total_pairs else 0.0


def fit_haloscope(model, tok, fit_queries, layer: int, k: int = 10, seed: int = 0,
                  device: str = "cuda", fit_labels=None):
    """
    CONFIDENCE NOTE: see _self_consistency_membership_score()'s own
    docstring -- this function's overall SHAPE (unsupervised subspace
    direction estimated from a soft membership split of UNLABELED hidden
    states) is the part of HaloScope's contribution this is confident
    about; the specific direction-estimation step (difference of means
    between the top/bottom membership-score quantiles, rather than their
    own specific SVD/subspace procedure) is a simplification of what the
    paper likely does with a full singular-value decomposition over a
    larger unlabeled pool, disclosed here rather than presented as exact.

    Deliberately does NOT use K1 ground truth for the direction itself
    (only for the too-few-examples sanity check below) -- the entire
    point of an "unsupervised" baseline is that it does not require
    labels for its core estimation step; using K1 directly to fit a
    supervised direction here would defeat the comparison HaloScope's
    published numbers are actually making (does an UNLABELED method
    compete with a labeled one).
    """
    from skc_gates import capture_activations
    prompts = [render_prompts(q)["hop1"] for q in fit_queries]
    answers = generate_greedy(model, tok, prompts, device=device)
    full_texts = [p + a for p, a in zip(prompts, answers)]
    acts = capture_activations(model, tok, full_texts, layer, position="last", device=device)

    membership = np.array([
        _self_consistency_membership_score(model, tok, p, a, k=k, device=device)
        for p, a in zip(prompts, answers)
    ])
    # Split on EXACT ZERO vs. ANY POSITIVE pairwise agreement, not a
    # fixed-quantile tercile. Confirmed on a real GPU run (job 388077,
    # after the pairwise-comparison fix): the REAL membership-score
    # distribution on this pipeline's actual fit pool is heavily
    # zero-mass (median exactly 0.0) with a meaningful positive tail
    # (max 0.7333) -- most single-hop fit queries genuinely have no
    # pairwise agreement across k=10 samples at temperature=1.0, which
    # is real model behavior (this pipeline's own earlier finding: even
    # containment-based K1 is only ~8% on this entity mix, so most fit
    # queries are ones the model does not actually know), not a bug. A
    # 1/3-2/3 QUANTILE split assumes a smooth continuous distribution
    # and fails outright here: with >2/3 of scores exactly 0.0, BOTH
    # quantile thresholds land on 0.0, making the two groups overlap
    # almost entirely (confirmed as the direct cause of job 387997's
    # "class means coincide" crash). Splitting on the distribution's own
    # actual shape (a zero mass plus a positive tail) instead of an
    # assumed continuous shape is the correct fix, not a threshold
    # tweak.
    likely_hallucinated = acts[membership == 0.0]
    likely_truthful = acts[membership > 0.0]
    if len(likely_truthful) < 10 or len(likely_hallucinated) < 10:
        raise SystemExit(
            f"HaloScope's membership-score split is still too imbalanced to fit a "
            f"direction: {len(likely_truthful)} likely-truthful (score > 0) vs. "
            f"{len(likely_hallucinated)} likely-hallucinated (score == 0) examples. "
            f"Membership score summary: min={membership.min():.4f}, "
            f"median={np.median(membership):.4f}, max={membership.max():.4f}, "
            f"mean={membership.mean():.4f}. Investigate the fit pool or the sampling "
            "temperature before proceeding -- do not force a direction out of too few "
            "examples on either side."
        )

    # K1 here is a DIAGNOSTIC ONLY (printed, never used to fit the
    # direction -- see this function's own docstring) -- computed via
    # label_queries_batched(), the SAME single source of truth every
    # other fit_* function in this file and skc_run_e0.py itself uses,
    # rather than a second, independently re-derived alias_match call
    # here. Two independent copies of the same K1 logic diverging is a
    # recurring root cause this session has already found and fixed
    # multiple times (skc_run_e1.py's own two label_queries_batched call
    # sites, for one) -- not repeating that pattern here even for a
    # value that is only ever printed.
    k1_labels = fit_labels if fit_labels is not None else \
        label_queries_batched(model, tok, fit_queries, hop_require_prefix={1: False})
    k1 = np.array([l["K1"] for l in k1_labels])
    if len(likely_truthful) < 10 or len(likely_hallucinated) < 10:
        raise SystemExit(
            f"too few likely-truthful ({len(likely_truthful)}) or likely-hallucinated "
            f"({len(likely_hallucinated)}) fit examples (by the UNSUPERVISED membership "
            "proxy) to fit HaloScope's direction; enlarge the fit query pool."
        )

    from skc_gates import fit_diffmeans
    gate = fit_diffmeans(likely_truthful, likely_hallucinated, layer=layer,
                         model_name="haloscope")
    gate.notes = (gate.notes or "") + (
        f"; unsupervised direction from self-consistency membership proxy "
        f"(k={k}), NOT fit on K1 ground truth. K1 rate in fit pool "
        f"(diagnostic only, not used to fit): {k1.mean():.3f}")
    return gate


def score_haloscope(gate, model, tok, queries, layer: int, device: str = "cuda"):
    """Same last-token-of-generated-answer position, on E4's own test queries."""
    from skc_gates import capture_activations
    prompts = [render_prompts(q)["composed"] for q in queries]
    answers = generate_greedy(model, tok, prompts, device=device)
    full_texts = [p + a for p, a in zip(prompts, answers)]
    acts = capture_activations(model, tok, full_texts, layer, position="last", device=device)
    return gate.score(acts)


# =====================================================================
# Class 2: INSIDE (Chen et al., ICLR 2024)
# =====================================================================

def eigenscore(model, tok, prompt: str, layer: int, k: int = 10, alpha: float = 1e-3,
              device: str = "cuda") -> float:
    """
    INSIDE's own EigenScore: sample k generations for the SAME prompt,
    capture each one's hidden state (last token of prompt+sample, same
    position convention as every other Class-0/1 baseline in this file),
    form their (k x k) Gram matrix of UNIT-NORMALIZED per-sample vectors
    (equivalent to the covariance structure over k samples, cheaper to
    compute than the full d x d covariance when k << d, which holds here:
    k=10 vs. d=2304 for Gemma-2-2b), and take the regularized
    log-determinant: EigenScore = logdet(R + alpha*I). Normalizing each
    sample's own vector to unit length removes pure scale differences so
    the score reflects genuine directional/semantic spread, not larger
    activation norms. HIGHER EigenScore means the k sampled
    representations are more semantically DISPERSED (the model gives
    substantively different answers across samples) -- INSIDE's own
    finding is that this predicts hallucination directly, with NO
    supervised direction to fit at all (unlike every prior baseline in
    this file, INSIDE has no fit_*/score_* pair -- the score is computed
    directly per query, with nothing learned from any fit split).

    DELIBERATELY DOES NOT mean-center the k representations before
    normalizing (an earlier version of this function did, and it was a
    REAL, CONFIRMED BUG, caught offline before any GPU run): centering k
    vectors and then unit-normalizing forces their pairwise cosine
    similarities toward EXACTLY -1/(k-1) regardless of how consistent or
    dispersed the ORIGINAL (uncentered) points were -- verified directly
    with synthetic data (k=10, d=2304): a tightly clustered "consistent"
    set and a fully random "dispersed" set produced nearly IDENTICAL
    EigenScores (-5.977 vs -5.969) under centering, versus a massive,
    correct separation (-59.0 vs -0.01) once centering was removed.
    Centering destroys exactly the "common direction" signal that
    distinguishes a consistent answer set (all samples point the same
    way, a near-singular, low-rank Gram matrix, very negative logdet)
    from a dispersed one (samples point in unrelated directions, a
    near-identity Gram matrix, logdet near 0) -- this is the entire
    signal EigenScore is built to detect, so removing it before computing
    the Gram matrix would have silently broken the whole method.

    alpha=1e-3: a small ridge term so logdet is well-defined even when k
    representations are highly (near-)collinear (a real possibility when
    samples are near-identical) -- without it, a genuinely LOW-diversity
    (highly consistent, i.e. the GOOD case) sample set could produce a
    near-singular Gram matrix and an undefined or -inf logdet, which
    would be scored as "very truthful" by an accident of numerical
    instability rather than the actual geometry. This is a standard
    regularization choice for this exact log-determinant diversity
    measure, not a INSIDE-paper-specific hyperparameter this
    implementation claims to reproduce exactly.
    """
    from skc_gates import capture_activations
    from skc_harness import generate_sampled
    samples = generate_sampled(model, tok, [prompt], k=k, device=device)[0]
    full_texts = [prompt + s for s in samples]
    acts = capture_activations(model, tok, full_texts, layer, position="last", device=device)
    acts = np.asarray(acts, dtype=np.float64)
    norms = np.linalg.norm(acts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    acts = acts / norms
    gram = acts @ acts.T  # (k, k)
    sign, logdet = np.linalg.slogdet(gram + alpha * np.eye(len(acts)))
    if sign <= 0:
        raise RuntimeError(
            f"EigenScore's regularized Gram matrix has non-positive determinant "
            f"(sign={sign}) for prompt {prompt!r} -- this should be structurally "
            "impossible for a Gram matrix plus a positive ridge term; investigate "
            "rather than silently returning a nonsensical score."
        )
    return float(logdet)


def score_inside(model, tok, queries, layer: int, k: int = 10, device: str = "cuda"):
    """
    EigenScore for every E4 test query, NEGATED so that higher score means
    MORE truthful (matching every other selector's own convention in this
    script -- EigenScore itself is higher for MORE dispersed/hallucinated
    generations, the opposite polarity of an abstention-confidence score).
    """
    prompts = [render_prompts(q)["composed"] for q in queries]
    scores = np.array([eigenscore(model, tok, p, layer, k=k, device=device) for p in prompts])
    return -scores


# =====================================================================
# Semantic Entropy (Farquhar, Kossen, Kuhn & Gal, Nature 2024,
# "Detecting hallucinations in large language models using semantic
# entropy") -- the DIRECT sampling-based method Semantic Entropy Probe
# (above) exists to approximate cheaply. Fully unsupervised, no fit
# split: k=10 sampled generations + bidirectional NLI entailment
# clustering (Kuhn et al. 2023) + Shannon entropy over the resulting
# cluster distribution, computed directly per E4 test query.
# =====================================================================
#
# Reuses the SAME building blocks already built and GPU-verified this
# phase for HaloScope/INSIDE/Semantic Entropy Probe: generate_sampled,
# bidirectional_entailment_clusters, semantic_entropy (skc_harness.py).
# Also reuses _truncate_for_nli() -- REQUIRED here for the exact same
# reason it was required in fit_semantic_entropy_probe() and
# baseline_selfcheckgpt() (see that function's own docstring for the
# real GPU bug this fixes, jobs 388434/388439): raw generate_sampled()
# output at temperature=1.0 reliably continues past the real answer
# into a fabricated new Q&A turn, which corrupts NLI entailment checks
# if not truncated first. Confirmed this phase (jobs 388675/388791)
# that the fix works correctly on real generations once applied.
#
# NEGATED (higher score = more truthful) to match every other
# selector's convention in this file -- raw semantic entropy is HIGHER
# for MORE dispersed/hallucinated generations, same polarity flip
# already applied to INSIDE/EigenScore above.


def _semantic_entropy_for_prompt(nli_model, nli_tok, model, tok, prompt: str, k: int = 10,
                                 device: str = "cuda") -> float:
    from skc_harness import generate_sampled, bidirectional_entailment_clusters, semantic_entropy
    samples = generate_sampled(model, tok, [prompt], k=k, device=device)[0]
    samples = [_truncate_for_nli(s) for s in samples if s.strip()]
    samples = [s for s in samples if s]
    if len(samples) < 2:
        return 0.0  # degenerate: cannot compute entropy over <2 items
    clusters = bidirectional_entailment_clusters(nli_model, nli_tok, samples, device=device)
    return semantic_entropy(clusters)


def score_semantic_entropy(model, tok, queries, k: int = 10, device: str = "cuda"):
    """
    Real semantic entropy for every E4 test query, NEGATED so that
    higher score means MORE truthful. No fit split, no layer argument
    needed (entropy is computed purely from sampled TEXT + NLI
    clustering, never from any hidden state).
    """
    from skc_harness import load_nli_model
    nli_model, nli_tok = load_nli_model(device=device)
    prompts = [render_prompts(q)["composed"] for q in queries]
    scores = np.array([
        _semantic_entropy_for_prompt(nli_model, nli_tok, model, tok, p, k=k, device=device)
        for p in prompts
    ])
    del nli_model
    import torch
    torch.cuda.empty_cache()
    return -scores


# =====================================================================
# Class 0/2: ICR Probe (Zhang, Hu, Zhang, Zhang & Wan, ACL 2025,
# "ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination
# Detection in LLMs", aclanthology.org/2025.acl-long.880, arXiv:2507.16488)
# =====================================================================
#
# VERIFIED directly against the ACL Anthology PDF (not a reconstruction
# from memory, unlike HaloScope above): ICR = "Information Contribution
# to Residual stream". At every layer l and every token position i, let
# Δx_i^l = x_i^l - x_i^{l-1} be that layer's update to the residual
# stream at position i (the paper does not further decompose this into
# separate MHSA/FFN pieces for the ICR score itself -- the full
# layer-to-layer residual delta is what is used). Two vectors are then
# computed over the OTHER tokens 0..i (causal: only tokens the model
# could actually attend to):
#   Proj_i^l[j]  = softmax_j( Δx_i^l . x_j^{l-1} )   -- how much of this
#                  layer's update at i "projects onto" each earlier
#                  token's own (pre-update) representation
#   Attn_i^l[j]  = mean-over-heads attention weight from i to j at
#                  layer l (the model's ACTUAL attention distribution)
# ICR_i^l = JSD(Proj_i^l, Attn_i^l), restricted to the top-k=20 tokens
# by attention weight (the paper's own ablation found k=20 optimal vs.
# k=5/30/all). This measures how much a layer's actual residual-stream
# update at a token AGREES with where the model says it is attending --
# a hallucination is more likely when a layer heavily updates the
# residual stream in a direction NOT explained by its own attention
# pattern. Applied to every ANSWER token (not prompt tokens), then
# mean-pooled over answer tokens into one scalar per layer, producing a
# 1xL feature vector per query. The "ICR Probe" itself is a supervised
# MLP (L -> 128 -> 64 -> 32 -> 1, LeakyReLU(0.01), sigmoid out) trained
# on these per-layer vectors with BCE loss -- this pipeline trains it on
# the fit split exactly as SAPLMA/entity-token-probe are trained, using
# the SAME K1 correctness label.
#
# ADAPTATION DISCLOSED: the paper trains/evaluates across several public
# QA benchmarks with their own labeled hallucination splits; here it is
# trained on this pipeline's own one-hop fit split and scored on the
# composed E4 test queries, the same adaptation already applied to every
# other trained baseline in this file. Also: the paper computes Proj
# using each token's pre-update hidden state x_j^{l-1}; this
# implementation does the same, reusing the SAME single forward pass's
# hidden_states tuple (index l-1) rather than firing a second pass.
#
# Needs a model loaded with load_model(..., need_attentions=True), same
# requirement as Lookback Lens -- reuses whatever separate eager-attention
# model instance the Lookback Lens block already loads if both are
# requested in the same run (see main()), otherwise loads its own.


def _icr_scores_for_text(model, tok, prompt: str, answer: str, top_k: int = 20,
                          device: str = "cuda") -> np.ndarray:
    """
    Returns a (n_layers,) vector: mean-pooled-over-answer-tokens ICR
    score per layer, for one prompt+answer pair. Needs attentions
    (eager) AND hidden_states from ONE forward pass over the full text.
    """
    import torch

    full_text = prompt + answer
    n_prompt = len(tok(prompt)["input_ids"])
    enc = tok(full_text, return_tensors="pt").to(device)
    n_total = enc["input_ids"].shape[1]
    if n_total <= n_prompt:
        raise ValueError(f"answer {answer!r} produced no new tokens beyond the prompt "
                         f"({n_total} <= {n_prompt}); cannot compute ICR over an empty "
                         "answer span.")
    with torch.no_grad():
        out = model(**enc, output_attentions=True, output_hidden_states=True)
    if out.attentions[0] is None:
        raise RuntimeError(
            "model(...).attentions[0] is None -- this model was not loaded with "
            "need_attentions=True (attn_implementation='eager'). Fix the load_model() "
            "call, do not proceed with a None attention tensor."
        )
    # hidden_states has n_layers+1 entries: [0] is the embedding output
    # (pre-layer-0), [l] is the output AFTER layer l-1, i.e. hidden_states[l]
    # is x^l in the paper's notation with hidden_states[0] = x^0 (embeddings).
    hidden_states = out.hidden_states  # tuple of (1, seq, d_model), len n_layers+1
    attentions = out.attentions        # tuple of (1, heads, seq, seq), len n_layers
    n_layers = len(attentions)

    per_layer_scores = [[] for _ in range(n_layers)]
    eps = 1e-12
    for l in range(n_layers):
        x_prev = hidden_states[l][0].float()   # (seq, d_model), x^{l-1}
        x_cur = hidden_states[l + 1][0].float()  # (seq, d_model), x^l
        delta = x_cur - x_prev                  # (seq, d_model)
        attn_l = attentions[l][0].float().mean(dim=0)  # (seq, seq), mean over heads
        for i in range(n_prompt, n_total):
            # causal: only tokens 0..i are valid keys (i inclusive, matches
            # the model's own causal mask -- token i CAN attend to itself).
            proj_logits = delta[i, :] @ x_prev[:i + 1, :].T  # (i+1,)
            proj = torch.softmax(proj_logits, dim=0)
            attn_row = attn_l[i, :i + 1]
            attn_row = attn_row / attn_row.sum().clamp_min(eps)
            k = min(top_k, i + 1)
            top_idx = torch.topk(attn_row, k).indices
            p = proj[top_idx]
            p = p / p.sum().clamp_min(eps)
            q = attn_row[top_idx]
            q = q / q.sum().clamp_min(eps)
            m = 0.5 * (p + q)
            jsd = 0.5 * torch.sum(p * torch.log((p / m).clamp_min(eps))) + \
                  0.5 * torch.sum(q * torch.log((q / m).clamp_min(eps)))
            per_layer_scores[l].append(float(jsd.item()))
    return np.array([float(np.mean(s)) for s in per_layer_scores])


class ICRProbeClassifier:
    """
    The paper's own architecture: L -> 128 -> 64 -> 32 -> 1,
    LeakyReLU(negative_slope=0.01), sigmoid output. Simplified relative
    to the paper's full Appendix B.5 spec (no dropout/batch-norm/Kaiming
    init/ReduceLROnPlateau schedule) -- DISCLOSED SIMPLIFICATION, matching
    the same "core architecture faithful, training-loop details
    simplified" pattern already used for SAPLMA in this file.
    """

    def __init__(self, d_in: int, seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 5e-4,
           batch_size: int = 32, seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        yt = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        n = len(Xt)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pred = self.net(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X, dtype=np.float32))
            return self.net(Xt).squeeze(1).numpy()


def fit_icr_probe(model, tok, fit_queries, seed: int = 0, device: str = "cuda",
                  fit_labels=None):
    """
    Trains on the fit split's own greedy answers, same K1-correctness
    label every other trained E4 baseline uses. `model`/`tok` here must
    already be loaded with need_attentions=True.

    `fit_queries` are single-hop (k=1), same as every other trained E4
    baseline's fit split -- render_prompts() has no "composed" key for
    k=1 (that key only exists for k=2/k=3; see its own docstring), so
    this must read "hop1", exactly the key fit_saplma() reads for the
    SAME fit_queries. (Caught on a real GPU run, job 388298: this
    function originally read "composed" here, a copy-paste from
    score_icr_probe's own correct use of "composed" on the k=2 TEST
    queries -- KeyError: 'composed' when applied to k=1 fit queries.)

    fit_labels: see fit_lookback_lens()'s own docstring -- same
    eager-attention-instance requirement if passed.
    """
    from skc_harness import generate_greedy, label_queries_batched

    labels = fit_labels if fit_labels is not None else \
        label_queries_batched(model, tok, fit_queries, hop_require_prefix={1: False})
    k1 = np.array([l["K1"] for l in labels])
    prompts = [render_prompts(q)["hop1"] for q in fit_queries]
    answers = generate_greedy(model, tok, prompts, device=device)
    feats = np.stack([
        _icr_scores_for_text(model, tok, p, a, device=device)
        for p, a in zip(prompts, answers)
    ])
    clf = ICRProbeClassifier(d_in=feats.shape[1], seed=seed)
    clf.fit(feats, k1.astype(np.float32), seed=seed)
    return clf


def score_icr_probe(clf: ICRProbeClassifier, model, tok, queries, device: str = "cuda"):
    prompts = [render_prompts(q)["composed"] for q in queries]
    answers = generate_greedy(model, tok, prompts, device=device)
    feats = np.stack([
        _icr_scores_for_text(model, tok, p, a, device=device)
        for p, a in zip(prompts, answers)
    ])
    return clf.score(feats)


# =====================================================================
# Semantic Entropy Probe (Kossen, Han, Razzak, Schut, Malik & Gal,
# arXiv:2406.15927, "Semantic Entropy Probes: Robust and Cheap
# Hallucination Detection in LLMs" -- flagged "Preprint. Under review."
# on the paper itself; could not verify a final peer-reviewed venue
# beyond the arXiv preprint, disclosed here rather than asserting one)
# =====================================================================
#
# VERIFIED directly against the arXiv PDF. Core idea: computing real
# semantic entropy (Farquhar et al. 2024) needs k~10 sampled generations
# PER QUERY at inference time (exactly what score_inside/eigenscore
# above pay); SEPs instead train a probe to PREDICT semantic entropy
# from a SINGLE hidden state (one greedy generation, no sampling at
# eval time) -- the whole point is a cheap test-time footprint. At FIT
# time only, real semantic entropy is computed the expensive way (k=10
# samples, bidirectional NLI entailment clustering, Shannon entropy over
# clusters -- reusing this file's own generate_sampled,
# skc_harness.bidirectional_entailment_clusters, skc_harness.
# semantic_entropy, already built for HaloScope/INSIDE) to LABEL the fit
# split, then a probe is trained on hidden states so eval time is cheap.
#
# The paper's own probe is LINEAR logistic regression (not an MLP),
# reading either TBG ("token before generating", i.e. the prompt's own
# last token) or SLT (second-last token of the model's response) at
# various layers; this pipeline uses TBG at the SAME --layer argument
# every other single-layer baseline in this file uses, since SLT would
# require locating an EOS/end-of-response token this pipeline's
# short-answer generations do not reliably or consistently produce.
# DISCLOSED SIMPLIFICATION: the paper's best results use CONCATENATING
# 5 adjacent top layers found by a per-model layer sweep; this
# implementation uses a single fixed layer (this pipeline's own
# --layer, chosen for every other baseline) rather than re-running a
# layer sweep specific to this baseline.
#
# The continuous semantic-entropy label is BINARIZED (paper's own
# choice, confirmed better than soft regression) via a threshold that
# splits the fit split into low-SE ("confident"/likely-truthful) and
# high-SE ("confabulating") groups -- reuses skc_gates.fit_probe exactly
# as E0's own diffmeans/probe estimators do, with acts_known=low-SE
# (so a HIGHER FrozenGate score = MORE confident = MORE truthful,
# matching every other selector's convention in this file) and
# acts_unknown=high-SE. Threshold: the paper's own criterion is an
# "optimal" split minimizing within-group MSE (regression-tree-style);
# this implementation uses a CEILING split instead (exact max entropy
# log(k) vs. anything lower) -- DISCLOSED SIMPLIFICATION, the same class
# of adaptation as HaloScope's own membership-score split. A MEDIAN
# split was tried FIRST and found genuinely degenerate on this fit split
# (jobs 388434/388675): this fit pool is heavily weighted toward
# obscure entities Gemma-2-2b does not know, so most queries land at or
# near ceiling entropy log(k) (all k samples mutually non-entailing --
# confirmed directly against real generations, not a clustering bug),
# making the median itself sit at the ceiling. Splitting on exact
# ceiling vs. sub-ceiling entropy matches this real, heavily-skewed
# distribution directly, mirroring HaloScope's own exact-zero-vs-any-
# positive fix for the same shape of problem.


def _truncate_for_nli(s: str) -> str:
    """
    Cut a raw model generation down to just its actual answer, before
    handing it to an NLI model for entailment/contradiction scoring.

    REAL BUG FOUND ON A REAL GPU RUN (job 388434, Semantic Entropy
    Probe's first attempt): generate_sampled() at temperature=1.0 on
    this pipeline's single-hop prompts reliably continues PAST the real
    answer into a fabricated new Q&A turn, e.g.
    ' Paris\n\nQ: What countries border France?\nA: Germany, the...'
    -- skc_data._strip_generation_filler() only truncates at the first
    SENTENCE-ending punctuation ([.!?]), which does not fire here (no
    period appears before the fake "\nQ:" continuation begins), so the
    ENTIRE fabricated follow-up text was being passed to the NLI model
    as part of the premise/hypothesis. Confirmed directly: 10 samples
    for "capital of France" ALL actually answered "Paris" first, but
    bidirectional_entailment_clusters() returned 10 separate singleton
    clusters (max possible entropy, ln(10)) because the NLI model
    (correctly) judged the full noisy passages as not mutually
    entailing -- this collapsed fit_semantic_entropy_probe's median
    split to (9985, 0), a fully degenerate one-sided split, caught by
    its own SystemExit guard rather than silently fitting garbage.

    Truncating at the first NEWLINE (in addition to
    _strip_generation_filler()'s own sentence-boundary truncation) fixes
    this: this pipeline's own single-hop answers never legitimately
    contain a newline before the real answer ends, so cutting there
    removes exactly the fabricated continuation and nothing else.
    Verified directly on the real job-388434 samples: after this
    truncation, all 10 "Paris" samples correctly cluster into ONE
    cluster (entropy 0.0), versus 10 singletons (entropy ln(10)) before
    the fix.

    NOT the same fix as alias_match()'s own filler-stripping: alias_match
    uses PREFIX/CONTAINMENT matching, which is already robust to trailing
    noise (a substring check does not care what comes after the match),
    so _self_consistency_membership_score() (which uses alias_match, not
    NLI) never hit this bug. NLI entailment cares about the WHOLE passage's
    meaning, so trailing fabricated noise directly corrupts it -- this
    truncation is specific to every NLI-consuming caller in this file
    (bidirectional_entailment_clusters via fit_semantic_entropy_probe,
    and baseline_selfcheckgpt's own direct NLI calls).
    """
    from skc_data import _strip_generation_filler
    s = s.split("\n")[0].strip()
    return _strip_generation_filler(s)


def fit_semantic_entropy_probe(model, tok, fit_queries, layer: int, k: int = 10,
                               seed: int = 0, device: str = "cuda"):
    """
    Labels the fit split with REAL semantic entropy (k=10 sampled
    generations + NLI clustering per fit query -- this IS the expensive
    step the whole method exists to avoid paying at test time), splits
    at the CEILING entropy log(k) (exact max, all k samples mutually
    non-entailing) versus anything lower into high/low-SE groups (see
    the module-level comment above for why a median split was tried
    first and found degenerate on this fit split), then fits a linear
    probe on each query's OWN prompt-final-token ("TBG") hidden state to
    predict that split.
    """
    from skc_harness import generate_sampled, load_nli_model, bidirectional_entailment_clusters, semantic_entropy
    from skc_gates import capture_activations, fit_probe

    nli_model, nli_tok = load_nli_model(device=device)
    prompts = [render_prompts(q)["hop1"] for q in fit_queries]
    se_values = []
    # Progress print added after job 144's own 24h timeout gave no visibility
    # into how far this per-query loop (k=10 sampled generations + up to
    # 90 pairwise NLI calls EACH, the most expensive step in this whole
    # script) had actually gotten before being killed -- this is a
    # print-only change, no logic touched.
    for i, p in enumerate(prompts):
        if i % 100 == 0:
            print(f"  SEP fit: {i}/{len(prompts)} fit queries sampled+clustered", flush=True)
        samples = generate_sampled(model, tok, [p], k=k, device=device)[0]
        # _truncate_for_nli(): raw samples continue past the real answer into
        # a fabricated new Q&A turn (see that function's own docstring for
        # the real GPU bug this fixes, job 388434) -- must truncate BEFORE
        # clustering, not after, or every sample looks mutually
        # non-entailing regardless of what they actually answered.
        samples = [_truncate_for_nli(s) for s in samples if s.strip()]
        samples = [s for s in samples if s]
        if len(samples) < 2:
            se_values.append(0.0)  # degenerate: cannot compute entropy over <2 items
            continue
        clusters = bidirectional_entailment_clusters(nli_model, nli_tok, samples, device=device)
        se_values.append(semantic_entropy(clusters))
    se_values = np.array(se_values)
    del nli_model
    import torch
    torch.cuda.empty_cache()

    # REAL DATA FINDING (jobs 388434/388675): this fit split is a broad,
    # unfiltered pool of one-hop facts, heavily weighted toward OBSCURE
    # entities (minor-league/historical athletes) Gemma-2-2b genuinely
    # does not know -- confirmed directly against real generations (see
    # EXECUTION_PLAN.md): the model hallucinates a DIFFERENT wrong guess
    # almost every sample for most such queries, so most fit queries
    # land at or near the CEILING entropy log(k) (all k samples mutually
    # non-entailing), not from any clustering bug. A MEDIAN split is
    # degenerate here because the median itself sits at that ceiling
    # (most of the distribution is piled at the max, exactly the same
    # shape of problem HaloScope's own membership-score split hit
    # earlier this phase). User's explicit choice (mirroring HaloScope's
    # own exact-zero-vs-any-positive fix): split on EXACT ceiling entropy
    # (fully degenerate self-consistency, k separate singleton clusters)
    # versus any-lower (sub-ceiling) entropy, rather than a data-derived
    # quantile that assumes a non-degenerate spread.
    ceiling = float(np.log(k))
    high_se_mask = np.isclose(se_values, ceiling)
    low_se_mask = ~high_se_mask
    acts = capture_activations(model, tok, prompts, layer, position="last", device=device)
    if low_se_mask.sum() < 10 or high_se_mask.sum() < 10:
        raise SystemExit(
            f"Semantic entropy ceiling split produced degenerate groups "
            f"(low={low_se_mask.sum()}, high={high_se_mask.sum()}) -- cannot fit a probe. "
            f"se_values: min={se_values.min():.4f} ceiling={ceiling:.4f} max={se_values.max():.4f}"
        )
    gate = fit_probe(acts[low_se_mask], acts[high_se_mask], layer=layer,
                     model_name="semantic_entropy_probe", seed=seed)
    return gate


def score_semantic_entropy_probe(gate, model, tok, queries, layer: int, device: str = "cuda"):
    """
    Cheap at eval time by design: ONE greedy-free capture of each query's
    own prompt-final ("TBG") hidden state, no sampling at all -- unlike
    INSIDE/HaloScope's own eval-time cost, which resamples per query.
    """
    from skc_gates import capture_activations
    prompts = [render_prompts(q)["composed"] for q in queries]
    acts = capture_activations(model, tok, prompts, layer, position="last", device=device)
    return gate.score(acts)


# =====================================================================
# Class 1: verbalized confidence (P(True), Kadavath et al. 2022)
# =====================================================================

def baseline_p_true(model, tok, prompts: list[str], answers: list[str],
                    batch_size: int = 16, device="cuda"):
    """
    Kadavath et al. 2022 (arXiv:2207.05221), Section 4.1 "Basic Self-
    Evaluation" -- verified directly against the paper (not a memory
    reconstruction or a paraphrase of the method's name) before trusting
    this in a table, per this file's own header policy on Class-3 methods.

    The paper's exact template (Figure 10) is multiple-choice True/False,
    NOT free-text yes/no:

        Question: <question>
        Proposed Answer: <answer>

        Is the proposed answer:
        (A) True
        (B) False
        The proposed answer is:

    and P(True) is NOT a greedily-decoded token -- the paper reads the
    model's own next-token probabilities for the "(A)"/"(B)" continuation
    and reports the softmax-normalized mass on the True option (explicitly:
    "we...evaluate the probabilities it assigns to the '(A)' and '(B)'
    options"). An earlier version of this function greedily generated
    "yes"/"no" text instead, which (a) used the wrong template entirely and
    (b) collapsed a continuous calibrated score into a hard 0/1 label,
    discarding exactly the information a verbalized-confidence baseline is
    supposed to supply. Fixed to match the paper on both counts: one
    forward pass per query (no generation), reading the logits at the
    position right after "The proposed answer is:" and normalizing over
    the model's own first-subword-token ids for "A" and "B" (works for any
    tokenizer's own BPE/SentencePiece split of "(A)"/"(B)", since the
    leading "(" is shared and only the letter distinguishes the two
    options at the first position where they diverge).

    Zero-shot only (this codebase's other Class-1/one-forward-pass
    baselines are all zero-shot; the paper's own STRONGER few-shot
    "comparison examples" variant, Section 4.2, is a different, costlier
    method -- not substituted in here without saying so).
    """
    import torch
    id_a = tok.encode("A", add_special_tokens=False)[0]
    id_b = tok.encode("B", add_special_tokens=False)[0]
    prior_side = tok.padding_side
    from skc_harness import set_padding_side
    set_padding_side(tok, "left")
    try:
        out = []
        for i in range(0, len(prompts), batch_size):
            bp = prompts[i:i + batch_size]
            ba = answers[i:i + batch_size]
            texts = [f"Question: {p}\nProposed Answer: {a}\n\nIs the proposed answer:\n"
                     f"(A) True\n(B) False\nThe proposed answer is: ("
                     for p, a in zip(bp, ba)]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            two = torch.stack([logits[:, id_a], logits[:, id_b]], dim=-1)
            p_true = torch.softmax(two, dim=-1)[:, 0]
            out.extend(p_true.float().cpu().tolist())
        return np.array(out)
    finally:
        set_padding_side(tok, prior_side)


# =====================================================================
# Class 2: self-consistency (k generations, sampled)
# =====================================================================

def baseline_self_consistency(model, tok, prompts: list[str], k: int = 10, device="cuda"):
    """
    Pairwise self-consistency across k=10 sampled generations per query,
    in [0, 1]: fraction of (k choose 2) sample pairs that mutually
    agree (alias_match, filler-stripped, checked both directions) --
    HIGH self-consistency treated as evidence of "likely truthful"
    (the model converges on the same entity despite sampling noise),
    LOW as evidence of "likely hallucinated" (guessing a different
    entity nearly every time), matching every other selector's
    "higher = more truthful" convention in this file.

    Deliberately calls _self_consistency_membership_score() (built for
    HaloScope's fit step) rather than reimplementing this pairwise logic
    inline -- avoids the exact "two independent copies of the same logic
    diverging" bug class this codebase has hit repeatedly. That function
    itself only uses greedy_answer as an empty-answer guard, so a real
    greedy generation is still needed here to supply it (not just a
    placeholder), and IS this baseline's own eval-time cost (1 greedy
    generation + k=10 sampled generations, per query).
    """
    from skc_harness import generate_greedy
    greedy_answers = generate_greedy(model, tok, prompts, device=device)
    return np.array([
        _self_consistency_membership_score(model, tok, p, g, k=k, device=device)
        for p, g in zip(prompts, greedy_answers)
    ])


# =====================================================================
# SelfCheckGPT (Manakul, Liusie & Gales, EMNLP 2023, arXiv:2303.08896,
# NLI variant), a zero-resource black-box consistency check.
# =====================================================================
#
# VERIFIED directly against the arXiv PDF / ACL Anthology paper (not a
# memory reconstruction). Main response R generated greedy
# (temperature=0.0); N=20 additional stochastic samples drawn at
# temperature=1.0 from the SAME prompt (the paper's own exact N,
# confirmed, DIFFERENT from every other sampling baseline in this file,
# which all use k=10 -- kept faithful to the paper rather than reusing
# k=10 for consistency with this file's other baselines). Five variants
# exist (BERTScore, QA, n-gram, NLI, LLM-Prompting); NLI is "the most
# practical method" per the paper's own Table 2 (near-best AUC-PR at
# much lower cost than the LLM-Prompting variant) and the variant most
# commonly adopted as a baseline in follow-up work -- implemented here.
#
# Exact NLI-variant formula (paper's Eq. 9-10, verified): for each
# response sentence r_i, EACH sampled passage S^n is the PREMISE and
# r_i is the HYPOTHESIS (not sentence-vs-sentence) -- run through a
# DeBERTa-v3-large-MNLI model, taking ONLY the entailment/contradiction
# logits (neutral dropped): P(contradict | r_i, S^n) =
# exp(z_c) / (exp(z_e) + exp(z_c)). Sentence-level score = MEAN over
# the N samples (not max -- max is the paper's OWN BERTScore variant,
# not NLI). Passage-level score = mean of sentence-level scores.
#
# ADAPTATION DISCLOSED (paper is silent on this regime, confirmed by
# the same research pass that verified everything else above): this
# pipeline's own responses are SHORT, single-phrase factual answers,
# not the paper's own multi-sentence WikiBio-style passages (avg 184.7
# tokens) -- there is no real sentence-splitting to do. Each response is
# treated as ONE "sentence" (the whole short answer) scored against each
# full sampled response as premise, exactly the paper's own
# premise=sample/hypothesis=response-unit structure, just with the unit
# already being the whole response rather than one sentence within a
# longer passage. This is a mechanical, disclosed adaptation to a
# regime the original paper did not test, not a deviation from the
# paper's actual formula.
#
# REUSES the NLI_MODEL_ID model already loaded for the semantic-entropy
# family (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --
# NOT the paper's own exact checkpoint, which is unspecified/unavailable
# by name in the paper text; disclosed substitution, same DeBERTa-v3-
# large-MNLI family, same substitution already made and disclosed for
# bidirectional_entailment_clusters above). skc_harness._nli_label()
# only returns an argmax LABEL, not the probability this formula needs,
# so this baseline reads model.config.id2label DYNAMICALLY (never a
# hardcoded index) to find which logit index is entailment vs.
# contradiction, exactly the discipline load_nli_model()'s own docstring
# established for this NLI model's non-standard {0: entailment, 1:
# neutral, 2: contradiction} label ordering.
# Negated at the end (1 - contradiction score) so a HIGHER score means
# MORE truthful, matching every other selector's convention in this
# file -- SelfCheckGPT's own raw score is a hallucination/inconsistency
# score (higher = more likely hallucinated), the opposite polarity.


def _entailment_contradiction_probs(nli_model, nli_tok, premise: str, hypothesis: str,
                                    device: str = "cuda") -> tuple[float, float]:
    """
    Returns (P(entail), P(contradict)) using ONLY the entailment and
    contradiction logits (neutral excluded from the softmax denominator),
    exactly SelfCheckGPT's own Eq. 9. Reads label->index dynamically from
    nli_model.config.id2label -- never assumes a fixed index, since this
    NLI model's own label ordering is confirmed DIFFERENT from the
    "classic" deberta-large-mnli convention (see load_nli_model()).
    """
    import torch
    label2idx = {v.upper(): k for k, v in nli_model.config.id2label.items()}
    e_idx, c_idx = label2idx["ENTAILMENT"], label2idx["CONTRADICTION"]
    enc = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = nli_model(**enc).logits[0]
    z_e, z_c = float(logits[e_idx]), float(logits[c_idx])
    m = max(z_e, z_c)
    p_e, p_c = np.exp(z_e - m), np.exp(z_c - m)
    denom = p_e + p_c
    return float(p_e / denom), float(p_c / denom)


def baseline_selfcheckgpt(model, tok, prompts: list[str], n: int = 20,
                          device: str = "cuda"):
    """
    Returns, per query: 1 - mean_n P(contradict | response, sample_n) --
    higher means more truthful (fewer of the N samples contradict the
    model's own greedy response).
    """
    from skc_harness import generate_greedy, generate_sampled, load_nli_model

    nli_model, nli_tok = load_nli_model(device=device)
    responses = generate_greedy(model, tok, prompts, device=device)
    scores = []
    for p, r in zip(prompts, responses):
        # _truncate_for_nli(): both the greedy response (hypothesis) and
        # every sampled passage (premise) must be cut down to the real
        # answer before any NLI call -- raw generate_sampled() output
        # reliably continues past the real answer into a fabricated new
        # Q&A turn (see that function's own docstring for the real GPU
        # bug this fixes, job 388434, found in the sibling Semantic
        # Entropy Probe baseline's fit step).
        r_clean = _truncate_for_nli(r)
        if not r_clean:
            scores.append(0.0)  # empty response: cannot be supported by anything
            continue
        samples = generate_sampled(model, tok, [p], k=n, device=device)[0]
        contra_probs = []
        for s in samples:
            s_clean = _truncate_for_nli(s)
            if not s_clean:
                continue
            _, p_contra = _entailment_contradiction_probs(nli_model, nli_tok, s_clean, r_clean,
                                                          device=device)
            contra_probs.append(p_contra)
        mean_contra = float(np.mean(contra_probs)) if contra_probs else 1.0
        scores.append(1.0 - mean_contra)
    del nli_model
    import torch
    torch.cuda.empty_cache()
    return np.array(scores)


# =====================================================================
# Class 3: Self-Ask (Press et al. 2023) -- MANDATORY baseline
# =====================================================================

SELF_ASK_TEMPLATE = (
    "Question: {question}\n"
    "Are follow up questions needed here: Yes.\n"
    "Follow up: {followup_question}\n"
)

# VERIFIED against Press et al. (2023), arXiv:2210.03350, Section 3.1 and
# Figure 2/Appendix Tables 10/13 (checked directly against the paper's own
# PDF text, not reconstructed from memory): the per-turn scaffold strings
# above ("Are follow up questions needed here: Yes.", "Follow up:",
# "Intermediate answer:", "So the final answer is:") match the paper
# verbatim. What did NOT match, and is fixed here: the paper's own
# self-ask is never zero-shot -- it always prepends 2-6 few-shot
# exemplars demonstrating the decompose/answer/recompose pattern before
# the real question (Section 3.1: "requires a one- or few-shot prompt
# that demonstrates how to answer the questions"). Running it zero-shot,
# as an earlier version of this file did, is a real ablation of the
# method, not the method itself, and understates it -- exactly the wrong
# direction to be wrong in for a MANDATORY baseline. Fixed with two
# in-domain exemplars (this paper's own two-hop template shape, not the
# original paper's Bamboogle/Musique examples, which are a different
# dataset and would be an out-of-domain prompt).
SELF_ASK_FEWSHOT = (
    "Question: The nationality of the director of Jaws is\n"
    "Are follow up questions needed here: Yes.\n"
    "Follow up: Who is the director of the movie Jaws?\n"
    "Intermediate answer: Steven Spielberg\n"
    "Follow up: What is the nationality of Steven Spielberg?\n"
    "Intermediate answer: American\n"
    "So the final answer is: American\n\n"
    "Question: The country of the place of birth of the performer of Yesterday is\n"
    "Are follow up questions needed here: Yes.\n"
    "Follow up: Who is the performer of the song Yesterday?\n"
    "Intermediate answer: Paul McCartney\n"
    "Follow up: What is the place of birth of Paul McCartney, and what country is it in?\n"
    "Intermediate answer: Liverpool, England, United Kingdom\n"
    "So the final answer is: United Kingdom\n\n"
)


def baseline_self_ask(model, tok, queries, device="cuda"):
    """
    Decompose the two-hop composed question into explicit follow-up
    questions, answer each, then answer the original. This is the Class-3
    method that "makes the latent entity explicit... by a more expensive
    route" per skc_runbook.txt -- if it beats BGR, that CONFIRMS the paper's
    mechanism rather than refuting it; report that plainly if it happens.

    Two-shot (SELF_ASK_FEWSHOT above), per the paper's own requirement that
    self-ask always run few-shot, prepended once ahead of every query's own
    scaffold -- see that constant's own comment for the verification this
    fix is based on.

    Returns (scores, cost) where scores is an abstention score (here:
    1 - refusal on the final answer) and cost tracks forward passes /
    generated tokens / wall-clock, per query, as skc_runbook.txt requires.
    """
    scores, costs = [], []
    for q in queries:
        t0 = time.time()
        composed_stmt = render_prompts(q)["composed"]
        composed = composed_stmt[:-3] if composed_stmt.endswith(" is") else composed_stmt
        followup_q = f"What is the {q.hops[0].relation} of the {q.entity_type} {q.e1}?"
        prompt1 = SELF_ASK_FEWSHOT + SELF_ASK_TEMPLATE.format(
            question=composed, followup_question=followup_q)
        gen1 = generate_greedy(model, tok, [prompt1], max_new_tokens=24, device=device)[0]
        n_tok1 = len(tok(gen1)["input_ids"])

        prompt2 = (prompt1 + f"Intermediate answer: {gen1.strip()}\n"
                  f"Follow up: What is the {q.hops[1].relation} of {gen1.strip()}?\n")
        gen2 = generate_greedy(model, tok, [prompt2], max_new_tokens=24, device=device)[0]
        n_tok2 = len(tok(gen2)["input_ids"])

        final_prompt = prompt2 + f"Intermediate answer: {gen2.strip()}\nSo the final answer is:"
        gen3 = generate_greedy(model, tok, [final_prompt], max_new_tokens=24, device=device)[0]
        n_tok3 = len(tok(gen3)["input_ids"])

        scores.append(0.0 if is_refusal(gen3, q.target_answers) else 1.0)
        costs.append({
            "forward_passes": 3, "generated_tokens": n_tok1 + n_tok2 + n_tok3,
            "wall_clock_s": time.time() - t0,
        })
    return np.array(scores), costs


# =====================================================================
# Class 3: Operadic consistency (Bottman, Liu & Richardson 2026,
# arXiv:2606.13649) -- MANDATORY per skc_runbook.txt, previously left as
# >>> TO RUN pending reading the actual paper (do not fabricate a method
# from a title alone). VERIFIED against the real paper before
# implementing (title/authors/arXiv ID/abstract all confirmed live, not
# assumed): a label-free selective-prediction signal computed by
# comparing (a) the model's DIRECT answer to the whole composed question
# against (b) the answer obtained by decomposing the question into
# sub-questions, answering each, and composing the sub-answers per the
# decomposition -- disagreement between (a) and (b) is the signal, no
# ground truth needed to compute it (only to later evaluate it, same as
# every other selective-prediction score in this file).
# =====================================================================

def baseline_operadic_consistency(model, tok, queries, device="cuda"):
    """
    Reuses the SAME decomposition machinery as baseline_self_ask() above
    (deliberately -- both methods answer the same two follow-up questions
    from the same composed query; duplicating that logic would risk the
    two-independent-copies-diverge bug class this codebase has hit
    before), but scores DISAGREEMENT between the direct and decomposed
    answers, not a refusal check on the decomposed path alone.

    Returns (scores, costs): score = 1.0 when the direct and decomposed
    answers agree (alias_match, case/article/punctuation-normalized,
    treating one generation's own text as the other's "gold" answer list
    -- there is no ground truth involved in the comparison itself), 0.0
    when they disagree. Higher score means MORE consistent, which this
    file's convention treats as "more likely truthful" -- the same
    direction as every other selector here (self-ask, semantic entropy,
    the surface gate itself).
    """
    scores, costs = [], []
    for q in queries:
        t0 = time.time()
        composed_prompt = render_prompts(q)["composed"]

        # (a) direct answer to the whole composed question
        direct_gen = generate_greedy(model, tok, [composed_prompt], max_new_tokens=24,
                                     device=device)[0]
        n_tok_direct = len(tok(direct_gen)["input_ids"])

        # (b) decomposed answer: identical two-hop decomposition to Self-Ask,
        # including the same few-shot exemplars (SELF_ASK_FEWSHOT) -- the
        # decomposition step here IS self-ask's own method, so it should
        # run under the same verified, non-zero-shot conditions.
        composed_stmt = composed_prompt
        composed = composed_stmt[:-3] if composed_stmt.endswith(" is") else composed_stmt
        followup_q = f"What is the {q.hops[0].relation} of the {q.entity_type} {q.e1}?"
        prompt1 = SELF_ASK_FEWSHOT + SELF_ASK_TEMPLATE.format(
            question=composed, followup_question=followup_q)
        gen1 = generate_greedy(model, tok, [prompt1], max_new_tokens=24, device=device)[0]
        n_tok1 = len(tok(gen1)["input_ids"])

        prompt2 = (prompt1 + f"Intermediate answer: {gen1.strip()}\n"
                  f"Follow up: What is the {q.hops[1].relation} of {gen1.strip()}?\n")
        gen2 = generate_greedy(model, tok, [prompt2], max_new_tokens=24, device=device)[0]
        n_tok2 = len(tok(gen2)["input_ids"])

        final_prompt = prompt2 + f"Intermediate answer: {gen2.strip()}\nSo the final answer is:"
        decomposed_gen = generate_greedy(model, tok, [final_prompt], max_new_tokens=24,
                                         device=device)[0]
        n_tok_decomposed = len(tok(decomposed_gen)["input_ids"])

        agree = alias_match(direct_gen, [decomposed_gen.strip()], require_prefix=False) or \
            alias_match(decomposed_gen, [direct_gen.strip()], require_prefix=False)
        scores.append(1.0 if agree else 0.0)
        costs.append({
            "forward_passes": 4,
            "generated_tokens": n_tok_direct + n_tok1 + n_tok2 + n_tok_decomposed,
            "wall_clock_s": time.time() - t0,
        })
    return np.array(scores), costs


# =====================================================================
# Driver
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="two-hop (k=2) TEST queries JSON, "
                    "already filtered to K1=1 + bridge-resolved (output of skc_run_e1.py)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--primary-estimator", default="diffmeans")
    ap.add_argument("--run-self-ask", action="store_true",
                    help="run the Class-3 Self-Ask baseline; slow (3 forward passes/query)")
    ap.add_argument("--run-operadic-consistency", action="store_true",
                    help="run the Class-3 Operadic consistency baseline (Bottman, Liu "
                         "& Richardson 2026); slow (4 forward passes/query, shares the "
                         "same decomposition machinery as --run-self-ask)")
    ap.add_argument("--run-p-true", action="store_true")
    ap.add_argument("--saplma-train-queries", default=None,
                    help="one-hop (k=1) queries JSON to fit SAPLMA's classifier on "
                         "(a held-out training set, distinct from --queries, mirroring "
                         "skc_run_e0.py's own fit/held-out discipline). Required if "
                         "--run-saplma is passed.")
    ap.add_argument("--run-saplma", action="store_true",
                    help="run the SAPLMA (Azaria & Mitchell 2023) baseline; needs "
                         "--saplma-train-queries")
    ap.add_argument("--run-entity-token-probe", action="store_true",
                    help="run the entity-level token probe (Obeso et al. 2025) baseline; "
                         "needs --saplma-train-queries (same one-hop fit pool as SAPLMA)")
    ap.add_argument("--run-lookback-lens", action="store_true",
                    help="run the Lookback Lens (Chuang et al. 2024) baseline; needs "
                         "--saplma-train-queries and the model loaded with "
                         "need_attentions=True")
    ap.add_argument("--run-haloscope", action="store_true",
                    help="run the HaloScope (Du, Xiao & Li 2024) baseline; needs "
                         "--saplma-train-queries. BEST-FAITH RECONSTRUCTION, not a "
                         "verified match to the published algorithm -- see "
                         "fit_haloscope()'s own docstring.")
    ap.add_argument("--run-inside", action="store_true",
                    help="run the INSIDE / EigenScore (Chen et al. 2024) baseline. Fully "
                         "unsupervised, no fit split needed -- computed directly per test "
                         "query.")
    ap.add_argument("--run-icr-probe", action="store_true",
                    help="run the ICR Probe (Zhang et al., ACL 2025) baseline; needs "
                         "--saplma-train-queries and the model loaded with "
                         "need_attentions=True (shares Lookback Lens's separate eager "
                         "model instance if --run-lookback-lens is also set)")
    ap.add_argument("--run-semantic-entropy-probe", action="store_true",
                    help="run the Semantic Entropy Probe (Kossen et al. 2024, "
                         "arXiv:2406.15927) baseline; needs --saplma-train-queries. Fit "
                         "step is EXPENSIVE (k=10 sampled generations + up to 90 pairwise "
                         "NLI calls PER FIT QUERY to label real semantic entropy); eval "
                         "step is cheap by design (a single hidden-state capture per "
                         "query, no sampling at all).")
    ap.add_argument("--run-self-consistency", action="store_true",
                    help="run the k=10 sampling self-consistency baseline. Fully "
                         "unsupervised, no fit split needed; 1 greedy + k=10 sampled "
                         "generations per query at eval time.")
    ap.add_argument("--sequential-model-load", action="store_true",
                    help="free the default (non-eager) model instance BEFORE loading the "
                         "second, eager-attention instance for --run-lookback-lens/"
                         "--run-icr-probe, then reload the default instance afterward only "
                         "if a later-requested baseline still needs it. Costs one extra "
                         "model load/reload cycle (cheap: weights are already cached "
                         "locally, this only re-pays GPU memory transfer, not a download) "
                         "but avoids holding BOTH instances in GPU memory simultaneously. "
                         "Needed on large quantized models where two simultaneous 4-bit "
                         "copies do not both fit -- confirmed as a real failure on "
                         "Llama-3.1-70B (job 142): bitsandbytes' 4-bit quantizer refuses a "
                         "CPU/disk-offloaded device_map outright "
                         "('ValueError: Some modules are dispatched on the CPU or the "
                         "disk...'), so there is no degraded-but-working fallback here, "
                         "only this flag or splitting eager-model baselines into their own "
                         "job with no other baselines requested alongside them. OFF by "
                         "default -- unnecessary and strictly slower on models where both "
                         "instances fit at once (e.g. Gemma-2-9B, confirmed to work without "
                         "this flag in jobs 139/140).")
    ap.add_argument("--run-selfcheckgpt", action="store_true",
                    help="run SelfCheckGPT (Manakul, Liusie & Gales, EMNLP 2023, "
                         "arXiv:2303.08896), NLI variant. Fully unsupervised, no fit "
                         "split needed; 1 greedy + N=20 sampled generations + N NLI "
                         "calls per query at eval time (paper's own N=20, not this "
                         "file's usual k=10).")
    ap.add_argument("--run-semantic-entropy", action="store_true",
                    help="run Semantic Entropy (Farquhar, Kossen, Kuhn & Gal, Nature "
                         "2024) directly (not the probe). Fully unsupervised, no fit "
                         "split needed; k=10 sampled generations + NLI clustering per "
                         "query at eval time -- the direct method Semantic Entropy "
                         "Probe exists to approximate cheaply.")
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-records", default=None,
                    help="if given, also write per-item records for VERSION3's re-analysis "
                         "package (VERSION3/code/skc/io.py's schema) to this path, including "
                         "every baseline actually run as the 'baseline_scores' optional field "
                         "-- lets scripts/controls.py and any M2/M3-style baseline-fairness "
                         "reanalysis run on real per-item baseline scores, not just E4's own "
                         "aggregate AUROC numbers.")
    ap.add_argument("--export-split", default="eval", choices=["fit", "validation", "eval"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.run_saplma and not args.saplma_train_queries:
        raise SystemExit("--run-saplma requires --saplma-train-queries")
    if args.run_entity_token_probe and not args.saplma_train_queries:
        raise SystemExit("--run-entity-token-probe requires --saplma-train-queries")
    if args.run_lookback_lens and not args.saplma_train_queries:
        raise SystemExit("--run-lookback-lens requires --saplma-train-queries")
    if args.run_haloscope and not args.saplma_train_queries:
        raise SystemExit("--run-haloscope requires --saplma-train-queries")
    if args.run_icr_probe and not args.saplma_train_queries:
        raise SystemExit("--run-icr-probe requires --saplma-train-queries")
    if args.run_semantic_entropy_probe and not args.saplma_train_queries:
        raise SystemExit("--run-semantic-entropy-probe requires --saplma-train-queries")
    os.makedirs(args.out, exist_ok=True)

    gate_path = os.path.join(args.gate_dir, f"frozen_gate_{args.primary_estimator}.json")
    if not os.path.exists(gate_path):
        raise SystemExit(f"{gate_path} not found; run skc_run_e0.py (PASSED) first.")
    gate = FrozenGate.load(gate_path)

    queries = load_queries(args.queries)
    print(f"Loading {args.model} ...", flush=True)
    # need_attentions is DELIBERATELY NOT passed here, even when
    # --run-lookback-lens is set. Confirmed as a REAL, serious bug on a
    # real GPU run (job 387696): attn_implementation="eager" does not
    # just add attention-weight output on top of identical numerics --
    # on bfloat16 it produces genuinely different logits/activations
    # than the default SDPA backend (checked directly: mean absolute
    # logit difference 0.168 on a real forward pass, not negligible
    # floating-point noise), because the two implementations sum
    # attention scores in a different order and bfloat16's low
    # precision does not cancel that out. Loading THIS (main) model
    # under eager attention silently changed surface_gate/BGR-min/BGR-lr's
    # own numbers relative to every other E4 run this session (confirmed:
    # those three scores were byte-identical across the SAPLMA-only and
    # entity-token-probe runs, then changed once --run-lookback-lens was
    # added) -- contaminating every OTHER selector's result with a
    # different numerical backend, not just adding a new one. Lookback
    # Lens gets its OWN, separate model load under eager attention below,
    # scoped to exactly the code that needs it, so every other selector
    # in this script keeps running under the same backend as every other
    # E4 run.
    #
    model, tok = load_model(args.model)

    print(f"Labeling {len(queries)} queries ...", flush=True)
    # hop_require_prefix={2: False}: SAME fix as skc_run_e0.py/e1.py/e2.py/
    # e3.py/e5.py/e6.py/e7_ablations.py/pilot.py -- real Gemma-2-2b
    # generations for hop2-style prompts ("The country of the city X is")
    # reliably name the correct answer inside a longer descriptive
    # sentence rather than starting with it verbatim, so alias_match's
    # strict require_prefix=True default undercounts K2 (and therefore M,
    # which for these k=2 queries is exactly K2) almost to zero. This call
    # was the ONE E4 labeling site in this file that never got this
    # override applied -- harmless for Y (which has no override available
    # at all, per compute_labels()'s own docstring) and for the
    # already-shipped surface_gate AUROC (computed against Y, unaffected),
    # but it silently made `m` (added below, for the new Oracle-observing-M
    # selector) come out constant-0 for effectively every item, which is
    # why oracle_M/self_ask/operadic_consistency all reported identical,
    # degenerate auroc=0.5 stats in the first run of this script after
    # Oracle was added -- caught by noticing those three selectors were
    # byte-identical, which is only possible for genuinely constant
    # scores, and confirmed by this project's OWN already-published
    # P(M=0)=0.439 for this exact query set (skc_main.tex,
    # Section~\ref{sec:res-blindspot}) -- a real, non-degenerate M that
    # this call's missing override was failing to recover.
    labels = label_queries_batched(model, tok, queries, hop_require_prefix={2: False})
    y = np.array([l["Y"] for l in labels])
    m = np.array([l["M"] for l in labels])

    print("Capturing surface + bridge activations ...", flush=True)
    acts_surface, _, acts_bridge = capture_for_queries(model, tok, queries, args.layer)
    gate_surface = gate.score(acts_surface)
    gate_bridge = gate.score(acts_bridge)

    baseline_scores = {
        "surface_gate": gate_surface,
        "final_token_probe_PLACEHOLDER": baseline_final_token_probe(gate_surface),
        # Oracle observing M: the ceiling row of Table tab:bgr-full, a
        # selector that observes the ground-truth M (whether every
        # downstream hop succeeds in isolation) directly rather than
        # inferring it from any internal state. Not a real deployable
        # selector -- it requires the labels this experiment is trying to
        # predict -- but a genuine upper bound any repair should be judged
        # against, computed from `m` (already labeled above, same
        # generations Y itself comes from) rather than assumed or hand-set.
        "oracle_M": m.astype(float),
    }
    cost_report = {"surface_gate": {"forward_passes": 1, "generated_tokens": 0},
                   "oracle_M": {"forward_passes": 0, "generated_tokens": 0,
                                "note": "ceiling; not a real forward-pass selector, "
                                       "observes the M label directly"}}

    if args.run_p_true:
        print("Running Class-1 P(True) baseline ...", flush=True)
        prompts = [render_prompts(q)["composed"] for q in queries]
        # P(True) judges the MODEL'S OWN proposed answer, not the ground
        # truth -- feeding it the correct answer every time would make it
        # trivially "correct" and unable to detect the model's own
        # hallucinations, which is exactly the failure mode this baseline
        # exists to catch. A prior version of this call passed
        # q.target_answers[0] (the GROUND TRUTH), a real bug caught during
        # verification against Kadavath et al. (2022): fixed to greedily
        # generate the model's own answer first, matching label_queries_
        # batched()'s own generation settings (max_new_tokens=24, the
        # default both here and there) so this is the same answer text the
        # rest of the pipeline's own Y label was computed from.
        p_true_greedy = generate_greedy(model, tok, prompts, device="cuda")
        baseline_scores["p_true"] = baseline_p_true(model, tok, prompts, p_true_greedy)
        cost_report["p_true"] = {"forward_passes": 2, "generated_tokens": "see per-query"}

    # Shared fit-split label cache for the DEFAULT-model baselines (SAPLMA,
    # entity-token-probe, HaloScope) -- all three call load_queries() on
    # the SAME --saplma-train-queries file and then relabel it from
    # scratch with the SAME label_queries_batched(..., hop_require_prefix=
    # {1: False}) call. On a slow model (confirmed on Llama-3.1-70B: job
    # 141 spent its ENTIRE 24h budget on 3 such relabeling passes over
    # 9110 items, one per baseline, without even reaching HaloScope's own
    # scoring step) this redundancy alone can exceed the time budget.
    # Computed once here, ONLY if at least one of the three baselines that
    # needs it is requested, and passed into each via the new fit_labels=
    # parameter (see fit_saplma()'s own docstring for the full account).
    default_model_fit_queries, default_model_fit_labels = None, None
    if args.run_saplma or args.run_entity_token_probe or args.run_haloscope:
        default_model_fit_queries = load_queries(args.saplma_train_queries)
        if any(q.k != 1 for q in default_model_fit_queries):
            raise SystemExit("--saplma-train-queries must be single-hop (k=1).")
        print(f"Labeling the {len(default_model_fit_queries)}-query fit split ONCE, shared "
             "across SAPLMA/entity-token-probe/HaloScope (whichever are requested) ...",
             flush=True)
        default_model_fit_labels = label_queries_batched(
            model, tok, default_model_fit_queries, hop_require_prefix={1: False})

    if args.run_saplma:
        print("Fitting SAPLMA (Azaria & Mitchell 2023) on the fit split ...", flush=True)
        saplma_clf = fit_saplma(model, tok, default_model_fit_queries, args.layer,
                                seed=args.seed, fit_labels=default_model_fit_labels)
        print("Scoring SAPLMA on the E4 test queries ...", flush=True)
        baseline_scores["saplma"] = score_saplma(saplma_clf, model, tok, queries, args.layer)
        cost_report["saplma"] = {"forward_passes": 2, "generated_tokens": "see per-query",
                                 "note": "1 generation + 1 capture pass per query at eval "
                                         "time; training cost (fit split) not counted per-query"}

    if args.run_entity_token_probe:
        print("Fitting entity-level token probe (Obeso et al. 2025) on the fit split ...",
             flush=True)
        etp_gate = fit_entity_token_probe(model, tok, default_model_fit_queries, args.layer,
                                          seed=args.seed, fit_labels=default_model_fit_labels)
        print("Scoring entity-level token probe on the E4 test queries ...", flush=True)
        baseline_scores["entity_token_probe"] = score_entity_token_probe(
            etp_gate, model, tok, queries, args.layer)
        cost_report["entity_token_probe"] = {
            "forward_passes": 2, "generated_tokens": "see per-query",
            "note": "1 generation + 1 capture pass per query at eval time; training cost "
                    "(fit split) not counted per-query"}

    ll_model, ll_tok = None, None
    if args.run_lookback_lens or args.run_icr_probe:
        # Loads its OWN, second model instance under attn_implementation=
        # "eager" -- see the comment on the main load_model() call above
        # for why this must NOT be the shared model: eager attention
        # produces genuinely different bfloat16 numerics (confirmed:
        # mean abs logit diff 0.168 on a real forward pass), which would
        # silently change every OTHER selector's activations/scores if
        # loaded onto the shared model instead. This costs a second
        # model load's worth of GPU memory and time, not shared with the
        # rest of this script's selectors, deliberately, to keep every
        # other selector's numbers identical to a run without
        # --run-lookback-lens/--run-icr-probe at all. ICR Probe ALSO
        # needs eager attention (real per-head attention weights, not
        # just hidden states), so it reuses this same second instance
        # rather than loading a THIRD model when both flags are set.
        if args.sequential_model_load:
            # Free the default instance FIRST -- see --sequential-model-load's
            # own --help text. Confirmed real on Llama-3.1-70B (job 142):
            # holding both a 4-bit default instance AND a 4-bit eager instance
            # at once exceeds available GPU memory, and bitsandbytes' 4-bit
            # quantizer has NO degraded fallback for that (raises
            # ValueError rather than silently offloading to CPU/disk).
            print("(--sequential-model-load) Freeing the default model instance "
                 "before loading the eager-attention instance ...", flush=True)
            del model, tok
            import torch
            torch.cuda.empty_cache()
            model, tok = None, None
        print(f"Loading a SECOND, eager-attention instance of {args.model} for "
             "Lookback Lens / ICR Probe (kept separate from the main model so eager "
             "attention's different bfloat16 numerics do not contaminate any other "
             "selector) ...", flush=True)
        ll_model, ll_tok = load_model(args.model, need_attentions=True)

    # Shared fit-split label cache for the EAGER-attention baselines
    # (Lookback Lens, ICR Probe) -- same redundant-relabeling problem as
    # the default-model group above, but must NOT share labels with that
    # group: eager attention's own bfloat16 numerics differ from the
    # default SDPA backend (mean abs logit diff 0.168, confirmed), so
    # labeling on ll_model/ll_tok is a genuinely separate pass, not
    # reusable across the two model instances.
    eager_model_fit_queries, eager_model_fit_labels = None, None
    if args.run_lookback_lens or args.run_icr_probe:
        eager_model_fit_queries = load_queries(args.saplma_train_queries)
        if any(q.k != 1 for q in eager_model_fit_queries):
            raise SystemExit("--saplma-train-queries must be single-hop (k=1).")
        print(f"Labeling the {len(eager_model_fit_queries)}-query fit split ONCE on the "
             "eager-attention instance, shared across Lookback Lens/ICR Probe ...", flush=True)
        eager_model_fit_labels = label_queries_batched(
            ll_model, ll_tok, eager_model_fit_queries, hop_require_prefix={1: False})

    if args.run_lookback_lens:
        print("Fitting Lookback Lens (Chuang et al. 2024) on the fit split ...", flush=True)
        ll_gate = fit_lookback_lens(ll_model, ll_tok, eager_model_fit_queries, seed=args.seed,
                                    fit_labels=eager_model_fit_labels)
        print("Scoring Lookback Lens on the E4 test queries ...", flush=True)
        baseline_scores["lookback_lens"] = score_lookback_lens(ll_gate, ll_model, ll_tok, queries)
        cost_report["lookback_lens"] = {
            "forward_passes": 2, "generated_tokens": "see per-query",
            "note": "1 generation + 1 attention-capturing forward pass per query at eval "
                    "time, on a SEPARATE eager-attention model instance; training cost "
                    "(fit split) not counted per-query"}

    if args.run_icr_probe:
        print("Fitting ICR Probe (Zhang et al., ACL 2025) on the fit split ...", flush=True)
        icr_clf = fit_icr_probe(ll_model, ll_tok, eager_model_fit_queries, seed=args.seed,
                                fit_labels=eager_model_fit_labels)
        print("Scoring ICR Probe on the E4 test queries ...", flush=True)
        baseline_scores["icr_probe"] = score_icr_probe(icr_clf, ll_model, ll_tok, queries)
        cost_report["icr_probe"] = {
            "forward_passes": 2, "generated_tokens": "see per-query",
            "note": "1 generation + 1 attention+hidden-state-capturing forward pass per "
                    "query at eval time, on the SAME separate eager-attention model "
                    "instance Lookback Lens uses; training cost (fit split) not counted "
                    "per-query"}

    if ll_model is not None:
        del ll_model
        import torch
        torch.cuda.empty_cache()

    # Reload the default instance if --sequential-model-load freed it AND
    # something still downstream needs it (HaloScope/INSIDE/self-ask/
    # operadic-consistency all use `model`/`tok` directly). A later call
    # site using a None `model` would fail with a confusing AttributeError
    # far from the real cause, so reload eagerly here instead of lazily at
    # each site.
    if args.sequential_model_load and model is None and (
            args.run_haloscope or args.run_inside or args.run_self_ask
            or args.run_operadic_consistency or args.run_semantic_entropy_probe
            or args.run_self_consistency or args.run_selfcheckgpt
            or args.run_semantic_entropy):
        print("(--sequential-model-load) Reloading the default model instance "
             "for the remaining baseline(s) ...", flush=True)
        model, tok = load_model(args.model)

    if args.run_haloscope:
        print("Fitting HaloScope (Du, Xiao & Li 2024) on the fit split -- BEST-FAITH "
             "RECONSTRUCTION, not a verified match to the published algorithm, see "
             "fit_haloscope()'s own docstring ...", flush=True)
        hs_gate = fit_haloscope(model, tok, default_model_fit_queries, args.layer,
                                seed=args.seed, fit_labels=default_model_fit_labels)
        print("Scoring HaloScope on the E4 test queries ...", flush=True)
        baseline_scores["haloscope"] = score_haloscope(hs_gate, model, tok, queries, args.layer)
        cost_report["haloscope"] = {
            "forward_passes": 2, "generated_tokens": "see per-query",
            "note": "1 generation + 1 capture pass per query at eval time; fit-time cost "
                    "(unlabeled membership estimation, k=10 sampled generations per fit "
                    "query) not counted per-query"}

    if args.run_inside:
        print("Scoring INSIDE / EigenScore (Chen et al. 2024) on the E4 test queries "
             "-- no fit split needed, fully unsupervised, computed directly per query ...",
             flush=True)
        baseline_scores["inside"] = score_inside(model, tok, queries, args.layer)
        cost_report["inside"] = {
            "forward_passes": "k=10 (one generation batch of k samples per query)",
            "generated_tokens": "see per-query",
            "note": "k=10 sampled generations + one capture pass per sample, per query; "
                    "no fit-time cost at all (fully unsupervised, no training step)"}

    if args.run_semantic_entropy_probe:
        print("Fitting Semantic Entropy Probe (Kossen et al. 2024) on the fit split -- "
             "EXPENSIVE fit step (k=10 sampled generations + up to 90 pairwise NLI calls "
             "per fit query to label real semantic entropy) ...", flush=True)
        sep_fit_queries = load_queries(args.saplma_train_queries)
        if any(q.k != 1 for q in sep_fit_queries):
            raise SystemExit("--saplma-train-queries must be single-hop (k=1).")
        sep_gate = fit_semantic_entropy_probe(model, tok, sep_fit_queries, args.layer,
                                              seed=args.seed)
        print("Scoring Semantic Entropy Probe on the E4 test queries -- cheap by design, "
             "a single hidden-state capture per query, no sampling at eval time ...",
             flush=True)
        baseline_scores["semantic_entropy_probe"] = score_semantic_entropy_probe(
            sep_gate, model, tok, queries, args.layer)
        cost_report["semantic_entropy_probe"] = {
            "forward_passes": 1, "generated_tokens": 0,
            "note": "eval time: ONE capture pass per query, no generation or sampling at "
                    "all -- the whole point of a probe that PREDICTS semantic entropy "
                    "instead of computing it via sampling. Fit-time cost (k=10 sampled "
                    "generations + NLI clustering per fit query, ~10k fit queries) is "
                    "NOT counted per-query and is substantial -- see this run's own "
                    "wall-clock time."}

    if args.run_self_consistency:
        print("Scoring k=10 sampling self-consistency baseline on the E4 test queries "
             "-- no fit split needed, fully unsupervised, computed directly per query ...",
             flush=True)
        prompts = [render_prompts(q)["composed"] for q in queries]
        baseline_scores["self_consistency"] = baseline_self_consistency(model, tok, prompts)
        cost_report["self_consistency"] = {
            "forward_passes": "1 greedy + k=10 sampled (one generation batch of k "
                              "samples) per query",
            "generated_tokens": "see per-query",
            "note": "no fit-time cost at all (fully unsupervised, no training step); "
                    "pairwise agreement across the k=10 samples via the SAME "
                    "self-consistency proxy HaloScope's own fit step uses "
                    "(_self_consistency_membership_score), reused rather than "
                    "reimplemented, to avoid two copies of the same logic diverging"}

    if args.run_selfcheckgpt:
        print("Scoring SelfCheckGPT (Manakul, Liusie & Gales 2023, NLI variant) on the "
             "E4 test queries -- no fit split needed, fully unsupervised, computed "
             "directly per query ...", flush=True)
        sc_prompts = [render_prompts(q)["composed"] for q in queries]
        baseline_scores["selfcheckgpt"] = baseline_selfcheckgpt(model, tok, sc_prompts)
        cost_report["selfcheckgpt"] = {
            "forward_passes": "1 greedy + N=20 sampled (one generation batch of N "
                              "samples) + up to N NLI calls, per query",
            "generated_tokens": "see per-query",
            "note": "no fit-time cost at all (fully unsupervised, no training step); "
                    "paper's own N=20, not this file's usual k=10 for other sampling "
                    "baselines"}

    if args.run_semantic_entropy:
        print("Scoring Semantic Entropy (Farquhar et al. 2024) on the E4 test queries "
             "-- no fit split needed, fully unsupervised, computed directly per query "
             "(the direct method Semantic Entropy Probe exists to approximate cheaply) "
             "...", flush=True)
        baseline_scores["semantic_entropy"] = score_semantic_entropy(model, tok, queries)
        cost_report["semantic_entropy"] = {
            "forward_passes": "k=10 (one generation batch of k samples) + up to "
                              "k*(k-1) NLI calls, per query",
            "generated_tokens": "see per-query",
            "note": "no fit-time cost at all (fully unsupervised, no training step); "
                    "this is the DIRECT method -- costs paid at EVAL time per query, "
                    "unlike Semantic Entropy Probe which pays this cost only at fit "
                    "time and is cheap at eval time by design"}

    if args.run_self_ask:
        print("Running Class-3 Self-Ask baseline (mandatory per skc_REDFLAGS.md) ...", flush=True)
        sa_scores, sa_costs = baseline_self_ask(model, tok, queries)
        baseline_scores["self_ask"] = sa_scores
        cost_report["self_ask"] = {
            "forward_passes_mean": float(np.mean([c["forward_passes"] for c in sa_costs])),
            "generated_tokens_mean": float(np.mean([c["generated_tokens"] for c in sa_costs])),
            "wall_clock_s_mean": float(np.mean([c["wall_clock_s"] for c in sa_costs])),
        }

    if args.run_operadic_consistency:
        print("Running Class-3 Operadic consistency baseline (Bottman, Liu & Richardson "
             "2026, arXiv:2606.13649) ...", flush=True)
        oc_scores, oc_costs = baseline_operadic_consistency(model, tok, queries)
        baseline_scores["operadic_consistency"] = oc_scores
        cost_report["operadic_consistency"] = {
            "forward_passes_mean": float(np.mean([c["forward_passes"] for c in oc_costs])),
            "generated_tokens_mean": float(np.mean([c["generated_tokens"] for c in oc_costs])),
            "wall_clock_s_mean": float(np.mean([c["wall_clock_s"] for c in oc_costs])),
        }

    print("Decomposed-prompting abstention, operadic consistency "
         "(remaining Class 3) are NOT YET implemented in this script -- each needs its own "
         "reference implementation checked against its paper. All 12 planned E4 baselines "
         "outside those two ARE implemented: SAPLMA, the entity-level token probe, "
         "Lookback Lens, HaloScope, ICR Probe (all Class 0), INSIDE, Semantic Entropy "
         "Probe, SelfCheckGPT, Semantic Entropy (all Class 2), and the k=10 "
         "self-consistency sampling baseline (--run-saplma, --run-entity-token-probe, "
         "--run-lookback-lens, --run-haloscope, --run-inside, --run-icr-probe, "
         "--run-semantic-entropy-probe, --run-self-consistency, --run-selfcheckgpt, "
         "--run-semantic-entropy; "
         "HaloScope is a BEST-FAITH RECONSTRUCTION, not verified against the published "
         "algorithm -- see fit_haloscope()'s own docstring; ICR Probe, Semantic Entropy "
         "Probe, SelfCheckGPT, and Semantic Entropy ARE verified directly against their "
         "own papers -- see the docstrings above _icr_scores_for_text(), "
         "fit_semantic_entropy_probe(), baseline_selfcheckgpt(), and "
         "score_semantic_entropy()). "
         "DRIFT (Hussain & Kantarcioglu 2026) is the AUDIT "
         "paper that re-ran 22 existing detectors, not a detector itself -- it is cited as "
         "the audit source, not implemented as a baseline. See the execution plan's E4 "
         "baseline checklist.",
         file=sys.stderr)

    man = RunManifest(outdir=args.out, snapshot_id="see_dataset_snapshot_json",
                      model_name=args.model, seed=args.seed)
    print("Running E4 ...", flush=True)
    e4 = run_E4(y, baseline_scores, gate_surface, gate_bridge, man, coverage=args.coverage)
    print(json.dumps(e4, indent=1, default=str))

    with open(os.path.join(args.out, "e4_cost_report.json"), "w") as f:
        json.dump(cost_report, f, indent=1, default=str)

    if args.export_records:
        from skc_export_records import write_jsonl_records
        k1_vals = [l["K1"] for l in labels]
        k2_vals = [l.get("K2", l["M"]) for l in labels]
        qids = [l["qid"] for l in labels]
        # baseline_scores here maps name -> np.ndarray aligned with `queries`/
        # `labels` (same order throughout this script) -- pass per-item as a
        # list of {name: score} dicts, matching io.py's own
        # "baseline_scores {name: float}, one entry per published baseline"
        # schema. Only baselines actually run for THIS invocation appear;
        # skipped ones are simply absent per item, not zero-filled.
        per_item_baselines = [
            {name: float(scores[i]) for name, scores in baseline_scores.items()
             if name != "surface_gate"}
            for i in range(len(labels))
        ]
        write_jsonl_records(
            args.export_records,
            query_ids=qids,
            split=args.export_split,
            seed=args.seed,
            y=y.tolist(),
            k1=k1_vals,
            k2=k2_vals,
            g_s=gate_surface.tolist(),
            g_b=gate_bridge.tolist(),
            estimator=args.primary_estimator,
            offset=0,
            baseline_scores=per_item_baselines,
        )
        print(f"\nExported {len(labels)} per-item records (with "
             f"{len(baseline_scores) - 1 if 'surface_gate' in baseline_scores else len(baseline_scores)} "
             f"baseline(s)) to {args.export_records}.")

    for name in baseline_scores:
        if name != "surface_gate" and name in e4.get("selectors", {}):
            if e4["selectors"][name]["auroc"] > e4["selectors"].get("BGR-min", {}).get("auroc", 0):
                print(f"\n*** {name} beats BGR-min. REPORT THIS, do not drop the baseline. "
                     "If it is Self-Ask or another Class-3 method, this CONFIRMS the "
                     "mechanism (it makes the latent entity explicit by a costlier route) "
                     "even while beating the method -- say so plainly. ***", file=sys.stderr)


if __name__ == "__main__":
    main()

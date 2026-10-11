#!/usr/bin/env python3
"""
skc_harness.py
Shared, model-facing harness for the skc_* experiment drivers.

WHY THIS FILE EXISTS
  skc_gates.py / skc_experiments.py ship only library functions
  (capture_activations, steer_at_position, patch_bridge all take an
  already-loaded `model, tokenizer` and never load one themselves). Nothing in
  the released bundle loads a real HuggingFace model, builds entity_spans from
  a tokenizer, or turns a ComposedQuery into greedy generations. That is the
  gap skc_REDFLAGS.md section 5 calls UNVERIFIED. This module is the first
  thing that closes it, and every skc_run_*.py driver in this directory
  imports from here instead of re-deriving the same fragile logic five times.

  This file has NEVER been run against a real transformer (no GPU was
  available while writing it). Treat the first real invocation, on the
  cheap pilot, as a debugging run -- see skc_README.md section 4 and
  skc_REDFLAGS.md section 5. Do not delete the shape/position assertions
  to silence an error; a silent position error produces a confident null
  that looks exactly like the paper's hypothesis being confirmed.

WHAT THIS FILE DOES NOT DO
  It does not decide which experiment to run or how to interpret results.
  That lives in skc_run_e0.py, skc_run_e1.py, etc. This file only turns
  (model_name, prompts) into (activations, generations, labels).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skc_data import ComposedQuery, Hop, render_prompts, compute_labels, is_refusal  # noqa: E402


# =====================================================================
# Model loading
# =====================================================================

# HF repo ids used verbatim in skc_runbook.txt section 0. Kept as a lookup so
# every driver script names models the same way instead of hand-typing repo
# strings that could drift.
MODEL_IDS = {
    "gemma-2-2b":        "google/gemma-2-2b",
    "gemma-2-2b-it":      "google/gemma-2-2b-it",
    "gemma-2-9b":         "google/gemma-2-9b",
    "gemma-2-9b-it":      "google/gemma-2-9b-it",
    "llama-3.1-8b":       "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-it":    "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b":      "meta-llama/Llama-3.1-70B",     # 4-bit scale check only
    "qwen3-32b":          "Qwen/Qwen3-32B",                # 4-bit scale check only
}

# SAE Gemma Scope / Llama Scope releases, per skc_runbook.txt section 0.
# `release` is sae_lens's own release-registry KEY (from pretrained_saes.yaml),
# NOT a Hugging Face repo id -- the two look similar but are not
# interchangeable (confirmed by reading pretrained_saes.yaml directly; see
# load_sae()'s own docstring for the bug this replaced). `sae_id_fn(layer)`
# builds the per-layer id, since Gemma Scope and Llama Scope use two
# unrelated naming schemes (confirmed the same way): Gemma Scope's canonical
# release uses "layer_{layer}/width_16k/canonical"; Llama Scope's
# llama_scope_lxa_32x release uses "l{layer}a_32x" (lowercase, no width/L0 in
# the id at all -- width is fixed at 32x per the release name itself).
SAE_RELEASE_IDS = {
    "gemma-2-2b": ("gemma-scope-2b-pt-res-canonical",
                  lambda layer: f"layer_{layer}/width_16k/canonical"),
    "gemma-2-9b": ("gemma-scope-9b-pt-res-canonical",
                  lambda layer: f"layer_{layer}/width_16k/canonical"),
    "llama-3.1-8b": ("llama_scope_lxa_32x", lambda layer: f"l{layer}a_32x"),
}

# Models that need 4-bit loading per skc_runbook.txt's scale check
# ("no SAE needed, difference-of-means and probes only").
FOUR_BIT_MODELS = {"llama-3.1-70b", "qwen3-32b"}


def load_model(model_key: str, device: str = "cuda", need_attentions: bool = False):
    """
    Load a HF causal LM + tokenizer by the short key used throughout this
    directory (see MODEL_IDS). Returns (model, tokenizer).

    need_attentions=True passes attn_implementation="eager" to
    from_pretrained(). Confirmed directly (not assumed) that this is
    REQUIRED, not merely faster-with-a-fallback: the default attention
    backend (SDPA on this installed transformers/torch) does not
    materialize per-head attention weights at all --
    `model(..., output_attentions=True).attentions[0]` is silently `None`
    under the default backend (verified on a real model, gpt2), not a
    smaller/aggregated tensor. Every caller that needs real attention
    weights (Lookback Lens, skc_run_e7_ablations.py's attention control)
    MUST pass need_attentions=True or `output_attentions=True` will
    appear to work (no error) while returning nothing usable -- exactly
    the kind of silent-null failure this pipeline's own discipline
    (skc_REDFLAGS.md) warns about elsewhere.

    Defaults to False (unchanged prior behavior, whatever attention
    backend the installed transformers picks by default) rather than
    always forcing "eager", since eager attention is slower and every
    model load this session BEFORE Lookback Lens never needed real
    attention weights -- changing the default would silently alter the
    performance profile of every already-verified run for no benefit to
    them.

    UNVERIFIED end-to-end: never run against a real checkpoint. The 4-bit
    branch in particular (bitsandbytes) has not been exercised here at all.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_key not in MODEL_IDS:
        raise ValueError(f"unknown model key {model_key!r}; add it to MODEL_IDS first")
    repo = MODEL_IDS[model_key]

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        # Gemma/Llama base checkpoints often ship without a pad token; batched
        # capture_activations() needs one for the attention_mask to be meaningful.
        tok.pad_token = tok.eos_token
    # padding_side is DELIBERATELY NOT set here. It must be set per call site,
    # not once at load time, because capture_activations() and generate_greedy()
    # need OPPOSITE settings:
    #   - capture_activations() (skc_gates.py) computes the read position as
    #     attention_mask.sum()-1, or via an entity span computed against the
    #     unpadded string -- both assume content starts at index 0, i.e.
    #     RIGHT-padding. This was already true of the shipped skc_gates.py
    #     code; it was never something this harness was free to choose.
    #   - generate_greedy() slices new tokens at a single shared
    #     enc["input_ids"].shape[1] offset across the whole batch, which is
    #     only correct if every prompt's real content ends at the same
    #     right-aligned position, i.e. LEFT-padding. transformers itself
    #     warns "right-padding was detected" for exactly this case, and
    #     wrong padding here does not crash -- it silently generates from
    #     the wrong offset for every prompt shorter than the batch's longest,
    #     which produces confidently wrong labels rather than an error. This
    #     was caught on the first real pilot run: K1 (do you know this
    #     player's birthplace) came back 0/300, an implausibly total
    #     failure rate for a 2B model on real NBA-caliber player names, and
    #     traced back to this bug rather than a genuine model limitation.
    # See set_padding_side() below; every call site must use it explicitly.

    kwargs = dict(torch_dtype=torch.bfloat16)
    if need_attentions:
        kwargs["attn_implementation"] = "eager"
    if model_key in FOUR_BIT_MODELS:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    if model_key not in FOUR_BIT_MODELS:
        model = model.to(device)
    model.eval()

    # capture_activations / steer_at_position / patch_bridge in skc_gates.py
    # and skc_experiments.py all index model.model.layers[layer] directly.
    # Confirmed true for Gemma-2 and Llama-3.1 HF implementations; assert here
    # so an architecture that does not expose this attribute fails loudly at
    # load time rather than deep inside a hook during the actual run.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(
            f"{repo} does not expose model.model.layers -- capture_activations, "
            "steer_at_position and patch_bridge all assume this HF module layout. "
            "Check the architecture before running anything downstream."
        )
    return model, tok


def set_padding_side(tokenizer, side: str) -> None:
    """
    Explicitly set tokenizer.padding_side before a batched call. Every call
    site in this file that pads a batch must call this first -- see
    load_model()'s comment for why there is no single correct default.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    tokenizer.padding_side = side


def load_sae(model_key: str, layer: int, device: str = "cuda"):
    """
    Load a Gemma Scope / Llama Scope JumpReLU SAE for one layer via sae_lens.

    VERIFIED against sae_lens==6.5.3, installed and imported directly (see
    EXECUTION_PLAN.md). Two real bugs were found and fixed here relative to
    the original guess:

    1. `SAE_IDS` held each model's Hugging Face REPO id (e.g.
       "google/gemma-scope-2b-pt-res"), but sae_lens's `release` argument
       wants the entry KEY from its own pretrained_saes.yaml, which is
       DIFFERENT from the repo id and has a "-canonical" suffix for the
       canonical (single, recommended L0 per layer/width) release --
       confirmed by reading sae_lens/pretrained_saes.yaml directly rather
       than guessing. See SAE_RELEASE_IDS below.
    2. `SAE.from_pretrained(...)` returns the SAE object DIRECTLY, not a
       (sae, cfg, _) tuple -- confirmed by reading sae_lens's own internal
       usage in sae_lens/evals.py. Config is `sae.cfg` (width: `sae.cfg.d_sae`,
       input dim: `sae.cfg.d_in`), not a separately returned value.

    The `layer_{layer}/width_16k/canonical` id path itself was correct in the
    original guess; only the release name and the return-value unpacking were
    wrong.
    """
    from sae_lens import SAE

    if model_key not in SAE_RELEASE_IDS:
        raise ValueError(
            f"no public SAE dictionary for {model_key!r}. skc_runbook.txt is explicit "
            "that the SAE analysis does not extend past 9B; use diffmeans/probe only "
            "for the 32B/70B scale check, and say so in the paper rather than implying "
            "SAE coverage there."
        )
    release, sae_id_fn = SAE_RELEASE_IDS[model_key]
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id_fn(layer),
        device=device,
    )
    return sae, sae.cfg


# =====================================================================
# Entity-span tokenization
# =====================================================================

def find_entity_span(tokenizer, prompt: str, entity_surface_form: str):
    """
    Token-index span [start, end) of the LAST occurrence of entity_surface_form
    in prompt, in the tokenizer's own token indices.

    capture_activations()'s docstring is explicit: "Compute spans with the
    tokenizer on the exact prompt string; do not estimate them by counting
    words." This is that computation. Every skc_data.py template
    (TEMPLATES in skc_data.py) places the subject entity at a fixed position
    near the end of the string, but never at the very end (the relation
    phrasing comes after it in some templates and before it in others), so
    this does a character-offset lookup via the tokenizer's offset mapping
    rather than assuming "last token before the final word".
    """
    char_start = prompt.rfind(entity_surface_form)
    if char_start < 0:
        raise ValueError(
            f"entity surface form {entity_surface_form!r} not found verbatim in "
            f"prompt {prompt!r}. Template rendering and span-finding have "
            "diverged; fix render_prompts or this function before trusting "
            "any downstream activation."
        )
    char_end = char_start + len(entity_surface_form)

    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s == e == 0 and i > 0:
            continue  # special token with no span
        if s <= char_start < e or (tok_start is None and s >= char_start and s < char_end):
            if tok_start is None:
                tok_start = i
        if s < char_end and e >= char_end:
            tok_end = i + 1
    if tok_start is None or tok_end is None:
        raise ValueError(
            f"could not map char span [{char_start},{char_end}) of "
            f"{entity_surface_form!r} onto token offsets for prompt {prompt!r}. "
            "Check tokenizer offset_mapping support (not all tokenizers provide "
            "it faithfully -- verify on your exact tokenizer before trusting this)."
        )
    return tok_start, tok_end


def entity_spans_for_prompts(tokenizer, prompts: list[str], entities: list[str]):
    """Batch wrapper: one (start,end) per (prompt, entity) pair, same order as capture_activations expects."""
    if len(prompts) != len(entities):
        raise ValueError(f"len(prompts)={len(prompts)} != len(entities)={len(entities)}")
    return [find_entity_span(tokenizer, p, e) for p, e in zip(prompts, entities)]


def find_entity_span_in_generated_tokens(tokenizer, generated_tokens: list[str],
                                         candidate_surface_forms: list[str]):
    """
    Locate a bridge entity's span WITHIN generated_tokens -- the exact token
    list capture_during_generation() returns alongside hidden_states, which
    hidden_states is indexed against -- rather than re-tokenizing the
    generated TEXT from scratch.

    THIS REPLACED A RE-TOKENIZATION-BASED DESIGN, WHICH WAS WRONG IN TWO
    WAYS, both found only by a real GPU run (see EXECUTION_PLAN.md): (1) the
    naive re-tokenization added a BOS token generated_tokens never had,
    shifting every index by +1; fixing JUST that (add_special_tokens=False)
    was not enough, because (2) decode-then-re-encode is not guaranteed to
    reproduce the SAME token boundaries as the original generation ids in
    the first place -- SentencePiece merge choices, leading-space markers,
    and multi-byte/accented characters (this dataset has several: "São
    Paulo", "Kaušaitė") can all make a fresh encoding of the decoded string
    split differently than the tokens that were actually generated. Both
    bugs produced a real, confirmed symptom: decoding a position ADJACENT
    to but not exactly matching the intended one (first generic noise, then
    the country instead of the city -- consistently one region further
    forward than correct, exactly consistent with an accumulating offset).
    The fix is to never re-tokenize at all: search for the candidate
    surface form's OWN tokenization (with a leading space, matching how
    SentencePiece/Gemma marks a token boundary after another word -- e.g.
    "Kaunas" mid-sentence tokenizes as ['▁Ka', 'unas'], not ['Ka', 'unas'])
    as a contiguous subsequence of generated_tokens directly.

    Tries each candidate (gold surface form first, then aliases --
    skc_data.py's Hop.answers convention) in order; for each, tries WITH a
    leading space first (the common case for an entity named mid-sentence,
    e.g. after "the city of"), then WITHOUT (in case the entity is the
    very first generated token, with no preceding space marker).

    Returns (tok_start, tok_end): tok_start is generated_tokens' own index
    of the entity's first token (usable directly against hidden_states,
    since hidden_states[tok_start] is defined as the state immediately
    BEFORE generated_tokens[tok_start] -- see capture_during_generation()'s
    docstring). Raises if no candidate's token sequence is found as a
    subsequence -- meant to be called only after alias_match() has already
    confirmed some answer appears in the generated text (e.g. after a K1=1
    check), so a raise here means the TOKEN-level search could not locate
    what the TEXT-level, normalization-tolerant alias_match() found -- worth
    surfacing rather than silently skipping.
    """
    for form in candidate_surface_forms:
        for prefixed in (True, False):
            probe = (" " + form) if prefixed else form
            probe_tokens = tokenizer(probe, add_special_tokens=False)["input_ids"]
            probe_tokens = tokenizer.convert_ids_to_tokens(probe_tokens)
            n = len(probe_tokens)
            if n == 0:
                continue
            for i in range(len(generated_tokens) - n + 1):
                if generated_tokens[i:i + n] == probe_tokens:
                    return i, i + n
    raise ValueError(
        f"none of {candidate_surface_forms!r} found as a token subsequence of "
        f"{generated_tokens!r}. If alias_match() found a match in the decoded text, "
        "its normalization (accents/punctuation-insensitive, or a decode/re-encode "
        "mismatch) matched something this stricter, token-level search could not -- "
        "decide whether to add a form here rather than silently skipping."
    )


# =====================================================================
# Generation (for labeling K1, K2, Y, refusal -- not for activation capture)
# =====================================================================

def generate_greedy(model, tokenizer, prompts: list[str], max_new_tokens: int = 24,
                    batch_size: int = 16, device: str = "cuda") -> list[str]:
    """
    Greedy (T=0) generations for labeling. Separate from capture_activations
    because labeling needs generated text, not a single forward pass's
    hidden state -- do not try to reuse one call for both purposes.

    Forces LEFT-padding: the decode slice below,
    gen[:, enc["input_ids"].shape[1]:], assumes every prompt in the batch
    ends its real content at the same right-aligned position, which is only
    true under left-padding. Under right-padding this silently generates
    from the wrong offset for every prompt shorter than the batch's longest
    -- transformers warns about this ("right-padding was detected") but does
    not raise, so it produces confidently wrong text rather than an error.
    Restores the tokenizer's PRIOR padding_side afterward, since
    capture_activations() (skc_gates.py) needs RIGHT-padding and callers may
    interleave calls to both functions on the same tokenizer object.
    """
    import torch
    prior_side = tokenizer.padding_side
    set_padding_side(tokenizer, "left")
    try:
        out = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            out.extend(texts)
        return out
    finally:
        set_padding_side(tokenizer, prior_side)


def generate_sampled(model, tokenizer, prompts: list[str], k: int = 10,
                     max_new_tokens: int = 24, temperature: float = 1.0, top_p: float = 1.0,
                     batch_size: int = 16, seed: int = 0, device: str = "cuda") -> list[list[str]]:
    """
    k temperature-sampled generations per prompt, for the Class-2/self-
    consistency family of baselines (self-consistency, SelfCheckGPT, semantic
    entropy, HaloScope, INSIDE) -- none of which this codebase could run
    before this function existed, since generate_greedy() is greedy-only
    (do_sample=False) by design and every one of these methods needs actual
    sampled diversity across generations of the SAME prompt, not k identical
    greedy repeats (which would manufacture a fake perfect-consistency score
    of 1.0 for every query, a real failure mode this function's absence was
    guarding against until now -- see baseline_self_consistency()'s own prior
    NotImplementedError in skc_run_e4.py).

    Returns a list of length len(prompts), each element a list of k decoded
    continuations for that prompt. Uses `model.generate(...,
    num_return_sequences=k)`, batching over PROMPTS (not prompt*k) to keep
    memory bounded on a 70B model -- each batch element still produces k
    sequences internally, so effective decode batch size is
    batch_size * k, worth watching if that OOMs on a low-memory GPU.

    Same LEFT-padding requirement and reasoning as generate_greedy() (this
    function's own decode slice makes the identical assumption), same
    save/restore discipline.

    Seeded via `torch.manual_seed` immediately before each batch's generate()
    call, not once globally, so results are reproducible per-batch regardless
    of how many batches precede it or what else touched the global RNG state
    in between -- global-only seeding would make batch 2's "reproducible"
    output depend on exactly how many random calls batch 1 happened to make,
    which is not what "seeded" should mean here.
    """
    import torch
    prior_side = tokenizer.padding_side
    set_padding_side(tokenizer, "left")
    try:
        out = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            torch.manual_seed(seed + i)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                     temperature=temperature, top_p=top_p,
                                     num_return_sequences=k,
                                     pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            # gen has batch_size(this chunk)*k rows, grouped as
            # [prompt0_sample0..k-1, prompt1_sample0..k-1, ...] -- transformers'
            # own documented num_return_sequences ordering, not assumed.
            for j in range(len(batch)):
                out.append(texts[j * k:(j + 1) * k])
        return out
    finally:
        set_padding_side(tokenizer, prior_side)


# =====================================================================
# Bidirectional-entailment clustering (Kuhn et al. 2023; used by
# Farquhar et al. 2024's semantic entropy and Kossen et al. 2024's
# semantic entropy probe, both cited in this bundle) -- shared
# infrastructure for every semantic-entropy-family baseline in
# skc_run_e4.py, not duplicated per baseline.
# =====================================================================

NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def load_nli_model(device: str = "cuda"):
    """
    Load the NLI classifier used to bidirectionally-entail cluster sampled
    generations. Originally microsoft/deberta-large-mnli, the classic choice
    -- but that repo ships ONLY a legacy pytorch_model.bin (no safetensors),
    which this environment's transformers refuses to load under this
    pipeline's pinned torch==2.4.1 (CVE-2025-32434's torch.load restriction,
    confirmed as a real load-time ValueError, not assumed). Switched to
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli: also DeBERTa-
    family, public, ungated, ships safetensors (confirmed via
    HfApi.model_info() before use), and trained on a superset of MNLI
    (MNLI+FEVER+ANLI+LingNLI+WANLI) -- a legitimate, arguably stronger
    substitute for the same entailment-classification role both
    semantic-entropy papers cited in this bundle need, not a downgrade.
    Not a from-the-model-under-test substitute either way -- using the
    model under test as its own judge would be a real deviation from
    Farquhar et al./Kossen et al.'s published method.

    3-way label head CONFIRMED via AutoConfig.from_pretrained(...).id2label
    (not assumed from the model name, and NOT assumed to match the
    microsoft/deberta-large-mnli convention either -- checked directly and
    found DIFFERENT): {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    (lowercase, and in a DIFFERENT order than deberta-large-mnli's
    {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}). Every caller in this
    file reads labels from `model.config.id2label` dynamically and
    compares case-insensitively -- never hardcodes an index or an exact
    string -- specifically so swapping the underlying NLI model (as just
    happened here) cannot silently break the entailment check by matching
    against the wrong label spelling or a stale index.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
    model.to(device)
    model.eval()
    return model, tok


def _nli_label(model, tok, premise: str, hypothesis: str, device: str = "cuda") -> str:
    """One NLI call, premise entails/contradicts/is neutral toward hypothesis."""
    import torch
    enc = tok(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    idx = int(logits.argmax(dim=-1))
    return model.config.id2label[idx]


def bidirectional_entailment_clusters(nli_model, nli_tok, generations: list[str],
                                      device: str = "cuda") -> list[int]:
    """
    Cluster `generations` (semantically-sampled continuations of the SAME
    prompt) by bidirectional entailment, following Kuhn et al. (2023)'s
    algorithm as used in Farquhar et al. (2024) and Kossen et al. (2024):
    two generations belong to the same semantic cluster iff each entails
    the other (entailment is checked in BOTH directions, since NLI
    entailment is not symmetric -- "a poodle is a dog" entails but is not
    entailed by "a dog"). CONTRADICTION or NEUTRAL in either direction
    keeps them in separate clusters.

    Returns a list of cluster ids, same length and order as `generations`
    (e.g. [0, 0, 1, 0, 2] means items 0/1/3 are one semantic cluster).

    Greedy union-find over pairs -- O(n^2) NLI calls for n generations,
    fine at the k~10 scale every citing baseline actually uses (Farquhar
    et al.'s own experiments use k=10); would need a smarter approach at
    much larger k, not needed here.
    """
    n = len(generations)
    cluster_id = list(range(n))

    def find(i):
        while cluster_id[i] != i:
            cluster_id[i] = cluster_id[cluster_id[i]]
            i = cluster_id[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            cluster_id[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            fwd = _nli_label(nli_model, nli_tok, generations[i], generations[j], device)
            bwd = _nli_label(nli_model, nli_tok, generations[j], generations[i], device)
            if fwd.upper() == "ENTAILMENT" and bwd.upper() == "ENTAILMENT":
                union(i, j)

    # Renumber to consecutive 0..k-1 rather than leaving union-find's own
    # arbitrary representative ids, so callers can treat the output as
    # plain cluster labels without knowing this function's internals.
    roots = sorted(set(find(i) for i in range(n)))
    remap = {r: k for k, r in enumerate(roots)}
    return [remap[find(i)] for i in range(n)]


def semantic_entropy(cluster_ids: list[int]) -> float:
    """
    Shannon entropy over cluster MEMBERSHIP COUNTS (Farquhar et al. 2024's
    definition: entropy of the distribution over semantic clusters, not
    over raw generations) -- one generation with 10 verbatim-different but
    semantically-identical phrasings has entropy 0 (all in one cluster,
    fully consistent), not the high entropy a naive string-level measure
    would report.
    """
    counts = np.bincount(cluster_ids)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# =====================================================================
# Query -> labeled record, end to end
# =====================================================================

def label_query(model, tokenizer, q: ComposedQuery, max_new_tokens: int = 24,
                device: str = "cuda") -> dict:
    """
    Render every prompt for one ComposedQuery, generate greedily, and compute
    labels via skc_data.compute_labels. One query at a time (not batched)
    because render_prompts returns a variable-length dict of prompt kinds
    per query (k=2 vs k=3 differ) -- batch across QUERIES of the same k using
    label_queries_batched below, not across prompt kinds of one query.
    """
    prompts = render_prompts(q)
    keys = list(prompts.keys())
    gens = generate_greedy(model, tokenizer, [prompts[k] for k in keys],
                           max_new_tokens=max_new_tokens, device=device)
    generations = dict(zip(keys, gens))
    return compute_labels(q, generations)


def label_queries_batched(model, tokenizer, queries: list[ComposedQuery],
                          max_new_tokens: int = 24, batch_size: int = 16,
                          device: str = "cuda", progress_every: int = 25,
                          save_generations_path: str | None = None,
                          hop_require_prefix: dict[int, bool] | None = None) -> list[dict]:
    """
    Label many queries. Flattens all (query, prompt-kind) pairs into one
    generation batch so the GPU is actually kept busy, then regroups before
    calling compute_labels per query.

    save_generations_path: if given, dump {qid: {prompt_key: (prompt, raw
    generation)}} to this path as JSON. Binary labels alone cannot
    distinguish "the model answered wrong" from "the scoring function
    rejected a correct-looking answer" from "the prompt template does not
    elicit an answer at all" -- this was needed on the first real pilot run,
    where K1 came back 0/300 and the raw text was the only way to tell
    which of those explanations actually applies.

    hop_require_prefix: passed straight through to skc_data.compute_labels;
    see that function's docstring for why hop2 in particular may need
    require_prefix=False on real generations.
    """
    flat_prompts, flat_index = [], []
    for qi, q in enumerate(queries):
        prompts = render_prompts(q)
        for key, text in prompts.items():
            flat_prompts.append(text)
            flat_index.append((qi, key))

    flat_gens = generate_greedy(model, tokenizer, flat_prompts,
                               max_new_tokens=max_new_tokens, batch_size=batch_size,
                               device=device)

    per_query: dict[int, dict] = {}
    for (qi, key), gen in zip(flat_index, flat_gens):
        per_query.setdefault(qi, {})[key] = gen

    if save_generations_path:
        dump = {}
        for (qi, key), prompt, gen in zip(flat_index, flat_prompts, flat_gens):
            dump.setdefault(queries[qi].qid, {})[key] = {"prompt": prompt, "generation": gen}
        with open(save_generations_path, "w") as f:
            json.dump(dump, f, indent=1)

    labels = []
    for qi, q in enumerate(queries):
        labels.append(compute_labels(q, per_query[qi], hop_require_prefix=hop_require_prefix))
        if progress_every and (qi + 1) % progress_every == 0:
            print(f"  labeled {qi + 1}/{len(queries)}", flush=True)
    return labels


def capture_for_queries(model, tokenizer, queries: list[ComposedQuery], layer: int,
                        device: str = "cuda", batch_size: int = 16):
    """
    For a list of two-hop ComposedQuery objects, capture activations at:
      - the surface entity (e1, in the composed prompt)
      - the hop1 entity (e1, in the isolated single-hop prompt)
      - the hop2 / bridge entity (bridge, in the isolated single-hop prompt)
    using skc_gates.capture_activations at position="last_entity_token".

    Returns (acts_surface, acts_hop1, acts_hop2), each (n_queries, d_model).

    This is the "surface entity position" and "hop1/hop2 prompts" referred to
    throughout skc_runbook.txt (E0, E1, the pilot). It does NOT capture the
    BRIDGE position inside the composed prompt itself -- that is a different,
    harder span-finding problem (the bridge is never written there, by
    construction; see skc_run_e3.py, which needs the Patchscopes bridge-
    resolution layer/position instead of a token span in the composed prompt).
    """
    from skc_gates import capture_activations
    from skc_data import render_prompts

    composed_prompts, composed_entities = [], []
    hop1_prompts, hop1_entities = [], []
    hop2_prompts, hop2_entities = [], []
    for q in queries:
        p = render_prompts(q)
        composed_prompts.append(p["composed"]); composed_entities.append(q.e1)
        hop1_prompts.append(p["hop1"]); hop1_entities.append(q.hops[0].subject)
        hop2_prompts.append(p["hop2"]); hop2_entities.append(q.hops[1].subject)

    surface_spans = entity_spans_for_prompts(tokenizer, composed_prompts, composed_entities)
    hop1_spans = entity_spans_for_prompts(tokenizer, hop1_prompts, hop1_entities)
    hop2_spans = entity_spans_for_prompts(tokenizer, hop2_prompts, hop2_entities)

    acts_surface = capture_activations(model, tokenizer, composed_prompts, layer,
                                       position="last_entity_token",
                                       entity_spans=surface_spans, batch_size=batch_size,
                                       device=device)
    acts_hop1 = capture_activations(model, tokenizer, hop1_prompts, layer,
                                    position="last_entity_token",
                                    entity_spans=hop1_spans, batch_size=batch_size,
                                    device=device)
    acts_hop2 = capture_activations(model, tokenizer, hop2_prompts, layer,
                                    position="last_entity_token",
                                    entity_spans=hop2_spans, batch_size=batch_size,
                                    device=device)
    return acts_surface, acts_hop1, acts_hop2


def capture_at_positions(model, tokenizer, prompts: list[str], layer: int,
                         positions: list[int], batch_size: int = 16, device: str = "cuda"):
    """
    Capture activations at an explicit, PER-PROMPT integer token position.

    skc_gates.capture_activations accepts position as "last_entity_token",
    "last", or a single int applied to every prompt in the call -- it has no
    mode for "a different explicit position per prompt in the same batch".
    This wrapper works around that by grouping prompts with identical target
    positions into sub-batches (each an eligible single-int call), running
    capture_activations once per group, and reassembling the result in the
    caller's original order. Needed by skc_run_e3.py's position-offset sweep,
    where "last token minus k" gives a different absolute index per prompt
    once prompt lengths vary.
    """
    from skc_gates import capture_activations

    if len(prompts) != len(positions):
        raise ValueError(f"len(prompts)={len(prompts)} != len(positions)={len(positions)}")
    if any(p < 0 for p in positions):
        raise ValueError("all positions must be >= 0; caller must clip before calling this")

    order = list(range(len(prompts)))
    results = [None] * len(prompts)
    by_pos: dict[int, list[int]] = {}
    for i, p in zip(order, positions):
        by_pos.setdefault(p, []).append(i)

    for p_val, idxs in by_pos.items():
        sub_prompts = [prompts[i] for i in idxs]
        for start in range(0, len(sub_prompts), batch_size):
            chunk_idxs = idxs[start:start + batch_size]
            chunk_prompts = [prompts[i] for i in chunk_idxs]
            acts = capture_activations(model, tokenizer, chunk_prompts, layer,
                                       position=p_val, batch_size=len(chunk_prompts),
                                       device=device)
            for i, row in zip(chunk_idxs, acts):
                results[i] = row
    if any(r is None for r in results):
        raise RuntimeError("internal error: not every prompt received an activation")
    return np.stack(results)


PATCHSCOPE_IDENTITY_PROMPT = "cat -> cat\n135 -> 135\nhello -> hello\nx ->"
# >>> TO RUN: this is the standard "identity"/few-shot-repeat prompt used in
# the Patchscopes literature (Ghandeharioun et al. 2024) to elicit a plain
# decode of whatever is patched into the final position -- confirm this
# exact template against the published paper before treating results as
# faithful to their protocol; this is a good-faith reconstruction from the
# paper's description, not a copy verified against their released code.
# The target position is always the LAST token of this prompt ("->").


def patchscope_decode(model, tokenizer, source_hidden_state: np.ndarray, layer: int,
                      identity_prompt: str = PATCHSCOPE_IDENTITY_PROMPT,
                      max_new_tokens: int = 8, device: str = "cuda") -> str:
    """
    The generic Patchscopes primitive (Ghandeharioun et al. 2024): patch a
    single captured hidden state into the LAST token position of a generic
    "identity" prompt at the same layer, then generate. If the source hidden
    state represents a specific entity in the format the model's own
    read-out machinery consumes, the patched generation should name that
    entity -- even though identity_prompt itself has nothing to do with
    whatever produced source_hidden_state.

    This is NOT steer_at_position() or patch_bridge() (skc_experiments.py):
    those add a direction to, or copy a donor's residual INTO, the model's
    OWN ongoing forward pass on the query it came from. This instead injects
    into a COMPLETELY UNRELATED prompt, which is what makes it a decode of
    "what entity is represented here" rather than an intervention on the
    original query's behavior.

    UNVERIFIED END TO END: never run against a real model (no GPU was
    available while writing it). Treat the first real call as a debugging
    run, same as every other model-facing function in this file -- and
    verify the decoded text on a handful of KNOWN-answer queries by eye
    before trusting it inside bridge_resolved_patchscope() below.
    """
    import torch
    d = torch.tensor(np.asarray(source_hidden_state, dtype=np.float32), device=device)
    call_count = {"n": 0}

    def hook(_m, _i, out):
        # model.generate() calls this hook ONCE PER FORWARD PASS: the first
        # call processes the full identity_prompt (prefill); every
        # subsequent call, one per newly generated token, only processes
        # the SINGLE new token (with the rest served from the KV cache).
        # The ORIGINAL version of this hook patched position -1 on EVERY
        # call unconditionally -- correct for the first call (the last
        # position of the prompt) but WRONG afterward, because it clamped
        # every subsequently generated token's hidden state at this layer
        # to the same fixed vector forever, instead of letting the model
        # process its own generated tokens normally after the initial
        # patch. That produced degenerate generations and was found to be
        # the direct cause of bridge_resolved_patchscope() returning False
        # on 100% of a large real sample. Patch ONLY the first call.
        call_count["n"] += 1
        if call_count["n"] > 1:
            return out
        h = out[0] if isinstance(out, tuple) else out
        assert h.ndim == 3, f"expected (batch, seq, d_model), got {tuple(h.shape)}"
        if d.shape[-1] != h.shape[-1]:
            raise ValueError(
                f"source_hidden_state dim {d.shape[-1]} does not match model hidden "
                f"dim {h.shape[-1]} at layer {layer}. Wrong layer or wrong model."
            )
        h[:, -1, :] = d
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        enc = tokenizer(identity_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        handle.remove()


def capture_during_generation(model, tokenizer, prompt: str, layer: int,
                              max_new_tokens: int = 8, device: str = "cuda"):
    """
    Capture the residual stream at EVERY forward-pass call while model.generate()
    autoregressively writes an answer to `prompt` -- not just a single forward
    pass over a static prompt, which is all capture_activations() (skc_gates.py)
    can address.

    WHY THIS EXISTS: an exhaustive sweep (5 layers x 8 offsets x hop1-only vs
    composed prompts, see EXECUTION_PLAN.md's "round 2" entry) found the bridge
    entity (e.g. a player's birth CITY) never decodable via Patchscopes at any
    PROMPT position tested -- every position either read the surface entity's
    own name or had already jumped to the composed query's FINAL answer (the
    country), even on the hop1-only prompt that never mentions country. All of
    that used capture_activations(), which only ever sees one forward pass over
    the prompt AS GIVEN -- it cannot address a position that only exists once
    the model starts writing its own answer. This function tests the one site
    the prior sweep structurally could not reach: is the city represented WHILE
    the model is generating it, even if it is not represented beforehand?

    Returns a list of (batch=1) hidden-state vectors, one per forward-pass call:
    index 0 is the prompt's own last token (the prefill call, same position
    capture_activations(..., position="last") would give you); index 1 is the
    first newly generated token; index 2 the second; and so on up to
    max_new_tokens. Same "hook fires once per forward-pass call" mechanics
    documented in patchscope_decode() above -- this hook only READS (does not
    patch), so unlike that function there is no call-count guard needed to
    avoid corrupting generation; every call's hidden state is simply recorded.
    """
    import torch
    hidden_states = []

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        assert h.ndim == 3, f"expected (batch, seq, d_model), got {tuple(h.shape)}"
        hidden_states.append(h[0, -1].float().cpu().numpy())

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        generated_text = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        generated_tokens = tokenizer.convert_ids_to_tokens(
            gen[0, enc["input_ids"].shape[1]:].tolist())
        return hidden_states, generated_text, generated_tokens
    finally:
        handle.remove()


def bridge_resolved_patchscope(model, tokenizer, q: ComposedQuery, layer: int,
                               max_new_tokens: int = 12, device: str = "cuda",
                               return_debug: bool = False):
    """
    Independent evidence the bridge was resolved internally, following the
    Patchscopes protocol (Biran et al. 2024) that skc_runbook.txt section
    "(2) CONDITION EVERY PRIMARY ANALYSIS ON K1 = 1" requires alongside K1=1.

    CAPTURE SITE (this is the corrected design; see EXECUTION_PLAN.md's
    "round 4" entry for the full empirical trail): captures WHILE the model
    generates its own greedy answer to the HOP1-ONLY prompt ("The {r1} of
    the {etype} {e1} is"), not a single forward pass over the composed
    two-hop prompt. The decodable position is the token IMMEDIATELY BEFORE
    the bridge entity's own span in that generation (e.g. the "of" in the
    model's own "the city of Kaunas" completion, one step before "Kaunas"
    itself) -- confirmed on 5/7 K1=1 queries to decode the bridge exactly,
    with the 2 misses explained by a different generated sentence shape
    (no "of"-then-city structure) rather than by a wrong layer or bug.

    An EARLIER design captured a single position on the STATIC composed
    two-hop prompt at an (layer, offset) pair taken from skc_run_e3.py's
    own AUROC sweep. That design is WRONG, not just less accurate: an
    exhaustive follow-up sweep (5 layers x 8 offsets x hop1-only vs
    composed prompts, zero city decodes anywhere) showed no STATIC prompt
    position ever exposes the bridge -- the composed prompt's word order
    ("...of the {etype} {e1} is") means nothing entity-specific can be
    read out before e1's own tokens, and by the time the model reaches
    e1 (the prompt's own last content), it already has enough context to
    jump straight to the FINAL composed answer, not stop at the
    intermediate bridge. The bridge only becomes locally decodable DURING
    the model's own generation of it, which capture_activations()
    (skc_gates.py, one forward pass over a fixed prompt) cannot address at
    all -- hence capture_during_generation() instead.

    This also means E3's own (layer, offset) sweep result is NOT the site
    this function uses anymore -- E3 answers a different question ("where
    does the frozen gate read K2 above chance on the composed prompt"),
    which this investigation found is highly likely to be the position
    right before the model emits the FINAL answer, not the bridge. `layer`
    here is still worth sourcing from a per-layer E0/E3 pass (this
    function only trusts a layer that has independently demonstrated a
    working K1/K2 gate), but the offset/position logic is no longer E3's.

    Requires q's hop1 answer to actually be findable in the model's own
    greedy generation -- i.e. this function is only meaningful to call
    after confirming K1=1 for q (skc_runbook.txt's own precondition,
    already required before this function regardless). Calling it on a
    K1=0 query is not an error in itself, but the empirical finding above
    was established ONLY on K1=1 queries -- an earlier, unfixed round of
    this investigation sampled queries WITHOUT a K1 filter and concluded
    (wrongly) that the bridge is never decodable at all, purely because
    most of that sample's hop1 answers were incorrect and had no correctly
    resolved bridge for Patchscopes to find in the first place. Raises
    (via find_entity_span_in_generated_tokens) if the bridge cannot be
    located in the generation, rather than silently returning False --
    a K1=0 caller will very likely hit this raise; callers should filter
    to K1=1 before calling, exactly as skc_runbook.txt already requires.

    return_debug=True returns (resolved, debug_dict) instead of just
    resolved, where debug_dict has "generated_text" and "decoded" -- for a
    caller that wants to save a human-readable sample (e.g.
    skc_run_e1.py's patchscope_sample.json) WITHOUT re-deriving the same
    capture a second time with its own copy of this function's logic. An
    earlier version of skc_run_e1.py's diagnostic-sample block did exactly
    that re-derivation, with its own separate call to
    capture_during_generation() using a DIFFERENT max_new_tokens default
    and no bounds check -- confirmed to crash with an IndexError on the
    first full-dataset run, for exactly the same reason two independent
    copies of the SAME position formula caused a crash earlier this
    session (see EXECUTION_PLAN.md): a second copy of logic that looks
    identical at a glance can silently diverge in a parameter default and
    is missing the real function's own safety check. return_debug exists
    so there only has to be one copy of this logic, ever.
    """
    from skc_data import alias_match, render_prompts

    hop1_prompt = render_prompts(q)["hop1"]
    hidden_states, generated_text, generated_tokens = capture_during_generation(
        model, tokenizer, hop1_prompt, layer, max_new_tokens=max_new_tokens, device=device)

    # q.hops[0].answers -- NOT q.hops[1].answers. hops[0] is (e1, r1,
    # answers=[bridge]): hop 1 is "birthplace of the player", and its
    # ANSWER is the bridge (the city). hops[1] is (bridge, r2,
    # answers=[final target]): hop 2 is "country of the city", and ITS
    # answer is the FINAL composed answer (the country), not the bridge.
    # q.bridge itself (== q.hops[1].subject) is the city as a plain
    # string, but this needs the bridge's FULL answer-alias list (gold
    # form first, then aliases) the same way K1 scoring already gets it
    # via q.hops[0].answers (see skc_data.compute_labels) -- q.hops[1]
    # .answers is a DIFFERENT list (the country's aliases) that happens to
    # share the "list of gold-form-then-aliases" shape, which is exactly
    # how this bug went unnoticed through three rounds of chasing what
    # looked like tokenizer/indexing bugs: every symptom (decoding the
    # country, decoding noise) was consistent with searching for the
    # WRONG STRING ENTIRELY, not a misplaced position for the right one.
    bridge_answers = q.hops[0].answers
    span_start, _span_end = find_entity_span_in_generated_tokens(
        tokenizer, generated_tokens, bridge_answers)
    # span_start is generated_tokens' OWN index of the bridge's first
    # token -- found by searching generated_tokens directly (see
    # find_entity_span_in_generated_tokens's docstring for why a
    # re-tokenize-the-decoded-text approach was tried first and found
    # wrong two ways). hidden_states[i] (i>=1) is the forward-pass hidden
    # state AT generated_tokens[i-1]'s own position (see
    # capture_during_generation's docstring) -- i.e. hidden_states[span_start]
    # is the state at the token immediately BEFORE the bridge's own first
    # generated token, exactly the "one step before the entity" site this
    # function's docstring above documents as the one that decodes cleanly.
    if span_start < 1 or span_start >= len(hidden_states):
        raise ValueError(
            f"{q.qid}: bridge span starts at generated-token index {span_start}, "
            f"but only {len(hidden_states)} generation positions were captured "
            f"(max_new_tokens={max_new_tokens}). Either the bridge is the very "
            "first generated token (no position before it exists to decode) or "
            "it was generated too late for this budget -- raise max_new_tokens "
            "rather than silently clipping."
        )
    decoded = patchscope_decode(model, tokenizer, hidden_states[span_start], layer, device=device)
    resolved = alias_match(decoded, [q.bridge], require_prefix=False)
    # require_prefix=False: Patchscopes decodes are short, generic completions
    # (max_new_tokens=8 by default) of an identity template, not free-form
    # narrative -- the same "narrative wanders past the answer" problem that
    # required require_prefix=False for K1/K2 elsewhere in this bundle is far
    # less likely here, but containment (bounded by the short token budget)
    # is still safer than an exact-prefix match against a name the model may
    # decode with a leading article or title.
    if return_debug:
        return resolved, {"generated_text": generated_text, "decoded": decoded}
    return resolved


# =====================================================================
# Small IO helpers shared by every driver
# =====================================================================

def save_activation_cache(path: str, acts: np.ndarray, meta: dict) -> None:
    """
    Cache activations to disk once, keyed by whatever (model, layer, position)
    string the caller builds into `path`. skc_runbook.txt section 0 is
    explicit: cache once, every downstream analysis reads the cache, do not
    recompute. Sidecar .json carries provenance (model, layer, position,
    snapshot_id) so a cached array can be traced to what produced it.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, acts)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)


def load_activation_cache(path: str):
    acts = np.load(path if path.endswith(".npy") else path + ".npy")
    meta_path = (path if path.endswith(".npy") else path + ".npy") + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return acts, meta

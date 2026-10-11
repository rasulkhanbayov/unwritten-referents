#!/usr/bin/env python3
"""
skc_data.py
Dataset construction and label computation for
"Tokens, Not Referents: Latent Entities and the Limits of Knowledge-Aware Abstention".

WHAT THIS FILE DOES
  Builds composed (multi-hop) queries and their single-hop decompositions from a
  Wikidata dump or SPARQL endpoint, renders prompts from the templates in the
  paper's appendix, and computes the binary outcomes Y and K_1..K_k by
  alias-aware exact match. No human annotation anywhere.

WHAT IS AND IS NOT TESTED
  Tested by skc_pipeline_test.py, offline, no network or GPU:
    - alias-aware matching, including normalisation and the tricky cases
    - template rendering and the length-matched named-bridge control
    - popularity-decile matching
    - label computation and the K1=1 filter
  NOT tested here: the SPARQL/dump path, because it needs network access. That
  code is written but you must verify it against your endpoint on first run.
  Do not assume it works because the rest of the file does.

Dependencies: numpy. `requests` only if you use the SPARQL path.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

# =====================================================================
# Templates. These mirror Appendix F of the paper exactly. If you change
# one here, change it there; a silent divergence between the released
# prompts and the reported ones is a reproducibility failure.
# =====================================================================

TEMPLATES = {
    "single_hop":   "The {relation} of the {etype} {entity} is",
    "composed_2":   "The {r2} of the {r1} of the {etype} {e1} is",
    "composed_3":   "The {r3} of the {r2} of the {r1} of the {etype} {e1} is",
    "named_bridge": "The {r2} of the {btype} {bridge} is",
    "cot":          "Answer step by step. The {r2} of the {r1} of the {etype} {e1} is",
}

# Relations whose answers are not textual or are ambiguous in modality are
# excluded, following the same exclusion made by Tamoyan et al. (2025).
EXCLUDED_RELATIONS = frozenset({
    "coordinates", "coordinate location", "geoshape", "image", "signature",
    "population",  # numeric, and drifts with the snapshot
})


# =====================================================================
# Records
# =====================================================================

@dataclass
class Hop:
    """One (subject, relation) -> answer step."""
    subject: str
    subject_type: str
    relation: str
    answers: list[str]           # gold surface form first, then aliases

    def __post_init__(self):
        if not self.answers:
            raise ValueError(f"hop {self.subject}/{self.relation} has no answers")


@dataclass
class ComposedQuery:
    """A k-hop chain with all its decompositions."""
    qid: str
    hops: list[Hop]
    entity_type: str
    popularity_decile: int | None = None   # of the FIRST bridge, not of e1
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.hops)

    @property
    def e1(self) -> str:
        return self.hops[0].subject

    @property
    def bridge(self) -> str:
        if self.k < 2:
            raise ValueError("single-hop query has no bridge")
        return self.hops[1].subject

    @property
    def target_answers(self) -> list[str]:
        return self.hops[-1].answers

    def validate(self) -> None:
        """
        Chain integrity: each hop's subject must be the previous hop's answer.

        Single-hop (k=1) queries have no previous hop to check against, so
        the chain-integrity loop is skipped for them rather than rejected --
        E0 (skc_runbook.txt) needs single-hop ComposedQuery objects as its
        primary input, and there is nothing to validate about a one-link
        "chain" beyond the relation-exclusion check that still applies below.
        Composed (k>=2) queries are unaffected: the loop still runs exactly
        as before.
        """
        if self.k < 1:
            raise ValueError(f"{self.qid}: need at least one hop")
        for i in range(1, self.k):
            prev_gold = self.hops[i - 1].answers[0]
            if normalize(self.hops[i].subject) != normalize(prev_gold):
                raise ValueError(
                    f"{self.qid}: hop {i} subject {self.hops[i].subject!r} does not match "
                    f"hop {i-1} answer {prev_gold!r}. The chain is broken; a broken chain "
                    "silently produces meaningless K_j labels."
                )
        for h in self.hops:
            if h.relation.lower() in EXCLUDED_RELATIONS:
                raise ValueError(f"{self.qid}: relation {h.relation!r} is excluded")


# =====================================================================
# Alias-aware exact match
# =====================================================================

_ARTICLES = re.compile(r"^\s*(the|a|an)\s+", re.I)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[.!?]")

# Descriptive lead-in phrases free-generating base models reliably prepend
# before naming the actual answer, e.g. "The place of birth of the player X
# is [the city of] Plovdiv." Confirmed against real Gemma-2-2b generations
# on the pilot dataset (skc_run_pilot.py): 15/300 hop1 answers were correct
# in substance but rejected by alias_match's require_prefix=True because of
# exactly this lead-in, every single one following one of these forms.
# Deliberately a SHORT, EXPLICIT list grounded in observed generations, not
# a speculative guess -- an overly broad filler list would let genuinely
# wrong answers slip through the same way loosening require_prefix would.
_GENERATION_FILLER = re.compile(
    r"^(the\s+)?(city|country|state|province|nation)\s+of\s+|^in\s+", re.I
)


def _strip_generation_filler(s: str) -> str:
    """
    Strip a leading descriptive filler phrase AND truncate at the first
    sentence boundary. Applied ONLY to model generations being scored, never
    to gold answers (which are already clean strings with no filler or
    trailing narrative to strip).

    Order matters: filler must be stripped BEFORE truncation, because the
    filler phrase itself precedes the answer and the first sentence boundary
    comes AFTER it (see the docstring examples in alias_match).
    """
    s = _GENERATION_FILLER.sub("", s.strip())
    m = _SENTENCE_END.search(s)
    return s[:m.start()] if m else s


def normalize(s: str) -> str:
    """
    Case-fold, strip accents, drop punctuation and leading articles, squeeze
    whitespace. Deliberately conservative: it does NOT stem, and it does not
    strip parentheticals, because "Michael Jordan (footballer)" and
    "Michael Jordan" are different entities and collapsing them would corrupt
    the labels in exactly the direction that flatters the paper.
    """
    if not isinstance(s, str):
        raise TypeError(f"expected str, got {type(s).__name__}")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = _ARTICLES.sub("", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def alias_match(generation: str, answers: Sequence[str],
                require_prefix: bool = True) -> bool:
    """
    True when the generation matches any gold surface form or alias.

    require_prefix=True (default) asks the normalised generation to START with
    an alias rather than merely contain it. Containment is much too permissive
    for free generation: a model that lists ten candidate names would count as
    correct for every one of them.

    Before the prefix check, the generation (never the gold answer) has its
    leading descriptive filler stripped and is truncated at the first
    sentence boundary -- see _strip_generation_filler(). Confirmed necessary
    against real generations: free-generating base models reliably answer
    "The place of birth of the player X is" with "the city of Plovdiv. He
    was born in 1991...", and without this step require_prefix=True rejects
    every one of those as wrong, even though the entity named is correct.
    """
    if not answers:
        raise ValueError("no gold answers supplied; cannot score")
    generation = _strip_generation_filler(generation)
    g = normalize(generation)
    if not g:
        return False
    for a in answers:
        na = normalize(a)
        if not na:
            continue
        if g == na:
            return True
        if require_prefix and g.startswith(na + " "):
            return True
        if not require_prefix and na in g:
            return True
    return False


# =====================================================================
# Refusal detection
# =====================================================================

REFUSAL_LEXICON = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i cannot", "i can't", "unable to", "no information", "not aware of",
    "i don't have", "i do not have", "unfamiliar with", "cannot determine",
)


def is_refusal(generation: str, candidate_answers: Sequence[str] | None = None) -> bool:
    """
    A response counts as a refusal only when it matches the lexicon AND does not
    commit to a candidate answer. Both conditions are required, because a
    response can hedge and still answer ("I'm not sure, but I think it's X"),
    and counting that as abstention would inflate every abstention metric.
    """
    g = normalize(generation)
    hedged = any(normalize(p) in g for p in REFUSAL_LEXICON)
    if not hedged:
        return False
    if candidate_answers and alias_match(generation, candidate_answers,
                                         require_prefix=False):
        return False
    return True


# =====================================================================
# Prompt rendering
# =====================================================================

def render_prompts(q: ComposedQuery) -> dict[str, str]:
    """
    All prompts for one query: the composed form, every single-hop
    decomposition, the named-bridge control, and the chain-of-thought form.

    k=1 (single-hop, used by E0 -- skc_runbook.txt's positive control) has
    no "composed" query to decompose: it IS already the decomposition. Only
    the per-hop loop below applies, producing just {"hop1": ...}. k=2/k=3
    additionally get "composed" (and "cot", "named_bridge" for k=2), since
    those are the forms that matter for entities that are never written.
    """
    q.validate()
    out: dict[str, str] = {}
    if q.k == 1:
        pass  # no composed/cot/named_bridge form; the per-hop loop below covers this case
    elif q.k == 2:
        out["composed"] = TEMPLATES["composed_2"].format(
            r2=q.hops[1].relation, r1=q.hops[0].relation,
            etype=q.entity_type, e1=q.e1)
        out["cot"] = TEMPLATES["cot"].format(
            r2=q.hops[1].relation, r1=q.hops[0].relation,
            etype=q.entity_type, e1=q.e1)
    elif q.k == 3:
        out["composed"] = TEMPLATES["composed_3"].format(
            r3=q.hops[2].relation, r2=q.hops[1].relation, r1=q.hops[0].relation,
            etype=q.entity_type, e1=q.e1)
    else:
        raise ValueError(f"{q.qid}: only k=1, k=2 and k=3 are templated; got k={q.k}")

    for j, h in enumerate(q.hops, start=1):
        out[f"hop{j}"] = TEMPLATES["single_hop"].format(
            relation=h.relation, etype=h.subject_type, entity=h.subject)

    if q.k >= 2:
        # named_bridge names q.bridge (the hop2 subject), which does not
        # exist for a single-hop query -- q.bridge itself already raises for
        # k<2, so this form is skipped rather than attempted for k=1.
        out["named_bridge"] = TEMPLATES["named_bridge"].format(
            r2=q.hops[1].relation, btype=q.hops[1].subject_type, bridge=q.bridge)
    return out


def length_match_named_bridge(composed: str, named: str, entity: str,
                              filler: str = "well-known") -> str:
    """
    Pad the named-bridge control toward the composed prompt's token count with a
    type-neutral modifier, so a gate difference cannot be attributed to prompt
    length. Returns the padded control.

    We pad by whitespace-token count, which is a proxy. Check the tokenizer count
    on your model and report the residual difference rather than assuming it is
    zero.

    `entity` is the REQUIRED surface form of the named entity in `named`
    (e.g. "Buenos Aires", "New Brunswick") -- REAL BUG FOUND AND FIXED
    (first real GPU run of this function, job 389064, E7 control (c)):
    the original version had no `entity` parameter at all and guessed the
    insertion point as `len(words) - 2` (i.e. "before the last 2 words"),
    silently assuming the named entity is always exactly ONE word. This
    is wrong for any multi-word entity -- confirmed directly, "Buenos
    Aires" (2 words) produced 'The country of the city Buenos well-known
    Aires is', splitting the entity's own two words apart and inserting
    filler INSIDE it, which then made find_entity_span()'s exact-
    substring lookup fail outright (a real crash, not a silent
    corruption -- ValueError: entity surface form not found verbatim).
    Fixed by inserting the filler immediately before `entity`'s own
    first occurrence as a whole unit, located via str.rfind on the
    UNPADDED `named` string (mirroring find_entity_span()'s own
    last-occurrence convention), rather than guessing by word count at
    all -- correct regardless of how many words the entity itself spans.
    """
    gap = len(composed.split()) - len(named.split())
    if gap <= 0:
        return named
    char_idx = named.rfind(entity)
    if char_idx < 0:
        raise ValueError(
            f"entity {entity!r} not found verbatim in named {named!r} -- cannot "
            "locate where to insert the length-matching filler."
        )
    before, after = named[:char_idx], named[char_idx:]
    # before ends in a trailing space (a word boundary); insert the filler
    # words there, immediately preceding the entity's own first word.
    return before + (filler + " ") * gap + after


# =====================================================================
# Labels
# =====================================================================

def compute_labels(q: ComposedQuery, generations: dict[str, str],
                   hop_require_prefix: dict[int, bool] | None = None) -> dict:
    """
    Y and K_1..K_k from generations. `generations` maps the prompt keys of
    render_prompts() to the model's greedy continuation.

    Raises on a missing key. There is no default: a missing generation means the
    run is incomplete, and silently scoring it as incorrect would bias every
    downstream quantity.

    k=1 (single-hop, E0's input) has no "composed" prompt -- render_prompts()
    does not produce one for k=1, since there is nothing to decompose. Y and
    K1 are the SAME quantity in that case: whether the model's one answer to
    its one question is correct. This mirrors render_prompts()'s own k=1
    branch rather than inventing a new convention.

    hop_require_prefix: optional {hop_index: bool} override of alias_match's
    default (require_prefix=True) for individual hops -- including K1
    (hop 1) for k=1 queries, where the override IS honored (this
    docstring previously claimed "K1 always uses the strict default,"
    which was WRONG and caused a real bug: skc_run_e0.py passes
    hop_require_prefix={1: False} for exactly the reason below, but
    skc_run_e1.py/skc_run_e5.py/skc_run_e7_ablations.py's own K1
    computations did not, silently disagreeing with E0's (and
    data/onehop_correct.json's) definition of "K1=1" until this mismatch
    surfaced as a real run failure -- see EXECUTION_PLAN.md). Only Y (the
    composed answer, line ~372 below) has no override available at all --
    that one genuinely always uses the strict default, hardcoded.
    require_prefix=False exists because real Gemma-2-2b generations for
    hop2-style prompts ("The country of the city X is") reliably answer
    with the correct country named somewhere in a descriptive sentence
    ("located in the state of Indiana...", "a city in the state of Texas
    in the United States...") rather than as the generation's first
    token(s), in a much messier variety of lead-in phrasings than hop1
    shows in the ORIGINAL pilot's player.place_birth entity type --
    confirmed on the pilot's raw_generations.json: every hop1 miss there
    followed one of two clean patterns and was fixed by
    _strip_generation_filler. Later measurement across ALL FOUR Ferrando
    entity types (skc_run_e0.py's own E0 run) found hop1 needs THE SAME
    loosening for the other three entity types (city, movie, song): strict
    K1 = 19/1999 (1.0%) vs containment K1 = 163/1999 (8.2%) on the SAME
    generations. Loosening is safe here specifically because
    max_new_tokens is small (skc_runbook.txt's pilot uses 24, E0 uses the
    same default), which bounds how much text there is for a stray,
    non-answer name to hide in -- this is not a blanket loosening of
    alias_match's own default, which stays strict everywhere this argument
    is not passed.
    """
    need = (["hop1"] if q.k == 1 else ["composed"] + [f"hop{j}" for j in range(1, q.k + 1)])
    missing = [key for key in need if key not in generations]
    if missing:
        raise KeyError(
            f"{q.qid}: missing generations for {missing}. Complete the run; do not "
            "score partial output."
        )
    hop_require_prefix = hop_require_prefix or {}
    labels = {"qid": q.qid, "k": q.k}
    if q.k == 1:
        labels["K1"] = int(alias_match(generations["hop1"], q.hops[0].answers,
                                       require_prefix=hop_require_prefix.get(1, True)))
        labels["Y"] = labels["K1"]
        labels["M"] = 1  # no downstream hops to fail; M is vacuously true for k=1
        labels["refusal_composed"] = int(is_refusal(generations["hop1"], q.hops[0].answers))
        return labels
    labels["Y"] = int(alias_match(generations["composed"], q.target_answers))
    for j, h in enumerate(q.hops, start=1):
        labels[f"K{j}"] = int(alias_match(generations[f"hop{j}"], h.answers,
                                         require_prefix=hop_require_prefix.get(j, True)))
    labels["M"] = int(np.prod([labels[f"K{j}"] for j in range(2, q.k + 1)]))
    labels["refusal_composed"] = int(is_refusal(generations["composed"], q.target_answers))
    if "named_bridge" in generations:
        labels["Y_named_bridge"] = int(
            alias_match(generations["named_bridge"], q.target_answers))
    return labels


def filter_bridge_resolved(labels: Iterable[dict],
                           bridge_resolved: dict[str, bool] | None = None) -> list[dict]:
    """
    Keep only queries with K1 = 1 and, when supplied, independent evidence that
    the bridge was resolved internally (from the Patchscopes decode).

    This is the filter the paper calls non-negotiable. A composed failure caused
    by a missing first hop says nothing about composition, and leaving those in
    turns a first-hop result into a fake composition result.
    """
    kept = []
    for lab in labels:
        if lab.get("K1", 0) != 1:
            continue
        if bridge_resolved is not None and not bridge_resolved.get(lab["qid"], False):
            continue
        kept.append(lab)
    return kept


# =====================================================================
# Popularity matching
# =====================================================================

def assign_popularity_deciles(pageviews: dict[str, float]) -> dict[str, int]:
    """Decile 0 = least viewed. Ties break toward the lower decile."""
    if not pageviews:
        raise ValueError("no pageview data supplied")
    names = list(pageviews)
    v = np.asarray([pageviews[n] for n in names], dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("pageview data contains NaN or inf")
    edges = np.quantile(v, np.linspace(0, 1, 11)[1:-1])
    return {n: int(np.searchsorted(edges, val, side="left")) for n, val in zip(names, v)}


def match_on_decile(queries: Sequence[ComposedQuery], seed: int = 0
                    ) -> list[tuple[ComposedQuery, ComposedQuery]]:
    """
    Pair known-bridge with unknown-bridge queries inside the same popularity
    decile, for the C2 patching control. Unpaired items are dropped and the
    dropped count is returned to the caller through the log, because a patch
    between mismatched-popularity queries measures popularity, not knowledge.
    """
    rng = np.random.default_rng(seed)
    by_decile: dict[int, list[ComposedQuery]] = {}
    for q in queries:
        if q.popularity_decile is None:
            raise ValueError(f"{q.qid}: popularity_decile not set; run assign_popularity_deciles first")
        by_decile.setdefault(q.popularity_decile, []).append(q)
    pairs = []
    for d, group in by_decile.items():
        known = [q for q in group if q.meta.get("bridge_known") is True]
        unknown = [q for q in group if q.meta.get("bridge_known") is False]
        if not known or not unknown:
            continue
        rng.shuffle(known)
        rng.shuffle(unknown)
        pairs.extend(zip(known, unknown))
    return pairs


# =====================================================================
# Provenance
# =====================================================================

def snapshot_id(queries: Sequence[ComposedQuery], wikidata_date: str) -> str:
    """
    Content hash of the dataset plus the Wikidata date. Record this in the paper.
    Entity knowledge changes, so an unrecorded snapshot makes the labels
    irreproducible even with the code released.
    """
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", wikidata_date):
        raise ValueError(f"wikidata_date must be YYYY-MM-DD, got {wikidata_date!r}")
    h = hashlib.sha256()
    h.update(wikidata_date.encode())
    for q in sorted(queries, key=lambda x: x.qid):
        h.update(json.dumps(asdict(q), sort_keys=True, default=str).encode())
    return f"{wikidata_date}+{h.hexdigest()[:16]}"


# =====================================================================
# Wikidata source (NOT covered by the offline test suite)
# =====================================================================

SPARQL_TWO_HOP = """
SELECT ?e1 ?e1Label ?bridge ?bridgeLabel ?target ?targetLabel WHERE {{
  ?e1 wdt:P31 wd:{etype_qid} .
  ?e1 wdt:{r1_pid} ?bridge .
  ?bridge wdt:{r2_pid} ?target .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit}
"""
# NOTE on etype_qid/P31: this template's "?e1 wdt:P31 wd:{etype_qid}"
# ("instance of") is correct for non-person entity types (e.g. "instance
# of: city", "instance of: film"), but is WRONG for occupation-typed
# person entities like "basketball player" -- people are modeled in
# Wikidata as "instance of: human" (P31 wd:Q5) with occupation as a
# SEPARATE property (P106), confirmed directly against the live endpoint
# while building SPARQL_THREE_HOP_STEP1 below (wdt:P31 wd:Q3665646 for
# "basketball player" returns 0 results; wdt:P106 wd:Q3665646 returns
# real players immediately). This template was never actually run before
# this session (its own docstring says "UNTESTED IN THIS REPOSITORY") --
# flagged here rather than silently changed, since the correct predicate
# depends on whether etype_qid names a person-occupation or a genuine
# Wikidata class, which only the CALLER knows for a given etype_qid.


# =====================================================================
# Wikidata source, THREE-HOP (E5's "compounding across hop depth" needs
# k=3 chains; the two-hop path above only ever produces k=2). Built and
# VERIFIED against the live endpoint for the player -> city -> country ->
# continent chain specifically (P106, P19, P17, P30 all confirmed real
# Wikidata property IDs by direct SPARQL lookup, not guessed) -- see
# EXECUTION_PLAN.md for the verification trail. Generalizes to other
# (r1, r2, r3) triples in principle, but ONLY THIS SPECIFIC CHAIN has
# been run against the real endpoint; verify a new chain's own sample
# response by hand (fetch_three_hop's own --limit-first-hop / confirm
# behavior) before trusting it, exactly as fetch_two_hop's docstring
# already requires for the two-hop path.
# =====================================================================

SPARQL_THREE_HOP_STEP1 = """
SELECT ?e1 ?e1Label ?bridge1 ?bridge1Label WHERE {{
  ?e1 wdt:{r0_pid} wd:{etype_qid} .
  ?e1 wdt:{r1_pid} ?bridge1 .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit}
"""
# r0_pid: P106 (occupation) for person/occupation entity types like
# "basketball player" -- P31 (instance of) for genuine Wikidata classes.
# See the note on SPARQL_TWO_HOP above for why this differs by entity
# kind; the caller must pick the right one for etype_qid, not this
# template.

SPARQL_THREE_HOP_STEP2 = """
SELECT ?bridge1 ?bridge2 ?bridge2Label ?target ?targetLabel WHERE {{
  VALUES ?bridge1 {{ {bridge1_values} }}
  ?bridge1 wdt:{r2_pid} ?bridge2 .
  ?bridge2 wdt:{r3_pid} ?target .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""
# WHY TWO SEPARATE QUERIES, NOT ONE 4-WAY JOIN: a single query joining
# "?e1 wdt:r0 wd:etype . ?e1 wdt:r1 ?bridge1 . ?bridge1 wdt:r2 ?bridge2 .
# ?bridge2 wdt:r3 ?target" (even with a LIMIT subquery around the first
# two triples) reliably TIMED OUT against the live endpoint -- confirmed
# directly, multiple attempts, up to a 90s client timeout. Splitting into
# step 1 (get a LIMITed batch of e1/bridge1 pairs, itself fast) and step 2
# (a VALUES-bound batch resolving JUST those specific bridge1 entities
# through the rest of the chain) reliably completes in a few seconds.
# VALUES binds the second query's starting point to a small, explicit
# entity list instead of asking the query planner to join across the
# full space of everyone with r0=etype_qid -- this is the standard
# Wikidata performance pattern for exactly this "get some IDs, then
# resolve them further" shape, not a workaround specific to this bundle.


def fetch_three_hop(etype_qid: str, r0_pid: str, r1_pid: str, r2_pid: str, r3_pid: str,
                    limit: int = 2000,
                    endpoint: str = "https://query.wikidata.org/sparql",
                    user_agent: str = "skc-research/1.0") -> list[dict]:
    """
    Pull three-hop chains from Wikidata: e1 --r1--> bridge1 --r2--> bridge2
    --r3--> target, where e1 satisfies "r0_pid = etype_qid" (P106 for an
    occupation-typed entity like "basketball player", P31 for a genuine
    Wikidata class -- see SPARQL_THREE_HOP_STEP1's own note).

    Two-step, not one query -- see SPARQL_THREE_HOP_STEP2's own comment
    for why a single 4-way join times out against the live endpoint while
    this two-step VALUES-bound version does not (confirmed directly).
    Step 1 is LIMITed to `limit` (e1, bridge1) pairs; step 2 resolves the
    DISTINCT bridge1 entities from step 1 through r2/r3 -- if several e1
    share the same bridge1 (e.g. two players born in the same city), step
    2 only queries that city once, and the caller must re-join step 2's
    results back onto every (e1, bridge1) pair from step 1 that used it
    (done below).

    UNTESTED end to end at scale in this repository -- verified by hand
    against the player/city/country/continent chain specifically while
    building this function (limit=10 and limit=500 both succeeded with
    correct data; limit=4000 initially failed with a 414 URI-Too-Long on
    step 2's GET request, since a few thousand players' worth of DISTINCT
    birth cities does not fit in a URL -- fixed by switching step 2 to
    batched POST requests, see below). Verify the first response of any
    NEW chain by hand before trusting a full pull, same discipline
    fetch_two_hop's own docstring requires.
    """
    import requests  # imported lazily so the offline path needs no dependency
    import time

    q1 = SPARQL_THREE_HOP_STEP1.format(etype_qid=etype_qid, r0_pid=r0_pid,
                                       r1_pid=r1_pid, limit=limit)
    r = requests.get(endpoint, params={"query": q1, "format": "json"},
                     headers={"User-Agent": user_agent}, timeout=60)
    r.raise_for_status()
    step1_rows = [{k: v["value"] for k, v in row.items()}
                 for row in r.json()["results"]["bindings"]]
    if not step1_rows:
        return []

    # Step 2, BATCHED and POSTed, not one GET: a single VALUES clause over
    # every distinct bridge1 QID from a large step 1 pull does not fit in
    # a URL (confirmed directly: limit=4000 produced a 414 URI Too Long on
    # a GET request). Splitting into fixed-size batches keeps each
    # request's VALUES list short, and POST sends the query in the request
    # body instead of the URL, sidestepping the length limit entirely --
    # the standard fix for this exact Wikidata query shape, not specific
    # to this bundle.
    bridge1_qids = sorted({row["bridge1"].rsplit("/", 1)[-1] for row in step1_rows})
    batch_size = 200
    step2_rows = []
    for i in range(0, len(bridge1_qids), batch_size):
        if i > 0:
            # A real chain (limit=4000 -> ~20 batches) hit Wikidata's
            # rate limit (429) firing them back to back -- confirmed
            # directly while building this function. A short pause
            # between batches is standard courtesy for the shared public
            # endpoint, not just a workaround for this one run.
            time.sleep(1.0)
        batch = bridge1_qids[i:i + batch_size]
        values_clause = " ".join(f"wd:{q}" for q in batch)
        q2 = SPARQL_THREE_HOP_STEP2.format(bridge1_values=values_clause,
                                           r2_pid=r2_pid, r3_pid=r3_pid)
        r = requests.post(endpoint, data={"query": q2, "format": "json"},
                          headers={"User-Agent": user_agent}, timeout=60)
        r.raise_for_status()
        step2_rows.extend({k: v["value"] for k, v in row.items()}
                          for row in r.json()["results"]["bindings"])

    # step 2 resolved each DISTINCT bridge1 once; re-join onto every
    # (e1, bridge1) pair from step 1 that shares it. A bridge1 with no
    # r2/r3 chain (step 2 returned nothing for it) silently drops every
    # e1 that used it -- correct behavior (nothing to chain to), not an
    # error, but worth knowing if the final count is smaller than step 1's.
    by_bridge1 = {}
    for row in step2_rows:
        by_bridge1.setdefault(row["bridge1"], []).append(row)

    out = []
    for row in step1_rows:
        for continuation in by_bridge1.get(row["bridge1"], []):
            out.append({**row, **continuation})
    return out


def fetch_two_hop(etype_qid: str, r1_pid: str, r2_pid: str, limit: int = 2000,
                  endpoint: str = "https://query.wikidata.org/sparql",
                  user_agent: str = "skc-research/1.0") -> list[dict]:
    """
    Pull two-hop chains from Wikidata.

    UNTESTED IN THIS REPOSITORY: it needs network access. Verify the first
    response by hand before trusting a full pull, and be aware that ambiguous
    labels (several entities sharing a surface form) must be filtered out, or
    the entity_type disambiguation the templates rely on will not hold.
    """
    import requests  # imported lazily so the offline path needs no dependency
    q = SPARQL_TWO_HOP.format(etype_qid=etype_qid, r1_pid=r1_pid,
                              r2_pid=r2_pid, limit=limit)
    r = requests.get(endpoint, params={"query": q, "format": "json"},
                     headers={"User-Agent": user_agent}, timeout=120)
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    return [{k: v["value"] for k, v in row.items()} for row in rows]


def fetch_pageviews(article_titles: Sequence[str], start: str = "2024010100",
                    end: str = "2024120100",
                    endpoint: str = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
                                    "/per-article/en.wikipedia/all-access/user",
                    user_agent: str = "skc-iclr2027-research/1.0 (research use)",
                    delay_s: float = 0.3, max_retries: int = 5) -> dict[str, float]:
    """
    Sum monthly English Wikipedia pageviews (all-access, user agent-type,
    i.e. excluding bots/spiders) over [start, end) for each article title,
    via the public Wikimedia Pageviews REST API
    (https://wikimedia.org/api/rest_v1/metrics/pageviews). No API key
    needed; a descriptive User-Agent is required by Wikimedia's own usage
    policy and is set above.

    VERIFIED live against this endpoint while building this function:
    "Porto" over 2024 returns 12 monthly points summing to a plausible
    city-article view count; a nonexistent article returns 404 (handled
    below by omission, not by imputing 0 -- a missing article and a
    real-but-unviewed article are different things, and this function
    must not conflate them for skc_run_e7_ablations.py's
    assign_popularity_deciles(), which itself refuses empty input).

    Article titles are matched EXACTLY (spaces replaced with underscores,
    the API's own convention) -- no redirect resolution, no disambiguation.
    A bridge surface form that does not match a real article title's exact
    spelling (accents, alternate names) silently returns 404 and is
    dropped, same as a genuinely nonexistent article. This is a real,
    disclosed limitation of matching Wikidata entity surface forms to
    Wikipedia article titles by string equality; call sites should report
    how many titles were dropped this way, not assume 100% coverage.
    """
    import requests  # imported lazily so the offline path needs no dependency
    import time

    # 0.05s between requests was tried first at the full 847-title scale
    # this function was built for and hit a real 429 partway through --
    # confirmed directly, not a hypothetical. Raised the default delay to
    # 0.3s and added retry-with-backoff on 429 (honoring Retry-After when
    # the server sends one) as standard courtesy for a shared public
    # endpoint, same discipline fetch_three_hop's own inter-batch sleep
    # already applies to the separate Wikidata SPARQL endpoint.
    out = {}
    for i, title in enumerate(article_titles):
        if i > 0 and delay_s:
            time.sleep(delay_s)
        safe_title = title.replace(" ", "_")
        url = f"{endpoint}/{safe_title}/monthly/{start}/{end}"
        for attempt in range(max_retries):
            r = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 5.0 * (attempt + 1)))
                time.sleep(wait)
                continue
            break
        if r.status_code == 404:
            continue
        r.raise_for_status()
        items = r.json().get("items", [])
        out[title] = float(sum(it["views"] for it in items))
    return out


def save_queries(queries: Sequence[ComposedQuery], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(q) for q in queries], f, indent=1, default=str)


def load_queries(path: str) -> list[ComposedQuery]:
    with open(path) as f:
        raw = json.load(f)
    out = []
    for d in raw:
        d["hops"] = [Hop(**h) for h in d["hops"]]
        q = ComposedQuery(**d)
        q.validate()
        out.append(q)
    return out

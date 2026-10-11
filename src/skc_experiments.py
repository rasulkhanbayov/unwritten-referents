#!/usr/bin/env python3
"""
skc_experiments.py
Interventions C1-C4 and experiment drivers E0-E7 for
"Tokens, Not Referents: Latent Entities and the Limits of Knowledge-Aware Abstention".

Run order matters. E0 is a STOP GATE: if the frozen estimators do not reproduce
the known single-hop separation, a null in E1 cannot be distinguished from a
broken probe, and nothing downstream is interpretable. run_all() enforces this
and refuses to continue.

WHAT IS AND IS NOT TESTED
  Tested offline by skc_pipeline_test.py with a mock model: the full driver
  sequence, the E0 gate, result schemas, the K1 filter, and every guard.
  NOT tested: anything that touches a real transformer, which means the hook
  bodies inside the intervention functions. Those are written and asserted but
  unverified. Treat the first real run as a debugging run.

Outputs: one JSON per experiment, plus a manifest recording the code hash, the
dataset snapshot id, and every seed, so a result can be traced to what produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

import skc_metrics as M
from skc_data import compute_labels, filter_bridge_resolved, render_prompts
from skc_gates import FrozenGate, gate_agreement

AGREEMENT_THRESHOLD = 0.50      # min pairwise Spearman across the three estimators
E0_AUROC_THRESHOLD = 0.65       # fixed in advance; see the runbook


# =====================================================================
# Run manifest
# =====================================================================

@dataclass
class RunManifest:
    outdir: str
    snapshot_id: str
    model_name: str
    seed: int = 0
    started: float = field(default_factory=time.time)
    entries: dict = field(default_factory=dict)

    def code_hash(self) -> str:
        h = hashlib.sha256()
        here = os.path.dirname(os.path.abspath(__file__))
        for fn in sorted(f for f in os.listdir(here) if f.endswith(".py")):
            with open(os.path.join(here, fn), "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:16]

    def record(self, name: str, payload: dict) -> None:
        self.entries[name] = payload
        os.makedirs(self.outdir, exist_ok=True)
        with open(os.path.join(self.outdir, f"{name}.json"), "w") as f:
            json.dump(payload, f, indent=1, default=str)

    def write(self) -> str:
        os.makedirs(self.outdir, exist_ok=True)
        path = os.path.join(self.outdir, "manifest.json")
        with open(path, "w") as f:
            json.dump({
                "snapshot_id": self.snapshot_id,
                "model_name": self.model_name,
                "seed": self.seed,
                "code_hash": self.code_hash(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "started": self.started,
                "finished": time.time(),
                "experiments": sorted(self.entries),
            }, f, indent=1)
        return path


# =====================================================================
# E0: positive control, and a stop gate
# =====================================================================

def run_E0(gates: dict[str, FrozenGate], acts_heldout: np.ndarray,
           known_heldout: np.ndarray, acts_named_bridge: np.ndarray,
           k2_named: np.ndarray, manifest: RunManifest) -> dict:
    """
    Confirm the instrument works before using it.

    (a) each frozen estimator separates known from unknown entities on HELD-OUT
        single-hop prompts;
    (b) the estimators agree with each other;
    (c) downstream knowledge is gate-visible when the bridge is NAMED, which is
        the premise whose absence the rest of the paper tests.
    """
    for g in gates.values():
        g.assert_not_fit_on(acts_heldout, "held-out single-hop data")

    per_est = {}
    for name, g in gates.items():
        s = g.score(acts_heldout)
        auc, lo, hi = M.auroc_ci(known_heldout, s, seed=manifest.seed)
        per_est[name] = {"auroc": auc, "ci": [lo, hi], "passes": auc >= E0_AUROC_THRESHOLD}

    agree = gate_agreement(gates, acts_heldout)
    named = {name: M.auroc(k2_named, g.score(acts_named_bridge))
             for name, g in gates.items()}

    n_pass = sum(v["passes"] for v in per_est.values())
    out = {
        "experiment": "E0",
        "threshold": E0_AUROC_THRESHOLD,
        "per_estimator": per_est,
        "n_estimators_passing": n_pass,
        "agreement": agree,
        "auroc_K2_named_bridge": named,
        "gate_open": bool(n_pass >= 2 and agree["min_pairwise_spearman"] >= AGREEMENT_THRESHOLD),
    }
    manifest.record("E0", out)
    return out


class E0Failure(RuntimeError):
    pass


def require_E0(e0: dict) -> None:
    if not e0["gate_open"]:
        raise E0Failure(
            "E0 did not pass. Fewer than two of three estimators cleared "
            f"AUROC {E0_AUROC_THRESHOLD} on held-out entities, or the estimators "
            f"disagreed (min pairwise Spearman "
            f"{e0['agreement']['min_pairwise_spearman']:.3f} < {AGREEMENT_THRESHOLD}).\n\n"
            "Do NOT proceed and do NOT tune the gate until it passes: tuning on the "
            "data you will later test on destroys the result. A null in E1 after a "
            "failed E0 is uninterpretable, because 'the model is blind' and 'my probe "
            "does not work' produce the same output. Flag back instead."
        )


# =====================================================================
# E1: the blind spot, and its silence
# =====================================================================

def run_E1(gate: FrozenGate, acts_surface: np.ndarray, acts_bridge: np.ndarray,
           labels: list[dict], acts_singlehop_correct: np.ndarray,
           manifest: RunManifest, coverages=(0.3, 0.5, 0.8),
           acts_surface_unfiltered: np.ndarray | None = None,
           k1_unfiltered: np.ndarray | None = None) -> dict:
    """
    (a) AUROC of the surface gate against K1, M, Y
    (b) slack terms and the Theorem 1 floor, tested by confidence criterion
    (c) the Lemma 1 decomposition at several coverages
    (d) the silence check: is the gate as confident on composed queries as on
        single-hop queries the model answers correctly?

    SUBSET DISCIPLINE, and the reason for the last two arguments.
    Everything about the downstream conjunction M must be computed on the
    K1 = 1 subset, or a first-hop failure is reported as a composition failure.
    But AUROC(gate -> K1) is UNDEFINED on that subset, because K1 is constant
    there. It has to come from the UNFILTERED set. Mixing the two silently is
    easy and would put an undefined or degenerate number in Table 1, so this
    function requires the unfiltered arrays explicitly and refuses to guess.
    """
    y  = np.array([l["Y"] for l in labels])
    k1 = np.array([l["K1"] for l in labels])
    m  = np.array([l["M"] for l in labels])
    gs = gate.score(acts_surface)
    gb = gate.score(acts_bridge)

    if len(set(k1.tolist())) > 1:
        auc_k1 = M.auroc(k1, gs)
        k1_source = "filtered set (K1 varies, so no filter was applied)"
    else:
        if acts_surface_unfiltered is None or k1_unfiltered is None:
            raise ValueError(
                "K1 is constant on this subset, so AUROC(gate -> K1) is undefined here. "
                "Pass acts_surface_unfiltered and k1_unfiltered so it can be computed "
                "before conditioning. Reporting it from the filtered set would be "
                "meaningless, and omitting it silently would leave a hole in Table 1."
            )
        k1u = np.asarray(k1_unfiltered)
        if len(set(k1u.tolist())) < 2:
            raise ValueError("k1_unfiltered is also constant; check the filter")
        auc_k1 = M.auroc(k1u, gate.score(acts_surface_unfiltered))
        k1_source = "unfiltered set (K1 is constant after conditioning)"

    out = {
        "experiment": "E1",
        "n": int(len(labels)),
        "auroc_K1_computed_on": k1_source,
        "auroc": {
            "gate_to_K1": auc_k1,
            "gate_to_M":  M.auroc(m, gs),
            "gate_to_Y":  M.auroc(y, gs),
        },
        "floor": M.floor_report(y, m, gs, seed=manifest.seed),
        "decomposition": {},
        "silence": _silence_check(gs, gate.score(acts_singlehop_correct)),
    }
    for c in coverages:
        out["decomposition"][str(c)] = M.blind_spot_decomposition(
            y, gs, gs.reshape(-1, 1), np.column_stack([gs, gb]),
            coverage=c, seed=manifest.seed)
    manifest.record("E1", out)
    return out


def _silence_check(gate_composed: np.ndarray, gate_singlehop_correct: np.ndarray) -> dict:
    """
    Two-sample distance between the gate's score distribution on composed
    queries and on single-hop queries the model gets right.

    If the distributions are indistinguishable while accuracy differs sharply,
    the failure is not merely present but SILENT: nothing in the model's own
    signal marks the composed case as riskier. That is the headline of E1.
    """
    a = np.sort(np.asarray(gate_composed, float))
    b = np.sort(np.asarray(gate_singlehop_correct, float))
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples must be non-empty for the silence check")
    grid = np.sort(np.concatenate([a, b]))
    ks = float(np.max(np.abs(np.searchsorted(a, grid, "right") / a.size
                             - np.searchsorted(b, grid, "right") / b.size)))
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) if min(a.size, b.size) > 1 else np.nan
    return {
        "ks_statistic": ks,
        "cohens_d": float((a.mean() - b.mean()) / pooled) if pooled and np.isfinite(pooled) else None,
        "n_composed": int(a.size), "n_singlehop_correct": int(b.size),
        "reading": "small KS with a large accuracy gap means the failure is silent",
    }


# =====================================================================
# C1-C4 interventions
# =====================================================================

def steer_at_position(model, tokenizer, prompts, direction, coefficient, layer,
                      positions, device="cuda", max_new_tokens=24, batch_size=16):
    """
    C1 and C3. Add `coefficient * direction` to the residual stream at the given
    positions and generate.

    batch_size=16: matches capture_activations()/label_queries_batched()'s own
    default elsewhere in this codebase. This function used to tokenize and
    generate on the ENTIRE prompt list in one `model.generate()` call, with no
    internal batching at all -- fine on a 2B/9B model with headroom to spare,
    but a REAL crash on a real GPU run: E7's random-direction control on
    Llama-3.1-70B (4-bit) called this function twice over 3503 prompts each,
    and the second call hit `torch.OutOfMemoryError: CUDA out of memory` with
    only ~5GB free on a 79GB GPU already holding ~74GB (the 4-bit model
    weights plus one full-batch generation's worth of activations/KV-cache).
    `cmd_named_bridge` in the same script succeeded because it batches
    internally via `label_queries_batched`; this function was the one
    remaining unbatched large-prompt-list generation call in the pipeline.
    Fixed at the source so every call site (this one, and skc_run_e2.py's
    C1/C3 uses) benefits, not just the one that happened to crash first.

    UNTESTED HERE (needs a GPU). Select the coefficient on a VALIDATION split,
    never on the test set, and report the whole sweep rather than the chosen
    value alone.
    """
    import torch
    d = torch.tensor(np.asarray(direction, np.float32), device=device)

    # KEEPS RIGHT-PADDING (this function's pre-existing behavior, now
    # made explicit rather than left to whatever the tokenizer happened
    # to inherit): `positions` are entity-span-relative indices computed
    # by entity_spans_for_prompts()/find_entity_span() BEFORE any
    # batching or padding -- i.e. "Nth token from the START of this
    # prompt's own real content" -- which only lands correctly under
    # RIGHT-padding (content starts at index 0, padding trails after).
    # Under LEFT-padding, content is pushed right by the pad amount, so
    # a content-relative `pos` would land inside the padding region
    # instead of on the entity token for any prompt shorter than its
    # batch's longest. Checked directly, not assumed: an earlier version
    # of this fix switched to LEFT-padding to fix generate()'s own
    # right-padding warning below, and would have silently broken the
    # steering injection itself in the process -- caught by testing
    # generate() under right-padding directly (a real transformers model,
    # not this codebase) before shipping either version.
    #
    # That warning ("A decoder-only architecture is being used, but
    # right-padding was detected!") is real but does NOT misalign the
    # extraction the way it first appeared to: tested directly against a
    # real model (gpt2, right-padded batch of ["The capital of France
    # is", "Hi"]) that `generate()` does NOT left-pad internally -- it
    # generates `max_new_tokens` new positions appended at the shared
    # `input_ids.shape[1]` boundary for every row, uniformly. For a row
    # shorter than the batch's longest, its own trailing PAD tokens are
    # simply generated as content first (verified: the "Hi" row's first
    # 4 new tokens were exactly the pad token, `<|endoftext|>`, before
    # genuinely new text began), silently costing that row real
    # generation budget rather than producing a misaligned decode. So
    # the ORIGINAL shared-offset slicing
    # (`gen[:, enc["input_ids"].shape[1]:]`) IS positionally correct;
    # the real defect is that short prompts in a mixed-length batch can
    # have their usable `max_new_tokens` silently eaten by echoed pad
    # tokens. Batching by similar prompt length (or accepting a larger
    # max_new_tokens margin) addresses that; it is a budget problem, not
    # a correctness bug in the slicing itself, and skip_special_tokens=True
    # already strips the echoed pad tokens from the returned text rather
    # than leaving visible garbage.
    prior_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    outputs = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_positions = positions[i:i + batch_size]
            call_count = {"n": 0}

            def hook(_m, _i, out):
                # model.generate() calls this hook ONCE PER FORWARD PASS: the
                # first call processes the full prompt (prefill), where `positions`
                # (prompt-relative indices) are valid; every subsequent call, one
                # per newly generated token, only processes the SINGLE new token
                # via the KV cache, so h.shape[1] == 1 and any positions[b] > 0 is
                # out of bounds. Confirmed as a real crash on the first real GPU
                # run (E7's random-direction control): "IndexError: index 13 is
                # out of bounds for dimension 1 with size 1" -- the exact same
                # bug class already found and fixed in patchscope_decode()'s hook
                # earlier this session (skc_harness.py), just never applied here
                # too. Patch ONLY the first call: the steered value written during
                # prefill is already part of that position's cached key/value, so
                # it does not need (and structurally cannot) be re-applied on
                # later single-token calls.
                call_count["n"] += 1
                if call_count["n"] > 1:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                assert h.ndim == 3, f"expected (batch, seq, d_model), got {tuple(h.shape)}"
                for b, pos in enumerate(batch_positions):
                    h[b, pos] = h[b, pos] + coefficient * d
                return (h,) + out[1:] if isinstance(out, tuple) else h

            handle = model.model.layers[layer].register_forward_hook(hook)
            try:
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
                outputs.extend(tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                      skip_special_tokens=True))
            finally:
                handle.remove()
    finally:
        tokenizer.padding_side = prior_padding_side
    return outputs


def patch_bridge(model, tokenizer, prompts, donor_states, layer_window, positions,
                 device="cuda", max_new_tokens=24):
    """
    C2, the decisive test. Replace the bridge-position residual with a donor
    from a matched query whose bridge is unknown, across a contiguous layer
    window, and generate.

    UNTESTED HERE (needs a GPU). Donors must be matched on relation type, prompt
    length, and popularity decile, or this measures popularity rather than
    knowledge. skc_data.match_on_decile() builds the pairs.
    """
    import torch
    handles = []
    # Same "hook fires once per forward-pass call, not once total" issue
    # already found and fixed in steer_at_position() (this file) and
    # patchscope_decode() (skc_harness.py) earlier this session --
    # model.generate() calls each layer's hook once per forward pass, and
    # only the FIRST (prefill) call has the full prompt in h.shape[1];
    # every later call is a single new token via the KV cache, where
    # `positions` (prompt-relative indices) would be out of bounds. Each
    # registered layer needs its OWN counter (one hook object per layer
    # in layer_window, each called once per forward pass independently).
    call_counts = {li: {"n": 0} for li in layer_window}

    def make_hook(li):
        def hook(_m, _i, out):
            call_counts[li]["n"] += 1
            if call_counts[li]["n"] > 1:
                return out
            h = out[0] if isinstance(out, tuple) else out
            for b, pos in enumerate(positions):
                h[b, pos] = torch.tensor(donor_states[li][b], device=device,
                                         dtype=h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    try:
        for li in layer_window:
            handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


def dissociation_ratio(d_recall: np.ndarray, d_refusal: np.ndarray,
                       n_boot: int = 2000, seed: int = 0) -> dict:
    """
    C2's summary: the ratio of STANDARDISED effects on recall and on refusal.

    Standardisation is not optional. The two outcomes have different scales, and
    an unstandardised ratio is not a dissociation, it is a units artefact.
    A large ratio means the bridge drives recall but not refusal.
    """
    r = np.asarray(d_recall, float)
    f = np.asarray(d_refusal, float)
    if r.shape != f.shape:
        raise ValueError(f"shape mismatch: {r.shape} vs {f.shape}")
    if r.size < 2:
        raise ValueError("need at least two paired observations")

    def ratio(a, b):
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        if sa < 1e-12 or sb < 1e-12:
            return np.nan
        za, zb = abs(a.mean() / sa), abs(b.mean() / sb)
        return np.inf if zb < 1e-12 else za / zb

    rng = np.random.default_rng(seed)
    point = ratio(r, f)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, r.size, r.size)
        v = ratio(r[idx], f[idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 20:
        raise RuntimeError("too few finite bootstrap ratios; effects are degenerate")
    v = np.asarray(vals)
    return {"ratio": float(point),
            "lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975)),
            "reading": "ratio >> 1 means the bridge drives recall but not refusal"}


def run_E2(recall_delta, refusal_delta, c1_refusal_by_M, c3_refusal_delta,
           manifest: RunManifest) -> dict:
    # C2_dissociation is computed defensively, not inline in the dict
    # literal: dissociation_ratio() legitimately raises when refusal_delta
    # is degenerate (zero variance across bootstrap resamples), and on a
    # real GPU run (job 383292) that is exactly what happened -- gemma-2-2b
    # confabulates a confident wrong answer rather than emitting any
    # REFUSAL_LEXICON phrase, on either the baseline or patched
    # generations, while recall_delta itself carried real, non-degenerate
    # signal (9/192 pairs flipped correct->incorrect after the bridge
    # patch). Letting that ONE degenerate sub-quantity crash the whole
    # dict literal silently discarded C1_selectivity and C3_injection_effect
    # too, even though both are fully computable from data that has
    # nothing wrong with it. This is itself a real, reportable finding
    # ("the model does not abstain on this template, it confabulates" --
    # the refusal axis of Table 1's dissociation test does not apply to
    # this model/template combination), not something to paper over by
    # loosening the guardrail.
    try:
        c2_dissociation = dissociation_ratio(recall_delta, refusal_delta,
                                             seed=manifest.seed)
    except RuntimeError as e:
        c2_dissociation = {
            "ratio": None, "undefined_reason": str(e),
            "recall_delta_mean": float(np.mean(recall_delta)),
            "recall_delta_std": float(np.std(recall_delta, ddof=1)),
            "refusal_delta_mean": float(np.mean(refusal_delta)),
            "refusal_delta_std": float(np.std(refusal_delta, ddof=1)),
            "reading": "the dissociation ratio is undefined when refusal_delta has "
                      "(near-)zero variance -- report recall_delta's own effect size "
                      "directly instead, and disclose that this model/template "
                      "combination does not exhibit the refusal axis this ratio needs.",
        }
    out = {
        "experiment": "E2",
        "C2_dissociation": c2_dissociation,
        "C1_selectivity": {
            "refusal_rate_M1": float(np.mean(c1_refusal_by_M.get(1, [np.nan]))),
            "refusal_rate_M0": float(np.mean(c1_refusal_by_M.get(0, [np.nan]))),
            "reading": "similar rates mean the intervention is not selective, which is H1",
        },
        "C3_injection_effect": float(np.mean(c3_refusal_delta)),
    }
    manifest.record("E2", out)
    return out


# =====================================================================
# E3: is the signal present but unused?
# =====================================================================

def run_E3(gate: FrozenGate, acts_by_layer_pos: dict, k2: np.ndarray,
           manifest: RunManifest, n_perm: int = 1000) -> dict:
    """
    Read the FROZEN direction across every (layer, position) and report the full
    map, with a permutation null over positions.

    Reporting only the argmax of a large sweep is selection on noise, so the
    full map is what goes in the paper and the maximum is calibrated against the
    null. Layer.Order-Inversion-style distributed computation would still show
    up here, because nothing assumes a single bridge-resolution site.
    """
    k2 = np.asarray(k2)
    grid, rng = {}, np.random.default_rng(manifest.seed)
    for key, acts in acts_by_layer_pos.items():
        gate.assert_not_fit_on(acts, "composed-query activations")
        grid[str(key)] = float(M.auroc(k2, gate.score(acts)))
    if not grid:
        raise ValueError("no (layer, position) activations supplied")

    best_key = max(grid, key=grid.get)
    best = grid[best_key]
    null_max = []
    for _ in range(n_perm):
        perm = rng.permutation(k2)
        null_max.append(max(M.auroc(perm, gate.score(a))
                            for a in acts_by_layer_pos.values()))
    null_max = np.asarray(null_max)
    out = {
        "experiment": "E3",
        "auroc_grid": grid,
        "peak": {"key": best_key, "auroc": best},
        "permutation_null": {
            "mean_max": float(null_max.mean()),
            "q95_max": float(np.quantile(null_max, 0.95)),
            "p_value": float((null_max >= best).mean()),
        },
        "reading": "a peak above the null's 95th percentile means the state is present "
                   "at a position the refusal mechanism does not consume",
    }
    manifest.record("E3", out)
    return out


# =====================================================================
# E4: repair
# =====================================================================

def run_E4(y: np.ndarray, baseline_scores: dict[str, np.ndarray],
           gate_surface: np.ndarray, gate_bridge: np.ndarray,
           manifest: RunManifest, coverage: float = 0.5) -> dict:
    """Compare BGR against baselines within inference-cost tiers."""
    y = np.asarray(y)
    scores = dict(baseline_scores)
    scores["BGR-min"] = M.bgr_min(gate_surface, gate_bridge)
    scores["BGR-lr"] = M.bgr_lr(gate_surface, gate_bridge, y, seed=manifest.seed)

    rows = {}
    for name, s in scores.items():
        cov, risk = M.risk_coverage_curve(y, s)
        idx = int(np.argmin(np.abs(cov - coverage)))
        rows[name] = {"auroc": M.auroc(y, s), "aurc": M.aurc(y, s),
                      "risk_at_coverage": float(risk[idx])}

    read = gate_surface.reshape(-1, 1)
    ref = np.column_stack([gate_surface, gate_bridge])
    # recovery_fraction() DELIBERATELY raises when Delta_blind is within
    # sampling noise -- rho is a share OF the blind term and is undefined
    # when there is no measurable blind term to be a share of (see that
    # function's own docstring/error message). This is a real, expected
    # outcome on a small or low-blind-term sample (e.g. E1's own n=158
    # filtered set had blind=0.025 at coverage 0.5, well within its own
    # computed noise_tolerance=0.169) -- not a bug to work around. The
    # function's own error message prescribes the correct response:
    # "Report Delta_blind with its interval and state that rho is
    # undefined" -- which requires catching it here, since letting it
    # propagate would silently lose the ENTIRE E4 report (the baseline
    # comparison table in `rows`, already computed successfully above)
    # to one metric that has a well-defined "undefined" state of its own.
    #
    # Computed for EVERY selector, not only BGR-min: rho itself is defined
    # for any candidate repair score (recovery_fraction() takes score_repair
    # as a plain array, with no dependence on how it was produced), and the
    # paper reports rho as a column for every row of Table tab:bgr/
    # tab:bgr-full, not only the two BGR variants. read/ref (and therefore
    # blind, r_read_star, r_ref_star) are identical across selectors since
    # they depend only on y and the gate features, not on score_repair --
    # only r_repair (and hence rho) varies by selector.
    rho_reports = {}
    for name, s in scores.items():
        if name in ("surface_gate", "final_token_probe_PLACEHOLDER"):
            continue  # rho is relative to the surface gate; it is not its own repair
        try:
            rho_reports[name] = M.recovery_fraction(y, s, read, ref,
                                                    coverage=coverage, seed=manifest.seed)
        except ValueError as e:
            # recovery_fraction() raises several DIFFERENT ValueErrors (bad
            # input shape, coverage out of range, features_ref not extending
            # features_read, ...) -- only the noise-guardrail one is expected
            # and meant to be caught here; every other one is a real bug and
            # must keep propagating loudly rather than being silently
            # relabeled as "rho undefined". Matched on the guardrail's own,
            # specific wording rather than catching every ValueError blindly.
            if "is within sampling noise" not in str(e):
                raise
            rho_reports[name] = {"rho": None, "undefined_reason": str(e)}
    out = {
        "experiment": "E4", "coverage": coverage, "selectors": rows,
        "rho_BGR_min": rho_reports.get("BGR-min"),
        "rho": rho_reports,
        "vs_surface_gate": {
            name: M.paired_bootstrap_auc_diff(y, s, gate_surface, seed=manifest.seed)
            for name, s in scores.items() if name.startswith("BGR")
        },
    }
    manifest.record("E4", out)
    return out


# =====================================================================
# E5: compounding
# =====================================================================

def run_E5(labels_k2: list[dict], labels_k3: list[dict],
           gate_k2: np.ndarray, gate_k3: np.ndarray, manifest: RunManifest) -> dict:
    def hops(labels, k):
        return [np.array([l[f"K{j}"] for l in labels]) for j in range(2, k + 1)]

    f2 = M.khop_floor(hops(labels_k2, 2), gate_k2,
                      y=np.array([l["Y"] for l in labels_k2]), seed=manifest.seed)
    f3 = M.khop_floor(hops(labels_k3, 3), gate_k3,
                      y=np.array([l["Y"] for l in labels_k3]), seed=manifest.seed)
    out = {
        "experiment": "E5", "k2": f2, "k3": f3,
        "leading_term_grows": bool(f3["p_M_zero"] >= f2["p_M_zero"]),
        "floor_grows": bool(f3["floor"] > f2["floor"]),
        "slack_growth": float((f3["delta_k"] + f3["v_k"]) - (f2["delta_k"] + f2["v_k"])),
        "reading": "the theorem bounds the leading term only. If the floor falls because "
                   "slack grew faster, that is a measurement, not a contradiction; report it.",
    }
    manifest.record("E5", out)
    return out


# =====================================================================
# E6: chain of thought
# =====================================================================

def run_E6(k2_direct: np.ndarray, gate_direct: np.ndarray,
           k2_cot: np.ndarray, gate_cot_selfwritten: np.ndarray,
           bridge_written_correct: np.ndarray, manifest: RunManifest) -> dict:
    """
    Does writing the bridge into the model's own chain of thought restore gating?

    Split by whether the SELF-WRITTEN bridge is correct. Gating on a
    confidently-wrong bridge is a different phenomenon and pooling the two would
    hide it. This experiment competes directly with Cheng et al. (2025), who
    report that chain of thought obscures hallucination cues.
    """
    ok = np.asarray(bridge_written_correct).astype(bool)
    if ok.sum() < 2 or (~ok).sum() < 2:
        raise ValueError(
            "too few items in one arm of the correct/incorrect self-written bridge split; "
            "do not pool them, collect more data"
        )
    out = {
        "experiment": "E6",
        "auroc_direct": M.auroc(k2_direct, gate_direct),
        "auroc_cot_bridge_correct": M.auroc(k2_cot[ok], gate_cot_selfwritten[ok]),
        "auroc_cot_bridge_wrong": M.auroc(k2_cot[~ok], gate_cot_selfwritten[~ok]),
        "n_correct": int(ok.sum()), "n_wrong": int((~ok).sum()),
        "reading": "gating restored only where the written bridge is correct supports our "
                   "account; no restoration anywhere supports Cheng et al. (2025)",
    }
    manifest.record("E6", out)
    return out


# =====================================================================
# Orchestration
# =====================================================================

def run_all(inputs: dict, outdir: str, snapshot_id: str, model_name: str,
            seed: int = 0) -> RunManifest:
    """
    Drive E0 through E6 in order, enforcing the E0 stop gate.

    `inputs` supplies the arrays each stage needs; see skc_pipeline_test.py for
    a complete worked example against a mock model.
    """
    man = RunManifest(outdir=outdir, snapshot_id=snapshot_id,
                      model_name=model_name, seed=seed)
    e0 = run_E0(inputs["gates"], inputs["acts_heldout"], inputs["known_heldout"],
                inputs["acts_named_bridge"], inputs["k2_named"], man)
    require_E0(e0)

    gate = inputs["gates"][inputs.get("primary_estimator", "diffmeans")]
    labels = filter_bridge_resolved(inputs["labels"], inputs.get("bridge_resolved"))
    if not labels:
        raise ValueError(
            "no queries survived the K1=1 and bridge-resolved filter. Without it you "
            "would be measuring first-hop failure and reporting it as composition."
        )
    keep = np.array([i for i, l in enumerate(inputs["labels"]) if l in labels])
    man.record("filter", {"n_before": len(inputs["labels"]), "n_after": len(labels),
                          "retained_fraction": len(labels) / len(inputs["labels"])})

    run_E1(gate, inputs["acts_surface"][keep], inputs["acts_bridge"][keep], labels,
           inputs["acts_singlehop_correct"], man,
           acts_surface_unfiltered=inputs["acts_surface"],
           k1_unfiltered=np.array([l["K1"] for l in inputs["labels"]]))
    if "E2" in inputs:
        run_E2(**inputs["E2"], manifest=man)
    if "acts_by_layer_pos" in inputs:
        run_E3(gate, inputs["acts_by_layer_pos"],
               np.array([l["M"] for l in labels]), man)
    if "baseline_scores" in inputs:
        run_E4(np.array([l["Y"] for l in labels]), inputs["baseline_scores"],
               gate.score(inputs["acts_surface"][keep]),
               gate.score(inputs["acts_bridge"][keep]), man)
    if "E5" in inputs:
        run_E5(**inputs["E5"], manifest=man)
    if "E6" in inputs:
        run_E6(**inputs["E6"], manifest=man)
    man.write()
    return man


if __name__ == "__main__":
    print(__doc__)
    print("This module is a library. Run skc_pipeline_test.py to exercise it offline,")
    print("and see skc_runbook.txt for the experimental protocol.")

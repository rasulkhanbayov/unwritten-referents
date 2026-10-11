#!/usr/bin/env python3
"""
skc_gates.py
Gate extraction and read-out for
"Tokens, Not Referents: Latent Entities and the Limits of Knowledge-Aware Abstention".

THE ONE RULE THIS FILE ENFORCES
  Gate directions are fit on SINGLE-HOP data and then frozen. A direction refit
  on composed data would show only that some linear readout exists. A frozen
  direction that transfers to a latent position shows the latent state is
  expressed in the format the refusal mechanism already reads. That distinction
  is the whole argument of Section 6.4, and it is easy to lose by accident, so
  FrozenGate refuses to be refit and records a hash of its fitting data.

WHAT IS AND IS NOT TESTED
  Tested offline by skc_pipeline_test.py: direction fitting for all three
  estimators, the freeze guarantee, scoring, agreement checking, and
  serialisation round-trips. All of this runs on synthetic activations.
  NOT tested: activation capture from a real transformer, because there is no
  GPU here. The hook code is written and commented; verify it on your model with
  the shape assertions left in place.

Dependencies: numpy. torch and transformers only for capture_activations().
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np


# =====================================================================
# Frozen gate
# =====================================================================

@dataclass
class FrozenGate:
    """
    A direction plus everything needed to audit where it came from.

    `fit_hash` is a hash of the activations it was fit on. skc_experiments.py
    checks it against the composed-data hash and aborts if they match, which is
    the mechanical guard against accidentally refitting on the evaluation set.
    """
    estimator: str            # "sae" | "diffmeans" | "probe"
    direction: np.ndarray     # (d_model,)
    bias: float
    layer: int
    model_name: str
    fit_hash: str
    n_fit: int
    sae_latent_id: int | None = None
    notes: str = ""

    def __post_init__(self):
        self.direction = np.asarray(self.direction, dtype=np.float64)
        if self.direction.ndim != 1:
            raise ValueError(f"direction must be 1-D, got {self.direction.shape}")
        n = np.linalg.norm(self.direction)
        if not np.isfinite(n) or n == 0:
            raise ValueError("direction is zero or non-finite")
        self.direction = self.direction / n     # unit norm, so scores compare across estimators

    def score(self, acts: np.ndarray) -> np.ndarray:
        """Project activations onto the frozen direction. Never modifies state."""
        a = np.asarray(acts, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2:
            raise ValueError(f"acts must be 1-D or 2-D, got {a.shape}")
        if a.shape[1] != self.direction.shape[0]:
            raise ValueError(
                f"activation dim {a.shape[1]} does not match direction dim "
                f"{self.direction.shape[0]}. Wrong layer or wrong model."
            )
        if not np.all(np.isfinite(a)):
            raise ValueError("activations contain NaN or inf")
        return a @ self.direction + self.bias

    def assert_not_fit_on(self, acts: np.ndarray, what: str = "evaluation data") -> None:
        """Abort if this gate was fit on the data now being scored."""
        if hash_activations(acts) == self.fit_hash:
            raise RuntimeError(
                f"This gate was fit on the same {what} you are now scoring. The frozen-"
                "direction argument in Section 6.4 does not survive that. Refit on the "
                "single-hop split, or score a different split."
            )

    def save(self, path: str) -> None:
        d = asdict(self)
        d["direction"] = self.direction.tolist()
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

    @staticmethod
    def load(path: str) -> "FrozenGate":
        with open(path) as f:
            d = json.load(f)
        d["direction"] = np.asarray(d["direction"])
        return FrozenGate(**d)


def hash_activations(acts: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(acts, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# =====================================================================
# Estimator 1: difference of means
# =====================================================================

def fit_diffmeans(acts_known: np.ndarray, acts_unknown: np.ndarray, *,
                  layer: int, model_name: str) -> FrozenGate:
    """
    Normalised difference of class means. No hyperparameters, so nothing to tune away.

    Standardises each dimension by its POOLED standard deviation before
    taking the class-mean difference, then folds that scaling back into the
    returned direction/bias exactly as fit_probe() does -- so FrozenGate.score()
    is still a plain projection, and this remains a fixed, hyperparameter-free
    preprocessing step, not a new knob.

    This is necessary, not cosmetic: a small number of real transformer
    activations carry "massive activation" outliers on certain residual-
    stream dimensions (documented in the literature, and confirmed directly
    on this bundle's own captured Gemma-2-2b activations at layers 8 and
    12 -- 7-14% of points with norm roughly an order of magnitude above
    typical). An unstandardised mean is not robust to this: those few
    high-magnitude points can dominate mu_k/mu_u on the very dimensions
    where the outlier lives, producing a direction that is mostly "does
    this point have the outlier" rather than "does the model know this
    entity." This was found empirically: cross-estimator agreement between
    this (previously unstandardised) function and fit_probe (which already
    standardises) swung unpredictably between roughly -0.20 and +0.47
    across seven real E0 runs at two different layers and five sample
    sizes, with no stable relationship to sample size or layer, before this
    fix -- see EXECUTION_PLAN.md for the full diagnostic trail.
    """
    ak = _check_2d(acts_known, "acts_known")
    au = _check_2d(acts_unknown, "acts_unknown")
    if ak.shape[1] != au.shape[1]:
        raise ValueError(f"dim mismatch: {ak.shape[1]} vs {au.shape[1]}")
    both = np.vstack([ak, au])
    sd = both.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    ak_z, au_z = ak / sd, au / sd
    mu_k, mu_u = ak_z.mean(0), au_z.mean(0)
    direction_z = mu_k - mu_u
    if np.linalg.norm(direction_z) < 1e-12:
        raise ValueError("class means coincide; there is no direction to find")
    direction = direction_z / sd            # undo standardisation, same pattern as fit_probe
    both_z = both / sd
    bias = -float(np.dot(direction_z / np.linalg.norm(direction_z), both_z.mean(0)))
    return FrozenGate("diffmeans", direction, bias, layer, model_name,
                      hash_activations(both), both.shape[0])


# =====================================================================
# Estimator 2: L2 logistic probe
# =====================================================================

def fit_probe(acts_known: np.ndarray, acts_unknown: np.ndarray, *,
              layer: int, model_name: str, l2: float = 1.0,
              iters: int = 800, lr: float = 0.5, seed: int = 0) -> FrozenGate:
    """
    L2 logistic regression by full-batch gradient descent, numpy only.

    Standardisation statistics are folded into the returned direction and bias
    so that FrozenGate.score() is a plain projection. That keeps the three
    estimators interchangeable at scoring time.
    """
    ak = _check_2d(acts_known, "acts_known")
    au = _check_2d(acts_unknown, "acts_unknown")
    X = np.vstack([ak, au])
    y = np.concatenate([np.ones(len(ak)), np.zeros(len(au))])
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=1e-3, size=Z.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        g = Z.T @ (p - y) / len(y) + l2 * w / len(y)
        w -= lr * g
        b -= lr * float((p - y).mean())
    direction = w / sd                      # undo standardisation
    bias = b - float(np.dot(w, mu / sd))
    return FrozenGate("probe", direction, bias, layer, model_name,
                      hash_activations(X), X.shape[0],
                      notes=f"l2={l2}, iters={iters}, lr={lr}, seed={seed}")


# =====================================================================
# Estimator 3: SAE latent selection (Ferrando et al. protocol)
# =====================================================================

def separation_scores(sae_known: np.ndarray, sae_unknown: np.ndarray) -> np.ndarray:
    """Per-latent difference in the fraction of positive activations."""
    k = _check_2d(sae_known, "sae_known")
    u = _check_2d(sae_unknown, "sae_unknown")
    if k.shape[1] != u.shape[1]:
        raise ValueError("latent dim mismatch")
    return (k > 0).mean(0) - (u > 0).mean(0)


def select_sae_latent(sae_by_type: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[int, float]:
    """
    Pick the latent maximising the MINIMUM separation across entity types.

    Max-min, not max-mean. A latent that separates superbly on one entity type
    and not at all on another is a type detector, not a knowledge gate, and
    max-mean would select it. Returns (latent_id, worst-case separation).
    """
    if not sae_by_type:
        raise ValueError("no entity types supplied")
    per_type = []
    for etype, (k, u) in sae_by_type.items():
        per_type.append(separation_scores(k, u))
    S = np.vstack(per_type)                 # (n_types, n_latents)
    worst = S.min(axis=0)
    j = int(np.argmax(worst))
    return j, float(worst[j])


def fit_sae_gate(sae_by_type: dict[str, tuple[np.ndarray, np.ndarray]],
                 decoder: np.ndarray, *, layer: int, model_name: str) -> FrozenGate:
    """
    Select a latent by max-min separation and take its decoder column as the
    direction, which is what makes it usable for both reading and steering.
    """
    j, worst = select_sae_latent(sae_by_type)
    W = np.asarray(decoder, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"decoder must be 2-D (n_latents, d_model), got {W.shape}")
    if j >= W.shape[0]:
        raise ValueError(f"latent {j} out of range for decoder with {W.shape[0]} rows")
    allacts = np.vstack([np.vstack(v) for v in sae_by_type.values()])
    return FrozenGate("sae", W[j], 0.0, layer, model_name,
                      hash_activations(allacts), allacts.shape[0],
                      sae_latent_id=j, notes=f"max-min separation {worst:.4f}")


# =====================================================================
# Agreement precondition
# =====================================================================

def gate_agreement(gates: dict[str, FrozenGate], acts: np.ndarray) -> dict:
    """
    Rank correlation between the three estimators' scores on the same
    activations. The paper makes agreement a PRECONDITION for reporting
    anything downstream, so this returns the minimum pairwise correlation and
    the caller decides. Disagreement means the estimators are not measuring the
    same thing and a downstream null would be uninterpretable.
    """
    if len(gates) < 2:
        raise ValueError("need at least two gates to check agreement")
    scores = {name: g.score(acts) for name, g in gates.items()}
    names = sorted(scores)
    out, pairs = {}, []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = _rankdata(scores[names[i]]), _rankdata(scores[names[j]])
            r = float(np.corrcoef(a, b)[0, 1])
            out[f"{names[i]}~{names[j]}"] = r
            pairs.append(r)
    out["min_pairwise_spearman"] = float(min(pairs))
    return out


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order.astype(float)


def _check_2d(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (examples x dim), got {a.shape}")
    if a.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    return a


# =====================================================================
# Activation capture (NOT covered by the offline test suite)
# =====================================================================

def capture_activations(model, tokenizer, prompts, layer: int,
                        position: str = "last_entity_token",
                        entity_spans=None, batch_size: int = 16,
                        device: str = "cuda"):
    """
    Residual-stream activations at a chosen position.

    UNTESTED HERE: needs a GPU and a real model. The shape assertions are
    deliberately left in; do not remove them on first run, because a silent
    position error is the single most likely way to produce a confident null.

    position:
      "last_entity_token" -- final token of the entity mention. Requires
                             entity_spans as [(start, end)] token indices per
                             prompt. This is where Ferrando et al. read the gate.
      "last"              -- final prompt token.
      int                 -- explicit index.

    FORCES RIGHT-PADDING internally (saves the tokenizer's prior
    padding_side, sets "right" for the duration of this call, restores it
    in a finally block) -- this function's own index arithmetic
    (`entity_spans[i+b][1]-1`, `attention_mask[b].sum()-1`) is only valid
    under right-padding, where real content starts at index 0 and any
    padding is appended after it. This was a REAL, CONFIRMED bug, not a
    theoretical one: this function used to trust every caller to have
    already set padding_side="right" before calling it, and nothing did
    so explicitly -- callers like skc_harness.capture_for_queries() never
    called set_padding_side() at all, silently inheriting whatever side a
    PRIOR call (e.g. skc_harness.generate_greedy(), which forces "left"
    internally then restores the tokenizer's own prior/default side
    afterward -- itself "left" if nothing upstream ever set it) happened
    to leave the tokenizer in. Confirmed directly: three pairs of
    genuinely different queries in a real E1 run produced BYTE-IDENTICAL
    captured activation vectors (same batch, alternating positions) --
    traced to exactly this, reproduced with a diagnostic
    (skc_diag_activation_collision.py) that showed zero collisions when
    padding_side was forced to "right" and the collision reproducing when
    left unforced. See EXECUTION_PLAN.md for the full investigation. This
    fix closes the bug at its actual source (this function, the one place
    that actually does the position arithmetic) rather than requiring
    every caller to remember to set the side correctly beforehand, which
    is exactly the discipline that already failed once.
    """
    import torch

    if position == "last_entity_token" and entity_spans is None:
        raise ValueError(
            "entity_spans is required for last_entity_token. Compute spans with the "
            "tokenizer on the exact prompt string; do not estimate them by counting words."
        )

    acts, hidden = [], {}

    def hook(_m, _i, out):
        hidden["h"] = out[0] if isinstance(out, tuple) else out

    prior_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                model(**enc)
            h = hidden["h"]
            assert h.ndim == 3, f"expected (batch, seq, d_model), got {tuple(h.shape)}"
            for b in range(h.shape[0]):
                if position == "last":
                    idx = int(enc["attention_mask"][b].sum()) - 1
                elif position == "last_entity_token":
                    idx = entity_spans[i + b][1] - 1
                else:
                    idx = int(position)
                if not (0 <= idx < h.shape[1]):
                    raise IndexError(
                        f"position {idx} outside sequence of length {h.shape[1]} for "
                        f"prompt {i+b}. Check tokenizer padding side and the span offsets."
                    )
                acts.append(h[b, idx].float().cpu().numpy())
    finally:
        handle.remove()
        tokenizer.padding_side = prior_padding_side
    return np.stack(acts)


def capture_activations_all_layers(model, tokenizer, prompts, layers: list[int],
                                   position: str = "last_entity_token",
                                   entity_spans=None, batch_size: int = 16,
                                   device: str = "cuda") -> dict:
    """
    Residual-stream activations at a chosen position, at MULTIPLE layers,
    in ONE forward pass per batch -- built for the per-layer E0 refit
    sweep (skc_run_e0_layersweep.py), where fitting fresh gates at all 26
    of Gemma-2-2b's layers would otherwise mean 26x the model-load and
    forward-pass cost of the single-layer capture_activations() above.
    The model computes every layer's residual stream on every forward
    pass regardless of how many of them a caller reads -- this function
    just hooks all of them at once instead of one at a time across 26
    separate passes, which is a real efficiency reuse of already-computed
    activations, not an approximation of the single-layer function's own
    result (registering N hooks in one pass and reading each layer's own
    hook output is exactly equivalent to N separate single-hook passes on
    the same input, since forward hooks do not affect the forward
    computation itself).

    Returns {layer: np.ndarray of shape (n_prompts, d_model)}, one entry
    per requested layer, each computed the SAME way
    capture_activations() computes its own single result (same position
    semantics, same right-padding requirement, same index arithmetic) --
    deliberately mirrors that function's own logic closely rather than
    diverging, to avoid the exact "two independent copies of the same
    logic diverging" bug class this project has hit repeatedly; the
    position-index computation below is copied from capture_activations()
    verbatim, not reimplemented differently.
    """
    import torch

    if position == "last_entity_token" and entity_spans is None:
        raise ValueError(
            "entity_spans is required for last_entity_token. Compute spans with the "
            "tokenizer on the exact prompt string; do not estimate them by counting words."
        )

    acts_by_layer = {layer: [] for layer in layers}
    hidden = {}

    def make_hook(layer):
        def hook(_m, _i, out):
            hidden[layer] = out[0] if isinstance(out, tuple) else out
        return hook

    prior_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    handles = [model.model.layers[layer].register_forward_hook(make_hook(layer))
              for layer in layers]
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                model(**enc)
            for layer in layers:
                h = hidden[layer]
                assert h.ndim == 3, f"expected (batch, seq, d_model), got {tuple(h.shape)}"
                for b in range(h.shape[0]):
                    if position == "last":
                        idx = int(enc["attention_mask"][b].sum()) - 1
                    elif position == "last_entity_token":
                        idx = entity_spans[i + b][1] - 1
                    else:
                        idx = int(position)
                    if not (0 <= idx < h.shape[1]):
                        raise IndexError(
                            f"position {idx} outside sequence of length {h.shape[1]} for "
                            f"prompt {i+b}, layer {layer}. Check tokenizer padding side "
                            "and the span offsets."
                        )
                    acts_by_layer[layer].append(h[b, idx].float().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
        tokenizer.padding_side = prior_padding_side
    return {layer: np.stack(acts_by_layer[layer]) for layer in layers}

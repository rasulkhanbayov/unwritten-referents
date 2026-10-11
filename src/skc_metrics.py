#!/usr/bin/env python3
"""
skc_metrics.py
Reference implementation of every quantity defined in
"Self-Knowledge Does Not Compose: Latent Entities Are a Blind Spot in
Language Model Knowledge Awareness".

DESIGN RULES (do not relax these)
  1. Nothing here fabricates a result. Every function computes from data
     you pass in. There is no default dataset and no synthetic fallback.
  2. Missing or malformed input raises. It never silently substitutes.
  3. The synthetic generator is reachable only through an explicit flag
     and writes files prefixed "SYNTHETIC_". Synthetic numbers must never
     appear in the paper.
  4. The smoke test verifies the PROPOSITIONS, not a hard-coded number.
     It checks the direction of each inequality on data whose ground
     truth is known by construction.

Dependencies: numpy only.

Usage
  python3 skc_metrics.py --smoke-test        # verify theory against code
  python3 skc_metrics.py --self-check        # internal consistency checks
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

__all__ = [
    "auroc", "auroc_ci", "delong_test", "paired_bootstrap_auc_diff",
    "separation_score", "risk_coverage_curve", "aurc",
    "estimate_kappa", "estimate_delta", "estimate_shortcut_v",
    "floor_from_proposition2", "kappa_gain_bound",
    "check_floor", "floor_report", "khop_floor", "recovery_fraction",
    "optimal_selective_risk", "blind_spot_decomposition",
    "corollary_blind_lower_bound",
    "kappa_gain_bound",
    "composition_gap", "bgr_min", "bgr_lr",
]


# =====================================================================
# Validation
# =====================================================================

def _check_binary(y: np.ndarray, name: str) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {y.shape}")
    if y.size == 0:
        raise ValueError(f"{name} is empty; no silent fallback is provided")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError(f"{name} must contain only 0/1, got {np.unique(y)[:8]}")
    return y.astype(np.int64)


def _check_scores(s: np.ndarray, n: int, name: str) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {s.shape}")
    if s.size != n:
        raise ValueError(f"{name} has length {s.size}, expected {n}")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{name} contains NaN or inf; fix upstream, do not impute")
    return s


# =====================================================================
# AUROC and significance
# =====================================================================

def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged. Needed for the DeLong covariance."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=np.float64)
    out[order] = r
    return out


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC. Ties contribute 0.5."""
    y = _check_binary(y, "y")
    s = _check_scores(s, y.size, "s")
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUROC undefined: need both classes present")
    r = _midrank(s)
    return float((r[y == 1].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def auroc_ci(y, s, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap over examples. Returns (point, lo, hi)."""
    y = _check_binary(y, "y")
    s = _check_scores(s, y.size, "s")
    rng = np.random.default_rng(seed)
    point = auroc(y, s)
    n, vals = y.size, []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(auroc(yb, s[idx]))
    if len(vals) < max(50, n_boot // 20):
        raise RuntimeError(
            f"Only {len(vals)} usable bootstrap resamples. The class balance is too "
            "extreme for a stable interval. Report this rather than widening silently."
        )
    v = np.asarray(vals)
    return point, float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def _delong_structural(y: np.ndarray, scores: np.ndarray):
    """V10/V01 structural components for k correlated scores on shared labels."""
    pos_mask = y == 1
    m, n = int(pos_mask.sum()), int((~pos_mask).sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong undefined: need both classes present")
    k = scores.shape[0]
    v10 = np.empty((k, m))
    v01 = np.empty((k, n))
    aucs = np.empty(k)
    for i in range(k):
        x, yy = scores[i][pos_mask], scores[i][~pos_mask]
        tx, ty, tz = _midrank(x), _midrank(yy), _midrank(np.concatenate([x, yy]))
        v10[i] = (tz[:m] - tx) / n
        v01[i] = 1.0 - (tz[m:] - ty) / m
        aucs[i] = (tz[:m].sum() / (m * n)) - (m + 1.0) / (2.0 * n)
    s = np.cov(v10) / m + np.cov(v01) / n
    return aucs, np.atleast_2d(s)


def delong_test(y, s1, s2) -> dict:
    """Correlated-ROC test (DeLong et al., 1988) via the fast midrank algorithm."""
    y = _check_binary(y, "y")
    s1 = _check_scores(s1, y.size, "s1")
    s2 = _check_scores(s2, y.size, "s2")
    aucs, s = _delong_structural(y, np.vstack([s1, s2]))
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": diff,
                "z": float("nan"), "p_value": float("nan"),
                "note": "degenerate variance; use the paired bootstrap instead"}
    z = diff / np.sqrt(var)
    # Two-sided normal tail via erfc, avoiding a scipy dependency.
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2.0))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": diff,
            "z": float(z), "p_value": float(p)}


def paired_bootstrap_auc_diff(y, s1, s2, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Paired bootstrap on AUC(s1) - AUC(s2). Resamples examples, not scores."""
    y = _check_binary(y, "y")
    s1 = _check_scores(s1, y.size, "s1")
    s2 = _check_scores(s2, y.size, "s2")
    rng = np.random.default_rng(seed)
    point, n, vals = auroc(y, s1) - auroc(y, s2), y.size, []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(auroc(yb, s1[idx]) - auroc(yb, s2[idx]))
    v = np.asarray(vals)
    return {"diff": float(point),
            "lo": float(np.quantile(v, alpha / 2)),
            "hi": float(np.quantile(v, 1 - alpha / 2)),
            "p_two_sided": float(2 * min((v <= 0).mean(), (v >= 0).mean()))}


# =====================================================================
# Gate separation (Ferrando et al., 2025 protocol)
# =====================================================================

def separation_score(act_known: np.ndarray, act_unknown: np.ndarray) -> np.ndarray:
    """Per-latent difference in the fraction of positive activations."""
    a, b = np.asarray(act_known, float), np.asarray(act_unknown, float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("activations must be 2-D (examples x latents)")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"latent dim mismatch: {a.shape[1]} vs {b.shape[1]}")
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("both activation sets must be non-empty")
    return (a > 0).mean(axis=0) - (b > 0).mean(axis=0)


# =====================================================================
# Selective prediction
# =====================================================================

def risk_coverage_curve(y: np.ndarray, s: np.ndarray):
    """
    Coverage and selective risk over all thresholds.
    Answer when s >= tau. Higher s means more confident.
    Returns (coverage, risk) sorted by increasing coverage.
    """
    y = _check_binary(y, "y")
    s = _check_scores(s, y.size, "s")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    n = y.size
    k = np.arange(1, n + 1)
    correct = np.cumsum(y_sorted)
    return k / n, 1.0 - correct / k


def aurc(y, s) -> float:
    """
    Area under the risk-coverage curve. Lower is better.

    Uses np.trapz, not np.trapezoid: sae-lens (needed for the SAE gate
    estimator) pulls in automated-interpretability, which hard-pins
    numpy<2.0.0, so this bundle's environment runs on numpy 1.x. trapezoid
    is numpy>=2.0 only; trapz is removed in numpy>=2.0 but present on 1.x.
    There is no single spelling that works on both -- pick the one that
    matches the environment this bundle actually runs in (numpy 1.26.4 per
    requirements.txt), not the newer name, until sae-lens's own dependency
    tree drops the numpy<2.0 pin.
    """
    cov, risk = risk_coverage_curve(y, s)
    return float(np.trapz(risk, cov) / (cov[-1] - cov[0])) if cov.size > 1 else float(risk[0])


# =====================================================================
# The paper's quantities
# =====================================================================

def _cross_fitted_eta(features: np.ndarray, y: np.ndarray, n_folds: int, seed: int,
                      fitter: Callable | None = None) -> np.ndarray:
    """Out-of-fold estimate of P(Y=1 | features). Default fitter: logistic regression."""
    n = y.size
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % n_folds
    eta = np.empty(n)
    fit = fitter if fitter is not None else _fit_logistic
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if y[tr].min() == y[tr].max():
            raise ValueError(
                f"fold {f} training split is single-class; use fewer folds or more data"
            )
        eta[te] = fit(features[tr], y[tr], features[te])
    return eta


def _fit_logistic(X_tr, y_tr, X_te, l2: float = 1.0, iters: int = 300, lr: float = 0.1):
    """Small L2 logistic regression by gradient descent. NumPy only."""
    X_tr = np.atleast_2d(X_tr)
    X_te = np.atleast_2d(X_te)
    if X_tr.shape[0] != y_tr.size:
        raise ValueError("feature/label length mismatch")
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
    Z_tr = np.hstack([Z_tr, np.ones((Z_tr.shape[0], 1))])
    Z_te = np.hstack([Z_te, np.ones((Z_te.shape[0], 1))])
    w = np.zeros(Z_tr.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z_tr @ w, -30, 30)))
        g = Z_tr.T @ (p - y_tr) / y_tr.size + l2 * np.r_[w[:-1], 0.0] / y_tr.size
        w -= lr * g
    return 1.0 / (1.0 + np.exp(-np.clip(Z_te @ w, -30, 30)))


def estimate_kappa(features: np.ndarray, y: np.ndarray, alpha: float = 0.05,
                   n_folds: int = 5, seed: int = 0) -> float:
    """
    Conservative plug-in for kappa = ess sup P(Y=1 | features).

    The essential supremum is not consistently estimable at finite n without
    smoothness assumptions, so we report the (1-alpha) quantile of a
    cross-fitted eta. This UNDER-estimates kappa, which RAISES the computed
    floor 1 - kappa. A higher floor is a stronger claim, so this direction
    FAVORS the paper's hypothesis and is anti-conservative. Do not treat the
    floor as a point estimate: report sensitivity across alpha and test the
    bound with check_floor(), which uses a one-sided confidence criterion.
    """
    y = _check_binary(y, "y")
    X = np.atleast_2d(np.asarray(features, float))
    if X.shape[0] != y.size:
        X = X.T
    if X.shape[0] != y.size:
        raise ValueError(f"features have {X.shape[0]} rows, y has {y.size}")
    eta = _cross_fitted_eta(X, y, n_folds, seed)
    return float(np.quantile(eta, 1.0 - alpha))


def estimate_delta(features: np.ndarray, k2: np.ndarray, alpha: float = 0.05,
                   n_folds: int = 5, seed: int = 0) -> float:
    """
    Gate leakage: how much the surface gate reveals about second-hop knowledge.
    delta = ess sup | P(K2=1 | S) - P(K2=1) |, same quantile plug-in.
    delta near 0 supports H1; large delta refutes it.
    """
    k2 = _check_binary(k2, "k2")
    X = np.atleast_2d(np.asarray(features, float))
    if X.shape[0] != k2.size:
        X = X.T
    if X.shape[0] != k2.size:
        raise ValueError("feature/label length mismatch")
    eta = _cross_fitted_eta(X, k2, n_folds, seed)
    return float(np.quantile(np.abs(eta - k2.mean()), 1.0 - alpha))


def estimate_shortcut_v(y: np.ndarray, k2: np.ndarray, features=None,
                        alpha: float = 0.05, n_folds: int = 5, seed: int = 0,
                        allow_marginal: bool = False) -> float:
    """
    Shortcut rate v = ess sup_s P(Y=1, K2=0 | S=s), Assumption 2 of the paper.

    WARNING, verified numerically by the smoke test. The MARGINAL rate
    P(Y=1, K2=0) is a LOWER bound on the essential supremum whenever the gate
    carries information about K1, because the shortcut concentrates where K1=1
    and the gate fires. Substituting the marginal inflates the Proposition 2
    floor and makes it ANTI-CONSERVATIVE, which can produce an apparent
    violation of a theorem that in fact holds. Always pass `features`.

    Set allow_marginal=True only for a descriptive summary statistic, never for
    the floor.
    """
    y, k2 = _check_binary(y, "y"), _check_binary(k2, "k2")
    if y.size != k2.size:
        raise ValueError("y and k2 length mismatch")
    z = ((y == 1) & (k2 == 0)).astype(np.int64)
    if features is None:
        if not allow_marginal:
            raise ValueError(
                "estimate_shortcut_v requires `features` (the surface gate). The marginal "
                "rate under-estimates the essential supremum and makes the Proposition 2 "
                "floor anti-conservative. Pass features=gate_surface, or set "
                "allow_marginal=True if you only want the descriptive marginal."
            )
        return float(z.mean())
    X = np.atleast_2d(np.asarray(features, float))
    if X.shape[0] != y.size:
        X = X.T
    if X.shape[0] != y.size:
        raise ValueError("feature/label length mismatch")
    if z.min() == z.max():
        return float(z.mean())
    return float(np.quantile(_cross_fitted_eta(X, z, n_folds, seed), 1.0 - alpha))


def floor_from_proposition2(k2: np.ndarray, delta: float, v: float) -> float:
    """R >= P(K2=0) - delta - v. Clipped at 0, which the proposition permits."""
    k2 = _check_binary(k2, "k2")
    if delta < 0 or v < 0:
        raise ValueError("delta and v are non-negative by definition")
    return float(max(0.0, (1.0 - k2.mean()) - delta - v))


def check_floor(y, s, floor: float, min_coverage: float = 0.05,
                conf: float = 0.99) -> dict:
    """
    Test whether the observed risk-coverage curve CONTRADICTS the Proposition 2
    floor, rather than comparing noisy point estimates.

    At small coverage the empirical risk rests on few examples, and taking a
    minimum over thousands of correlated thresholds biases that minimum
    downward. A raw point comparison therefore false-alarms. We instead flag a
    violation only when the one-sided upper confidence limit of the empirical
    risk lies strictly below the floor.
    """
    y = _check_binary(y, "y")
    s = _check_scores(s, y.size, "s")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor={floor} outside [0,1]")
    cov, risk = risk_coverage_curve(y, s)
    keep = cov >= min_coverage
    if not keep.any():
        raise ValueError(f"no threshold reaches coverage {min_coverage}")
    cov, risk = cov[keep], risk[keep]
    m = np.maximum(np.round(cov * y.size).astype(int), 1)
    z = {0.95: 1.6449, 0.99: 2.3263, 0.999: 3.0902}.get(conf)
    if z is None:
        raise ValueError("conf must be one of 0.95, 0.99, 0.999")
    ucl = risk + z * np.sqrt(np.maximum(risk * (1 - risk), 1e-12) / m)
    viol = ucl < floor - 1e-12
    j = int(np.argmin(ucl))
    return {"floor": float(floor),
            "min_risk": float(risk.min()),
            "min_ucl": float(ucl[j]),
            "coverage_at_min_ucl": float(cov[j]),
            "n_at_min_ucl": int(m[j]),
            "violated": bool(viol.any()),
            "n_violating_thresholds": int(viol.sum())}


def floor_report(y, k2, gate_surface, alpha: float = 0.05,
                 min_coverage: float = 0.05, seed: int = 0) -> dict:
    """
    End-to-end Proposition 2 check. Use this rather than assembling the pieces
    by hand: it always uses the conditional shortcut estimator.
    """
    delta = estimate_delta(gate_surface, k2, alpha=alpha, seed=seed)
    v = estimate_shortcut_v(y, k2, features=gate_surface, alpha=alpha, seed=seed)
    floor = floor_from_proposition2(k2, delta, v)
    out = check_floor(y, gate_surface, floor, min_coverage=min_coverage)
    out.update({"delta": float(delta), "v": float(v),
                "p_k2_zero": float(1.0 - _check_binary(k2, "k2").mean())})
    return out


def optimal_selective_risk(eta: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """
    R*(Z, c): the minimal selective risk at coverage c over sigma(Z)-measurable
    rules, attained by thresholding eta = P(Y=1 | Z) at its (1-c) quantile
    (Chow's rule).

    Pass an OUT-OF-FOLD eta. Passing an in-sample eta makes R* optimistically
    biased and can drive the decomposition negative.
    """
    y = _check_binary(y, "y")
    eta = _check_scores(eta, y.size, "eta")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0,1], got {coverage}")
    k = max(1, int(round(coverage * y.size)))
    keep = np.argsort(-eta, kind="mergesort")[:k]
    return float(1.0 - y[keep].mean())


def blind_spot_decomposition(y, score, features_read, features_ref,
                             coverage: float = 0.5, n_folds: int = 5,
                             seed: int = 0) -> dict:
    """
    Lemma 1 (blind-spot decomposition). Splits the selective risk of `score` at a
    given coverage into
      irreducible  = R*(H_ref, c)
      blind        = R*(H_read, c) - R*(H_ref, c)     >= 0
      subopt       = R(score, c)   - R*(H_read, c)    >= 0

    features_read: what the deployed rule actually reads (e.g. the surface gate)
    features_ref : the enlarged reference read set (e.g. surface + bridge)

    The reference must be a SUPERSET of the read set, or the sign guarantee does
    not apply and this function raises.
    """
    y = _check_binary(y, "y")
    Xr = np.atleast_2d(np.asarray(features_read, float))
    if Xr.shape[0] != y.size:
        Xr = Xr.T
    Xf = np.atleast_2d(np.asarray(features_ref, float))
    if Xf.shape[0] != y.size:
        Xf = Xf.T
    if Xf.shape[1] < Xr.shape[1]:
        raise ValueError(
            "features_ref must extend features_read; got "
            f"{Xf.shape[1]} reference columns vs {Xr.shape[1]} read columns"
        )
    eta_read = _cross_fitted_eta(Xr, y, n_folds, seed)
    eta_ref = _cross_fitted_eta(Xf, y, n_folds, seed)
    r_read = optimal_selective_risk(eta_read, y, coverage)
    r_ref = optimal_selective_risk(eta_ref, y, coverage)
    r_obs = optimal_selective_risk(_check_scores(score, y.size, "score"), y, coverage)
    blind, subopt = r_read - r_ref, r_obs - r_read

    # Non-negativity of both terms is a POPULATION statement. R*(., c) is
    # estimated by a cross-fitted plug-in, not the true Bayes rule, so fold-wise
    # calibration differences can make an estimated term slightly negative even
    # when the proposition holds. Verified on synthetic data: violations sit
    # around 0.1 binomial standard errors. We therefore tolerate violations
    # inside sampling noise, report them, and raise on anything larger, since a
    # large negative term signals an estimation bug rather than a discovery.
    m = max(1, int(round(coverage * y.size)))
    se = float(np.sqrt(0.25 / m))
    tol = 3.0 * se
    worst = min(blind, subopt)
    if worst < -tol:
        raise RuntimeError(
            f"decomposition term is negative beyond sampling noise "
            f"(blind={blind:.5f}, subopt={subopt:.5f}, tolerance={-tol:.5f}). "
            "Lemma 1 guarantees non-negativity in population, so this "
            "indicates an estimation problem: check that eta is out-of-fold, "
            "that features_ref truly extends features_read, and that folds are "
            "large enough. Do not report these numbers."
        )
    return {"coverage": float(coverage), "observed_risk": float(r_obs),
            "irreducible": float(r_ref),
            "blind": float(blind), "subopt": float(subopt),
            "blind_clipped": float(max(0.0, blind)),
            "subopt_clipped": float(max(0.0, subopt)),
            "noise_tolerance": tol,
            "within_noise": bool(worst >= -tol),
            "identity_residual": float(r_obs - (r_ref + blind + subopt)),
            "blind_share": float(max(0.0, blind) / r_obs) if r_obs > 0 else 0.0}


def recovery_fraction(y, score_repair, features_read, features_ref,
                      coverage: float = 0.5, n_folds: int = 5, seed: int = 0) -> dict:
    """
    rho, as the revised paper defines it: the share of the blind term that a
    repair closes.

        rho = ( R*(H_read, c) - R(repair, c) ) / Delta_blind(read -> ref, c)

    IMPORTANT, and the reason this does NOT call blind_spot_decomposition:
    the repair is expected to read the ENLARGED set, so Lemma 1's guarantee
    Delta_subopt >= 0 does not apply to it. Routing a repair through the
    decomposition therefore trips that guardrail spuriously. Lemma 1's sign
    guarantee is for a rule that reads only R; rho is for a rule that reads R'.

    rho = 1 when the repair is optimal for the enlarged read set, and rho < 0
    when it does worse than the best surface-only rule. Values above 1 beyond
    noise indicate an estimation problem, not a very good repair.

    NOTE: this replaced an earlier essential-supremum definition of rho. That
    version is retained below as recoverable_fraction_rho for estimator
    sanity-checking only; it is NOT the quantity reported in the paper.
    """
    y = _check_binary(y, "y")
    s_rep = _check_scores(score_repair, y.size, "score_repair")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0,1], got {coverage}")

    Xr = np.atleast_2d(np.asarray(features_read, float))
    if Xr.shape[0] != y.size:
        Xr = Xr.T
    Xf = np.atleast_2d(np.asarray(features_ref, float))
    if Xf.shape[0] != y.size:
        Xf = Xf.T
    if Xf.shape[1] < Xr.shape[1]:
        raise ValueError("features_ref must extend features_read")

    r_read_star = optimal_selective_risk(_cross_fitted_eta(Xr, y, n_folds, seed), y, coverage)
    r_ref_star = optimal_selective_risk(_cross_fitted_eta(Xf, y, n_folds, seed), y, coverage)
    blind = r_read_star - r_ref_star

    k = max(1, int(round(coverage * y.size)))
    tol = 3.0 * float(np.sqrt(0.25 / k))
    if blind <= tol:
        raise ValueError(
            f"Delta_blind ({blind:.5f}) is within sampling noise ({tol:.5f}). rho is a share "
            "OF the blind term and is undefined when there is no measurable blind term. "
            "Report Delta_blind with its interval and state that rho is undefined, rather "
            "than dividing by a noise-level number."
        )

    keep = np.argsort(-s_rep, kind="mergesort")[:k]
    r_repair = float(1.0 - y[keep].mean())
    rho = (r_read_star - r_repair) / blind
    return {"rho": float(rho), "blind": float(blind),
            "r_read_star": float(r_read_star), "r_ref_star": float(r_ref_star),
            "r_repair": r_repair, "coverage": float(coverage),
            "noise_tolerance": tol,
            "exceeds_one": bool(rho > 1.0 + tol / max(blind, 1e-12))}


def corollary_blind_lower_bound(y, m, gate_surface, features_ref, coverage: float = 0.5,
                                alpha: float = 0.05, n_folds: int = 5, seed: int = 0) -> dict:
    """
    Corollary 1: the floor certifies a blind share without trusting the
    reference optimum.

        Delta_blind(R -> R', c)  >=  P(M=0) - delta - v - R*(H_R', c)

    Returns the certified lower bound alongside the directly estimated
    Delta_blind. The point of the corollary is that these are independent routes
    to the same quantity, so a directly estimated Delta_blind that falls BELOW
    the certified bound signals an estimation problem rather than a small blind
    term. `bound_respected` reports that check.
    """
    y = _check_binary(y, "y")
    m = _check_binary(m, "m")
    gs = _check_scores(gate_surface, y.size, "gate_surface")
    delta = estimate_delta(gs, m, alpha=alpha, n_folds=n_folds, seed=seed)
    v = estimate_shortcut_v(y, m, features=gs, alpha=alpha, n_folds=n_folds, seed=seed)
    floor = floor_from_proposition2(m, delta, v)

    Xf = np.atleast_2d(np.asarray(features_ref, float))
    if Xf.shape[0] != y.size:
        Xf = Xf.T
    r_ref = optimal_selective_risk(_cross_fitted_eta(Xf, y, n_folds, seed), y, coverage)

    certified = floor - r_ref
    dec = blind_spot_decomposition(y, gs, gs.reshape(-1, 1), Xf,
                                   coverage=coverage, n_folds=n_folds, seed=seed)
    tol = 3.0 * float(np.sqrt(0.25 / max(1, int(round(coverage * y.size)))))
    return {"floor": float(floor), "delta": float(delta), "v": float(v),
            "R_star_ref": float(r_ref),
            "certified_lower_bound": float(certified),
            "estimated_blind": float(dec["blind"]),
            "bound_respected": bool(dec["blind"] >= certified - tol),
            "tolerance": tol}


def khop_floor(hop_outcomes: list, gate_surface, y=None, alpha: float = 0.05,
               n_folds: int = 5, seed: int = 0) -> dict:
    """
    Theorem 1 (surface-gate floor) for a k-hop chain.

    hop_outcomes: [K_2, K_3, ..., K_k] as binary arrays. Hop 1 is excluded
    because it is the only gate-visible hop.

    To compare across k, pass NESTED chains: the k-hop outcomes must come
    from the same items as the (k+1)-hop outcomes, with the shorter chain a
    prefix of the longer. The monotonicity clause of Theorem 1 is pointwise
    (M_{k+1} = M_k * K_{k+1} <= M_k) and says nothing about two unrelated
    query sets, where a rise could be a dataset difference instead.

    Returns P(M_k=0), delta_k, v_k and the resulting floor. M_k is the product
    of all downstream hop outcomes.
    """
    if not hop_outcomes:
        raise ValueError("need at least one downstream hop (K_2 onward)")
    arrs = [_check_binary(h, f"hop_{i+2}") for i, h in enumerate(hop_outcomes)]
    n = arrs[0].size
    if any(a.size != n for a in arrs):
        raise ValueError("all hop outcome arrays must have the same length")
    m_k = arrs[0].copy()
    for a in arrs[1:]:
        m_k = m_k * a
    gs = _check_scores(gate_surface, n, "gate_surface")
    delta_k = estimate_delta(gs, m_k, alpha=alpha, n_folds=n_folds, seed=seed)
    v_k = (estimate_shortcut_v(y, m_k, features=gs, alpha=alpha, n_folds=n_folds, seed=seed)
           if y is not None else 0.0)
    return {"k": len(arrs) + 1,
            "p_M_zero": float(1.0 - m_k.mean()),
            "delta_k": float(delta_k),
            "v_k": float(v_k),
            "floor": float(max(0.0, (1.0 - m_k.mean()) - delta_k - v_k))}


def kappa_gain_bound(kappa_s: float, kappa_sb: float) -> float:
    """
    RETIRED FROM THE PAPER. The reported rho is now recovery_fraction(), defined
    from the Lemma 1 decomposition at a stated coverage. This essential-supremum
    version is kept only because its monotonicity is a useful check that the
    cross-fitted estimators behave. Do not report it.

        (kappa_{S,B} - kappa_S) / (1 - kappa_S)
    """
    for nm, val in (("kappa_s", kappa_s), ("kappa_sb", kappa_sb)):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{nm}={val} outside [0,1]")
    if kappa_s >= 1.0:
        raise ValueError("rho undefined when kappa_S = 1")
    if kappa_sb < kappa_s - 1e-9:
        raise ValueError(
            f"kappa_sb ({kappa_sb:.4f}) < kappa_s ({kappa_s:.4f}) violates Proposition 3. "
            "This indicates an estimation bug, not a real effect. Investigate before reporting."
        )
    return float(max(0.0, (kappa_sb - kappa_s) / (1.0 - kappa_s)))


def composition_gap(y, gate_surface, gate_bridge, n_boot: int = 2000, seed: int = 0) -> dict:
    """Delta = AUC(S,B -> Y) - AUC(S -> Y), with a paired bootstrap interval."""
    y = _check_binary(y, "y")
    gs = _check_scores(gate_surface, y.size, "gate_surface")
    gb = _check_scores(gate_bridge, y.size, "gate_bridge")
    joint = _cross_fitted_eta(np.column_stack([gs, gb]), y, 5, seed)
    out = paired_bootstrap_auc_diff(y, joint, gs, n_boot=n_boot, seed=seed)
    out["auc_surface"] = auroc(y, gs)
    out["auc_joint"] = auroc(y, joint)
    return out


# =====================================================================
# Bridge-Gate Readout
# =====================================================================

def bgr_min(gate_surface, gate_bridge, fit_gate_surface=None, fit_gate_bridge=None) -> np.ndarray:
    """
    Parameter-free readout: s = min(g_S, g_B) after per-feature standardization.
    Implements the conjunctive behavior H2 predicts and H1 denies. No fitting,
    so the effect cannot be attributed to a learned model.

    fit_gate_surface/fit_gate_bridge: if given, the standardization mean/std
    are computed from THESE arrays (a genuinely separate fit-split
    population) rather than from gate_surface/gate_bridge themselves. This
    was a REAL bug, not a stylistic choice: the paper's own stated protocol
    (skc_main.tex, sec:bgr, step 2) requires "statistics computed on the fit
    split, not on the evaluation items" -- standardizing on the eval set's
    own mean/std, as the old (still-default, for backward compatibility)
    behavior does, leaks eval-set information into the score before min() is
    even taken. Found by comparing this function's own output
    (E4's BGR-min=0.620 on Gemma-2-9B) against VERSION3's independent
    ablations.py (which DOES use a genuine fit-split Standardizer,
    bgr_min=0.450 on the identical data) -- a 0.17 AUROC gap entirely
    attributable to this leakage. Old behavior (fit_gate_* omitted) is kept
    as the default so past results remain reproducible byte-for-byte from
    the same call; NEW callers (skc_run_e4.py, once it has a real fit-split
    g_S/g_B to pass) should supply fit_gate_surface/fit_gate_bridge.
    """
    gs = np.asarray(gate_surface, float)
    gb = np.asarray(gate_bridge, float)
    if gs.shape != gb.shape:
        raise ValueError(f"shape mismatch: {gs.shape} vs {gb.shape}")
    if not (np.all(np.isfinite(gs)) and np.all(np.isfinite(gb))):
        raise ValueError("non-finite gate scores; fix upstream")
    if (fit_gate_surface is None) != (fit_gate_bridge is None):
        raise ValueError("fit_gate_surface and fit_gate_bridge must be given together, or not at all")

    def z(a, fit_a=None):
        ref = np.asarray(fit_a, float) if fit_a is not None else a
        sd = ref.std()
        if sd < 1e-12:
            raise ValueError("zero-variance gate score; the direction is not firing")
        return (a - ref.mean()) / sd

    return np.minimum(z(gs, fit_gate_surface), z(gb, fit_gate_bridge))


def bgr_lr(gate_surface, gate_bridge, y, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Cross-fitted two-feature logistic readout. Reported alongside bgr_min."""
    y = _check_binary(y, "y")
    gs = _check_scores(gate_surface, y.size, "gate_surface")
    gb = _check_scores(gate_bridge, y.size, "gate_bridge")
    return _cross_fitted_eta(np.column_stack([gs, gb]), y, n_folds, seed)


# =====================================================================
# Synthetic generator (explicitly gated)
# =====================================================================

@dataclass
class SyntheticSpec:
    n: int = 8000
    p_k1: float = 0.70          # first-hop success rate
    q_k2: float = 0.55          # second-hop success rate
    gate_noise: float = 0.85    # how noisy the surface gate is about K1
    bridge_noise: float = 0.95  # how noisy the bridge readout is about K2
    leak: float = 0.0           # how much the surface gate leaks about K2
    shortcut: float = 0.0       # P(Y=1 | K1=1, K2=0), the composite-memorization rate
    seed: int = 0


def make_synthetic(spec: SyntheticSpec):
    """
    SYNTHETIC DATA. For verifying the propositions only.
    Numbers produced from this generator must never appear in the paper.

    Ground truth by construction:
      K1 ~ Bern(p_k1); K2 ~ Bern(q_k2) independent of K1
      Y  = K1 * K2, except with prob `shortcut` when K1=1, K2=0, Y=1
      S  = signal about K1, plus `leak` * signal about K2
      B  = signal about K2
    """
    rng = np.random.default_rng(spec.seed)
    n = spec.n
    k1 = (rng.random(n) < spec.p_k1).astype(np.int64)
    k2 = (rng.random(n) < spec.q_k2).astype(np.int64)
    y = k1 * k2
    if spec.shortcut > 0:
        flip = (k1 == 1) & (k2 == 0) & (rng.random(n) < spec.shortcut)
        y = np.where(flip, 1, y)
    gate_s = k1 + spec.gate_noise * rng.standard_normal(n) + spec.leak * k2
    gate_b = k2 + spec.bridge_noise * rng.standard_normal(n)
    # Third hop is drawn last so that adding it does not perturb the random
    # stream used by the two-hop quantities above. Seeds stay reproducible.
    k3 = (rng.random(n) < spec.q_k2).astype(np.int64)
    y3 = k1 * k2 * k3          # correctness of the corresponding 3-hop query
    return {"k1": k1, "k2": k2, "k3": k3, "y": y.astype(np.int64),
            "y3": y3.astype(np.int64),
            "gate_surface": gate_s, "gate_bridge": gate_b}


# =====================================================================
# Smoke test: verify the propositions numerically
# =====================================================================

def smoke_test(verbose: bool = True) -> bool:
    """
    Checks the DIRECTION of each inequality on data whose truth is known by
    construction. It does not check a hard-coded number, so it stays valid if
    the generator changes.
    """
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    print("=" * 74)
    print("SMOKE TEST. Synthetic data only. These numbers are NOT results.")
    print("=" * 74)

    # -------- Proposition 1 and 2: the surface-gate floor ----------------
    print("\n[1] Propositions 1-2: selective risk cannot fall below the floor")
    print("    Violation criterion: the one-sided 99% upper confidence limit of the")
    print("    empirical risk lies strictly below the floor. A raw point comparison")
    print("    false-alarms, because the minimum over many correlated thresholds is")
    print("    biased downward and the tail rests on few examples.")
    spec = SyntheticSpec(n=40000, leak=0.0, shortcut=0.0, seed=1)
    d = make_synthetic(spec)
    y, k1, k2, gs = d["y"], d["k1"], d["k2"], d["gate_surface"]

    rep = floor_report(y, k2, gs, seed=1)
    report("no threshold contradicts the floor", not rep["violated"],
           f"min UCL={rep['min_ucl']:.3f} vs floor={rep['floor']:.3f} "
           f"at coverage {rep['coverage_at_min_ucl']:.2f}")
    report("delta near zero when the gate does not leak", rep["delta"] < 0.10,
           f"delta={rep['delta']:.4f}")
    report("v near zero when there is no shortcut", rep["v"] < 0.05, f"v={rep['v']:.4f}")
    report("floor tracks the second-hop failure rate",
           abs(rep["floor"] - rep["p_k2_zero"]) < 0.12,
           f"floor={rep['floor']:.3f}  P(K2=0)={rep['p_k2_zero']:.3f}")

    # H1 signature: the gate sees hop 1 but not hop 2.
    a_k1, a_k2 = auroc(k1, gs), auroc(k2, gs)
    report("gate predicts K1 well above chance", a_k1 > 0.65, f"AUC(S->K1)={a_k1:.3f}")
    report("gate is near chance on K2 (H1 signature)", abs(a_k2 - 0.5) < 0.05,
           f"AUC(S->K2)={a_k2:.3f}")

    # -------- The bound must survive leakage and shortcuts ---------------
    print("\n[2] Proposition 2 with non-zero slack terms")
    for leak, short in ((0.8, 0.0), (0.0, 0.25), (0.8, 0.25), (1.5, 0.40)):
        dd = make_synthetic(SyntheticSpec(n=40000, leak=leak, shortcut=short, seed=7))
        r = floor_report(dd["y"], dd["k2"], dd["gate_surface"], seed=7)
        report(f"bound holds (leak={leak}, shortcut={short})", not r["violated"],
               f"min UCL={r['min_ucl']:.3f} floor={r['floor']:.3f} "
               f"delta={r['delta']:.3f} v={r['v']:.3f}")

    # -------- The marginal shortcut estimator is anti-conservative --------
    print("\n[2b] Regression test: the marginal v estimator must be refused")
    dd = make_synthetic(SyntheticSpec(n=40000, leak=0.0, shortcut=0.25, seed=7))
    try:
        estimate_shortcut_v(dd["y"], dd["k2"])
        report("estimate_shortcut_v refuses the marginal by default", False, "did not raise")
    except ValueError:
        report("estimate_shortcut_v refuses the marginal by default", True)
    v_marg = estimate_shortcut_v(dd["y"], dd["k2"], allow_marginal=True)
    v_cond = estimate_shortcut_v(dd["y"], dd["k2"], features=dd["gate_surface"], seed=7)
    report("marginal v under-estimates the conditional v", v_marg < v_cond,
           f"marginal={v_marg:.4f} < conditional={v_cond:.4f}")
    de = estimate_delta(dd["gate_surface"], dd["k2"], seed=7)
    f_marg = floor_from_proposition2(dd["k2"], de, v_marg)
    f_cond = floor_from_proposition2(dd["k2"], de, v_cond)
    bad, good = (check_floor(dd["y"], dd["gate_surface"], f_marg),
                 check_floor(dd["y"], dd["gate_surface"], f_cond))
    # The invariant is the ORDERING of the two floors. Whether the inflated
    # floor tips into a detected violation depends on sample noise, so we
    # assert the ordering and merely report the violation flags.
    report("the marginal floor is inflated above the conditional one",
           f_marg > f_cond, f"marginal={f_marg:.3f} > conditional={f_cond:.3f}")
    report("the conditional floor is never contradicted", not good["violated"],
           f"violated={good['violated']}")
    print(f"       (marginal floor violated={bad['violated']}; when it is True the "
          f"inflated floor has manufactured a false violation)")

    # -------- Proposition 1: the decomposition ---------------------------
    print("\n[1b] Proposition 1: the blind-spot decomposition")
    dd = make_synthetic(SyntheticSpec(n=40000, leak=0.0, shortcut=0.0, seed=21))
    yy, gs_, gb_ = dd["y"], dd["gate_surface"], dd["gate_bridge"]
    for cov in (0.3, 0.5, 0.8):
        dec = blind_spot_decomposition(yy, gs_, gs_.reshape(-1, 1),
                                       np.column_stack([gs_, gb_]),
                                       coverage=cov, seed=21)
        ok_id = abs(dec["identity_residual"]) < 1e-9
        report(f"identity is exact (c={cov})", ok_id,
               f"residual={dec['identity_residual']:.2e}")
        report(f"both terms non-negative within sampling noise (c={cov})",
               dec["within_noise"],
               f"blind={dec['blind']:+.4f} subopt={dec['subopt']:+.4f} "
               f"tol={-dec['noise_tolerance']:.4f}")
    dec = blind_spot_decomposition(yy, gs_, gs_.reshape(-1, 1),
                                   np.column_stack([gs_, gb_]), coverage=0.5, seed=21)
    report("blind term is large when the bridge is informative and unread",
           dec["blind"] > 0.02, f"blind={dec['blind']:.3f}")
    # control: an uninformative reference must not create a blind term
    rng2 = np.random.default_rng(99)
    dec0 = blind_spot_decomposition(yy, gs_, gs_.reshape(-1, 1),
                                    np.column_stack([gs_, rng2.standard_normal(yy.size)]),
                                    coverage=0.5, seed=21)
    report("an uninformative reference yields a near-zero blind term",
           dec0["blind"] < dec["blind"], f"{dec0['blind']:.3f} < {dec['blind']:.3f}")
    try:
        blind_spot_decomposition(yy, gs_, np.column_stack([gs_, gb_]), gs_.reshape(-1, 1))
        report("raises when the reference is not a superset", False, "did not raise")
    except ValueError:
        report("raises when the reference is not a superset", True)
    # A deliberately broken call (score far better than what the read set allows)
    # must trip the negativity guard rather than silently returning.
    try:
        blind_spot_decomposition(yy, yy.astype(float), gs_.reshape(-1, 1),
                                 np.column_stack([gs_, gb_]), coverage=0.5, seed=21)
        report("raises when a term is negative beyond noise", False, "did not raise")
    except RuntimeError:
        report("raises when a term is negative beyond noise", True)

    # -------- Corollary 1: the floor certifies a blind share --------------
    print("\n[1c] Corollary 1: floor-certified lower bound on the blind share")
    dc = make_synthetic(SyntheticSpec(n=40000, leak=0.0, shortcut=0.0, seed=23))
    yc, mc, gsc, gbc = dc["y"], dc["k2"], dc["gate_surface"], dc["gate_bridge"]
    for cov in (0.3, 0.5):
        cb = corollary_blind_lower_bound(yc, mc, gsc, np.column_stack([gsc, gbc]),
                                         coverage=cov, seed=23)
        report(f"estimated blind share respects the certified bound (c={cov})",
               cb["bound_respected"],
               f"estimated={cb['estimated_blind']:.3f} >= certified="
               f"{cb['certified_lower_bound']:.3f}")
    cb = corollary_blind_lower_bound(yc, mc, gsc, np.column_stack([gsc, gbc]),
                                     coverage=0.5, seed=23)
    report("the certified bound is a real constraint, not vacuous",
           cb["certified_lower_bound"] > -0.5,
           f"certified={cb['certified_lower_bound']:.3f}, floor={cb['floor']:.3f}")

    # -------- Proposition 4: the blind spot compounds --------------------
    print("\n[2c] Proposition 4: the uncovered fraction grows with hop count")
    d3 = make_synthetic(SyntheticSpec(n=40000, leak=0.0, shortcut=0.0, seed=13))
    f2 = khop_floor([d3["k2"]], d3["gate_surface"], y=d3["y"], seed=13)
    f3 = khop_floor([d3["k2"], d3["k3"]], d3["gate_surface"], y=d3["y3"], seed=13)
    report("P(M_k=0) is non-decreasing in k", f3["p_M_zero"] >= f2["p_M_zero"],
           f"k=2: {f2['p_M_zero']:.3f}  k=3: {f3['p_M_zero']:.3f}")
    report("the k=3 floor exceeds the k=2 floor when slack does not grow",
           f3["floor"] > f2["floor"],
           f"floor k=2: {f2['floor']:.3f}  k=3: {f3['floor']:.3f}")
    c3 = check_floor(d3["y3"], d3["gate_surface"], f3["floor"])
    report("the k=3 bound is not contradicted", not c3["violated"],
           f"min UCL={c3['min_ucl']:.3f} vs floor={f3['floor']:.3f}")
    report("k=2 case of Prop 4 matches Prop 2", abs(f2["floor"] - floor_report(
               d3["y"], d3["k2"], d3["gate_surface"], seed=13)["floor"]) < 1e-9,
           "identical by construction")

    # -------- Proposition 3: the bridge readout can only help ------------
    print("\n[3] Estimator sanity: kappa monotone under refinement")
    print("    (kappa-based rho was RETIRED from the paper; kept only as an estimator check)")
    d = make_synthetic(SyntheticSpec(leak=0.0, shortcut=0.0, seed=3))
    y, gs, gb = d["y"], d["gate_surface"], d["gate_bridge"]
    k_s = estimate_kappa(gs.reshape(-1, 1), y, seed=3)
    k_sb = estimate_kappa(np.column_stack([gs, gb]), y, seed=3)
    report("kappa_(S,B) >= kappa_S", k_sb >= k_s - 1e-6, f"{k_sb:.3f} >= {k_s:.3f}")
    rho = kappa_gain_bound(k_s, k_sb)
    report("rho in [0,1]", 0.0 <= rho <= 1.0, f"rho={rho:.3f}")
    report("rho > 0 when the bridge is informative", rho > 0.0, f"rho={rho:.3f}")

    # Control: an uninformative B must not manufacture recovery.
    rng = np.random.default_rng(11)
    k_sb_null = estimate_kappa(np.column_stack([gs, rng.standard_normal(y.size)]), y, seed=3)
    rho_null = kappa_gain_bound(k_s, max(k_sb_null, k_s))
    report("rho stays small for a random B (no false recovery)", rho_null < rho,
           f"rho_null={rho_null:.3f} < rho={rho:.3f}")

    # -------- BGR ---------------------------------------------------------
    print("\n[4] Bridge-Gate Readout beats the surface gate when H1 holds")
    s_bgr = bgr_min(gs, gb)
    a_surface, a_bgr = auroc(y, gs), auroc(y, s_bgr)
    report("BGR-min improves AUROC", a_bgr > a_surface,
           f"BGR={a_bgr:.3f} vs surface={a_surface:.3f}")
    report("BGR-min lowers AURC", aurc(y, s_bgr) < aurc(y, gs),
           f"{aurc(y, s_bgr):.3f} < {aurc(y, gs):.3f}")
    pb = paired_bootstrap_auc_diff(y, s_bgr, gs, n_boot=500, seed=5)
    report("improvement excludes zero", pb["lo"] > 0, f"CI=[{pb['lo']:.3f},{pb['hi']:.3f}]")

    # -------- rho as the paper now defines it -----------------------------
    print("\n[4b] Recovery fraction derived from the Lemma 1 decomposition")
    read = gs.reshape(-1, 1)
    ref = np.column_stack([gs, gb])
    rr = recovery_fraction(y, s_bgr, read, ref, coverage=0.5, seed=3)
    report("rho <= 1", rr["rho"] <= 1.0 + 1e-9, f"rho={rr['rho']:.3f}")
    report("BGR closes a positive share of the blind term", rr["rho"] > 0,
           f"rho={rr['rho']:.3f}  blind={rr['blind']:.4f}")
    rr_bad = recovery_fraction(y, gs, read, ref, coverage=0.5, seed=3)
    report("a surface-only rule closes less of it than BGR",
           rr_bad["rho"] < rr["rho"], f"surface rho={rr_bad['rho']:.3f} < BGR rho={rr['rho']:.3f}")
    report("rho does not exceed one beyond noise", not rr["exceeds_one"],
           f"rho={rr['rho']:.3f}")
    try:
        recovery_fraction(y, s_bgr, read, read, coverage=0.5, seed=3)
        report("rho refuses a zero blind term", False, "did not raise")
    except ValueError:
        report("rho refuses a zero blind term", True)
    # Lemma 1's sign guarantee is only for a rule that READS the smaller set.
    try:
        blind_spot_decomposition(y, s_bgr, read, ref, coverage=0.5, seed=3)
        report("decomposition guardrail fires on a rule reading the larger set", False,
               "did not raise")
    except RuntimeError:
        report("decomposition guardrail fires on a rule reading the larger set", True)

    # -------- Estimator sanity -------------------------------------------
    print("\n[5] Estimator sanity")
    report("AUROC of a perfect score is 1", abs(auroc(y, y.astype(float)) - 1.0) < 1e-9)
    report("AUROC of a constant score is 0.5", abs(auroc(y, np.zeros_like(y, float)) - 0.5) < 1e-9)
    yy = np.array([0, 0, 1, 1])
    report("AUROC is invariant to monotone rescaling",
           abs(auroc(yy, np.array([1., 2, 3, 4])) - auroc(yy, np.array([10., 20, 30, 40]))) < 1e-12)
    dl = delong_test(y, s_bgr, gs)
    report("DeLong agrees in sign with the paired bootstrap",
           np.sign(dl["diff"]) == np.sign(pb["diff"]),
           f"DeLong diff={dl['diff']:.4f} p={dl['p_value']:.2e}")
    report("DeLong and bootstrap agree in magnitude",
           abs(dl["diff"] - pb["diff"]) < 0.05,
           f"|{dl['diff']:.4f} - {pb['diff']:.4f}|")
    sep = separation_score(np.array([[1.0, -1.0], [2.0, -2.0]]), np.array([[-1.0, 1.0]]))
    report("separation score has the right sign", sep[0] > 0 and sep[1] < 0, f"{sep}")

    # -------- Guardrails must fire ---------------------------------------
    print("\n[6] Guardrails: malformed input must raise, never impute")
    for name, fn in [
        ("empty labels", lambda: auroc(np.array([]), np.array([]))),
        ("non-binary labels", lambda: auroc(np.array([0, 1, 2]), np.array([0., 1, 2]))),
        ("NaN scores", lambda: auroc(np.array([0, 1]), np.array([np.nan, 1.0]))),
        ("single class", lambda: auroc(np.array([1, 1]), np.array([0.1, 0.2]))),
        ("length mismatch", lambda: auroc(np.array([0, 1]), np.array([0.1]))),
        ("kappa_sb < kappa_s", lambda: kappa_gain_bound(0.9, 0.5)),
        ("negative delta", lambda: floor_from_proposition2(np.array([0, 1]), -0.1, 0.0)),
    ]:
        try:
            fn()
            report(f"raises on {name}", False, "did not raise")
        except (ValueError, RuntimeError):
            report(f"raises on {name}", True)

    print("\n" + "=" * 74)
    print("SMOKE TEST " + ("PASSED" if ok else "FAILED"))
    print("Reminder: every number above is synthetic and belongs in no table.")
    print("=" * 74)
    return ok


def self_check() -> bool:
    """Quick internal consistency pass used in CI."""
    d = make_synthetic(SyntheticSpec(n=2000, seed=42))
    cov, risk = risk_coverage_curve(d["y"], d["gate_surface"])
    checks = [
        ("coverage ends at 1", abs(cov[-1] - 1.0) < 1e-12),
        ("coverage increasing", bool(np.all(np.diff(cov) > 0))),
        ("risk in [0,1]", bool(np.all((risk >= -1e-12) & (risk <= 1 + 1e-12)))),
        ("full-coverage risk equals base error", abs(risk[-1] - (1 - d["y"].mean())) < 1e-12),
        ("aurc in [0,1]", 0.0 <= aurc(d["y"], d["gate_surface"]) <= 1.0),
    ]
    for n, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}")
    return all(p for _, p in checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-test", action="store_true",
                    help="verify Propositions 1-3 numerically on synthetic data")
    ap.add_argument("--self-check", action="store_true", help="internal consistency checks")
    ap.add_argument("--emit-synthetic", metavar="PATH",
                    help="write a synthetic dataset (files are prefixed SYNTHETIC_)")
    a = ap.parse_args()

    if not (a.smoke_test or a.self_check or a.emit_synthetic):
        ap.print_help()
        return 0
    ok = True
    if a.self_check:
        ok &= self_check()
    if a.smoke_test:
        ok &= smoke_test()
    if a.emit_synthetic:
        import os
        p = os.path.join(os.path.dirname(a.emit_synthetic) or ".",
                         "SYNTHETIC_" + os.path.basename(a.emit_synthetic))
        d = make_synthetic(SyntheticSpec())
        np.savez(p, **d)
        with open(p + ".README.txt", "w") as f:
            f.write("SYNTHETIC DATA. Generated by skc_metrics.py for testing only.\n"
                    "Numbers derived from this file must never appear in the paper.\n"
                    f"spec = {json.dumps(asdict(SyntheticSpec()))}\n")
        print(f"wrote {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

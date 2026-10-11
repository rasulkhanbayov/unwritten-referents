"""
Selective-prediction metrics for the availability/scoring decomposition.

Every function here computes from arrays the caller supplies. Nothing in this
module contains, defaults to, or falls back on a number from the paper. If the
data needed for a quantity is absent, the function raises.

numpy + scipy only.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "auroc", "delong_var", "delong_test", "aurc", "selective_risk",
    "risk_coverage_curve", "r_star", "bootstrap_ci", "paired_bootstrap_diff",
    "benjamini_hochberg", "class_balance",
]


# --------------------------------------------------------------------------
# basic quantities
# --------------------------------------------------------------------------

def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ")
    if s.size == 0:
        raise ValueError("empty input")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary 0/1")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN; fix upstream rather than dropping")
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("AUROC undefined: one class is empty")
    return s, y


def class_balance(labels) -> dict:
    """Positive rate and counts. The paper could not state these; report them."""
    y = np.asarray(labels)
    n1 = int(y.sum())
    return {"n": int(y.size), "n_pos": n1, "n_neg": int(y.size - n1),
            "pos_rate": float(n1 / y.size)}


def auroc(scores, labels) -> float:
    """Rank-based AUROC with ties handled at 0.5, higher score = higher label."""
    s, y = _check(scores, labels)
    r = stats.rankdata(s)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def delong_var(scores, labels) -> float:
    """DeLong variance of a single AUROC via placement values."""
    s, y = _check(scores, labels)
    x, z = s[y == 1], s[y == 0]
    m, n = x.size, z.size
    # V10[i] = P(z < x_i) + 0.5 P(z == x_i)
    v10 = np.array([(z < xi).mean() + 0.5 * (z == xi).mean() for xi in x])
    v01 = np.array([(x > zj).mean() + 0.5 * (x == zj).mean() for zj in z])
    return float(v10.var(ddof=1) / m + v01.var(ddof=1) / n)


def delong_test(scores_a, scores_b, labels) -> dict:
    """
    Two-sided DeLong test for two correlated AUROCs on the same items.
    Returns the two AUROCs, their difference, the z statistic and the p value.
    """
    s_a, y = _check(scores_a, labels)
    s_b, _ = _check(scores_b, labels)
    a, b = auroc(s_a, y), auroc(s_b, y)
    x_a, z_a = s_a[y == 1], s_a[y == 0]
    x_b, z_b = s_b[y == 1], s_b[y == 0]
    m, n = x_a.size, z_a.size
    v10 = np.column_stack([
        [(z_a < xi).mean() + 0.5 * (z_a == xi).mean() for xi in x_a],
        [(z_b < xi).mean() + 0.5 * (z_b == xi).mean() for xi in x_b]])
    v01 = np.column_stack([
        [(x_a > zj).mean() + 0.5 * (x_a == zj).mean() for zj in z_a],
        [(x_b > zj).mean() + 0.5 * (x_b == zj).mean() for zj in z_b]])
    S = np.cov(v10, rowvar=False) / m + np.cov(v01, rowvar=False) / n
    var_d = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_d <= 0:
        raise ValueError("non-positive variance for the AUROC difference")
    zstat = (a - b) / np.sqrt(var_d)
    return {"auroc_a": a, "auroc_b": b, "diff": a - b,
            "se": float(np.sqrt(var_d)), "z": float(zstat),
            "p": float(2 * stats.norm.sf(abs(zstat)))}


# --------------------------------------------------------------------------
# selective prediction
# --------------------------------------------------------------------------

def selective_risk(scores, labels, coverage: float) -> float:
    """
    Risk among the top-`coverage` fraction by score, with fractional inclusion
    at the boundary so every coverage is exactly attainable (the randomization
    U of Section 3). labels: 1 = correct, so risk = 1 - precision.
    """
    s, y = _check(scores, labels)
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    order = np.argsort(-s, kind="stable")
    y = y[order].astype(float)
    k = coverage * y.size
    whole = int(np.floor(k))
    frac = k - whole
    acc = y[:whole].sum() + (frac * y[whole] if whole < y.size else 0.0)
    return float(1.0 - acc / k)


def risk_coverage_curve(scores, labels, coverages) -> np.ndarray:
    return np.array([selective_risk(scores, labels, c) for c in coverages])


def aurc(scores, labels, n_points: int = 100) -> float:
    """Area under the risk-coverage curve, trapezoid over coverage in (0, 1]."""
    cov = np.linspace(1.0 / n_points, 1.0, n_points)
    return float(np.trapz(risk_coverage_curve(scores, labels, cov), cov))


def r_star(eta_hat, labels, coverage: float) -> float:
    """
    R*(Z, c): the least selective risk over rules seeing only Z.

    The infimum is attained by thresholding eta(Z) = P(Y=1 | Z) and accepting
    where it is largest (verified exhaustively in tests/test_smoke.py), so this
    is a plug-in: pass a CROSS-FITTED estimate of eta, never one fit on the
    same items it scores.
    """
    return selective_risk(eta_hat, labels, coverage)


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------

def bootstrap_ci(fn, *arrays, groups=None, n_resamples=2000, ci=0.95, seed=0):
    """
    Percentile bootstrap of fn(*arrays).

    `groups` is the resampling unit. Pass the query id so a composed item and
    its decompositions move together; passing None resamples rows, which
    understates uncertainty on this data.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = arrays[0].shape[0]
    if groups is None:
        raise ValueError(
            "groups is required. Row-level resampling understates uncertainty "
            "here because decompositions of one query are not independent. "
            "Pass query ids explicitly, or groups=np.arange(n) to opt out.")
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by_group = {u: np.flatnonzero(g == u) for u in uniq}
    point = fn(*arrays)
    draws = []
    for _ in range(n_resamples):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_group[c] for c in chosen])
        try:
            draws.append(fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue  # a resample with one class empty contributes nothing
    if len(draws) < 0.5 * n_resamples:
        raise ValueError(
            f"only {len(draws)}/{n_resamples} resamples were usable; the "
            "sample is too small or too imbalanced for a bootstrap interval")
    lo, hi = np.percentile(draws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "se": float(np.std(draws, ddof=1)), "n_resamples": len(draws)}


def paired_bootstrap_diff(fn, scores_a, scores_b, labels, groups,
                          n_resamples=2000, ci=0.95, seed=0):
    """Paired bootstrap of fn(a, y) - fn(b, y) on the same resampled items."""
    rng = np.random.default_rng(seed)
    a, b, y, g = map(np.asarray, (scores_a, scores_b, labels, groups))
    uniq = np.unique(g)
    idx_by_group = {u: np.flatnonzero(g == u) for u in uniq}
    point = fn(a, y) - fn(b, y)
    draws = []
    for _ in range(n_resamples):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_group[c] for c in chosen])
        try:
            draws.append(fn(a[idx], y[idx]) - fn(b[idx], y[idx]))
        except ValueError:
            continue
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    # two-sided bootstrap p: how often the difference crosses zero
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"diff": float(point), "lo": float(lo), "hi": float(hi),
            "p": float(min(p, 1.0)), "n_resamples": int(draws.size)}


def benjamini_hochberg(pvals, alpha=0.05):
    """BH-adjusted p values and rejection flags, for the baseline sweep."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        running = min(running, p[order[rank]] * m / (rank + 1))
        adj[order[rank]] = running
    return {"p_adjusted": adj, "reject": adj <= alpha}

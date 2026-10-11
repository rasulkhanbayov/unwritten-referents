"""
Slack terms, the written-position floor, and the availability decomposition.

Definition 1 (paper):
    delta = esssup_s ( P(M=1 | S=s) - P(M=1) )^+      one-sided, positive part
    v     = esssup_s   P(Y=1, M=0 | S=s)

Theorem 1:
    R >= P(M=0) - delta - v   for any rule whose accept/abstain decision
                              depends on the input only through S.

Two traps this module refuses to fall into, both found by running the smoke
test on synthetic data with known ground truth:

  1. Substituting the MARGINAL P(Y=1, M=0) for v under-estimates the essential
     supremum whenever the gate carries information about K1. That inflates the
     computed floor and can manufacture an apparent violation of a theorem that
     holds. `estimate_v(..., marginal=True)` raises unless you pass
     `i_know_this_is_wrong=True`, which exists only so the test can demonstrate
     the failure.

  2. A two-sided delta is larger than the proof needs, which shrinks the bound
     for nothing. On a gate that is anti-predictive of M the two differ a lot.
     `estimate_delta` is one-sided by default.
"""
from __future__ import annotations

import numpy as np

__all__ = ["estimate_delta", "estimate_v", "floor", "p_m0",
           "delta_blind", "delta_subopt", "decompose"]


def _bins(gate, n_bins, min_count):
    """Equal-mass bins over the gate score, merged until each has min_count."""
    g = np.asarray(gate, float)
    if g.size < min_count * 2:
        raise ValueError(
            f"{g.size} items cannot support conditional estimation at "
            f"min_count={min_count}. Do not fall back to a marginal estimate.")
    n_bins = int(min(n_bins, max(2, g.size // min_count)))
    edges = np.quantile(g, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(g, edges[1:-1]), 0, n_bins - 1)
    return idx, n_bins


def p_m0(M) -> float:
    """P(M = 0), the leading term of the floor."""
    m = np.asarray(M)
    if not np.isin(m, (0, 1)).all():
        raise ValueError("M must be binary")
    return float(1.0 - m.mean())


def estimate_delta(gate, M, one_sided=True, n_bins=10, min_count=20,
                   quantile=0.99) -> float:
    """
    Conditional-quantile plug-in for delta. The essential supremum is taken
    over the law of the gate, approximated by the `quantile` of the per-bin
    deviations so a single sparse bin cannot drive the estimate.
    """
    g, m = np.asarray(gate, float), np.asarray(M)
    idx, nb = _bins(g, n_bins, min_count)
    p_marg = m.mean()
    devs = []
    for b in range(nb):
        sel = idx == b
        if sel.sum() < min_count:
            continue
        d = m[sel].mean() - p_marg
        devs.append(max(d, 0.0) if one_sided else abs(d))
    if not devs:
        raise ValueError("no bin reached min_count; increase n or lower n_bins")
    return float(np.quantile(devs, quantile))


def estimate_v(gate, Y, M, n_bins=10, min_count=20, quantile=0.99,
               marginal=False, i_know_this_is_wrong=False) -> float:
    """
    Conditional-quantile plug-in for the shortcut rate v.

    `marginal=True` computes P(Y=1, M=0) over all items. That is NOT a
    conservative simplification; it under-estimates v and inflates the floor.
    It is available only to reproduce the failure in the smoke test.
    """
    g, y, m = np.asarray(gate, float), np.asarray(Y), np.asarray(M)
    shortcut = ((y == 1) & (m == 0)).astype(float)
    if marginal:
        if not i_know_this_is_wrong:
            raise ValueError(
                "The marginal estimator for v is wrong, not conservative "
                "(Appendix D). It under-states the essential supremum and "
                "inflates the floor. Use the conditional estimator.")
        return float(shortcut.mean())
    idx, nb = _bins(g, n_bins, min_count)
    vals = [shortcut[idx == b].mean() for b in range(nb)
            if (idx == b).sum() >= min_count]
    if not vals:
        raise ValueError("no bin reached min_count")
    return float(np.quantile(vals, quantile))


def floor(gate, Y, M, clip=False, **kw) -> dict:
    """
    The written-position floor of Theorem 1.

    `clip=False` by default and deliberately. A negative value is a VACUOUS
    bound, not a bound of zero, and reporting 0.000 for it is how the earlier
    draft made four vacuous cells look satisfied. The caller must decide to
    clip and must say so.
    """
    d = estimate_delta(gate, M, **{k: v for k, v in kw.items()
                                   if k in ("one_sided", "n_bins", "min_count", "quantile")})
    v = estimate_v(gate, Y, M, **{k: v for k, v in kw.items()
                                  if k in ("n_bins", "min_count", "quantile")})
    pm0 = p_m0(M)
    raw = pm0 - d - v
    return {"p_m0": pm0, "delta": d, "v": v, "floor_raw": float(raw),
            "floor": float(max(raw, 0.0)) if clip else float(raw),
            "vacuous": bool(raw <= 0), "clipped": bool(clip and raw < 0)}


# --------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------

def delta_blind(r_star_small: float, r_star_large: float) -> float:
    """
    R*(g_R, c) - R*(g_R', c) for nested read sets R subset R'.

    Non-negative only when g_R is a SUB-TUPLE of g_R', i.e. the read sets are
    read as per-position tuples. A single scalar pooled over positions does not
    refine the smaller read set and this term can go negative by construction
    (tests/test_smoke.py demonstrates the counterexample).
    """
    return float(r_star_small - r_star_large)


def delta_subopt(risk_rule: float, r_star_small: float) -> float:
    return float(risk_rule - r_star_small)


def decompose(risk_rule, r_star_small, r_star_large, tol=None) -> dict:
    """
    risk = R*(large) + Delta_blind + Delta_subopt.

    Both bracketed terms are non-negative in population. Cross-fitted plug-ins
    can drive an estimate slightly negative; `tol` (a sampling-noise bound the
    caller supplies, e.g. one binomial SE) marks that as tolerated. Anything
    beyond tol raises, because a large negative term signals an estimation
    fault and not a finding.
    """
    db = delta_blind(r_star_small, r_star_large)
    ds = delta_subopt(risk_rule, r_star_small)
    out = {"irreducible": float(r_star_large), "delta_blind": db,
           "delta_subopt": ds, "total": float(r_star_large + db + ds),
           "risk_rule": float(risk_rule)}
    if abs(out["total"] - risk_rule) > 1e-9:
        raise AssertionError("decomposition is not an identity; check inputs")
    for name, val in (("delta_blind", db), ("delta_subopt", ds)):
        if val < 0:
            if tol is None or val < -abs(tol):
                raise ValueError(
                    f"{name} = {val:.4f} is negative beyond tolerance "
                    f"{tol}. This is an estimation fault, not a result.")
            out.setdefault("tolerated_negative", []).append(name)
    return out

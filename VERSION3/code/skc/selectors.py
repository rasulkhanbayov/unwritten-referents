"""
Selectors, including the ablation arms the paper is missing.

The paper argues BGR-min works because the MINIMUM implements a conjunction:
answering a composed query needs both hops, so one low reading should suffice
to abstain. That argument has never been tested. `bridge_only` is the arm that
tests it, and it is the cheapest falsification available:

    if bridge_only ~= bgr_min, the minimum contributes nothing, the finding is
    "read at a different position", and the motivation in Section 4.3 is wrong.

Standardization uses FIT-SPLIT statistics. Passing evaluation statistics is a
leak and `Standardizer` refuses it.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

__all__ = ["Standardizer", "surface_gate", "bridge_only", "bgr_min", "bgr_mean",
           "bgr_max", "bgr_lr", "SELECTORS", "ablation_arms"]


class Standardizer:
    """z-score with statistics frozen from one split."""

    def __init__(self, split_name: str):
        if split_name != "fit":
            raise ValueError(
                f"standardization statistics must come from the fit split, got "
                f"'{split_name}'. Using evaluation statistics leaks and changes "
                "min(.,.) because the minimum is not scale invariant.")
        self.split_name = split_name
        self.mu_: dict[str, float] = {}
        self.sd_: dict[str, float] = {}

    def fit(self, **columns) -> "Standardizer":
        for k, v in columns.items():
            v = np.asarray(v, float)
            sd = v.std(ddof=1)
            if not np.isfinite(sd) or sd == 0:
                raise ValueError(f"column '{k}' has zero or non-finite spread")
            self.mu_[k], self.sd_[k] = float(v.mean()), float(sd)
        return self

    def transform(self, **columns) -> dict[str, np.ndarray]:
        missing = set(columns) - set(self.mu_)
        if missing:
            raise ValueError(f"never fit on {sorted(missing)}")
        return {k: (np.asarray(v, float) - self.mu_[k]) / self.sd_[k]
                for k, v in columns.items()}


# --------------------------------------------------------------------------
# one-pass selectors. All take standardized scores.
# --------------------------------------------------------------------------

def surface_gate(g_S, g_B=None):
    """Reference: the gate read at the written entity only."""
    return np.asarray(g_S, float)


def bridge_only(g_S, g_B):
    """THE MISSING ARM. Read at the bridge position only, ignore the gate."""
    return np.asarray(g_B, float)


def bgr_min(g_S, g_B):
    """Conjunction: one low reading is enough to abstain."""
    return np.minimum(np.asarray(g_S, float), np.asarray(g_B, float))


def bgr_mean(g_S, g_B):
    """Compensatory control: a high gate reading can offset a low bridge one."""
    return 0.5 * (np.asarray(g_S, float) + np.asarray(g_B, float))


def bgr_max(g_S, g_B):
    """Disjunction control: either reading alone suffices to answer."""
    return np.maximum(np.asarray(g_S, float), np.asarray(g_B, float))


def bgr_lr(g_S_fit, g_B_fit, y_fit, g_S_eval, g_B_eval, folds=5,
           C_grid=(0.01, 0.1, 1.0, 10.0), seed=0):
    """
    Cross-fitted two-feature logistic regression.

    Cross-fitting is over the EVALUATION items so no item contributes to the
    model that scores it. The penalty is chosen inside each training fold, not
    once on everything, which would leak the choice.
    """
    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold
    except ImportError as e:  # pragma: no cover
        raise ImportError("bgr_lr needs scikit-learn; pip install scikit-learn") from e
    X_eval = np.column_stack([np.asarray(g_S_eval, float), np.asarray(g_B_eval, float)])
    X_fit = np.column_stack([np.asarray(g_S_fit, float), np.asarray(g_B_fit, float)])
    y_fit = np.asarray(y_fit)
    out = np.full(X_eval.shape[0], np.nan)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    # placeholder labels only to stratify the eval folds; scoring uses fit data
    for tr, te in skf.split(X_eval, np.zeros(X_eval.shape[0], dtype=int)):
        clf = LogisticRegressionCV(Cs=list(C_grid), cv=folds, penalty="l2",
                                   scoring="roc_auc", max_iter=2000,
                                   random_state=seed)
        clf.fit(X_fit, y_fit)
        out[te] = clf.decision_function(X_eval[te])
    if np.isnan(out).any():
        raise ValueError("cross-fitting left items unscored")
    return out


SELECTORS = {
    "surface_gate": surface_gate,
    "bridge_only": bridge_only,
    "bgr_min": bgr_min,
    "bgr_mean": bgr_mean,
    "bgr_max": bgr_max,
}

#: The four arms that together decide whether the conjunction does any work.
ablation_arms = ("surface_gate", "bridge_only", "bgr_min", "bgr_mean")


def interpret_ablation(auroc_by_arm: dict, tol: float) -> str:
    """
    Turn the ablation into the sentence the paper should print.

    `tol` is the minimum resolvable AUROC difference on this sample; get it
    from scripts/reconcile.py, do not guess it.
    """
    need = set(ablation_arms) - set(auroc_by_arm)
    if need:
        raise ValueError(f"missing arms: {sorted(need)}")
    b, m = auroc_by_arm["bridge_only"], auroc_by_arm["bgr_min"]
    mean_ = auroc_by_arm["bgr_mean"]
    if abs(m - b) <= tol:
        return ("bridge_only matches bgr_min within the resolvable difference: "
                "the minimum contributes nothing and the finding is about read "
                "position, not the conjunction. Section 4.3's motivation does "
                "not survive and must be rewritten.")
    if m - b > tol and m - mean_ > tol:
        return ("bgr_min beats both bridge_only and bgr_mean beyond the "
                "resolvable difference: the conjunction is doing work, as "
                "Section 4.3 argues.")
    return ("the arms separate but not in the pattern the conjunction argument "
            "predicts; report all four and rewrite the motivation to match.")

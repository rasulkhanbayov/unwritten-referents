#!/usr/bin/env python3
"""
Smoke test on SYNTHETIC data with known ground truth.

Nothing here reproduces a number from the paper and nothing here is evidence
for any claim in it. The job is to show the library computes what it says, and
to pin the two estimation traps so a future edit cannot quietly reintroduce
them.

Run:  python tests/test_smoke.py
Exit: 0 if every check passes, 1 otherwise.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skc import metrics, selectors, slack

RNG = np.random.default_rng(0)
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- metrics
y = (RNG.random(4000) < 0.3).astype(int)
s = y + RNG.normal(0, 1.0, y.size)
a = metrics.auroc(s, y)
check("auroc is in range and beats chance on separable data", 0.7 < a < 0.95, f"{a:.3f}")
check("auroc of a constant score is 0.5", abs(metrics.auroc(np.zeros_like(s), y) - 0.5) < 1e-9)
check("auroc is invariant to a monotone transform",
      abs(metrics.auroc(s, y) - metrics.auroc(np.exp(s), y)) < 1e-9)

d = metrics.delong_test(s, -s, y)
check("delong detects a reversed score", d["p"] < 1e-6, f"z={d['z']:.1f}")

cb = metrics.class_balance(y)
check("class balance reports counts and rate",
      cb["n_pos"] + cb["n_neg"] == cb["n"], str(cb))

check("selective risk at full coverage equals the error rate",
      abs(metrics.selective_risk(s, y, 1.0) - (1 - y.mean())) < 1e-9)
check("selective risk falls as coverage tightens on a useful score",
      metrics.selective_risk(s, y, 0.2) < metrics.selective_risk(s, y, 0.9))

# --- R* is attained by thresholding eta. Exhaustive over acceptance sets.
bad = 0
for _ in range(2000):
    n = int(RNG.integers(2, 6))
    pZ = RNG.dirichlet(np.ones(n)); eta = RNG.uniform(0, 1, n)
    c = float(RNG.uniform(0.05, 0.95))
    def risk_of(order):
        rem, mm, cc = c, 0.0, 0.0
        for i in order:
            t = min(pZ[i], rem); mm += t; cc += t * eta[i]; rem -= t
            if rem <= 1e-12: break
        return 1 - cc / mm if mm > 0 else np.inf
    thr = risk_of(np.argsort(-eta))
    best = min(risk_of(list(p)) for r in range(1, n + 1)
               for S in itertools.combinations(range(n), r)
               if pZ[list(S)].sum() >= c - 1e-9
               for p in [sorted(S, key=lambda i: -eta[i])])
    if thr > best + 1e-9: bad += 1
check("R*(Z,c) is attained by thresholding eta(Z), so the plug-in is licensed",
      bad == 0, f"{bad}/2000 counterexamples")

# ---------------------------------------------------------------- bootstrap
ci = metrics.bootstrap_ci(metrics.auroc, s, y, groups=np.arange(y.size),
                          n_resamples=300, seed=1)
check("bootstrap interval brackets the point estimate",
      ci["lo"] < ci["point"] < ci["hi"], f"[{ci['lo']:.3f},{ci['hi']:.3f}]")
try:
    metrics.bootstrap_ci(metrics.auroc, s, y, n_resamples=50)
    check("bootstrap refuses row-level resampling by default", False)
except ValueError:
    check("bootstrap refuses row-level resampling by default", True)

# ---------------------------------------------------------------- slack
n = 6000
gate = RNG.normal(size=n)
M = (RNG.random(n) < 0.55).astype(int)              # independent of the gate
K1 = (RNG.random(n) < 0.5 + 0.3 * np.tanh(gate)).astype(int)
# Shortcuts: the model sometimes answers correctly with M=0, and they CONCENTRATE
# where the gate fires. That concentration is exactly what a marginal estimator
# for v averages away, which is the trap Appendix D describes.
p_short = 0.45 / (1 + np.exp(-2.5 * gate))
shortcut = ((M == 0) & (RNG.random(n) < p_short)).astype(int)
Y = np.maximum(K1 * M, shortcut)
f = slack.floor(gate, Y, M, clip=False, min_count=100)
check("floor is not violated when M is independent of the gate",
      metrics.selective_risk(gate, Y, 0.5) >= f["floor_raw"] - 1e-6,
      f"risk {metrics.selective_risk(gate, Y, 0.5):.3f} vs floor {f['floor_raw']:+.3f}")
check("a negative floor is reported as vacuous, not clipped to zero",
      "vacuous" in f and (f["floor_raw"] <= 0) == f["vacuous"])

# --- TRAP 1: the marginal estimator for v is wrong, not conservative.
try:
    slack.estimate_v(gate, Y, M, marginal=True)
    check("marginal estimator for v is refused by default", False)
except ValueError:
    check("marginal estimator for v is refused by default", True)
v_cond = slack.estimate_v(gate, Y, M, min_count=100)
v_marg = slack.estimate_v(gate, Y, M, marginal=True, i_know_this_is_wrong=True)
check("marginal v under-states the conditional one, inflating the floor",
      v_marg < v_cond, f"marginal {v_marg:.3f} < conditional {v_cond:.3f}")

# --- TRAP 2: a pooled scalar breaks the non-negativity of Delta_blind.
h1, h2 = RNG.normal(size=3000), RNG.normal(size=3000)
g_small, g_pooled = h1, h1 + h2
order = np.argsort(g_pooled)
gap = np.abs(np.diff(g_small[order]))[np.abs(np.diff(g_pooled[order])) < 1e-3]
check("a pooled scalar over a larger read set does not determine the smaller one",
      gap.size > 0 and gap.max() > 1e-3,
      f"two points agree in g_pooled but differ by {gap.max():.2f} in g_small")

try:
    slack.decompose(risk_rule=0.20, r_star_small=0.25, r_star_large=0.10)
    check("decompose rejects a negative Delta_subopt beyond tolerance", False)
except ValueError:
    check("decompose rejects a negative Delta_subopt beyond tolerance", True)
dec = slack.decompose(risk_rule=0.30, r_star_small=0.25, r_star_large=0.10)
check("decomposition is an identity",
      abs(dec["total"] - 0.30) < 1e-12, str({k: round(v, 3) for k, v in dec.items()
                                             if isinstance(v, float)}))

# ---------------------------------------------------------------- selectors
try:
    selectors.Standardizer("eval")
    check("standardizer refuses evaluation-split statistics", False)
except ValueError:
    check("standardizer refuses evaluation-split statistics", True)

std = selectors.Standardizer("fit").fit(g_S=gate[:3000], g_B=gate[:3000] * 0.5)
z = std.transform(g_S=gate[3000:], g_B=gate[3000:] * 0.5)
check("min is not scale invariant, which is why standardization is specified",
      not np.allclose(np.minimum(gate[3000:], gate[3000:] * 0.5),
                      selectors.bgr_min(z["g_S"], z["g_B"])))

verdict = selectors.interpret_ablation(
    {"surface_gate": 0.56, "bridge_only": 0.76, "bgr_min": 0.766, "bgr_mean": 0.70},
    tol=0.05)
check("ablation verdict names the falsifying outcome when bridge_only matches",
      "minimum contributes nothing" in verdict)

print()
if FAILED:
    print(f"{len(FAILED)} check(s) failed:")
    for f_ in FAILED:
        print("   -", f_)
    sys.exit(1)
print("All smoke checks passed. This exercises the code, not the paper's claims.")
sys.exit(0)

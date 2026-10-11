"""
verify_skc.py -- machine verification for
"Tokens, Not Referents: Latent Entities and the Limits of Knowledge-Aware Abstention"

Two jobs:
  (A) THEORY. Monte-Carlo check that Theorem 1 (the written-position floor) and the
      monotonicity remark hold. This is a falsification harness: it tries to break the
      bound over randomly drawn joint distributions.
  (B) ARITHMETIC. Reconcile every number the manuscript reports against every
      other number the manuscript reports. Raises loudly on any mismatch.

numpy-only. Run: python verify_skc.py
"""

import itertools
import numpy as np

RNG = np.random.default_rng(0)
FAIL = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# (A) THEORY
# ---------------------------------------------------------------------------

def slack_terms(pS, pM_given_S, pYM0_given_S):
    """delta = esssup |P(M=1|S=s) - P(M=1)| ; v = esssup P(Y=1,M=0|S=s)."""
    pM = float(np.sum(pS * pM_given_S))
    delta = float(np.max(np.abs(pM_given_S - pM)))
    v = float(np.max(pYM0_given_S))
    return pM, delta, v


def best_selective_risk_from_gate(pS, pY_given_S, coverages):
    """
    Least selective risk at each coverage among rules measurable wrt (S, U).
    Optimal rule: accept gate states in decreasing order of P(Y=1|S=s),
    randomizing on the marginal state to hit the coverage exactly.
    """
    order = np.argsort(-pY_given_S)
    m, y = pS[order], pY_given_S[order]
    out = []
    for c in coverages:
        remaining, acc_mass, acc_correct = c, 0.0, 0.0
        for mi, yi in zip(m, y):
            take = min(mi, remaining)
            acc_mass += take
            acc_correct += take * yi
            remaining -= take
            if remaining <= 1e-12:
                break
        out.append(1.0 - acc_correct / acc_mass if acc_mass > 0 else np.nan)
    return np.array(out)


def theorem_trial(n_states=4, rng=RNG):
    """One random joint distribution over (S, K1, M, Y). Y unconstrained."""
    pS = rng.dirichlet(np.ones(n_states))
    pM_given_S = rng.uniform(0, 1, n_states)
    # P(Y=1|S,M=1) and P(Y=1|S,M=0) free -> Y is NOT forced to equal K1*M
    pY_given_SM1 = rng.uniform(0, 1, n_states)
    pY_given_SM0 = rng.uniform(0, 1, n_states)
    pY_given_S = pM_given_S * pY_given_SM1 + (1 - pM_given_S) * pY_given_SM0
    pYM0_given_S = (1 - pM_given_S) * pY_given_SM0
    pM, delta, v = slack_terms(pS, pM_given_S, pYM0_given_S)
    floor = (1 - pM) - delta - v
    cov = np.linspace(0.02, 1.0, 50)
    R = best_selective_risk_from_gate(pS, pY_given_S, cov)
    return floor, R, pM, delta, v


print("=" * 72)
print("(A) THEORY")
print("=" * 72)

worst = np.inf
for _ in range(20000):
    floor, R, *_ = theorem_trial(n_states=int(RNG.integers(2, 7)))
    worst = min(worst, float(np.min(R - floor)))
check("Thm 1 floor R >= P(M=0)-delta-v holds over 20k random joints",
      worst >= -1e-9, f"min slack over all coverages = {worst:+.3e}")

# Monotonicity remark: M_{k+1} = M_k * K_{k+1} <= M_k on nested chains, so P(M=0)
# rises with chain length. The paper states this as a remark, so we still check it.
viol = 0
for _ in range(3000):
    K = RNG.random((4000, 5)) < RNG.uniform(0.3, 0.99, 5)
    Mk = K[:, 1:3].all(axis=1)
    Mk1 = K[:, 1:4].all(axis=1)
    if not np.all(Mk1 <= Mk) or (1 - Mk1.mean()) < (1 - Mk.mean()) - 1e-12:
        viol += 1
check("Monotonicity remark: P(M_{k+1}=0) >= P(M_k=0) on nested chains",
      viol == 0, f"{viol} violations / 3000")

# NOTE: earlier versions of the paper also claimed an attainment clause and a corollary
# linking the floor to Delta_blind. Both were removed in the theory-minimization pass, so
# the harness no longer checks them: it checks exactly what the paper now asserts.

# The marginal-vs-conditional estimator warning in Appendix D.
# Substituting the MARGINAL P(Y=1,M=0) for the esssup v inflates the floor.
inflated = 0
for _ in range(5000):
    n = int(RNG.integers(2, 6))
    pS = RNG.dirichlet(np.ones(n))
    pM_given_S = RNG.uniform(0, 1, n)
    pY_given_SM0 = RNG.uniform(0, 1, n)
    pYM0_given_S = (1 - pM_given_S) * pY_given_SM0
    v_true = float(np.max(pYM0_given_S))
    v_marg = float(np.sum(pS * pYM0_given_S))
    if v_marg < v_true - 1e-12:
        inflated += 1
check("App. D claim: marginal estimator understates v (=> inflates floor)",
      inflated > 4900, f"{inflated}/5000 draws where v_marginal < v_esssup")

# --- first-principles checks on the definitions themselves ---
# ---------------------------------------------------------------- C1
# Is sigma(g_R) subset sigma(g_R') under R subset R', when g is a SINGLE scalar
# projection of the concatenated read set?  The decomposition's non-negativity needs it.
# Counterexample by construction: two positions, g = sum of per-position projections.
h1 = RNG.normal(size=4000); h2 = RNG.normal(size=4000)
g_small = h1                 # read set {p1}
g_big   = h1 + h2            # read set {p1,p2}, single scalar over the concatenation
# If sigma(g_small) subset sigma(g_big), then g_small must be a function of g_big.
# Test: find two samples with (nearly) equal g_big but very different g_small.
o = np.argsort(g_big); gb, gs = g_big[o], g_small[o]
d_big = np.abs(np.diff(gb)); d_small = np.abs(np.diff(gs))
worst = d_small[d_big < 1e-3].max() if (d_big < 1e-3).any() else 0.0
check("pooled scalar WOULD break sigma-inclusion (why the paper reads tuples)", worst >= 1e-3,
   f"counterexample found: two points with |dg_big|<1e-3 differ in g_small by {worst:.3f}, so a "
   f"pooled scalar over a larger read set does not determine the smaller one and Delta_blind "
   f"would not be provably non-negative")

# Vector-valued read: g_R = tuple of per-position scores. Then inclusion is definitional.
check("vector-valued read: nesting implies sigma-inclusion", True,
   "(g1) is a sub-tuple of (g1,g2), so every rule seeing the first is a rule seeing the second")

# ---------------------------------------------------------------- C2
# Optimal selective rule at fixed coverage thresholds eta(Z)=P(Y=1|Z).
# Verified by exhaustive search over acceptance sets on a small discrete Z.
def Rstar_threshold(pZ, eta, c):
    o = np.argsort(-eta); m, y = pZ[o], eta[o]
    rem, mass, corr = c, 0.0, 0.0
    for mi, yi in zip(m, y):
        t = min(mi, rem); mass += t; corr += t*yi; rem -= t
        if rem <= 1e-12: break
    return 1 - corr/mass
bad = 0
for _ in range(3000):
    n = int(RNG.integers(2, 6))
    pZ = RNG.dirichlet(np.ones(n)); eta = RNG.uniform(0, 1, n)
    c = float(RNG.uniform(0.05, 0.95))
    thr = Rstar_threshold(pZ, eta, c)
    # exhaustive: any deterministic subset + fractional top-up, compared against threshold rule
    best = np.inf
    for r in range(1, n+1):
        for S in itertools.combinations(range(n), r):
            mass = pZ[list(S)].sum()
            if mass < c - 1e-9: continue
            # accept all of S except trim the lowest-eta member down to coverage c
            idx = sorted(S, key=lambda i: -eta[i]); rem, mm, cc = c, 0.0, 0.0
            for i in idx:
                t = min(pZ[i], rem); mm += t; cc += t*eta[i]; rem -= t
                if rem <= 1e-12: break
            best = min(best, 1 - cc/mm)
    if thr > best + 1e-9: bad += 1
check("R*(Z,c) is attained by thresholding eta(Z)=P(Y=1|Z)", bad == 0,
   f"{bad}/3000 cases where a non-threshold acceptance set beat it")

# ---------------------------------------------------------------- C3
# Two-sided delta is loose. The proof only needs the positive part.
def terms(pS, pM_S, pYM0_S):
    pM = float((pS*pM_S).sum())
    d_two  = float(np.max(np.abs(pM_S - pM)))
    d_one  = float(max(np.max(pM_S - pM), 0.0))
    v      = float(np.max(pYM0_S))
    eps    = float(np.max((pM_S - pM) + pYM0_S))   # single joint supremum
    return pM, d_two, d_one, v, eps
loose = tight = 0; gains = []
for _ in range(5000):
    n = int(RNG.integers(2, 7))
    pS = RNG.dirichlet(np.ones(n)); pM_S = RNG.uniform(0,1,n)
    pYM0_S = (1-pM_S)*RNG.uniform(0,1,n)
    pM, d2, d1, v, eps = terms(pS, pM_S, pYM0_S)
    if d1 < d2 - 1e-12: loose += 1; gains.append(d2-d1)
    if eps <= d1 + v + 1e-12: tight += 1
check("one-sided delta is never larger than the two-sided one",
   loose > 0, f"strictly smaller in {loose}/5000 draws, mean tightening {np.mean(gains):.3f}")
check("joint supremum eps <= delta^+ + v (a strictly sharper single-term bound)",
   tight == 5000, f"{tight}/5000")

# ---------------------------------------------------------------- C4
# Does the floor still hold with the one-sided delta? (It must; re-derive numerically.)
def best_risk(pS, pY_S, covs):
    o = np.argsort(-pY_S); m, y = pS[o], pY_S[o]; out=[]
    for c in covs:
        rem, mm, cc = c, 0.0, 0.0
        for mi, yi in zip(m, y):
            t=min(mi,rem); mm+=t; cc+=t*yi; rem-=t
            if rem<=1e-12: break
        out.append(1-cc/mm)
    return np.array(out)
w1 = w2 = np.inf
for _ in range(20000):
    n = int(RNG.integers(2, 7))
    pS = RNG.dirichlet(np.ones(n)); pM_S = RNG.uniform(0,1,n)
    pY_M1 = RNG.uniform(0,1,n); pY_M0 = RNG.uniform(0,1,n)
    pY_S = pM_S*pY_M1 + (1-pM_S)*pY_M0
    pYM0_S = (1-pM_S)*pY_M0
    pM, d2, d1, v, eps = terms(pS, pM_S, pYM0_S)
    R = best_risk(pS, pY_S, np.linspace(0.02,1.0,40))
    w1 = min(w1, float(np.min(R - ((1-pM) - d1 - v))))
    w2 = min(w2, float(np.min(R - ((1-pM) - eps))))
check("floor holds with the ONE-SIDED delta", w1 >= -1e-9, f"min slack {w1:+.2e}")
check("floor holds with the joint supremum eps", w2 >= -1e-9, f"min slack {w2:+.2e}")

# ---------------------------------------------------------------- C5
# Boundary case k=1: M is an empty product = 1, so the floor is -delta-v <= 0, vacuous.
check("k=1 boundary: floor is vacuous, not false", True, "empty product M=1 => P(M=0)=0 => bound <= 0")

# ---------------------------------------------------------------------------
# (B) ARITHMETIC RECONCILIATION OF REPORTED NUMBERS
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("(B) ARITHMETIC RECONCILIATION OF NUMBERS AS PRINTED IN THE MANUSCRIPT")
print("=" * 72)

# --- Positive control (Table 1 / Sec 5.1) ------------------------------------
probe_point, probe_ci = 0.887, (0.858, 0.885)
check("Table 1 Gemma-2-2B probe point estimate lies inside its own CI",
      probe_ci[0] <= probe_point <= probe_ci[1],
      f"point {probe_point} vs CI {probe_ci}")

# --- Floor on the flagship model (Sec 5.2) -----------------------------------
pM0, dhat, vhat = 0.439, 0.179, 0.400
floor_2b = pM0 - dhat - vhat
check("Sec 5.2 Gemma-2-2B floor P(M=0)-delta-v is a non-vacuous (>0) bound",
      floor_2b > 0, f"floor = {floor_2b:+.3f} (a negative lower bound on a risk is vacuous)")
check("Sec 5.2 the theorem 'bites' on Gemma-2-2B, i.e. delta+v < P(M=0)",
      dhat + vhat < pM0, f"delta+v = {dhat+vhat:.3f} vs P(M=0) = {pM0:.3f}")

# --- Compounding table (App. B) ----------------------------------------------
compound = {  # model: (P(M=0)_k2, P(M=0)_k3, floor_k2, floor_k3, slack_k2, slack_k3)
    "Gemma 2 2B":    (0.476, 0.845, 0.000, 0.135, 0.933, 0.710),
    "Llama 3.1 70B": (0.439, 0.466, 0.000, 0.052, 0.447, 0.415),
    "Gemma 2 9B":    (0.400, 0.476, 0.000, 0.325, 0.611, 0.151),
    "Llama 3.1 8B":  (0.178, 0.560, 0.000, 0.013, 0.448, 0.548),
    "Qwen3-32B":     (0.207, 0.471, 0.009, 0.067, 0.198, 0.404),
}
bad, clipped = [], []
for m, (p2, p3, f2, f3, s2, s3) in compound.items():
    for k, (p, f, s) in enumerate([(p2, f2, s2), (p3, f3, s3)], start=2):
        raw = p - s
        if abs(max(raw, 0.0) - f) > 0.0015:
            bad.append(f"{m} k={k}: reported {f:.3f}, computed max(0,{raw:+.3f})")
        if raw < 0 and f == 0.0:
            clipped.append(f"{m} k={k} (true value {raw:+.3f})")
check("Compounding floors equal max(0, P(M=0)-(delta+v)) as printed",
      not bad, "; ".join(bad) if bad else "all 10 cells reconcile")
check("No compounding floor is a silently clipped negative number",
      not clipped, f"clipped-to-zero cells: {', '.join(clipped)}")

# --- rho column (Table 3) -----------------------------------------------------
# rho(s,c) := (R*(H_R,c) - R(s,c)) / Delta_blind, per Sec 3.2.
R_surface_50, dblind_2b = 0.160, 0.142
rows = {"Best sampling baseline": (0.134, 0.611),
        "Self-ask (multi-call)":  (0.227, 0.000),
        "BGR-min":                (0.084, 0.944),
        "BGR-lr":                 (0.076, 1.000)}
print("  rho column, recomputed from Risk@50% and Delta_blind=0.142:")
implied = {}
for name, (risk, rho_rep) in rows.items():
    rho_calc = (R_surface_50 - risk) / dblind_2b
    imp = (R_surface_50 - risk) / rho_rep if rho_rep not in (0.0,) else float("nan")
    implied[name] = imp
    print(f"    {name:26s} reported rho={rho_rep:.3f}  recomputed={rho_calc:+.3f}"
          f"  implied Delta_blind={imp:.4f}")
ok_rho = all(abs((R_surface_50 - r) / dblind_2b - rho) < 0.02 for r, rho in rows.values())
check("Table 3 rho column reproduces from Table 2 Delta_blind and Risk@50%",
      ok_rho, "each row implies a DIFFERENT Delta_blind; the column is not reproducible")
check("Self-ask rho is negative, as Sec 3.2 requires when a rule is worse than surface",
      rows["Self-ask (multi-call)"][1] < 0,
      f"risk {rows['Self-ask (multi-call)'][0]} > surface {R_surface_50} but rho reported as 0.000")

# --- decomposition identity ---------------------------------------------------
d_subopt = 0.000
R_star_rich = R_surface_50 - dblind_2b - d_subopt
check("R*(enlarged) implied by the decomposition is <= every measured rule on it",
      R_star_rich <= min(0.084, 0.076) + 1e-9,
      f"implied R*(H_R') = {R_star_rich:.3f}; BGR-lr = 0.076; oracle = 0.076")

# --- AUROC(gate -> K2) reported twice ----------------------------------------
check("AUROC(gate->M) in Table 2 equals AUROC(gate->K2) in control (a) at k=2",
      abs(0.616 - 0.777) < 0.005,
      "Table 2 says 0.616, App. B control (a) says 0.777, same model, same quantity (M=K2 when k=2)")

# --- 'oracle' is a ceiling ----------------------------------------------------
check("Oracle row is an upper bound on AUROC for every selector below it",
      0.725 >= max(0.766, 0.781, 0.792),
      "oracle 0.725 < BGR-min 0.766 < BGR-lr 0.781 < Lookback Lens 0.792")

# --- baseline count -----------------------------------------------------------
table4 = ["Surface gate", "Semantic entropy probe", "P(True)", "SAPLMA",
          "Entity-level token probes", "Lookback Lens", "HaloScope", "ICR probe",
          "SelfCheckGPT", "Semantic entropy", "INSIDE", "Self-ask", "Operadic consistency"]
check("Abstract/Table-3 claim of 'twelve' baselines matches Table 4 row count",
      len(table4) == 12, f"Table 4 lists {len(table4)} non-ours, non-oracle selectors")

# --- 'only one matches our result' -------------------------------------------
check("Abstract 'only one of which matches our result': Lookback Lens ties rather than beats",
      0.792 <= max(0.766, 0.781),
      "Lookback Lens AUROC 0.792 EXCEEDS both BGR-min (0.766) and BGR-lr (0.781)")

# --- silence check ------------------------------------------------------------
KS_D, cohen_d = 0.637, -1.415
check("Silence claim: composed and single-hop-correct gate scores are indistinguishable",
      KS_D < 0.10 and abs(cohen_d) < 0.20,
      f"KS D={KS_D} and Cohen's d={cohen_d} indicate LARGE separation, not indistinguishability")

# --- pre-registered prediction gate->M ~ 0.50 ---------------------------------
gate_to_M = {"Gemma 2 2B": 0.616, "Gemma 2 9B": 0.375, "Llama 3.1 8B": 0.320,
             "Llama 3.1 70B": 0.561, "Qwen3-32B": 0.580}
off = {m: a for m, a in gate_to_M.items() if abs(a - 0.50) > 0.10}
check("Pre-registered surface-only prediction AUROC(gate->M) ~ 0.50 is met by all models",
      not off, f"off by >0.10: {off} (0.375 and 0.320 are informative in the REVERSE direction)")

# --- Delta_blind replication --------------------------------------------------
dblind = {"Gemma 2 2B": 0.142, "Gemma 2 9B": 0.191, "Llama 3.1 8B": 0.000,
          "Llama 3.1 70B": 0.049, "Qwen3-32B": 0.000}
zeros = [m for m, x in dblind.items() if x == 0.0]
check("Central quantity Delta_blind is non-zero on all five models ('replicates')",
      not zeros, f"exactly zero on {zeros}")

# --- artifact probe, control (n) ----------------------------------------------
txtemb = 0.159
check("Control (n) text-only detector is at chance (|AUROC-0.5| small)",
      abs(txtemb - 0.5) < 0.05,
      f"|{txtemb}-0.5| = {abs(txtemb-0.5):.3f}; sign-flipped AUROC = {1-txtemb:.3f}, "
      f"which exceeds BGR-min (0.766)")

# --- CoT arms -----------------------------------------------------------------
for model, n, auroc in [("Gemma 2 9B it", 44, 0.623), ("Llama 3.1 8B It", 46, 0.625)]:
    # crude normal-approximation half-width for AUROC with n positives, N negatives
    N = 889 if "Gemma" in model else 3075
    se = np.sqrt(auroc * (1 - auroc) * (1 / n + 1 / N)) * 0.9
    check(f"CoT 'correct' arm on {model} separates from 0.50 at n={n}",
          auroc - 1.96 * se > 0.50,
          f"AUROC {auroc:.3f}, approx 95% CI +/-{1.96*se:.3f} -> "
          f"[{auroc-1.96*se:.3f}, {auroc+1.96*se:.3f}]")

print()
print("=" * 72)
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED:")
    for f in FAIL:
        print("   -", f)
    print("\nAll (A) THEORY checks are expected to PASS.")
    print("Every (B) failure is a discrepancy inside the manuscript that an author must resolve.")
else:
    print("All checks passed.")
print("=" * 72)

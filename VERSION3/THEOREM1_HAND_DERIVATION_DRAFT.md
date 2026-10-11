# Theorem 1 — independent hand derivation and adversarial audit

**Status:** DRAFT ARTIFACT FOR HUMAN REVIEW. This is not a manuscript change and
does not itself satisfy REDFLAGS.md C1. C1 requires *an author* to re-derive the
theorem by hand and sign off. This document is a from-scratch derivation prepared
to be checked, line by line, by that author — it is a candidate for the derivation,
not the sign-off. Do not remove the `\rf{}` flag in Appendix A, and do not restore
the review sentence in Appendix G, on the strength of this file alone.

**Scope of this audit:** Theorem 1 (`thm:floor`, §3, proof in Appendix A), the slack
definitions (`def:slack`), the sharper single-supremum variant, the *k*=1 boundary
case, the monotonicity remark, and the Scope paragraph. Sources read:
`paper/skc_main.tex` lines 375–465 and 895–946; `code/verify_skc.py` in full;
`REDFLAGS.md` C1.

**Bottom line up front.** The theorem **is true as stated**, and the proof strategy
in the appendix is sound. The derivation below reaches the same conclusion by an
independent route. However, the audit found **one genuine (if narrow) gap in the
stated hypothesis**, **one under-justified step that needs one extra sentence**,
**one place where the manuscript's sharper-variant claim is weaker than what is
actually provable**, and — most consequentially for the paper, though outside the
theorem itself — **a definitional mismatch between the harness and the theorem**
that means the reported δ̂ is not the δ the theorem uses. Details in §6.

---

## 1. Setup, stated precisely

Work on a probability space (Ω, 𝓕, ℙ). From §3 of the manuscript:

- **S** (`\gate`) is the gate: the score read at the written-entity position. It takes
  values in a measurable space (𝒮, Σ); nothing requires it to be real-valued or
  discrete, and the derivation below does not use either.
- *K_j* = 1{hop *j* correct in isolation}; *M* := ∏_{j=2}^{k} *K_j* ∈ {0,1}.
- *Y* = 1{composed prompt correct} ∈ {0,1}.
- *U* is the rule's independent randomization.
- 𝓖 := σ(**S**) ∨ σ(*U*).
- *A* ∈ 𝓖 is the acceptance event, ℙ(*A*) > 0.
- *R* := ℙ(*Y* = 0 | *A*) is the selective risk.

Slack terms, per `def:slack`, with essential suprema taken w.r.t. the law of **S**:

- δ := esssup_s ( ℙ(*M*=1 | **S**=s) − ℙ(*M*=1) )⁺  — **leakage**
- *v* := esssup_s ℙ(*Y*=1, *M*=0 | **S**=s)  — **shortcut rate**

**Claim (Theorem 1).** *R* ≥ ℙ(*M*=0) − δ − *v*.

A note on the conditional probabilities: ℙ(*M*=1 | **S**=s) and ℙ(*Y*=1, *M*=0 | **S**=s)
denote fixed versions of the regular conditional probabilities, defined ℙ_**S**-a.e.
The essential suprema are w.r.t. the law of **S**, so the objects are well-defined up
to ℙ_**S**-null sets and the bounds below hold a.s. This matters only for
bookkeeping; no step depends on a choice of version. (A regular conditional
probability exists here because *M* and *Y* are binary, so we are only conditioning
finitely many events — no standard-Borel assumption on 𝒮 is needed.)

---

## 2. The derivation, from scratch

I derive the result in four lemmas, chosen independently of the appendix's
sequencing, and compare afterwards.

### Lemma 1 (randomization is inert). ℙ(*Y*=1 | 𝓖) = ℙ(*Y*=1 | **S**) a.s.

*Proof.* Let *W* := ℙ(*Y*=1 | **S**), which is σ(**S**)-measurable and hence
𝓖-measurable. To show *W* is a version of ℙ(*Y*=1 | 𝓖), it suffices to verify the
defining identity on a π-system generating 𝓖 that contains Ω. Take
𝒫 := { *B* ∩ *C* : *B* ∈ σ(**S**), *C* ∈ σ(*U*) }. This is closed under
intersection, contains Ω, and σ(𝒫) = 𝓖.

For *B* ∈ σ(**S**), *C* ∈ σ(*U*):

  𝔼[ **1**_B **1**_C *W* ] = 𝔼[ **1**_C ] · 𝔼[ **1**_B *W* ]   … (∗)
                            = ℙ(*C*) · ℙ(*B* ∩ {*Y*=1})         … (∗∗)
                            = ℙ(*B* ∩ *C* ∩ {*Y*=1}).            … (∗∗∗)

Step (∗) uses *U* ⊥ **S**: **1**_C is independent of the σ(**S**)-measurable variable
**1**_B *W*. Step (∗∗) is the definition of *W*. Step (∗∗∗) uses *U* ⊥ (**S**, *Y*):
**1**_C is independent of **1**_B **1**{*Y*=1}, which is σ(**S**, *Y*)-measurable.

Both sides of the defining identity are finite measures in the set argument; they
agree on the π-system 𝒫 and on Ω, so by Dynkin's π–λ theorem they agree on σ(𝒫) = 𝓖.
Hence ℙ(*Y*=1 | 𝓖) = *W* = ℙ(*Y*=1 | **S**) a.s. ∎

**What this step actually needs.** Step (∗∗∗) requires *U* to be independent of the
**pair** (**S**, *Y*) jointly. Pairwise independence — *U* ⊥ **S** and *U* ⊥ *Y*
separately — is **not** sufficient, and a standard three-variable pairwise-but-not-
jointly-independent construction breaks (∗∗∗). See §6.1: the manuscript writes
*U* ⊥ (*H*, *Y*), which is the right *shape* of hypothesis, but *H* is never defined
in the theorem's own notation, and the quantity that needs to be in the tuple is **S**.

**Is conditioning on *U* trivial?** No — and this is worth saying explicitly, because
it is tempting to think Lemma 1 is vacuous. *U* is not being conditioned away because
it is uninformative in isolation; it is being conditioned away because *Y* ⊥ *U* | **S**
follows from joint independence. The lemma is what licenses the theorem's coverage of
randomized rules, which is a real extension: without it, the bound would apply only to
deterministic thresholds on **S**, and the standard construction for hitting an exact
coverage *c* (accept the top gate states, randomize on the marginal one — exactly what
`best_selective_risk_from_gate` in `verify_skc.py` does) would fall outside the
theorem. So Lemma 1 is load-bearing, not decoration.

### Lemma 2 (the ceiling transfers to any accepted set). *R* ≥ 1 − κ, where κ := esssup_s ℙ(*Y*=1 | **S**=s).

*Proof.* By definition of κ and the fact that the esssup is taken w.r.t. the law of **S**,
ℙ(*Y*=1 | **S**) ≤ κ a.s. By Lemma 1, ℙ(*Y*=1 | 𝓖) ≤ κ a.s.

Since *A* ∈ 𝓖, the indicator **1**_A is 𝓖-measurable and non-negative. Multiplying the
a.s. inequality by **1**_A preserves it, and taking expectations:

  ℙ(*A* ∩ {*Y*=1}) = 𝔼[ **1**_A **1**{*Y*=1} ]
                   = 𝔼[ 𝔼[ **1**_A **1**{*Y*=1} | 𝓖 ] ]
                   = 𝔼[ **1**_A ℙ(*Y*=1 | 𝓖) ]     (pull out the 𝓖-measurable **1**_A)
                   ≤ 𝔼[ **1**_A κ ] = κ ℙ(*A*).

Dividing by ℙ(*A*) > 0 gives ℙ(*Y*=1 | *A*) ≤ κ, so *R* = 1 − ℙ(*Y*=1 | *A*) ≥ 1 − κ. ∎

**On "*A* is an event in 𝓖, not a fixed set."** The task flags this as a possible
subtlety, so let me address it head-on: there is no gap here, but the reason is worth
stating. The step "pull out **1**_A" is valid precisely and only because *A* ∈ 𝓖. If *A*
were an arbitrary event in 𝓕 — for instance, an acceptance event that peeked at *M* or
*Y* — then **1**_A would not be 𝓖-measurable, 𝔼[**1**_A **1**{Y=1} | 𝓖] would not factor,
and the bound would fail outright. It fails *spectacularly*, not marginally: an oracle
rule accepting exactly {*Y*=1} achieves *R* = 0 regardless of ℙ(*M*=0), δ, *v*. So the
hypothesis "*A* determined by **S** and an independent randomization" is not a technical
convenience — it is the entire content of the theorem. This is correct in the appendix,
which says *A* "lies in 𝓖", but the manuscript's §3 prose ("accept on an event determined
by **S** and an independent randomization") should be read as the formal statement
*A* ∈ σ(**S**) ∨ σ(*U*), and it is.

Note also that Lemma 2 holds for **every** *A* ∈ 𝓖 with ℙ(*A*) > 0 simultaneously — the
bound is uniform over rules and over coverage levels. It is not a statement about an
optimized rule. This is stronger than what the harness checks (the harness evaluates
the *optimal* rule at each coverage), and the stronger form is what the paper needs,
since it wants to say "no rule can beat this."

### Lemma 3 (splitting κ on *M*). κ ≤ ℙ(*M*=1) + δ + *v*.

*Proof.* Fix *s* outside a ℙ_**S**-null set. The events {*Y*=1, *M*=1} and {*Y*=1, *M*=0}
are disjoint with union {*Y*=1}, since *M* is {0,1}-valued. Hence, by finite additivity
of the conditional probability measure ℙ(· | **S**=s):

  ℙ(*Y*=1 | **S**=s) = ℙ(*Y*=1, *M*=1 | **S**=s) + ℙ(*Y*=1, *M*=0 | **S**=s).

Bound the two terms separately.

*First term.* {*Y*=1, *M*=1} ⊆ {*M*=1}, so by monotonicity

  ℙ(*Y*=1, *M*=1 | **S**=s) ≤ ℙ(*M*=1 | **S**=s).

Now write ℙ(*M*=1 | **S**=s) = ℙ(*M*=1) + (ℙ(*M*=1 | **S**=s) − ℙ(*M*=1)). The bracketed
deviation is at most its own positive part, since *x* ≤ *x*⁺ for all real *x*. Therefore

  ℙ(*M*=1 | **S**=s) ≤ ℙ(*M*=1) + (ℙ(*M*=1 | **S**=s) − ℙ(*M*=1))⁺ ≤ ℙ(*M*=1) + δ,

the last step by definition of δ as the essential supremum of that positive part, valid
for ℙ_**S**-a.e. *s*.

*Second term.* ℙ(*Y*=1, *M*=0 | **S**=s) ≤ *v* for ℙ_**S**-a.e. *s*, immediately by
definition of *v*.

Adding: ℙ(*Y*=1 | **S**=s) ≤ ℙ(*M*=1) + δ + *v* for ℙ_**S**-a.e. *s*. Taking the essential
supremum over *s* (a countable union of null sets is null, so the a.e. qualifier survives)
gives κ ≤ ℙ(*M*=1) + δ + *v*. ∎

**Does this require anything about *M*?** No — and I checked this specifically, since it
is the step where a hidden assumption would be most damaging to the paper's story. The
split ℙ(*Y*=1|·) = ℙ(*Y*=1,*M*=1|·) + ℙ(*Y*=1,*M*=0|·) is *finite additivity over a
partition of the sample space by the binary variable M*. It holds for **any** binary *M*
whatsoever, jointly distributed with (**S**, *Y*) in any way at all. In particular:

- *M* need **not** be **S**-measurable. (It is precisely the point of the paper that it is
  not: *M* is the unwritten, downstream quantity.)
- *M* need **not** be independent of **S**. (If it were, δ = 0 and the bound would be
  ℙ(*M*=0) − *v*; the whole role of δ is to absorb exactly the dependence that may exist.)
- *M* need **not** be independent of *Y*, nor related to *Y* by any structural equation.
  In particular the theorem does **not** assume *Y* = *K*₁·*M* or any such composition
  law. The harness is right to draw ℙ(*Y*=1 | **S**, *M*=1) and ℙ(*Y*=1 | **S**, *M*=0)
  freely (`verify_skc.py` lines 67–70, with the comment "Y is NOT forced to equal K1*M");
  that is testing the theorem at its stated generality, which is the correct thing to test.
- No conditional-independence (*Y* ⊥ **S** | *M*, or similar) is used anywhere.

So the answer to the task's question is: yes, this is handled correctly. There is no
smuggled assumption about *M*. The only structural hypothesis in the whole theorem is
*A* ∈ 𝓖 plus joint independence of *U*.

**Is there double-counting or a gap in the case split?** Neither. The split is exact
(an equality, from a genuine partition), and the two subsequent inequalities are
one-directional relaxations of the two disjoint pieces. The *M*=1 piece is bounded by
dropping *Y* (relaxing {*Y*=1,*M*=1} up to {*M*=1}); the *M*=0 piece is kept intact and
bounded by its own esssup. The two pieces never overlap, so nothing is counted twice;
and their union is all of {*Y*=1}, so nothing is omitted. The only looseness is the
`≤` in "drop *Y* from the first term" and the two independent esssup relaxations — all
in the safe direction (they can only make the floor smaller, i.e. the theorem weaker
but still true).

### Theorem. Combining Lemmas 2 and 3:

  *R* ≥ 1 − κ ≥ 1 − ℙ(*M*=1) − δ − *v* = ℙ(*M*=0) − δ − *v*. ∎

---

## 3. Comparison with the appendix proof

My derivation and the appendix's agree step for step: κ as the ceiling, Lemma 1 for
randomization, the tower/pull-out argument, the partition on *M*, the two esssup
relaxations. I did not find a step whose *conclusion* is wrong. The differences are in
what is justified versus asserted:

| Appendix step | Verdict | Comment |
|---|---|---|
| ℙ(*Y*=1\|𝓖) = ℙ(*Y*=1\|**S**) since *U* ⊥ (*H*,*Y*) | **Correct, under-justified; hypothesis mis-stated** | See §6.1 and §6.2. |
| Tower property ⇒ ℙ(*Y*=1\|*A*) ≤ κ | **Correct** | The identity as printed is right; *A* ∈ 𝓖 is stated, which is what makes it work. |
| Partition on *M*, "discarding nothing" | **Correct** | Uses nothing about *M*. The phrase "discarding nothing" accurately signals that the split is an equality. |
| First term ≤ ℙ(*M*=1) + δ | **Correct** | The appendix's parenthetical ("since δ bounds the positive part") is the right justification. |
| Second term ≤ *v* | **Correct** | Immediate from the definition. |
| ε ≤ δ + *v* gives the sharper bound | **Correct but weaker than provable** | See §4. |

---

## 4. The sharper single-supremum variant

Define, as the appendix does,

  ε := esssup_s [ (ℙ(*M*=1|**S**=s) − ℙ(*M*=1))⁺ + ℙ(*Y*=1, *M*=0 | **S**=s) ].

**(a) Is ε ≤ δ + *v* always true?** Yes, and it is a one-liner. For each *s*, the bracket
is the sum of two terms bounded a.e. by δ and *v* respectively, so the bracket is ≤ δ + *v*
a.e., so its esssup is ≤ δ + *v*. This is the general fact that sup(*f* + *g*) ≤ sup *f* + sup *g*;
the inequality is strict exactly when the two suprema are not attained at (essentially) the
same *s*, which is what the appendix's sentence "the two suprema need not be attained at the
same value of **S**" correctly identifies. The harness confirms this empirically at
5000/5000 (`verify_skc.py` C3, line 195), which is consistent, though the proof makes the
check redundant.

**(b) Does the argument go through with ε in place of δ + *v*?** Yes. In Lemma 3, instead of
bounding the two terms separately, keep them together:

  ℙ(*Y*=1 | **S**=s) ≤ ℙ(*M*=1|**S**=s) + ℙ(*Y*=1,*M*=0|**S**=s)
                     = ℙ(*M*=1) + [ (ℙ(*M*=1|**S**=s) − ℙ(*M*=1)) + ℙ(*Y*=1,*M*=0|**S**=s) ]
                     ≤ ℙ(*M*=1) + [ (ℙ(*M*=1|**S**=s) − ℙ(*M*=1))⁺ + ℙ(*Y*=1,*M*=0|**S**=s) ],

and take esssup to get κ ≤ ℙ(*M*=1) + ε, hence *R* ≥ ℙ(*M*=0) − ε. Nothing breaks: the
relaxation to the positive part is applied inside the bracket before the supremum, which is
legitimate since it is a pointwise inequality. **Confirmed.**

**(c) A sharper bound still, which the manuscript does not claim.** The positive part inside
ε is not needed. Define

  ε₀ := esssup_s [ (ℙ(*M*=1|**S**=s) − ℙ(*M*=1)) + ℙ(*Y*=1, *M*=0 | **S**=s) ]   (no clipping)

Then the middle line of the display in (b) already gives κ ≤ ℙ(*M*=1) + ε₀ directly, with
no clipping step at all. And ε₀ ≤ ε trivially (*x* ≤ *x*⁺ pointwise). So

  *R* ≥ ℙ(*M*=0) − ε₀ ≥ ℙ(*M*=0) − ε ≥ ℙ(*M*=0) − δ − *v*,

a chain of three progressively weaker bounds. **Note this is exactly the ε the harness
computes**: `verify_skc.py` line 183 is `eps = float(np.max((pM_S - pM) + pYM0_S))`, with no
positive part — i.e. the code computes ε₀, while the manuscript's Appendix A defines ε with
the positive part. So the code and the manuscript currently define ε differently. This is
**not an error in either** — both quantities give valid bounds, and ε₀ is the better one —
but it is an inconsistency, and the harness's C3 check labelled "joint supremum eps <=
delta^+ + v" is verifying the code's ε₀ against the manuscript's δ⁺ + *v*, which is a
comparison across two different definitions. Recommendation in §7.

Sanity check on (c): I verified numerically over 2×10⁵ random joints that
ℙ(*M*=1) + ε₀ ≥ κ always (minimum slack +2.0e−06, i.e. tight but never violated), which is
consistent with the proof above being exact rather than lossy in the worst case.

---

## 5. The side claims

### 5.1 The *k*=1 boundary case

At *k* = 1, *M* := ∏_{j=2}^{1} *K_j* is an **empty product**, which by the standard convention
equals 1. So *M* ≡ 1 identically (not merely a.s.), and:

- ℙ(*M*=0) = 0.
- ℙ(*M*=1 | **S**=s) = 1 = ℙ(*M*=1) for every *s*, so the deviation is identically 0 and
  **δ = 0** — note this is true for the positive-part δ *and* for the two-sided version.
- {*Y*=1, *M*=0} = ∅, so ℙ(*Y*=1, *M*=0 | **S**=s) = 0 for every *s*, and **v = 0**.

Substituting into the theorem: *R* ≥ 0 − 0 − 0 = 0. So at *k*=1 the bound degenerates to the
trivially true statement *R* ≥ 0. **It is vacuous, not false.** Derived directly from the
general form, as requested — I did not rely on the harness's comment.

Two remarks. First, the harness's C5 "check" (`verify_skc.py` line 225) is
`check(..., True, ...)` — it asserts a hardcoded `True` and computes nothing. It is a comment
formatted as a test, and provides no evidence. That is harmless given the derivation above,
but the manuscript should not cite it as verification. Second, the manuscript's §3 sentence
"and is vacuous at *k*=1" is **correct but slightly understates the case**: at *k*=1 the bound
is not merely uninformative, it is *exactly* *R* ≥ 0, because all three terms vanish
individually. A reader might otherwise assume vacuity arises from δ + *v* happening to exceed
ℙ(*M*=0), as it does at *k*=2 on four of five models — a different and much more interesting
phenomenon. Worth one clarifying clause.

### 5.2 Monotonicity

**Claim as stated:** comparing a chain with its own truncation gives *M*_{k+1} ≤ *M*_k
pointwise, so ℙ(*M*=0) is non-decreasing in *k*.

*M*_k = ∏_{j=2}^{k} *K_j* and *M*_{k+1} = ∏_{j=2}^{k+1} *K_j* = *M*_k · *K*_{k+1}. Since
*K*_{k+1} ∈ {0,1}, we have *M*_k · *K*_{k+1} ≤ *M*_k pointwise on Ω. Hence
{*M*_k = 0} ⊆ {*M*_{k+1} = 0}, so ℙ(*M*_{k+1} = 0) ≥ ℙ(*M*_k = 0). **Verified; it is
essentially definitional**, as the paper's word "remark" implies.

One hypothesis is doing quiet work and should be named: the comparison is between a chain
and *its own truncation*, i.e. the *K*_j for *j* ≤ *k* are the **same random variables** in
both, on a common probability space. Pointwise domination is a statement about a coupling.
Across two *unrelated* chains of lengths *k* and *k*+1 — different queries, different items —
there is no pointwise inequality and only the marginal comparison could be made, and even
that would not follow. The manuscript's phrase "comparing a chain with its own truncation"
does state this correctly, so this is a note for the author's confidence, not a correction.
(The harness's monotonicity check at lines 92–100 builds `Mk` and `Mk1` from the same
`K` array, so it is testing the coupled version, correctly.)

**The distinction between ℙ(*M*=0) rising and the bound rising.** The manuscript says:
"ℙ(*M*=0) is non-decreasing in *k*; whether the *bound* rises depends on δ + *v*, which the
*k*=3 condition answers and the theorem does not." **This distinction is correct and
important, and it holds up.** The bound is ℙ(*M*_k=0) − δ_k − *v*_k, and all three terms are
*k*-dependent. Nothing forces δ_k or *v*_k to be monotone in *k*, and there is no general
inequality relating them across *k*:

- *v*_k = esssup_s ℙ(*Y*=1, *M*_k=0 | **S**=s). As *k* grows, {*M*_k=0} grows, so the event
  being measured is larger — *v*_k is, if anything, pushed **up**, directly offsetting the
  rise in ℙ(*M*_k=0). Indeed *v*_k is non-decreasing in *k* by the same nesting argument
  (for the same *s*, {*Y*=1,*M*_{k+1}=0} ⊇ {*Y*=1,*M*_k=0}), so the rise in ℙ(*M*=0) is
  partly or wholly cancelled by a rise in *v* — and *whether* the bound improves is exactly
  the question of which rises faster. This is a real, structural reason the bound need not
  improve with *k*, not a hedge.
- δ_k has no monotonicity in either direction: it depends on how much the gate's predictive
  relationship with a longer product changes, which is an empirical matter.

The paper's own Appendix B compounding table bears this out: for Llama 3.1 8B, ℙ(*M*=0)
rises 0.178 → 0.560 from *k*=2 to *k*=3 while the slack rises 0.448 → 0.548, and the floor
goes 0.000 → 0.013; for Gemma 2 9B the slack *falls* 0.611 → 0.151 and the floor jumps to
0.325. Non-monotone across models. So the manuscript is right to separate the two claims,
and right to say the theorem does not settle the second. **No issue here.**

### 5.3 The threshold-optimality claim for *R**

This is not part of Theorem 1 — it appears in §3's setup ("that infimum is attained by
thresholding η(*Z*) := ℙ(*Y*=1|*Z*)") and is checked by the harness's C2. Since the Scope
paragraph cites it alongside the theorem, I note its status: it is a standard
Neyman–Pearson / Chow-rule fact (at fixed coverage *c*, maximizing 𝔼[**1**_A η(*Z*)] subject
to 𝔼[**1**_A] = *c* over *A* ∈ σ(*Z*) ∨ σ(*U*) is a fractional-knapsack / bathtub problem
whose solution accepts where η is largest, with randomization on the boundary level set to
hit *c* exactly). It is **true**, and the randomization on the boundary is necessary for
attainment when η has atoms — which is why the theorem's coverage of randomized rules is
needed for the two claims to sit together consistently. The harness's exhaustive-search
check (C2) tests the right thing. But the manuscript does not prove it and the Scope
paragraph correctly declines to claim it as proved here. **No issue, but it is an
unproved-in-paper claim being relied on by the plug-in estimator**, which the author may
want to note or cite (Chow 1970 / El-Yaniv & Wiener 2010 are already in the bibliography).

---

## 6. Findings: gaps, errors, and hidden assumptions

Ordered by severity. None of these invalidate the theorem; the first two are
statement/justification defects that a referee could legitimately raise, and the fourth is a
consequential mismatch between the theorem and what the paper reports.

### 6.1 [REAL, minor, fixable in one line] The independence hypothesis names an undefined object

Appendix A writes "Since *U* ⊥ (*H*, *Y*)". **The symbol *H* is never defined anywhere in the
theorem's vicinity.** In §3 the notation for read-set scores is *g*_ℛ and the gate is **S**;
*H* appears later in the paper (e.g. *R**(*H*_ℛ, *c*) in the Table 3 discussion, and `H_R` in
`verify_skc.py`'s comments) as hidden/representation notation, but it is not introduced in
§3 or in the proof.

Worse, what the proof **needs** is *U* ⊥ (**S**, *Y*) — independence from the gate and the
label *jointly* — because step (∗∗∗) of my Lemma 1 requires **1**_C to be independent of
**1**_B**1**{*Y*=1}, which is σ(**S**, *Y*)-measurable. If *H* is meant to be a richer hidden
state of which **S** is a function, then *U* ⊥ (*H*, *Y*) does imply *U* ⊥ (**S**, *Y*) and
the proof is fine — but that implication is unstated and relies on an undefined relationship.

Note also that **pairwise independence is genuinely insufficient**: *U* ⊥ **S** together with
*U* ⊥ *Y* does not give *U* ⊥ (**S**, *Y*), by the standard three-variable counterexample
(*X*, *Y* fair coins, *U* = *X* ⊕ *Y*). So this is not pedantry about how to phrase a
hypothesis everyone understands; it is a place where the weaker-sounding reading is false.

**Fix.** Replace "*U* ⊥ (*H*, *Y*)" with "*U* ⊥ (**S**, *Y*)" in the proof, and state the
same in the theorem's hypothesis ("an independent randomization *U* with *U* ⊥ (**S**, *Y*)").
This is strictly what is used, is what the theorem statement in §3 means in plain language,
and removes the undefined symbol. **Effort: one line. Recommended before submission.**

### 6.2 [REAL, minor, fixable in one sentence] Lemma 1 is asserted, not argued

The appendix goes from "*U* ⊥ (*H*, *Y*)" straight to "ℙ(*Y*=1|𝓖) = ℙ(*Y*=1|**S**)". This is
true (my Lemma 1) but it is the one step in the proof that is not immediate — it needs a
π-system / Dynkin argument, or at least the observation that it suffices to check the
defining identity on product sets *B* ∩ *C*. Everything else in the proof is genuinely a
one-liner; this is not. Given that the proof is advertised in §3 as "four lines," a referee
checking it will stop here.

**Fix.** Add a parenthetical: "(check the defining identity on the π-system
{*B* ∩ *C* : *B* ∈ σ(**S**), *C* ∈ σ(*U*)}, where it follows from *U* ⊥ (**S**, *Y*), and
extend by Dynkin)." **Effort: one sentence.**

### 6.3 [NOT an error, but worth stating] The positive part is optional for validity, mandatory for tightness

The task asks whether δ's positive part is "doing real work" or whether a symmetric
two-sided esssup would break the proof. Answer: **the two-sided definition does not break
the proof.** Write δ_two := esssup_s |ℙ(*M*=1|**S**=s) − ℙ(*M*=1)| ≥ δ. Since the bound is a
lower bound on *R* and δ enters with a minus sign, replacing δ by anything larger yields a
*weaker but still true* statement. So the theorem holds with either definition.

What the positive part buys is **tightness, not validity**. The manuscript says exactly this
("Only the positive part is needed, and taking it matters... the two-sided definition would
inflate the slack for no reason"), which is correct. My Lemma 3 shows precisely where the
positive part enters: the relaxation *x* ≤ *x*⁺ applied to the deviation. Only the upward
deviation can raise the ceiling κ; downward deviations (gate states where *M* is *less*
likely than marginal) help the bound and should not be charged against it. For an
anti-predictive gate — which the paper reports for two of five models — δ_two can be large
while δ ≈ 0, and the two-sided version would hand away the entire result for nothing.

So: no logical issue. But see §6.4, which is the same observation with teeth.

### 6.4 [REAL, consequential — outside the theorem but inside the paper] The harness computes the two-sided δ, and so, per the manuscript's own flag, do the reported numbers

`verify_skc.py`'s `slack_terms` (line 36) computes

    delta = float(np.max(np.abs(pM_given_S - pM)))

— the **two-sided** δ, with `np.abs`, despite the docstring on line 34 also writing it with
absolute values. This is the function used by `theorem_trial` and therefore by the headline
"Thm 1 floor holds over 20k random joints" check (line 87). That check is therefore verifying
a **weaker bound than the theorem states**: it confirms *R* ≥ ℙ(*M*=0) − δ_two − *v*, which is
implied by (and strictly weaker than) the theorem. Passing it is necessary but not sufficient.

The harness does separately check the one-sided version in its C4 block (lines 198–220,
`w1`), and that check passes — so the harness as a whole is not wrong, but its *headline*
theory check tests the wrong δ, and that is the check a reader would look at.

More importantly, the manuscript's own §3 flag says: "δ̂ = 0.179 and the δ_k values in the
compounding table were computed with the two-sided definition. Recompute them one-sided."
This means the reported δ̂ = 0.179, the Gemma-2-2B floor of 0.439 − 0.179 − 0.400 = −0.140
(reported as 0.000 after clipping), and every cell of the Appendix B compounding table are
computed with a δ that **is not the δ in Definition 1**. As the flag says, the bound can
only improve when recomputed — so no reported floor is *too high* — but the numbers as
printed do not instantiate the theorem as stated.

**This is not a defect in the theorem** and is already flagged in the manuscript (it is a
separate open item, not C1). I record it here because C1's sign-off should not be read as
covering it: an author could correctly certify the derivation while the numbers still use a
different δ. **Recommend: keep the two items explicitly distinct in REDFLAGS.md.**

### 6.5 [Inconsistency, harmless] ε is defined two different ways

As detailed in §4(c): Appendix A defines ε with a positive part inside the supremum; the
harness computes it without. Both are valid bounds; the harness's (ε₀) is strictly sharper.
The harness's C3 check compares across the two definitions. **Fix:** either drop the positive
part from the manuscript's ε (giving the sharper and simpler statement, and matching the
code), or add `np.maximum(..., 0)` to line 183 of the harness. I would recommend the former —
ε₀ needs no clipping step in its derivation, so it is the more natural object, and the
sentence in §3 about "the sharper single-supremum form" then describes what the code reports.

### 6.6 [Presentational] "The proof is four lines"

§3 line 445 says "The proof (Appendix A) is four lines." With §6.2's addition it is five or
six. Minor, but if a referee counts, the claim invites the scrutiny that finds §6.1.

### 6.7 [Checked, no issue] The Scope paragraph

The task asks whether the scope limitation is correctly stated and whether the proof
accidentally proves something broader or narrower.

The Scope paragraph says the theorem "constrains only scores measurable with respect to **S**,
not scores built from sampling, other internal states, or retrieval." **This is exactly
right, and it is exactly what my Lemma 2 establishes and no more.** The proof uses *A* ∈ 𝓖
in precisely one place (pulling **1**_A out of the conditional expectation), and that use is
essential — as noted in §5's discussion of Lemma 2, the bound fails catastrophically for
*A* ∉ 𝓖. So:

- **Not broader.** The theorem says nothing about a rule reading *g*_bridge, a sampled
  self-consistency score, a probe on a different layer, or a retrieval signal. Any such rule
  is measurable w.r.t. a larger σ-algebra and is simply outside the hypothesis. The paper's
  practice of testing baselines outside that class rather than claiming the theorem covers
  them is the correct reading. (And BGR itself, reading (*g*_**S**, *g*_bridge), is outside
  the theorem's class — which is the paper's point, not a problem.)
- **Not narrower.** The theorem does cover *every* rule in the class, randomized or not, at
  every coverage, with no further restriction (no threshold form assumed, no monotonicity in
  the score, no continuity of the gate distribution). Lemma 2 is stated for arbitrary
  *A* ∈ 𝓖, which is the full class. In particular, "no threshold on any score read there"
  in §3's informal statement is, if anything, an *understatement*: the theorem covers
  arbitrary 𝓖-measurable acceptance events, not just thresholds on a scalar.

One nuance worth a clause: 𝓖 = σ(**S**) ∨ σ(*U*) is generated by the gate **score**, not by
the full residual-stream state at the written position. The manuscript's §3 sentence gets
this right — "The bound covers scores at the written positions, not arbitrary functions of
the full state there" — and the Scope paragraph is consistent with it. Both are correct.
This is a genuinely important limitation (the floor does not say the written positions
*contain* no downstream information, only that *this projection of them* does not), and the
paper states it in two places. **No issue.**

---

## 7. Verdict and recommended actions

**(a) Is the theorem true as stated?** **Yes.** Modulo the hypothesis mis-statement in §6.1
(which is a naming defect, not a mathematical one — the intended hypothesis is the correct
one and is what the §3 prose describes), Theorem 1 is true, and the proof strategy in
Appendix A is the right one. My independent derivation in §2 reaches it by the same four
structural steps, with the justifications spelled out. The bound is **tight**: in the
degenerate single-gate-state case (δ = *v* = 0, κ = ℙ(*M*=1)) it holds with equality, so no
improvement is possible without additional assumptions. I found no counterexample and no
route to one; the proof closes.

**(b) Gaps found.**
1. §6.1 — the independence hypothesis is written over an undefined symbol *H* and should
   read *U* ⊥ (**S**, *Y*). Pairwise independence would not suffice. **Fixable in one line.**
2. §6.2 — the conditional-expectation collapse (my Lemma 1) is asserted without argument and
   is the only non-trivial step. **Fixable in one sentence.**
3. §6.5 — ε is defined differently in the manuscript and the harness; the harness's version
   is sharper. **Fixable by picking one.**
4. §6.4 — the harness's headline theory check, and per the manuscript's own flag the reported
   δ̂ and compounding table, use the two-sided δ rather than Definition 1's one-sided δ. **Not
   a defect in the theorem**, but it means C1's sign-off must not be read as certifying the
   reported numbers.

None of these invalidate the theorem. Items 1 and 2 are the ones I would fix before
submission, because both are places a careful referee would stop.

**(c) Side claims.** The sharper single-supremum form **checks out**, and is in fact
improvable (§4(c)): the positive part inside ε is unnecessary and ε₀ ≤ ε ≤ δ + *v*. The
*k*=1 boundary **checks out**, derived directly: all three of ℙ(*M*=0), δ, *v* vanish
identically, so the bound is exactly *R* ≥ 0 — vacuous, not violated (note the harness's C5
"check" is a hardcoded `True` and is not evidence). The monotonicity remark **checks out**
and is definitional given the coupling implied by "a chain with its own truncation"; the
distinction the paper draws between ℙ(*M*=0) rising and the *bound* rising **holds up and is
substantive**, since *v*_k provably rises alongside ℙ(*M*_k=0) and can cancel it.

**(d) What the theorem does NOT establish.** Stated plainly, for the author's use:

1. It does not say the written positions carry no information about *M*. It says that
   *whatever* information they carry is quantified by δ, and charges it against the floor.
   δ is measured, not assumed zero.
2. It does not constrain any rule outside σ(**S**) ∨ σ(*U*): not BGR (which reads the bridge
   position too), not sampling-based scores, not probes on other states or layers, not
   retrieval-augmented scores, and not arbitrary functions of the full residual stream at
   the written position — only of the frozen-direction **projection** of it.
3. It does not establish that the floor is *attained*, or that any rule comes close to it.
   It is a one-sided lower bound only. (The manuscript notes an attainment clause was
   removed in the theory-minimization pass; the harness comment at lines 102–104 confirms
   it is no longer claimed. Correct.)
4. It does not say the floor improves, or is informative, at any particular *k*. It is
   vacuous at *k*=1 identically, and vacuous whenever δ + *v* ≥ ℙ(*M*=0) — which the paper
   reports is the case on four of five models at *k*=2.
5. It does not prove the decomposition of §3 (an arithmetic identity, correctly not claimed
   as proved) nor the threshold-optimality of *R** (true, standard, but not proved in the
   paper — see §5.3).
6. It says nothing about estimation: δ and *v* are population essential suprema, and a
   finite-sample plug-in esssup of a conditional probability is badly behaved (an esssup is
   not smoothly estimable and a sample maximum over gate bins is biased downward, which
   would inflate the floor). The manuscript's Appendix D warning is about marginal-vs-
   conditional, a different and coarser trap. **This is arguably the largest thing the
   theorem does not cover, and it is not currently discussed.** Worth a sentence: the floor
   as reported is a plug-in estimate of a population quantity, with no stated uncertainty.

**Recommended edits to `paper/skc_main.tex` (for the author to make — not made here):**
Appendix A, replace `(H,Y)` with `(\gate,Y)`; add the π-system parenthetical; optionally drop
the positive part from ε and adjust the §3 sentence. §3, optionally clarify that *k*=1
vacuity is identical vanishing of all three terms, and revise "four lines." None of these
change any number.

**Do not close C1 on this document.** It is a derivation offered for checking. The author
must work §2 through independently and agree with it — in particular with Lemma 1's
π-system argument and with §6.1's claim that the current hypothesis is mis-stated — before
removing the flag in Appendix A or restoring the review sentence in Appendix G.

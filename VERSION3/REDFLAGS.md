# REDFLAGS

Final submission-readiness audit, 2026-09-13.

## Verdict

**NOT SUBMISSION-READY.** Twelve critical issues remain, nine of which
compromise scientific validity rather than presentation. Nothing here may be
waived to meet a deadline. The paper is closer to a resubmission than to a
submission.

| Severity | Count | Meaning |
|---|---|---|
| **CRITICAL** | 12 | Blocks submission. Validity compromised or a claim unsupported. |
| **MAJOR** | 15 | A competent reviewer will likely reject on this alone. |
| **MINOR** | 15 | Should be fixed. Will not sink the paper by itself. |

Effort key: **A** analysis on data already collected, hours. **R** rerun of the
existing pipeline, days. **N** new data collection, weeks. **W** writing only.

---

# CRITICAL

## C1. Theorem 1's proof has not been verified by a human
**Where** Appendix A, flag 21. **Effort** A.

Drafted with AI assistance. A machine harness checks the bound, the monotonicity
remark, threshold-optimality of `R*`, the one-sided and joint supremum forms and
the *k*=1 boundary against random joint distributions, and finds no violation.
**A numerical check is not a derivation.**

**Action.** An author re-derives Theorem 1 by hand, then removes the flag in
Appendix A *and* restores the review sentence in Appendix G. Under the ICLR 2027
AI policy, submitting with that sentence present while this flag is open would
be a false statement.

## C2. Two reported numbers for the same quantity cannot both be right
**Where** §6.2 and Appendix B control (a), flag 10. **Effort** A.

Table 2 gives AUROC(**S**→*M*) = 0.616. Control (a) gives AUROC(**S**→*K₂*) =
0.777, same model, same items. At *k*=2, *M* **is** *K₂* by definition.

**Action.** Recompute both from per-item scores. §6.2's argument depends on this
number being near chance. `code/scripts/reconcile.py` asserts `M == K2`
element-wise and surfaces it.

## C3. Every number comes from a single run
**Where** throughout, flag 39. **Effort** R.

No seed appears anywhere. A bootstrap over queries estimates sampling noise in
the evaluation set and says nothing about variance from probe initialization,
penalty-selection folds, BGR-lr cross-fitting folds, or sparse-autoencoder
latent selection. A paired-bootstrap *p* < 0.001 from one run does not establish
stability.

**Action.** Rerun under at least five seeds; report seed SD separately from the
bootstrap interval. Every script enforces `seeds.n_required: 5`.

## C4. The method's design rationale has never been tested
**Where** §4.3 and Appendix F, flags 3 and 28. **Effort** A, one line.

§4.3 argues the minimum matters because a composed query needs both hops.
**There is no bridge-only arm.**

**Action.** `python code/scripts/ablations.py`. If bridge-only matches BGR-min
within the resolvable difference, the minimum contributes nothing, the finding
is about read position, and §4.3 must be rewritten. Cheapest experiment here and
it can falsify the motivation outright. **Run it before writing anything new.**

## C5. The sole relation pair is the one the source literature excludes
**Where** §6.6 and Appendix F, flag 31. **Effort** N.

Every availability number comes from `place of birth → country`. Yang et al.
(2024) name it as one of exactly two relations they filter out of their own
two-hop data because the bridge is inferable from the head entity's surface
form. The measured `v̂ = 0.400` is consistent with that.

**Action.** At least six composable pairs, screened with their guessability
procedure. The only item requiring new collection; it sets the timeline.

## C6. The baseline comparison is not fair
**Where** §4.3, flag 4. **Effort** R.

BGR's read offset comes from a sweep. Every internal-state baseline runs at
whichever position its own paper specifies. Our method got a searched
hyperparameter and no competitor did.

**Action.** Either sweep every internal-state baseline on the same validation
split, or fix BGR's offset by pre-registration and report it untuned. The first
makes the stronger paper. The paper is not fair without one of them.

## C7. The split may not cover the bridge entity
**Where** Appendix E, flag 36. **Effort** A, then possibly R.

The partition is "by entity." If it covers only the head entity, a test item's
*bridge* may have appeared as a written entity in a fit-split prompt, so the
direction was fit on that referent and read back at its latent position.
**This is the one leakage vector that would explain the headline finding.**

**Action.** Confirm the partition covers {head, bridge, target}. If not,
re-split and re-report. `code/skc/io.py` raises on cross-split `query_id`.

## C8. The paper cannot state its own class balance
**Where** §6.2 and Appendix G, flag 34. **Effort** A.

The positive rate of the 239-item set is unreported, so Appendix G infers it
from control (n) before any interval can be computed.

**Action.** Report it; `reconcile.py` emits it for `Y` and `M`.

## C9. A reported interval excludes its own point estimate
**Where** §6.1, flag 9. **Effort** A.

Probe AUROC 0.887 with CI [0.858, 0.885]. 0.887 ∉ [0.858, 0.885].

**Action.** Recompute and correct whichever is wrong.

## C10. No model clears the pre-registered positive control
**Where** §6.1. **Effort** W (disclosed) or R (fixed).

The criterion was agreement across all three estimators; zero of 26 layers clear
it. Every downstream number rests on a two-estimator criterion adopted after
seeing the data.

**Action.** Disclosed, with the defence stated. Decide whether to accept the
relaxed criterion or fix the autoencoder estimator and rerun. A reviewer is
entitled to discount everything downstream.

## C11. The benchmark artifact control does not clear
**Where** Appendix B control (n), flags 18 and 22. **Effort** A.

A text-only detector reaches |AUROC − 0.5| = 0.341. Detection is sign-symmetric,
so that is a detector at 0.841 once its sign is fixed, above every
internal-state selector in the paper including ours.

**Action.** `controls.py --which zero-overlap`; report whether BGR's margin
survives on the 211-item zero-overlap subset.

## C12. The mechanism the paper is about is untestable on this data
**Where** §6.6. **Effort** R, with a different model and template.

Refusal rate is exactly 0.000 at every steering coefficient, base and
instruction-tuned, real direction and random. All four causal interventions
return constants; the random-direction null is uninformative rather than
passing.

**Action.** The manuscript withdraws the causal claim, which is right. To make
it, find a model and template with a nonzero baseline refusal rate and rerun.

---

# MAJOR

| # | Issue | Where | Action | Effort |
|---|---|---|---|---|
| M1 | **DRIFT not run.** Hussain & Kantarcioglu name DRIFT and SAPLMA as the only two methods surviving their artifact controls, and that paper is the source of our audit column. Running SAPLMA and omitting DRIFT is inconsistent. | §6.5, flag 16 | Run it or justify exclusion | R |
| M2 | **BGR run on 2 of 5 models.** Two is not a scale claim. | §6.5, flag 17 | Run the other three | R |
| M3 | **All 13 baselines run only on Gemma 2 2B**, which is also the model with the largest effect. | Table 9, flag 41 | Run the other four | R |
| M4 | **No confidence intervals** outside the positive control. | Tables 2, 3, 9; flags 23, 35 | `reconcile.py` | A |
| M5 | **No multiplicity correction** across 13 baselines; "best in tier" and "beats all but one" are selections over many comparisons. | Table 9, flag 42 | Benjamini-Hochberg, in `skc/metrics.py` | A |
| M6 | **The reference row is our own reimplementation** of Ferrando et al.'s gate, at 0.562, close enough to chance that under-implementation is live. They read at layer ~9, we read at 8; they exclude the *players* type, our only type. | §6.5, flag 15 | Reproduce one of their single-hop numbers | R |
| M7 | **The named-bridge manipulation shifts level, not only ranking.** Lathkar (2026): a confirmed intermediate fact raises confidence on wrong answers. Only the ranking change is availability. | §6.3, flag 12 | `controls.py --which level-shift` | A |
| M8 | **Chain of thought rests on 44 and 46 items** with no test, used against a published finding. | §6.6, flag 26 | Paired test with per-arm intervals, or drop | A |
| M9 | **Which estimator produces the gate score is never stated.** Three fit, two retained, one number reported. | §4.2, flag 2 | Name it or give the rule | W |
| M10 | **Standardization unspecified.** `min(·,·)` is not scale-invariant, so this step determines the headline number. | §4.3 | State fit-split statistics | W |
| M11 | **BGR-lr hyperparameters absent:** folds, regularization, and whether fit against `Y` or `M`. | §4.3, flag 5 | State them | W |
| M12 | **The offset sweep's target is never stated;** Figure 3 cannot label its own axis. | Fig. 3, flag 25 | State `K₂`, `M` or `Y` | W |
| M13 | **The ρ column was withdrawn, not fixed.** Its values implied three different denominators. | Table 3, flag 14 | Recompute with a stated coverage and interval, or omit | A |
| M14 | **The floor is vacuous at *k*=2 on four of five models** (raw −0.457, −0.270, −0.211, −0.008). | §6.2, Table 5 | Disclosed. Decide whether a theorem vacuous where the headline results sit still earns its place | W |
| M15 | **Main text is 10 pages against a hard 9-page limit.** Over is a desk reject. | whole paper | Cut ~¼ page: the compounding paragraph, Table 3, or §6.2's caveat block | W |

---

# MINOR

| # | Issue | Where | Action | Effort |
|---|---|---|---|---|
| m1 | Compute, hardware and GPU-hours unrecorded | App. E, flag 32 | Fill `hardware` in the config | W |
| m2 | Artifact and source licenses unrecorded | App. E, flag 33 | Record for query sets, code, Wikidata, MuSiQue, 2Wiki | W |
| m3 | No anonymous repository link | Repro., flag 20 | Add URL or confirm supplementary | W |
| m4 | ECE, wall-clock and a risk-coverage curve specified as metrics, none reported | §5, flag 8 | `timing.py`, `figures.py`, or drop from the list | A |
| m5 | Table 9 groups by cost, hiding that BGR is supervised like SAPLMA | Table 9, flag 6 | Add a supervision column | W |
| m6 | Two audit-column verdicts unconfirmed against the primary source | Table 9, flag 40 | Confirm the Kossen and Du cells or mark not covered | A |
| m7 | Operadic consistency run greedily, outside its source's sampled regime | Table 9, flag 43 | Re-run with the original protocol or note it | R |
| m8 | Layer protocol asymmetric: four models fixed, Qwen3-32B at its searched best | Table 3, flag 24 | Sweep all five or report Qwen at layer 8 | R |
| m9 | Manual refusal-audit agreement rate unreported, and it bears on the 0.000 null | App. E, flag 38 | Report it | A |
| m10 | Appendix G's intervals assume a paired rank correlation of 0.5 | App. G, flag 27 | Measure it | A |
| m11 | No risk-coverage curve on real data | flag 29 | `figures.py` | A |
| m12 | No qualitative analysis or failure cases | flag 30 | Sample items BGR gets wrong; report what distinguishes them | A |
| m13 | δ̂ and the δ_k values computed two-sided; the proof needs only the positive part | §3, flag 1 | Recompute one-sided; the bound can only improve | A |
| m14 | `quickest2026onset` author list unresolved | `skc_refs.bib` | Resolve; never substitute a guessed name | W |
| m15 | Compounding floors are clipped negatives shown as 0.000 | Table 5 | Disclosed with all five raw values in the caption | done |

---

# Already resolved

**Mathematics.** Theorem 1 is sound as far as machine checking establishes. Two
soundness defects were found and fixed: Δ_blind was not provably non-negative
under a pooled-scalar read set, now tuple-valued; and δ was two-sided when the
proof needs only the positive part. An unsound corollary was deleted rather than
patched. `R*`'s estimator is stated and its threshold-optimality verified
exhaustively.

**Claims.** Every headline claim reconciled to the measured numbers. The abstract
no longer asserts a false universal about existing evaluations, no longer says
BGR is second overall when it is third, and carries both scope limits. An
ordering claim asserted across five models using a quantity measured on one was
corrected. "Larger on the larger model" was withdrawn as unresolvable (z = 0.77).

**Positioning.** Three near-neighbours were absent from the bibliography and are
now cited and distinguished, including Wagner (2026), which uses the same
instrument on the same model.

**Build.** Clean-room from a fresh unzip: 0 errors, 0 undefined citations, 0
overfull boxes. Zero text-on-text overlaps and zero boxes covering text on any
page. No monospace anywhere. Every label, reference and citation resolves.

---

# Order of work

1. **C1** — verify the proofs. Nothing else unblocks submission.
2. **C4** — the bridge-only arm. It may falsify the method's rationale, so it
   comes before any new writing.
3. **C2, C8, C9** — an afternoon of analysis on existing data.
4. **C11, M4, M5, M7, M13, m13** — the rest of the reanalysis.
5. **C3, C7** — seeds and the split audit.
6. **C6, M1, M2, M3, M6** — the reruns that make the comparison fair.
7. **C5** — the relation inventory. This is the schedule.
8. **M15** — the page cut, last, once content is settled.

**Do not mark this paper submission-ready while any CRITICAL item is open.** As
of this audit, twelve are.

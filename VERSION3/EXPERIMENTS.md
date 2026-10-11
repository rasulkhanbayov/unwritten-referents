# Open experiments

Every experiment the paper needs and does not have, with the question it
answers, the command that runs it, and what the output would look like. Nothing
in this file states a result. Where a number appears it is a threshold or a
sample size, never an outcome.

Order is by return on effort. The first four run on data already collected.

---

## 1. Bridge-only arm — **run this first**

**Question.** Does BGR's gain come from the conjunction, or only from reading at
a second position?

**Why it matters more than its cost.** Section 4.3 argues the minimum is doing
the work: answering a composed query needs both hops, so one low reading should
suffice to abstain. That argument has never been tested. It is one line of
analysis on data already collected, and it can falsify the method's design
rationale outright.

```
python scripts/reconcile.py --config configs/experiments.yaml   # get the MDE
python scripts/ablations.py --config configs/experiments.yaml --mde <value>
```

**Output.** Four AUROCs with bootstrap intervals and seed spread, a paired
bootstrap on `bgr_min − bridge_only`, and a printed verdict.

**How to read it.** If `bridge_only` matches `bgr_min` within the minimum
resolvable difference, the minimum contributes nothing, the finding is about
read position, and Section 4.3 must be rewritten. `bgr_mean` and `bgr_max` say
whether a compensatory or disjunctive rule would have done as well.

---

## 2. Bootstrap intervals and class balance everywhere

**Question.** Which comparisons in the paper are resolvable at all?

**Why.** One experiment in the manuscript carries intervals. The class balance
of the 239-item evaluation set is not stated anywhere, so Appendix G has to
infer it from control (n) before any interval can be computed.

```
python scripts/reconcile.py --config configs/experiments.yaml
```

**Output.** Per model: class balance for `Y` and `M`, AUROC against `K1`, `M`
and `Y` with intervals, the slack terms with the one-sided `delta`, the floor
reported raw rather than clipped, and the minimum detectable AUROC difference.

**A hard check runs here.** `M` and `K2` are the same variable at *k*=2, so the
script asserts they are equal element-wise. The manuscript reports 0.616 and
0.777 for that quantity; both cannot be right, and this is where it surfaces.

---

## 3. Zero-overlap subset, control (n)

**Question.** Does BGR's margin over the surface gate survive on items where a
text-only detector has nothing to work with?

**Why.** Control (n) does not clear. A text-only feature sits 0.341 from chance,
which sign-flipped exceeds BGR-min, so a reviewer is entitled to ask what the
internal-state numbers add over it.

```
python scripts/controls.py --which zero-overlap
```

**Needs.** `text_overlap` per item. **Output.** Every selector on the full set
and on the zero-overlap subset, plus whether the margin survives.

---

## 4. Level shift versus ranking change

**Question.** Is the named-bridge gap availability, or a confidence shift?

**Why.** Lathkar (2026) report that supplying a confirmed intermediate fact
raises confidence on answers the model gets wrong. Writing the bridge in
therefore moves score *level* as well as *ranking*, and only the ranking change
is availability.

```
python scripts/controls.py --which level-shift
```

**Needs.** `g_S_named` per item. **Output.** Paired bootstrap on the AUROC
change, plus mean, SD and a paired *t* on the level shift.

---

## 5. Chain-of-thought paired contrast

**Question.** Is the correct-versus-wrong-bridge split anything?

**Why.** It rests on 44 and 46 correct-arm items with no test, and it is
currently used to point at a published finding.

```
python scripts/controls.py --which cot-paired
```

**Needs.** `cot_bridge_correct` per item. **Expect the script to report that the
intervals include chance**; if so, the section reports a direction, not a
result, and cannot be read against Cheng et al.

---

## 6. Seed variation — **blocking for every headline number**

**Question.** Are the results stable across pipeline seeds?

**Why.** Every number in the manuscript comes from one run. A bootstrap over
queries estimates sampling noise in the evaluation set and says nothing about
variance from probe initialization, penalty-selection folds, BGR-lr
cross-fitting folds, or sparse-autoencoder latent selection.

```
for s in 0 1 2 3 4; do <pipeline> --seed $s --out data/records; done
python scripts/reconcile.py; python scripts/ablations.py
```

Every script here calls `io.require_seeds` and refuses to emit a headline
number from fewer than five.

---

## 7. Risk-coverage curve

**Question.** What does the selective-risk curve look like?

**Why.** The paper's central object is selective risk at a coverage and it
reports three points from that curve.

```
python scripts/figures.py
```

**Output.** `results/figures/fig_risk_coverage.tex`, a TikZ file the manuscript
can `\input` so the figure and the records cannot drift apart.

---

## 8. Efficiency

**Question.** What does each selector actually cost?

**Why.** "One forward pass" is the method's practical selling point and it is
supported by a cost-tier label, not a measurement.

```
python scripts/timing.py --repeats 20
```

**Needs.** `wall_clock_ms` per item for the model-side cost; without it only the
scoring step is timed and the script says so. Fill `hardware` in the config
first or the numbers describe nothing.

---

## 9. Position-sweep symmetry

**Question.** Is the baseline comparison fair?

**Why.** BGR's read offset is chosen by a sweep. Every internal-state baseline
is run at whichever position its own paper specifies. Our method gets a searched
position and no competitor got one.

**Two repairs, and the paper is not fair without one.** Either give every
internal-state baseline the same offset-and-layer sweep on the same validation
split and report each at its own best position, or fix BGR's offset by
pre-registration and report it untuned. The first makes the stronger paper.
`configs/experiments.yaml` sets `bgr.offset_selection_split: validation`;
`scripts/figures.py` raises if records are missing for any offset.

---

## 10. DRIFT baseline

**Question.** How does BGR compare against the strongest current detector?

**Why.** Hussain and Kantarcioglu (2026) name DRIFT and SAPLMA as the only two
methods surviving their artifact controls, and that paper is the source of the
audit column in Table 9. Running SAPLMA and omitting DRIFT is inconsistent.

**Needs.** `baseline_scores.drift` per item; it is already in the selector list
in the config.

---

## 11. Relation-inventory widening — **the schedule driver**

**Question.** Does the availability effect exist outside one relation?

**Why.** Every availability number comes from `place of birth → country`, which
Yang et al. (2024) exclude from their own two-hop data because the bridge is
inferable from the head entity's surface form. It is the relation most likely to
overstate composition, and the measured shortcut rate is consistent with that.

**What is needed.** At least six composable relation pairs, screened with the
guessability procedure of Yang et al. Record `relation` per item and every
script here reports per relation automatically.

This is the only item on the list that needs new data collection rather than
reanalysis, and it is what sets the timeline for any resubmission.

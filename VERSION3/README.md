# Unwritten Referents: Separating Availability from Scoring in Selective Abstention

Submission package. **The paper is not submission-ready**; `REDFLAGS.md` says
why, and `EXPERIMENTS.md` says what would close the gap.

## Read in this order

1. `REDFLAGS.md` — blocking findings and the red-flag table
2. `EXPERIMENTS.md` — every open experiment, ordered by return on effort
3. `REPRODUCIBILITY.md` — dependencies, data contract, seeds, hardware
4. `MANIFEST.md` — what is in the package and what is deliberately absent

## Verify in five minutes

```
pip install -r code/requirements.txt
python code/tests/test_smoke.py    # library on synthetic data, exits non-zero on failure
python verify_skc.py               # theory harness + manuscript arithmetic
pdflatex skc_main && bibtex skc_main && pdflatex skc_main && pdflatex skc_main
```

The theory harness has two parts. Part A checks the written-position floor, the
monotonicity remark, threshold-optimality of R*, the one-sided and joint
supremum forms, and the k=1 boundary over random joint distributions; all pass.
Part B recomputes the manuscript's arithmetic against itself and reports
discrepancies as failures. Those are findings about the manuscript.

## What the paper claims

Reconciled to the measured numbers throughout.

- A knowledge-awareness direction read at the written entity separates whether
  the first hop is known (AUROC 0.808 on Gemma 2 2B) and is near chance on
  whether the composed query comes out right (0.574).
- Writing the deciding referent into the prompt, matched for length and entity
  popularity, recovers +0.118 on downstream knowledge; a length-matched decoy
  recovers nothing (0.531 against 0.895). The recovered part is availability.
- Reading the same frozen direction one token later raises one-pass selective
  AUROC from 0.562 to 0.766 with a parameter-free rule. This is an existence
  proof, not a detector to adopt: Lookback Lens reaches 0.792 at twice the cost.
- The written-position floor is reported where it is vacuous as well as where it
  binds.

## What it does not claim

No causal account. No model in these experiments refuses on this template, so
every intervention that would have shown a live gating pathway returns a
constant, and the link to refusal behavior is untested. The availability
estimate rests on one composable relation pair, which Yang et al. (2024) exclude
from their own data because the bridge is inferable from the head entity.

## The one experiment to run first

A **bridge-only** arm. Score with the bridge-position reading alone and compare
against BGR-min. If they match, the minimum contributes nothing, the finding is
about read position rather than the conjunction, and Section 4.3's motivation is
wrong. One line of analysis on data already collected.

```
python code/scripts/reconcile.py                 # get the resolvable difference
python code/scripts/ablations.py --mde <value>   # run the arm, print the verdict
```

## Build variants

`\redflagstrue` on line 104 of `skc_main.tex` is the default and produces
`skc_main_ANNOTATED.pdf` with every red flag visible. `\redflagsfalse` produces
`skc_main_CLEAN.pdf`, main text 10 pages against a 9-page ICLR limit.

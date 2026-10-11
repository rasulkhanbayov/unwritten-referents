# Start here

You are picking up a paper that is **written but not finished**. The prose,
theory, figures and tables are done and audited. What is missing is
experimental: several claims rest on evidence that has not been produced yet,
and one of them may not survive being tested.

Your job is the experiments, not the writing. Nothing in `paper/` needs your
attention until a result comes back that changes what it should say.

---

## The situation in one paragraph

The paper asks whether a model's failure to abstain on a composed question is a
*scoring* failure or an *availability* failure, and separates the two with a
controlled manipulation: write the deciding referent into the prompt, matched
for length and popularity, and re-measure. It reports that a knowledge-awareness
direction read at the written entity separates whether the first hop is known
but is near chance on whether the composed query succeeds, that writing the
referent in recovers part of that, and that reading the same direction one token
later recovers part of it without touching the prompt. Twelve critical issues
stand between that and a submission. They are in `REDFLAGS.md`.

---

## Your first task, and why it comes first

**Run the bridge-only arm.** It is one line of analysis on data that already
exists, and it can invalidate the paper's central design argument.

Section 4.3 claims the method works because it takes the **minimum** of two
readings: answering a composed query needs both hops, so one low reading should
be enough to abstain. Nobody has ever checked whether the minimum does anything.
If reading at the bridge position *alone* performs as well as the minimum, then
the conjunction contributes nothing, the finding is "read somewhere else," and
Section 4.3 has to be rewritten.

```bash
cd code
pip install -r requirements.txt
python tests/test_smoke.py                 # 20 checks, should all pass
python scripts/reconcile.py                # prints the resolvable difference
python scripts/ablations.py --mde <value>  # runs the arm, prints a verdict
```

`ablations.py` prints one of three verdicts in plain words. You do not have to
interpret it. Send whichever it prints.

**You need data first.** See "Getting data in" below. Until the records exist,
every script will decline with a message naming the missing file. That is
deliberate: nothing here will invent a number.

---

## What to do after that

`EXPERIMENTS.md` has all eleven open experiments in priority order, each with
the question it answers, the exact command, and what the output should look
like. Work down the list. The short version:

| Order | Task | Effort |
|---|---|---|
| 1 | Bridge-only arm | one line of analysis |
| 2 | Bootstrap intervals and class balance everywhere | an afternoon |
| 3 | Zero-overlap subset, control (n) | an afternoon |
| 4 | Level shift versus ranking change | an afternoon |
| 5 | Chain-of-thought paired contrast | an afternoon |
| 6 | Rerun under five seeds | days of compute |
| 7 | Risk-coverage curve | minutes once data exists |
| 8 | Timing and memory per selector | an hour |
| 9 | Position-sweep symmetry for baselines | days |
| 10 | DRIFT baseline | days |
| 11 | Six screened relation pairs | weeks, and this sets the timeline |

Items 1 to 5 and 7 run on data already collected. Item 11 is the one that needs
new collection and it is the reason this is not a submission next week.

---

## Getting data in

The analysis code never touches model weights. It reads per-item records:

```
data/records/{model}/{family}.jsonl        one JSON object per evaluation item
```

The directories are already there. The full field list is at the top of
`code/skc/io.py` and in `data/records/README.md`. Required on every record:
`query_id`, `split`, `seed`, `Y`, `K1`, `K2`, `g_S`, `g_B`, `estimator`,
`offset`.

Three things the loader will reject, each because it is a way a wrong number
could reach the paper:

- **A `query_id` in two splits.** If a test item's bridge appeared as a written
  entity in a fit-split prompt, the direction was fit on that referent and the
  result is partly memorization. This is the single most dangerous defect the
  paper could have.
- **Fewer than five seeds.** Every number currently comes from one run.
- **Mixed estimators in one file.** Three directions are fit and two retained,
  but one number per item is reported. The paper never says which.

---

## Folder map

```
START_HERE.md        this file
README.md            what the paper claims and does not claim
REDFLAGS.md          42 issues by severity, each with the exact action
EXPERIMENTS.md       the task list, in priority order
REPRODUCIBILITY.md   dependencies, data contract, seeds, hardware
MANIFEST.md          what is in the package and what is deliberately absent

paper/               LaTeX source, bibliography, style files, build.sh
pdf/                 ANNOTATED (all 44 red flags visible) and CLEAN builds
code/                analysis package, scripts, tests, config, verify_skc.py
data/records/        empty; drop the exported JSONL here
results/             empty; scripts write here on first run
```

Build the paper with `cd paper && ./build.sh`. It writes both PDFs into `pdf/`.

---

## Three things worth knowing before you start

**The code will not lie to you.** No module contains a number from the paper.
Every function raises rather than substituting a default. If a script prints a
number, it computed it from your data.

**Two estimation traps are already pinned by tests.** A marginal estimator for
the shortcut rate is wrong rather than conservative, and a pooled scalar breaks
the decomposition's non-negativity. Both raise if you reach for them, with an
explanation. Do not work around them; they are there because someone already
made the mistake.

**A negative result on task 1 is a good outcome, not a failure.** If the
bridge-only arm matches, you have saved the paper from a claim it could not
defend, and you found it in an afternoon rather than in review. Report it
plainly and we rewrite Section 4.3.

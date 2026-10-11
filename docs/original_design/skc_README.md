# Start here

This is a complete, submission-ready ICLR 2027 paper bundle **pending experiments**. The
manuscript, theory, code, and protocol are written. The numbers are not, because nobody has run
the experiments yet. Every quantitative claim in the paper is a visible red placeholder.

This file explains what the paper claims, how to build it, what to run, and what to check before
submitting. It is the entry point. `skc_runbook.txt` is the detailed experimental protocol.

---

## 1. What the paper says, in one paragraph

Language models carry a linear signal that encodes whether they can recall facts about an entity,
and that signal causally decides whether they answer or refuse. The catch: that signal lives at the
token positions where an entity is *written*. In a two-hop question like *"who is the mother of the
performer of Imagine?"*, the written entity is *Imagine*, but the entity that actually decides
whether the model can answer is *John Lennon*, who is never written down. The model resolves him
internally, in early layers, at no token position. So the machinery that would say "I don't know
about this person" never gets a chance to run on him.

**The one-sentence thesis:** recognition is keyed to *tokens* while correctness depends on
*referents*; the two coincide only when every entity that matters is written down, so on composed
questions the model releases its refusal brake on the wrong entity. This creates a measurable floor
on how well any abstention method built on recognition can work, and the uncovered part grows with
the length of the chain.

**Title note.** The paper is titled *Tokens, Not Referents: Latent Entities and the Limits of
Knowledge-Aware Abstention*. The main title states the mechanism in three words; the subtitle carries
the coined term (**latent entity**: determined by the prompt, not tokenized in it) that the rest of
the paper uses, plus the theoretical contribution. Keep that term consistent everywhere. A coined
term only becomes a citation handle if it is used the same way every time.

This is not a curiosity of two-hop trivia. Latent entities are the normal case in agentic and
retrieval pipelines, where each step's referent is produced by the previous step. That is where
abstention matters most, and where this failure is silent.

### Why this is a paper and not a truism

The obvious objection is "of course a model can't check knowledge it never looked up." The paper is
built around that objection. Two results answer it:

1. The bridge state **is** there. Taking the entity-gate direction learned on one-hop data,
   freezing it, and reading it at the bridge position gives an above-chance readout of whether the
   model knows the unnamed entity. The information exists, in the same format the refusal
   mechanism already reads.
2. Activation patching gives a **double dissociation**: swapping the bridge representation changes
   what the model recalls but not much whether it refuses.

So the model computes the relevant state and does not route it to the decision. That is a claim
about *wiring*, not about information, and wiring can be repaired. That turns a truism into a
finding with a fix.

### What is new, precisely

This matters because a reviewer will ask. The delta against each closest neighbor:

| Closest work | What they did | What we change |
|---|---|---|
| Ferrando et al., ICLR 2025 | Entity-level knowledge gate, causal on refusal | We change the gated object from a **written** entity to a **latent** one, and show the gating does not follow |
| Tamoyan et al., 2025 | Entity-relation level probe, correlational | We change the failure regime to an unrecalled attribute of an **unnamed** entity, and add the causal analysis they leave open |
| Yang et al. 2024; Biran et al. 2024 | How the **answer** is computed across hops | We analyze how the **self-knowledge** is computed, and show the two dissociate |
| Abstention / uncertainty literature | Score the whole query with one number | We give a **lower bound** those scores must obey, and a readout that escapes it |
| Cheng et al., Findings EMNLP 2025 | CoT *obscures* hallucination cues | We predict CoT *restores* the entity gate by writing latent entities down. The two are separable on the same items (E6) |
| Lindsey et al. 2025 (attribution graphs, Claude 3.5 Haiku) | Refusal is the **default**; a "known entity" feature inhibits it. Misfires when a name is recognized but nothing is known about it | Same mechanism, independently found. We show a systematic case: in a composed query the brake is released by the **surface** entity while the deciding entity is never recognized. The brake is released by the *wrong* entity |

**This was checked, not assumed.** The obvious extension of Ferrando et al. (relation-level
knowledge awareness) is already published by Tamoyan et al. The fine-tuning angle is saturated. The
open-ended-generation extension is being actively worked on by the original authors. Those three
directions were dropped for those reasons. Do not quietly wander back into them.

---

## 2. The files

| File | What it is |
|---|---|
| **`skc_REDFLAGS.md`** | **Every known risk and blocker. Read this first.** |
| `skc_main.pdf` / `.tex` | The manuscript, 9 pages of main text |
| `skc_refs.bib` | Bibliography. **29 `% VERIFY` flags you must clear** |
| `skc_main.bbl` | Precompiled bibliography, so one `pdflatex` pass resolves citations |
| `iclr2026_conference.sty` / `.bst` | Official ICLR template files |
| `skc_data.py` | Dataset construction, prompt rendering, alias-aware scoring, labels |
| `skc_gates.py` | Gate extraction (SAE / diff-of-means / probe), frozen after fitting |
| `skc_experiments.py` | Interventions C1-C4 and drivers E0-E7, with the E0 stop gate |
| `skc_metrics.py` | Every metric in the paper, numpy only, 54-check self-test |
| `skc_pipeline_test.py` | End-to-end offline test against a mock model, 50 checks |
| `skc_runbook.txt` | The experimental protocol |
| `requirements.txt` | Pinned dependencies |
| `skc_iclr2027_submission.zip` | Self-contained; compiles from a clean extraction |

### Running the code

```bash
python3 skc_metrics.py --smoke-test     # 54 checks: verifies the theory numerically
python3 skc_pipeline_test.py            # 50 checks: verifies the whole pipeline offline
```

Both must end in `PASSED` before you touch a GPU. Neither needs one.

**What those tests do and do not prove.** They verify the *plumbing*: labels,
filters, the frozen-gate guarantee, the E0 stop gate, result schemas, and every
guard that is supposed to raise. They prove nothing about language models.
Everything that touches a real transformer (`capture_activations`,
`steer_at_position`, `patch_bridge`) has **never been executed** and is listed as
UNVERIFIED in `skc_REDFLAGS.md`.

### Building the paper

```bash
pdflatex skc_main
bibtex   skc_main
pdflatex skc_main
pdflatex skc_main
```

Four passes, in that order. It currently compiles with zero errors, zero undefined citations, zero
multiply-defined labels, and no overfull boxes above 15pt. If you see `[?]` where citations should
be, the `.bib` filename stopped matching `\bibliography{skc_refs}`.

**Template note.** ICLR has not published the 2027 style file yet, so this uses the official ICLR
2026 one. When 2027 drops, put the new `.sty` and `.bst` in this directory and change the two lines
marked `SWAP FOR 2027` in `skc_main.tex`. Nothing else depends on the year.

### Where each claim lives

- The formal claims are Propositions 1-3 in Section 3, proofs in Appendix A.
- The hypotheses that the experiments decide between are Section 3.3 (H1, H2, H3).
- The causal design is Table 1.
- The result placeholders are Section 6, one subsection per experiment.

---

## 3. The theory, in words

**Proposition 1 is the framework, and it is the part other people can reuse.** It splits the error
of *any* abstention rule into three parts: what is irreducible, what is lost purely because the rule
reads the wrong internal positions, and what is lost because the rule uses badly what it does read.

That middle term is the point. Every abstention paper implicitly assumes all error is of the first
and third kinds, so the fix is always a better score. If the middle term is large, no better score on
the same positions can help, because the information was never read. Our empirical claim is that on
composed queries the middle term is large and sits at latent-entity positions.

**Proposition 2** says: if your abstention score is computed only from the gates of written
entities, then no matter where you set the threshold, your error rate on the answers you *do* give
cannot go below a certain number.

**Proposition 2** says what that number is: roughly the rate at which the model fails the second
hop. Minus two slack terms: `delta` (how much the surface gate accidentally leaks about the second
hop) and `v` (how often the model gets a composed query right by memorizing it whole).

The practical reading: **abstaining more aggressively does not help**, because the thing you are
thresholding is uninformative about the thing that is actually going wrong.

**Proposition 1** (the decomposition) is the one to understand first. It splits the error of any
abstention rule into three parts: what is irreducible, what is lost by being **blind** to certain
positions, and what is lost by being **suboptimal** with what it already reads. Better estimators fix
the third. Nothing fixes the second except reading more positions. Abstention methods are routinely
evaluated as though all error were of the first and third kinds. Report all three.

`rho`, the fraction of the blind spot a repair recovers, is now just a **ratio of quantities the
decomposition already defines**: how much of the blind term the repair closes. It used to be a
separate proposition built on essential suprema. That was redundant with the decomposition and has
been cut, which removed a proposition, a proof, and the messiest estimation caveat in the paper.
Fewer moving parts, same content.

Use `recovery_fraction()`. The old `recoverable_fraction_rho()` is retained only as an estimator
sanity check and is **not** what the paper reports.

Proposition 2 also covers `k` hops and shows the uncovered part, `P(M=0)`, grows with `k`. Read the
caveat carefully: it proves the *leading term* grows, **not** that the floor itself grows, because
the slack terms could in principle grow faster. That gap is deliberate and is what E5 measures. Do
not describe it as proving the floor rises.

Both slack terms are *measured*, not assumed. That matters: if `delta` comes back large, the paper's
central hypothesis is **false**, and you report that.

---

## 4. The code

```bash
python3 skc_metrics.py --self-check    # quick internal consistency
python3 skc_metrics.py --smoke-test    # verifies Propositions 1-3 numerically
```

Both should end in `PASSED`. The smoke test builds synthetic data where the ground truth is known
by construction and checks that each inequality in the paper holds in the right direction. It
checks *directions*, not hard-coded numbers, so it stays valid if you change the generator.

**Every number the smoke test prints is synthetic. None of it belongs in a table.** The script says
so on every run.

### The one function you should use

```python
from skc_metrics import floor_report
rep = floor_report(y, k2, gate_surface)   # delta, v, floor, and the violation check
```

Use this rather than assembling the pieces yourself. Here is why, and it is a real lesson.

While verifying the code against the theory, the smoke test **failed**. Two failures. Both were
investigated rather than tolerated:

1. **Tail noise.** At 2% coverage the risk estimate rested on 160 samples. The apparent violation
   was under one standard error, and it disappeared at larger sample size. The theorem was fine;
   the *test* was reading noise. Fixed by flagging a violation only when the upper confidence limit
   of the risk falls below the floor.
2. **A real bug in the estimator.** Using the *marginal* rate for `v` under-estimates it, because
   shortcut answers concentrate exactly where the gate fires. That inflates the computed floor and
   **manufactures an apparent violation of a theorem that actually holds**. Conditional estimation
   gave `v = 0.126` instead of `0.078`, and the bound held.

3. **A third issue, found when the decomposition was added.** The non-negativity of the two
   decomposition terms is a *population* statement. Cross-fitted plug-in estimates can go slightly
   negative even where the proposition holds; on synthetic data the violations sit near a tenth of a
   binomial standard error. The code now tolerates violations inside sampling noise, reports the raw
   values, and **raises** on anything larger, because a large negative term is an estimation fault,
   not a discovery.

The code refuses the marginal estimator by default, regression tests lock both bugs in, and the paper
states each issue. This is what verification is for. If your smoke test never fails, you are probably
not testing anything.

---

## 5. What to run, and what the outcomes mean

Full detail is in `skc_runbook.txt`. The short version:

**Do the cheap pilot first.** One afternoon, Gemma-2-2B, 300 queries. It tells you whether the idea
is alive before you spend weeks.

**Then E0, which is a stop gate.** E0 checks that your gate directions reproduce the known
one-hop separation. If they do not, your instrument is broken and a null result in E1 would be
uninterpretable: you could not distinguish "the model is blind" from "my probe does not work."

**Then E1 through E7.** E1 is the main result, E2 the causal tests, E3 the experiment that makes the
paper non-trivial, E4 the repair, E5 the compounding across hops, E6 the chain-of-thought test, E7
the ablations.

**E6 deserves attention.** It is a head-to-head test against a published prediction rather than an
ablation. Cheng et al. found CoT obscures hallucination cues; we predict it restores the entity gate.
Both are measured on the same items. A test that could go either way against a named prior finding is
worth far more to reviewers than another confirmatory ablation.

### On baselines, which is where this paper is most exposed

Baselines are compared **within inference-cost classes**, not on one leaderboard. BGR-min is the
cheapest possible class: one forward pass, no trained parameters. A method that beats it by sampling
ten generations has not beaten it in the class that matters, and reviewers in this area know that
distinction.

The dangerous class is **Class 3, decomposition pipelines**. "Why do I need a bridge-gate readout
when I can just prompt the model to decompose the question?" is the obvious objection, and it now has
published methods behind it (Self-Ask; Madhwal et al. 2026; operadic consistency). You must run them.
If they beat BGR, report it: they work by making the latent entity explicit, which is our mechanism
by a more expensive route, so they confirm the account even while beating the method. Say that
plainly rather than burying it.

### What "promising" looks like

- Gate predicts the first hop well above chance, and the second hop near chance
- `delta` near zero
- The risk-coverage curve **flattens** onto the floor instead of descending
- The frozen direction transfers to the bridge position above chance
- Patching changes recall but not refusal
- BGR-min beats every baseline at matched coverage

If **BGR-min matches BGR-lr**, that is the *stronger* outcome, not a disappointment: it means the
effect is structural rather than something a learned model absorbed. Say so explicitly.

### What "negative" looks like, and what to do

If `AUC(gate -> K2)` comes back clearly **above** chance, H1 is false. The model does compose
self-knowledge. **This is a real result, not a failure.**

Do not try to rescue the hypothesis. Do not slice the data until it goes away. The honest paper
becomes a negative result that corrects a plausible structural prediction and still contributes the
floor, the benchmark, and the diagnostics. Revise the title and abstract to match. The paper is
written so this pivot is possible: Section 7 already commits to it in print.

---

## 5b. File integrity note

During the final audit, the manuscript was found to contain substantive content that had not been
authored in the session that produced the rest of the bundle: the two-converging-lines framing built
on Lindsey et al. (2025), and the C4 attribution-graph intervention. Every factual claim in that
material was independently verified against primary sources before it was allowed to remain, and two
bibliography errors it carried were corrected (an incomplete author list, and missing page numbers).

The material is sound and improves the paper, so it was kept rather than reverted. But the incident
is the reason for the first checklist item below: **do not assume any part of this bundle is what you
last saw.** Re-read the manuscript end to end before submitting, and treat the audit as yours, not as
something already done for you.

## 6. Before you submit: the checklist

Nothing here is optional. Unresolved items are blocking.

- [ ] **Read the whole manuscript yourself, start to finish.** Content has changed between audits in
      this project before. Verify that every claim is one you are willing to defend.
- [ ] **Re-run the prior-art search.** This wedge was open in August 2026. Interpretability moves
      monthly. Check arXiv for the exact topic and near neighbors, several rounds, past page one.
- [ ] **Replace all 71 `[RES:]` placeholders** with measured numbers. Grep the `.tex` for `RES`.
- [ ] **Clear all 12 `>>> TO RUN` items.**
- [ ] **Verify the proofs.** Appendix A was AI-drafted and is not yet checked. At least two authors,
      line by line. The smoke test verifies the inequalities numerically, which is necessary and
      **not** sufficient. Numerical agreement is not a proof.
- [ ] **Verify the method and design independently.** They were AI-assisted.
- [ ] **Fix the two placeholder authors** (`layerorder2026`, `pretrainexposure2026` in the `.bib`).
      Both are papers the manuscript positions *against*, so citing them wrongly is worse than not
      citing them. Clear all 17 `% VERIFY` comments. Never invent an author name.
- [ ] **Reconcile every claim with the actual numbers.** Abstract, title framing, contribution list.
      If the effect is softer than hoped, walk the claims back.
- [ ] **Complete the AI disclosure** in Appendix D with what you actually did. ICLR 2027 requires
      it, it does not count against the page limit, and authors are held responsible for
      AI-produced content.
- [ ] **Anonymize.** Double blind. No institution, no identifying repository link.
- [ ] **Check the page limit.** 9 pages main text at submission. The body currently ends exactly at
      page 9, so *any* added sentence pushes it over. Re-check after every edit. Reproducibility and
      ethics statements do not count.
- [ ] **Get 2-3 independent human mock reviews** and resolve every issue raised. This is the single
      step most correlated with turning a strong draft into an accept, and no automated process can
      do it for you.

---

## 7. Honest expectations

Read this part.

**No process makes acceptance certain.** ICLR rejects most submissions and the reviewer draw is
noisy. Nothing about how this paper is written or structured changes that.

**Your results are the ceiling.** The dominant factor in the decision is the strength and robustness
of the evidence, and that comes from your runs, not from the manuscript. Until the numbers exist,
this is "complete pending validation."

**What this bundle does reliably:** it identifies a genuinely open contribution and positions it
precisely, neutralizes the objection most likely to sink it, makes the theory sound with explicit
assumptions, and removes the self-inflicted failure modes that sink otherwise-good papers, which are
scoop, overclaim, weak positioning, broken build, missing baselines, and integrity slips.

**What changes when results arrive.** If the headline effect is real, the positive control separates
cleanly, and it survives seeds, ablations, and the popularity confound, then the biggest risk factor
flips from a ceiling into a strength. A real open wedge, plus this process, plus independent human
mock review, is genuinely competitive at a top venue. That is a wide-error-bar judgement, not a
promise. The residual gap is reviewer variance and whether the results hold up under expert
pressure, and neither is closable by any process.

**On the timeline.** ICLR 2027 abstracts are due 18 September 2026, papers 25 September 2026. That
is roughly six weeks from when this was written. E0 through E4 are feasible in that window on the
listed models with public SAEs, but only if E0 starts immediately. If the pilot is ambiguous or E0
fails, the right call is to target a later venue rather than submit thin evidence. A rejected ICLR
submission stays public with your name attached.

**Do not let the assumption of good results become permission to overclaim.** Keep the placeholders
until the numbers are real.

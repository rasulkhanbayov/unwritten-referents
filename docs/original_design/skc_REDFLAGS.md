# RED FLAGS

Everything in this bundle that could embarrass you, ordered by how much damage it
does. Read it before you read anything else. Nothing here is hypothetical: each
item is a real defect, gap, or unverified assumption in the files you were given.

Two words are used precisely throughout:

- **BLOCKING** — do not submit until this is closed.
- **UNVERIFIED** — written but never executed. It may be wrong.

---

## 1. BLOCKING — the bibliography previously contained fabrications

During a deep audit, the reference file was found to contain **two fabricated
titles and one invented author**. They were corrected, but this is the single
most serious thing in the bundle and it tells you something about how to treat
the rest.

| Entry | What it said | What is true |
|---|---|---|
| Kirichenko et al. | Title *"Reasoning Fine-Tuning Degrades Abstention"* | **Fabricated.** The real work is **AbstentionBench** (NeurIPS 2025). That phrase is a *finding*, not a title. |
| Madhusudhan et al. | *"Do LLMs Know When to Abstain? Abstain-QA and the AUCM Protocol"* | **Fabricated.** Real title: *"Do LLMs Know When to NOT Answer? Investigating Abstention Abilities of Large Language Models"*, COLING 2025, pp. 9329–9345. |
| Layer-Order Inversion | Author `Liu and others` | **Invented.** The paper at arXiv:2601.03542 is real; its authors were never verified. |
| `madhwal2026decomposed` | No identifier, bare surname | **Unverifiable.** No primary source found. Entry and citation removed. |

A fabricated citation found by a reviewer is a Code of Ethics matter under
ICLR's AI-use policy, and the **authors** carry that responsibility, not any
tool.

**What you must do:** open every reference's primary source and check the author
list, title, venue, year, and pages by eye. There are **29 `% VERIFY` comments**
in `skc_refs.bib` marking entries where some field was not confirmed. Clear all
of them.

**Two placeholder authors remain and are hard blockers:**

- `layerorder2026` — a paper the manuscript **argues against**. Citing it wrongly
  is worse than not citing it.
- `kirichenko2025abstentionbench` — the benchmark name and venue are confirmed;
  the subtitle and full author list are not.

## 2. BLOCKING — files in this project have changed between audits

Across sessions, the manuscript has twice contained substantive content that was
not written in the session that produced the rest of the bundle. On the second
occasion that content carried the fabrications above. The material was verified
and kept where it was sound.

**What you must do:** read the manuscript end to end yourself before submitting.
Do not assume any section is what you last saw, and treat the audit as yours.

## 3. BLOCKING — the proofs are AI-drafted and unverified by a human

Appendix A contains three proofs (Lemma 1, Theorem 1, Proposition 1). They were
drafted with AI assistance. `skc_metrics.py` checks the inequalities numerically
on synthetic data with known ground truth, which is **necessary and not
sufficient**: numerical agreement is not a proof.

**What you must do:** at least two authors check each proof line by line.

## 4. BLOCKING — the AI-use disclosure is incomplete

Appendix H has a `>>> TO RUN` where the verification description belongs. ICLR
2027 requires disclosure and holds authors responsible for AI-produced content.
It must describe what *you* actually did, not the template.

## 5. UPDATE (2026-08-27): model-facing code has now run for real, on a GPU

This section originally read "nothing that touches a real transformer has ever
been executed," written when this bundle had no GPU access. That is no longer
true. A full A100/H200 Slurm run of E0 through E7 (base model gemma-2-2b at
layer 8, plus a separate gemma-2-9b-it chain for E6) has since completed, and
the prediction in the paragraph below this table ("the most likely failure is
a silent position error... which produces a confident null that looks exactly
like the hypothesis being confirmed") was **correct** — that exact bug class
was found and fixed repeatedly before the results below could be trusted (see
`EXECUTION_PLAN.md` for the full trail; the single most consequential one was
a padding-side collision in `capture_activations` that silently produced
byte-identical captured activations for different queries in the same batch,
invalidating every result across the whole pipeline until found and fixed).

| File | Function | Status | Evidence |
|---|---|---|---|
| `skc_gates.py` | `capture_activations` | **VERIFIED, real GPU runs** | Used by every E0/E1/E3/E4/E5/E6/E7 job this session (e.g. E0 job 382897, agreement 0.547). A real padding-side bug was found and fixed here (see `EXECUTION_PLAN.md`) — the fix changed downstream numbers materially, so this is exercised, not just executed. |
| `skc_experiments.py` | `steer_at_position` (C1, C3) | **VERIFIED, real GPU runs** | E2 job 383331 (C1 coefficient sweep + C3 injection), E7 job 382906 (random-direction control). A real crash (hook fired on every KV-cached token instead of only the prefill call) was found and fixed here before it produced trustworthy output. |
| `skc_experiments.py` | `patch_bridge` (C2) | **VERIFIED, real GPU run** | E2 job 383331: 192 donor pairs patched, real non-degenerate recall effect (9/192 pairs flipped correct→incorrect, d_recall std=0.212). Same prefill-only hook-counter fix as `steer_at_position` needed applying here too (per-layer, since this one hooks multiple layers). |
| `skc_data.py` | `fetch_two_hop` (SPARQL) | **VERIFIED, real network calls** | `data/twohop_queries.json` (n=3503) built from it. A real bug (`P31` used for occupation-typed entities where `P106` is correct) was found by querying the live endpoint directly — flagged, not yet fixed as of this update; see the dataset's own build notes. |
| `skc_data.py` | `fetch_three_hop` (SPARQL) | **VERIFIED, real network calls** | `data/threehop_queries.json` (n=3961 chains) built from it against the live Wikidata endpoint; hit and fixed a real 414 (URI too long) and 429 (rate limit) at scale. |
| `skc_data.py` | `fetch_pageviews` (Wikimedia REST API) | **VERIFIED, real network calls** | `data/pageviews_twohop.json` (847/847 bridge entities resolved) built from it against the live public endpoint; hit and fixed a real 429 at scale. |

Everything above was found to have at least one real bug that would have
silently produced a wrong or degenerate result if not caught — none of them
passed on the first real GPU/network attempt. This is exactly the failure
mode this section originally warned about, not a hypothetical.

Also worth noting directly: the "result that would refute the paper" named in
section 6 below actually occurred once, at layer 8, under a disclosed
methodological weakening (`--skip-bridge-resolution`, K1=1 only, no
independent Patchscopes confirmation) — `AUROC(gate→M)=0.616`, clearly above
chance. It was reported alongside the Patchscopes-verified near-chance result
rather than sliced away, per this file's own instruction in section 6.

`skc_pipeline_test.py` (50 checks) and `skc_metrics.py --smoke-test` (57
checks as of this update, up from 54 originally) still pass and were re-run
after every code change described above —
they remain necessary, proving the plumbing, but real GPU/network execution
is what actually caught the bugs listed in this table; the offline suites
caught none of them (they cannot, by construction — a mock model has no
padding-side bug to have).

**What is still genuinely unrun**, not just under-verified: E7(c) length-matched
distractor control (not implemented anywhere in this bundle), (h) attention
control (stubbed in `skc_run_e7_ablations.py::cmd_attention` but deliberately
`raise NotImplementedError` — needs new `output_attentions=True` capture code
that does not exist yet), (i) counterfactual chains, (l) cross-type
generalization, (m) template/paraphrase robustness, (n) a second
reasoning-tuned model beyond the one already used for E6 (gemma-2-9b-it), (o)
C4 faithfulness fraction, (p) a pre-registered power calculation. C4
(attribution graphs) itself was also never run — it needs a separate
toolchain (circuit-tracer), deliberately kept out of this pipeline (see
`skc_run_e2.py`'s own docstring).

## 6. The result that would refute the paper, and how you will be tempted to hide it

If `AUC(gate → M)` comes back clearly **above chance**, H1 is false, the model
does compose self-knowledge, and the central claim is wrong.

This is a real outcome, not a failure. The temptation will be to slice by entity
type, or relation, or model, until some subset shows the predicted pattern. That
is p-hacking and it will not survive review.

Section 7 already commits in print to reporting this under a revised title. Honor
it. The floor, the benchmark, and the diagnostics all survive a negative result.

## 7. Three estimation traps, all found by running the code

Each of these was a live bug caught by the test suites. They are guarded now,
but you can reintroduce them by editing around the guards.

1. **Marginal `v` is anti-conservative.** Substituting the marginal rate
   `P(Y=1, M=0)` for `v` under-estimates the essential supremum whenever the
   gate is informative about K₁, inflating the computed floor and
   **manufacturing an apparent violation of a theorem that holds**.
   `estimate_shortcut_v` refuses the marginal by default.

2. **Non-negativity is a population statement.** The two decomposition terms can
   go slightly negative in finite samples (about 0.1 binomial standard errors on
   synthetic data), because `R*` is a cross-fitted plug-in and not the true Bayes
   rule. `blind_spot_decomposition` tolerates violations inside sampling noise
   and **raises** beyond it. A large negative term is an estimation fault, not a
   discovery.

3. **Subset discipline in Table 1.** `AUC(gate → K₁)` is **undefined** on the
   K₁ = 1 subset, because K₁ is constant there. It must come from the unfiltered
   set; every other column comes from the filtered set. `run_E1` now demands both
   arrays and refuses to guess. Mixing them silently would put a degenerate
   number in the paper's main table.

## 8. Design traps that will cost you the paper if you get them wrong

- **E0 is a stop gate, not a warm-up.** If the frozen estimators do not reproduce
  the single-hop separation, a null in E1 cannot be distinguished from a broken
  probe. `require_E0` raises. Do not tune the gate until it passes; tuning on
  data you will later test on destroys the result.
- **Never refit a "frozen" direction.** The Section 6.4 argument depends
  entirely on the direction being fit on single-hop data and never touched
  again. `FrozenGate.assert_not_fit_on` enforces this by hashing.
- **Condition on K₁ = 1 with independent bridge-resolution evidence.** Otherwise
  you measure first-hop failure and report it as composition failure. Reviewers
  will find this immediately and it is fatal.
- **Match on popularity.** Without pageview-decile matching, a reviewer will say
  you measured entity frequency, not knowledge. The named-bridge control must
  also be length-matched.
- **Attribution graphs are hypothesis generators.** Lindsey et al. (2025)
  validate every graph-derived claim by intervention, and so must you. "The graph
  shows it" is not an answer to a reviewer who knows that literature.
- **Standardize before taking the C2 dissociation ratio.** Recall and refusal have
  different scales; an unstandardized ratio is a units artefact, not a
  dissociation.
- **Do not pool the E6 split.** Gating on a confidently-wrong self-written bridge
  is a different phenomenon from gating on a correct one. `run_E6` raises if one
  arm is too small.

## 9. Scope limits a reviewer will probe

- **No frontier models.** SAE analysis needs public dictionaries; the probe path
  reaches 70B and stops. Stated in Section 7; do not let it drift out.
- **No retrieval-augmented setting**, where the bridge may be written into
  context and the effect may vanish. That is a fair question and the honest
  answer is that we did not test it.
- **Sparse autoencoder features are dataset dependent and contested.** Three
  independent estimators with an agreement precondition mitigate but do not
  eliminate this. A negative result resting on SAE latents alone would be weak.
- **Wikipedia-derived multi-hop sets over-state composition** because models
  memorize composites. That is exactly what `v` measures; report it.

## 10. Submission mechanics

- **The body has zero slack at 9 pages.** Any added sentence pushes it over.
  Re-check after every edit. Real numbers are usually shorter than the
  `[RES: ]` tags, which should give a little room back.
- **96 `[RES: ]` placeholders and 18 `>>> TO RUN` items** remain. None may
  survive submission. Grep for `RES` and `TORUN` in the `.tex`.
- **Anonymize.** Double blind. No institution, no identifying repository link.
- **The Conclusion is folded into Section 7** to save space. If you want a
  standalone Section 8, it costs about four lines you must find elsewhere.
- **The template is ICLR 2026**, the newest published at build time. Swap in the
  2027 style file when it appears; the two lines are marked `SWAP FOR 2027`.
- **Re-run the prior-art search before submitting.** The wedge was open in
  August 2026. Interpretability moves monthly.

## 11. The honest bottom line

With no results, the realistic acceptance probability is near zero, and no
amount of manuscript quality changes that. What this bundle does is make sure
that **if** the experiments come back strong, the paper converts that into an
accept rather than stalling at borderline: the wedge is genuinely open, the
killer objection is answered structurally, the theory is sound with stated
assumptions, and the self-inflicted failure modes are removed.

The one step most correlated with turning a strong draft into an accept is
**two or three independent human mock reviews with every raised issue resolved**.
No tool can do that for you.

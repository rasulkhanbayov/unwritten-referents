# Reproducibility

What is reproducible today, what is not, and exactly what has to be supplied to
close the gap. Written so a reader who has never seen this project can decide
which numbers to trust.

## Honest summary

| Component | State |
|---|---|
| Data construction from Wikidata | fully specified (Appendix E, query dates recorded) |
| Correctness labels | fully specified, alias-aware exact match, no human annotation |
| Gate fitting | specified except **which of three estimators produces the reported score** |
| BGR scoring | specified except **how the two scores are standardized** |
| Read-position selection | **split not stated**; if the peak was chosen on the evaluation items, BGR is tuned on test |
| Metrics and statistics | fully specified and implemented in `code/skc` |
| Seeds | **none recorded**; every reported number is a single run |
| Hardware and compute | **not recorded** |
| Artifact licenses | **not recorded** |

The three bolded implementation gaps are the difference between "someone else
can rerun this" and "someone else can guess at this." They are flagged in the
manuscript at the point each occurs.

## Environment

```
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
python code/tests/test_smoke.py     # ~30 s, exercises the library, not the paper
```

Python 3.10 or newer. The analysis package needs numpy, scipy and pyyaml only.
`torch`, `transformers` and `sae-lens` are commented out in `requirements.txt`
and are needed only to regenerate the per-item records, not to reanalyse them.
`scikit-learn` is needed only for `bgr_lr`.

## Data contract

The analysis code never reads model weights. It consumes per-item records the
model-side pipeline exports:

```
data/records/{model}/{family}.jsonl        one JSON object per evaluation item
```

The full field list, with which analysis needs which field, is at the top of
`code/skc/io.py`. Required on every record: `query_id`, `split`, `seed`, `Y`,
`K1`, `K2`, `g_S`, `g_B`, `estimator`, `offset`.

Three things about the schema are load-bearing.

**`query_id` is the bootstrap unit.** A composed item and its decompositions
must share one id, because they are not independent. `metrics.bootstrap_ci`
refuses to run without groups rather than silently resampling rows.

**`estimator` forces an answer the paper does not give.** Three directions are
fit and two are retained, but one number per item is reported. The loader
refuses a file that mixes estimators.

**Split membership is checked for leakage.** A `query_id` appearing in two
splits is a hard failure. This is the leakage vector that would explain the
headline result: if a test item's *bridge* appeared as a written entity in a
fit-split prompt, the direction was fit on that referent and the transfer result
is partly memorization. The partition must be over the union of head entity,
bridge and target, not the head entity alone.

## Seeds

`configs/experiments.yaml` sets `seeds.n_required: 5`, and every script calls
`io.require_seeds`. Fewer than five seeds is a hard failure, not a warning.

A bootstrap over queries and a spread over seeds measure different things. The
bootstrap estimates sampling noise in the evaluation set. Seed variation covers
probe initialization, the folds that pick the probe penalty, BGR-lr
cross-fitting folds, and sparse-autoencoder latent selection. The manuscript
currently has the first and none of the second.

## Hardware

`configs/experiments.yaml → hardware` is `null` and must be filled before any
timing number is reported. `scripts/timing.py` prints a warning and proceeds,
because a timing run with unrecorded hardware describes nothing.

Record: GPU model and count, total GPU-hours, and quantization per model.
Quantization is already recorded for the two models that use it.

## Commands and expected outputs

| Command | Writes | Purpose |
|---|---|---|
| `python code/tests/test_smoke.py` | stdout | verify the library; exits non-zero on failure |
| `python code/scripts/reconcile.py` | `results/reconcile/{model}.json` | every reported number, with intervals and class balance |
| `python code/scripts/ablations.py` | `results/ablations/{model}.json` | the bridge-only arm and its verdict |
| `python code/scripts/controls.py --which all` | `results/controls/{model}.{control}.json` | zero-overlap, level shift, CoT paired |
| `python code/scripts/figures.py` | `results/figures/*.tex` | risk-coverage curve and offset sweep as TikZ |
| `python code/scripts/timing.py` | `results/timing/{model}.json` | per-selector scoring cost |
| `python verify_skc.py` | stdout | theory harness and manuscript arithmetic |

Every script skips a model whose records are absent and prints what is missing.
None substitutes a default.

## Two estimation traps the code refuses

Both were found by running the smoke test on synthetic data with known ground
truth, and both are pinned by tests so an edit cannot reintroduce them.

**A marginal estimator for the shortcut rate is wrong, not conservative.**
Substituting the marginal `P(Y=1, M=0)` for the essential supremum understates
it whenever the gate carries information about the first hop, because shortcut
answers concentrate where the gate fires. That inflates the computed floor and
can manufacture an apparent violation of a theorem that holds.
`slack.estimate_v(marginal=True)` raises.

**A pooled scalar breaks the decomposition.** The non-negativity of
`Delta_blind` needs the smaller read set to be a *sub-tuple* of the larger one.
A single scalar summed over positions does not refine the smaller read set, and
the term can go negative by construction. The smoke test produces the
counterexample.

## Theory harness

`verify_skc.py` has two parts. Part A checks the written-position floor, the
monotonicity remark, the threshold-optimality of `R*`, the one-sided and joint
supremum forms of the bound, and the *k*=1 boundary, over randomly drawn joint
distributions. All pass. Part B recomputes the manuscript's arithmetic against
itself and reports the discrepancies as failures; those are findings about the
manuscript, not bugs in the harness.

A numerical check is not a derivation. Theorem 1's proof still carries an open
human-verification flag.

## What would make this fully reproducible

1. Record seeds and rerun under five (`seeds.values` in the config).
2. State which estimator produces the reported gate score.
3. State how the two BGR scores are standardized.
4. State which split the read offset was selected on.
5. Record hardware, GPU-hours and artifact licenses.
6. Publish `data/records/` in the documented schema, or the code that emits it.

Items 2 through 4 are sentences. Item 1 is compute. Item 6 is the release.

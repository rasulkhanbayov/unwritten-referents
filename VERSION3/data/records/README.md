# Per-item records

Drop exported JSONL here, one file per (model, family):

```
data/records/{model}/{family}.jsonl
```

Models: `gemma-2-2b`, `llama-3.1-8b`, `gemma-2-9b`, `qwen3-32b`,
`llama-3.1-70b`. Families: `f1_twohop`, `f2_threehop`, `f3_cot`.

One JSON object per evaluation item, one per line.

## Required on every record

| Field | Type | Notes |
|---|---|---|
| `query_id` | str | Stable id. **The bootstrap resamples this**, so a composed item and its decompositions must share one id. |
| `split` | str | `fit`, `validation` or `eval`. Analyses read `eval`; the offset sweep reads `validation`; standardization reads `fit`. |
| `seed` | int | Which pipeline seed produced this record. At least five are required. |
| `Y` | 0/1 | Composed prompt answered correctly, alias-aware exact match. |
| `K1` | 0/1 | Hop 1 correct in isolation. |
| `K2` | 0/1 | Hop 2 correct in isolation, asked of the **gold** bridge. |
| `g_S` | float | Gate score at the final token of the written entity. |
| `g_B` | float | Same frozen direction at the bridge-resolution position. |
| `estimator` | str | `diffmeans`, `probe` or `sae`. One per file; the loader rejects mixtures. |
| `offset` | int | Read offset used for `g_B`, relative to the written entity. |

## Required only for specific analyses

| Field | Needed by |
|---|---|
| `g_S_named` | level-shift control, `controls.py --which level-shift` |
| `g_S_decoy` | distractor control |
| `text_overlap` | zero-overlap subset, `controls.py --which zero-overlap` |
| `cot_bridge_correct` | chain-of-thought contrast, `controls.py --which cot-paired` |
| `relation` | per-relation breakdown once the inventory widens |
| `baseline_scores` | object, one entry per published baseline, including `drift` |
| `wall_clock_ms` | efficiency table, `timing.py` |

Absent optional fields are fine; the script that needs one will say so and skip.

## Example line

```json
{"query_id": "f1_00417", "split": "eval", "seed": 0, "Y": 1, "K1": 1, "K2": 1,
 "g_S": 0.8123, "g_B": 1.2047, "estimator": "diffmeans", "offset": 1,
 "g_S_named": 1.4410, "text_overlap": 0.0, "relation": "birthplace_country",
 "baseline_scores": {"lookback_lens": 0.71, "p_true": 0.44}}
```

## Three hard failures

The loader raises rather than warning on each of these, because each is a way a
wrong number could reach the paper.

**A `query_id` in more than one split.** The partition must be over the union of
head entity, bridge and target. If a test item's *bridge* appeared as a written
entity in a fit-split prompt, the direction was fit on that referent and read
back at its latent position, so the transfer result is partly memorization.

**Fewer than five seeds.** A bootstrap over queries measures sampling noise in
the evaluation set. It says nothing about probe initialization, penalty-selection
folds, cross-fitting folds, or sparse-autoencoder latent selection.

**Mixed estimators in one file.** Three directions are fit and two are retained,
but one number per item is reported, and the paper never says which. Export one
file per estimator.

## Known issue with the exported `g_B` field (2026-09-29)

In every record file here, `g_B` comes from
`src/driver_scripts/skc_harness.py::capture_for_queries()`'s hop-2 activation. That prompt
**names the gold bridge** ("The country of the city <bridge> is"), so it is a named-bridge
read, not the latent read inside the composed prompt that the paper describes.

Consequences:

- **Unaffected:** `g_S`, Y/K1/K2/M and everything computed from them (e.g. `reconcile.py`'s
  AUROC(S→M/Y) and Theorem-1 floor).
- **Not the paper's quantity:** `scripts/ablations.py`'s bridge-only/BGR outputs.
- **Corrected latent version:** `results/e3_*_latent_*/peak_activations.npy` (repo root),
  combined by `src/driver_scripts/skc_rebuild_c4_latent.py` into `results/c4_latent_rebuild/`.

Files per model:

- `f1_twohop.jsonl`: merged seeds 0-4 (eval) + fit split.
- `f1_twohop_seed{N}.jsonl`: per-seed exports.
- `f1_twohop_fitsplit.jsonl`: fit split.
- `f1_twohop_baselines_*.jsonl`: E4 baseline scores.
- `gemma-2-2b/`: out of scope (single seed, disclosed null).

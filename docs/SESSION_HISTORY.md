# Project history (condensed)

This is the condensed version. The full dated log, with every job ID, error message and
reasoning step, is [`EXECUTION_PLAN_full.md`](EXECUTION_PLAN_full.md); search it for a job
number or bug name. Work ran 2026-08-11 to 2026-10-10 on QCRI's Panther cluster (A100,
later H200), with an AI coding assistant doing most implementation and the author making
every scope decision.

---

## 1. Arc

### Phase A, 2026-08-11 to 09-12: original paper, "Tokens, Not Referents"

- Built the model-side pipeline (`src/`, `src/driver_scripts/`): E0 positive control,
  E1 gate vs. composition, E2 causal interventions, E3 offset sweep, E4 baselines, E5
  three-hop compounding, E6 chain of thought, E7 controls.
- Ran it on 5 models: Gemma 2 2B/9B, Llama 3.1 8B/70B (4-bit), Qwen3 32B (4-bit).
- Wrote and length-cut the paper to 9 pages. It is now `archive/iclr2027_tokens_not_referents/`.
- Most of the effort went into infrastructure bugs (§2a).

### Phase B, 2026-09-13 onward: VERSION3, "Unwritten Referents"

- The supervisor delivered `VERSION3/`, a re-framed paper plus a model-free re-analysis
  package and an audit, `VERSION3/REDFLAGS.md`, with 12 critical, 15 major and 15 minor items.
  VERSION3 replaced the Phase A paper.
- The plan was to export per-item records to VERSION3's schema, then work down REDFLAGS:
  - C4: the bridge-only ablation.
  - C2/C8/C9: reanalysis.
  - C1: hand-derivation of Theorem 1.
  - C3: 5 seeds.
  - C7: the split audit.
  - M2/M3: baselines on more models.
  - C5: relation inventory.
- Grounding the export in real code exposed the **C7 split leak** (§2b). Fixing it forced
  a positive-control layer search on every model, which narrowed scope from 5 models to 2.
- Late audits found the **named-gold-bridge `g_B` bug**. The C4 result was recomputed on a
  genuinely latent read, and the core finding survived with smaller margins.

---

## 2. Bugs found and fixed

### 2a. Phase A (infrastructure)

| Bug | Symptom | Fix |
|---|---|---|
| Left-padding in `capture_activations()` | Activations read at the wrong token for shorter prompts | Force right-padding internally |
| Strict prefix alias match for K1/K2 | K2/M nearly all 0, because models answer inside a longer sentence | `hop_require_prefix={j: False}` at every labelling site, including two that were missed at first |
| bfloat16 → numpy | Crashes in three separate places (e.g. Lookback Lens) | Cast to float32 first |
| Eager vs. SDPA attention | Loading Lookback Lens's eager model silently changed every other selector's scores (mean abs logit diff 0.168) | Separate eager model instance |
| `module load cuda12.1` | `undefined symbol: __nvJitLinkComplete_12_4` | Never load a CUDA module; torch cu124 ships its own runtime |
| Loose `torch>=2.4` pin | pip upgraded torch to 2.13 with torchvision 0.19.1, so `torchvision::nms does not exist` | Pin `torch==2.4.1` exactly; never `pip install` into the shared env |
| HF 10s read timeout | The 70B download never finished in 16h | `HF_HUB_ETAG_TIMEOUT=120`, `HF_HUB_ENABLE_HF_TRANSFER=1` |
| HF cache filled the disk | Jobs crashed on disk full | Cleared 323 GB of caches |
| Wikidata `P31` for people | 0 rows for "basketball player" | Use `P106` (occupation); split the 4-way join; batched POST to avoid 414 |
| E3 offsets counted from prompt end | Read the next word-piece of the player's own surname | Anchor offsets to the entity span's end |
| Patchscopes capture site | Bridge undecodable at every static position | Capture during the model's own hop-1 generation |
| P(True) | Was judging the ground-truth answer, not the model's own | Judge the model's greedy answer |

### 2b. Phase B (methodology)

| Bug | How found | Effect | Fix |
|---|---|---|---|
| **C7 split leakage** in `skc_run_e0.py` | Grounding the export | 402/847 (47.5%) bridge entities were in the fit split. The gate was partly fit on test referents and inflated agreement | `split_entities(..., exclude_from_fit=...)` excludes every head/bridge/target entity of the evaluation chains |
| Same leak, independently, in `skc_run_e0_layersweep.py` | Reading it as a template | Every layer-sweep ranking was leaky | Same fix |
| Single-seed positive control | 5-seed reruns (C3) | Gemma 2 2B / layer 16's pass at seed 0 (0.706) failed seeds 1-4 | Require all 5 seeds to pass |
| `reconcile.py` crash | First real run | AUROC(S→K1) undefined: K1 is constant after K1=1 filtering | Guard, and report as structurally undefined |
| No usable C4 fit split | `ablations.py` needs `split="fit"` | The C7 fix excludes all 3503 two-hop chains from fit, and the Ferrando source is exhausted | Fit split = k=2 prefixes of 1253 entity-disjoint three-hop Wikidata chains (`data/c4_fit_queries.json`) |
| Merge double-counting | Row counts (692 = 2×346) | Re-merging into an already-merged file doubled seeds 1-4 | Dedupe on (query_id, seed, split) |
| `bgr_min()` eval-set standardization | Disagreed with VERSION3's `ablations.py` | Standardized on eval mean/std, which leaks | Optional fit-split args added; not yet threaded through `skc_run_e4.py` |
| E4 relabels the fit split once per baseline | 24h timeout, nothing saved | 3× redundant 9110-item generation on 70B | Label once per model instance (`fit_labels=`) |
| Two 4-bit 70B copies don't fit | `ValueError: ... dispatched on the CPU or the disk` | Lookback Lens / ICR probe impossible on 70B | `--sequential-model-load` |
| Semantic Entropy Probe silent for hours | 24h timeout with no progress information | Unknowable | Progress print every 100 queries; dedicated 36h job |
| A100 vs. H200 | Cross-job check | surface_gate/BGR differ by 0.001-0.011 AUROC across hardware | Re-ran all of Llama 3.1 70B's E4 on H200 |
| **Named-gold-bridge `g_B`** | Background audit, then verified by hand | `capture_for_queries()`'s hop-2 prompt *names the gold bridge*, so every `g_B` was a named-bridge read, not latent. AUROC(g_B→K2) was 0.853 on 70B | Recaptured `g_B` with `skc_run_e3.py`'s composed-prompt offset read (jobs 152-155); C4 rebuilt (`skc_rebuild_c4_latent.py`) |
| Transcription error in `tab:bgr-full` | Recomputed instead of copying | Bridge-only's AURC/R@50 had been pasted into the BGR-min row | Every cell recomputed from per-item data |

---

## 3. Scope decisions (all made by the author)

| Date | Decision | Why |
|---|---|---|
| 09-13 | VERSION3 replaces the Phase A paper | Supervisor's re-framing |
| 09-13 | Fix C7 before anything else | Every downstream number depended on the gate |
| 09-14 | Search other layers rather than stop | Layer 8 failed after the fix |
| 09-15 | Full layer sweeps for failing models | Rigor over blind guesses |
| 09-16 | Narrow to models with a passing gate | Llama 3.1 8B and Qwen3 32B: 0 layers pass |
| 09-18 | Require 5 seeds; Gemma 2 2B becomes a disclosed null | Its only pass was a seed-0 fluke; 0/130 layer×seed pass |
| 09-18 | **Final scope: Gemma 2 9B (layer 8), Llama 3.1 70B (layer 25)** | Both stable on all 5 seeds |
| 09-19 | Fresh, disjoint fit split for C4 (not a relaxed disjointness) | Keep the C7 guarantee |
| 09-21 | Full 13-baseline rerun before rewriting Section 4.3 | No stale Gemma 2 2B numbers in the main table |
| 09-24 | Let the SEP job run rather than kill it | Then split it out when it timed out |
| 09-25 | Move to H200 | A100s full |
| 09-27 | Re-run Lookback Lens on H200 | Hardware-consistent table |
| 09-28 | Stop paper edits; rebuild C4 on the latent read first | Named-bridge `g_B` bug |

---

## 4. Job numbers → purpose

| Jobs | Purpose |
|---|---|
| 00-02, 23 | Environment, offline tests, dataset builds |
| 03-88 | Phase A pipeline (pre-C7-fix). Archived; see `slurm_jobs/archive/README.md` |
| 89-93 | Gemma 2 2B under the fixed split, layers 8/12/16/19/7 |
| 94-95 | Gemma 2 2B E1 at 16 (Patchscopes works) and 8 (fails) |
| 96-99 | Fixed-split E0 for Llama 3.1 8B, Gemma 2 9B, Qwen3 32B, Llama 3.1 70B at their old layers |
| 100-102 | Full layer sweeps: Llama 3.1 8B, Qwen3 32B, Llama 3.1 70B |
| 103 | Llama 3.1 70B layer-25 gate |
| 104-107 | E1 on the in-scope models (Patchscopes attempt, then K1=1-only) |
| 108-131 | Seeds 1-4: E0 + E1 for Gemma 2 2B (E1 never run), Gemma 2 9B, Llama 3.1 70B |
| 132-136 | Gemma 2 2B, 26 layers × 5 seeds |
| 137-138 | Disjoint fit split for C4 |
| 139-149 | E4, 13 baselines (141-144 timed out or superseded; archived) |
| 150-151 | E7 named-bridge + distractor |
| 152-155 | E3-latent: the corrected `g_B` |

---

## 5. Current results

| Quantity | Gemma 2 9B (L8, n=346) | Llama 3.1 70B (L25, n=494) | Source |
|---|---|---|---|
| Positive-control agreement, 5 seeds | 0.502-0.523 | 0.599-0.633 | `results/e0_*` |
| AUROC(g_S→M), 5 seeds | 0.309-0.345 (below chance) | 0.318-0.333 (below chance) | `results/e1_*` |
| Theorem 1 floor, one-sided δ | −0.039 (vacuous) | **+0.062** (binding; see open issue 5) | `VERSION3/code/results/reconcile/` |
| E3-latent peak AUROC(gate→K2) | 0.657 at offset −1 (p<0.001) | 0.682 at offset −1 (p<0.001) | `results/e3_*_latent_splitfix` |
| **Bridge-only vs. BGR-min (latent, seed 0)** | **0.686 vs 0.415**, +0.271 [0.209, 0.336] | **0.606 vs 0.506**, +0.100 [0.036, 0.166] | `results/c4_latent_rebuild/` |
| BGR-mean (latent) | 0.490 | 0.548 | same |
| Best baseline | INSIDE 0.538 | Lookback Lens 0.670 (**beats** bridge-only) | `results/e4_*` |
| Named-bridge gap (a) | 0.007 (≈ none) | 0.304 | `results/e7ac_*` |
| Distractor AUROC(K2) (c) | 0.523 | 0.567 | same |

Disclosed nulls: Gemma 2 2B (0/130 layer×seed), Llama 3.1 8B (0/32 layers), Qwen3 32B
(0/64 layers). Patchscopes confirms bridge resolution only at Gemma 2 2B / layer 16.

Theorem 1 is true by an independent hand derivation (`VERSION3/THEOREM1_HAND_DERIVATION_DRAFT.md`),
with two small presentation gaps (an undefined symbol `H` in the hypothesis; one asserted
π-system step). It still needs a human author's sign-off.

---

## 6. Open TODOs

1. **BGR-lr still uses the named-bridge `g_B`.** The 0.827/0.669 in `tab:bgr` and
   `tab:bgr-full` are contaminated. Recompute it with cross-fitting on the latent `g_B`.
2. **The latent bridge-only result is single-seed.** Run E3-latent for seeds 1-4 (both splits),
   and check that offset −1 stays the peak.
3. **Most of the manuscript still describes Gemma 2 2B.** That covers the abstract, Fig. 1,
   contributions, §5.1-5.4, `tab:e0-allmodels`, `tab:gap`, `tab:compound-allmodels`,
   `tab:power`, `tab:bench`, `tab:map` and `fig:offsets`. Only §4.3 + `tab:bgr` + `tab:bgr-full`
   are rewritten. The abstract's availability claim needs jobs 150/151's numbers, including
   Gemma 2 9B's near-zero named-bridge gap.
4. **Baseline sign conventions.** Several baselines are well below 0.5 on Gemma 2 9B
   (SelfCheckGPT 0.156, SEP 0.203); flipped, they would beat bridge-only. Fix the sign on
   the fit split and report it.
5. **Llama 3.1 70B labels look suspect.** P(M=0)=0.71 on city→country, and Y=1 with M=0 on 115
   items. Re-score K2/Y with an LLM judge from the saved raw generations (on HF). The +0.062
   floor depends on this.
6. **Position-selection fairness.** Bridge-only's offset is swept; the baselines' positions are
   not. Sweep the baselines too, or pre-register the offset.
7. **Theorem 1 needs human sign-off.** Then fix the two gaps and restore the AI-policy sentence.
8. **There is one relation pair** (place-of-birth→country), which Yang et al. exclude as
   guessable (C5).
9. **Thread `bgr_min`'s fit-split standardization** through `skc_run_e4.py`.
10. **Missing statistics:** CIs on every table cell, a multiplicity correction across the 13
    baselines, seed SDs for the baselines (single-seed now).
11. **Baselines not yet run:** DRIFT, and operadic consistency with sampling.
12. **`VERSION3/REDFLAGS.md` is dated 2026-09-13.** Many items are resolved or superseded (see
    above); use the manuscript's own `\rf{}`/`\rv{}` flags as the live list.

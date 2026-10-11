# INDEX: where things are

New here? Read [`README.md`](README.md) first, then [`docs/SESSION_HISTORY.md`](docs/SESSION_HISTORY.md).

## The 20 files that matter most

| # | File | Why |
|---|---|---|
| 1 | `README.md` | What the paper claims, current status, how to reproduce |
| 2 | `docs/SESSION_HISTORY.md` | Condensed history: bugs, decisions, current numbers, open TODOs |
| 3 | `VERSION3/paper/skc_main.tex` | **The live manuscript** ("Unwritten Referents") |
| 4 | `src/skc_gates.py` | Gate estimators (diffmeans / probe / SAE), `FrozenGate`, activation capture |
| 5 | `src/skc_metrics.py` | AUROC/AURC/selective risk, Theorem-1 floor, `bgr_min`/`bgr_lr` |
| 6 | `src/skc_data.py` | `ComposedQuery`, prompt templates, alias-aware labelling, `filter_bridge_resolved` |
| 7 | `src/skc_experiments.py` | `run_E0` ... `run_E7`: the experiment logic the drivers call |
| 8 | `src/driver_scripts/skc_harness.py` | Model loading (4-bit for 32B/70B), batched labelling, `capture_for_queries` (**its `g_B` names the gold bridge**, see history) |
| 9 | `src/driver_scripts/skc_run_e0.py` | Positive control; `split_entities()` with the C7 leakage fix |
| 10 | `src/driver_scripts/skc_run_e3.py` | Composed-prompt offset sweep, **the correct latent `g_B`** (`--save-peak-activations`) |
| 11 | `src/driver_scripts/skc_rebuild_c4_latent.py` | Headline C4 result: bridge-only vs BGR-min/mean on the latent read |
| 12 | `src/driver_scripts/skc_run_e4.py` | 13 baselines |
| 13 | `src/driver_scripts/skc_run_e7_ablations.py` | Controls: named-bridge, distractor, random direction, label validation, ... |
| 14 | `VERSION3/code/skc/io.py` | Per-item record schema for the re-analysis package |
| 15 | `VERSION3/code/scripts/reconcile.py` | Class balance, AUROC CIs, one-sided Theorem-1 floor |
| 16 | `VERSION3/THEOREM1_HAND_DERIVATION_DRAFT.md` | Independent derivation of Theorem 1 (needs human sign-off) |
| 17 | `slurm_jobs/README.md` | Every current GPU job, grouped by experiment |
| 18 | `scripts/run_analysis.sh` | CPU analysis chain, verified to reproduce every current output |
| 19 | `results/README.md` | Which of the 140 result directories are current / null evidence / superseded |
| 20 | `docs/EXECUTION_PLAN_full.md` | Full dated work log (11k lines; search by job number or bug) |

## Directories

| Path | Contents | In git? |
|---|---|---|
| `src/` | Core library (`skc_*.py`) + `skc_pipeline_test.py` (51 offline checks) | yes |
| `src/driver_scripts/` | One CLI per experiment (`skc_run_e*.py`), dataset builders (`skc_build_*.py`), record export/merge, one-off diagnostics (`skc_diag_*.py`, kept, not used by current results) | yes |
| `slurm_jobs/` | 63 current job scripts | yes |
| `slurm_jobs/archive/` | 92 superseded job scripts, one reason each | yes |
| `scripts/` | `download_data.sh`, `upload_to_hf.sh`, `download_from_hf.sh`, `run_analysis.sh` | yes |
| `data/` | Query sets and fit splits (all < 16 MB each); `ferrando_entities/` is recreated by `scripts/download_data.sh` | yes (except `ferrando_entities/`) |
| `results/` | One directory per job run | small JSON yes; `*.npy`, `*.pt`, `*_raw_generations.json` on HF |
| `VERSION3/` | Supervisor's package: manuscript (`paper/`), re-analysis code (`code/`), per-item records (`data/records/`, 8 MB), audit docs, pre-built PDFs | yes |
| `docs/` | History, full log, restructure log, `original_design/` (first README, REDFLAGS, runbook) | yes |
| `logs/slurm/` | Original job stdout/stderr | no (HF) |
| `archive/old_paper_v0/` | First paper draft | yes |
| `archive/iclr2027_tokens_not_referents/` | Superseded Phase A paper | yes |
| `archive/local_only/` | Third-party clones (`sae_entities`, `circuit-tracer`), Ferrando data copy, PDFs | no (re-downloadable) |
| `environment.yml`, `requirements-freeze.txt` | Exact env export (Python 3.11, torch 2.4.1+cu124) | yes |
| `requirements.txt` | Hand-maintained minimal pins, with the reasons for each pin | yes |
| `CLAUDE.md` | Working notes for an AI coding assistant on this repo | yes |

## Which file produces which part of the paper

**Rewritten for the two-model scope (current):**

| Paper element | Produced by | Results |
|---|---|---|
| §4.3 C4 ablation; `tab:bgr` and `tab:bgr-full` rows for bridge-only / BGR-min / BGR-mean | `skc_run_e3.py` (jobs 152-155) → `skc_rebuild_c4_latent.py` | `results/e3_*_latent_*`, `results/c4_latent_rebuild/` |
| `tab:bgr` / `tab:bgr-full`: baselines, oracle-M, surface gate | `skc_run_e4.py` (jobs 139, 140, 145-149) | `results/e4_gemma9b_layer8_*`, `results/e4_llama70b_layer25_{partA1,partA2,partB1,partB2,lookback_h200}` |
| `tab:bgr` / `tab:bgr-full`: BGR-lr row | `skc_run_e4.py` | same. **Uses the named-bridge `g_B`; not yet recomputed** |
| Positive-control scope (2 models in, 3 nulls) | `skc_run_e0.py`, `skc_run_e0_layersweep.py` | `results/e0_*splitfix`, `_seed*`, `e0_layersweep_*` |
| Class balance, AUROC(S→M/Y) CIs, one-sided floor | `VERSION3/code/scripts/reconcile.py` | `VERSION3/code/results/reconcile/` |

**Not yet rewritten (still Gemma 2 2B / pre-fix numbers; see README → Open issues):**

| Paper element | Produced by | Results |
|---|---|---|
| `tab:e0-allmodels` | `skc_run_e0.py` | `results/e0_*` (pre-fix) |
| `tab:gap` | `skc_run_e1.py` | `results/e1_*` (pre-fix) |
| `fig:offsets` | `skc_run_e3.py` | `results/e3_*` (non-latent, pre-fix) |
| `tab:compound-allmodels` | `skc_run_e5.py` | `results/e5_*` |
| `tab:power` | `reconcile.py` MDE | n=239 (Gemma 2 2B) |
| Controls appendix (a)-(n) | `skc_run_e7_ablations.py`, `skc_run_e7j_crosstype.py` | `results/e7*` (new (a)/(c) for in-scope models: `results/e7ac_*`) |
| §5 chain of thought | `skc_run_e6.py` | `results/e6_*` |
| §5 causal arms, `fig:causal`, `tab:causal` | `skc_run_e2.py`, `skc_run_e2_c4_graphs.py` | `results/e2_*`, `results/c4_graphs_gemma2b` |
| `fig:teaser`, `tab:map`, `tab:bench`, `tab:templates` | Hand-written in `skc_main.tex` (TikZ / text) | — |

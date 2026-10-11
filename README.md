# Unwritten Referents: Separating Availability from Scoring in Selective Abstention

Code, data and results for the paper in [`VERSION3/paper/skc_main.tex`](VERSION3/paper/skc_main.tex).

> **Status: research in progress, not submission-ready.** This README states what the
> current data supports and what is still open. [`docs/SESSION_HISTORY.md`](docs/SESSION_HISTORY.md)
> has the full story; [`INDEX.md`](INDEX.md) is the map of the repo.

## The question

In *"the mother of the performer of Imagine"*, the entity that decides correctness (John
Lennon) is never written. Knowledge-awareness probes (Ferrando et al., 2025) are read at
positions where an entity **is** written. Can such a probe tell whether a composed,
multi-hop query will be answered correctly? If not, is the information missing at those
positions (**availability**), or present but mis-scored (**scoring**)?

## Method in one paragraph

1. **Fit the gate.** A frozen "do I know this entity?" direction (difference of means, fit
   once on single-hop known/unknown prompts with an entity-disjoint split) is the *gate*.
2. **Read it at two positions** on two-hop *player → birthplace → country* queries:
   - `g_S`: at the written surface entity;
   - `g_B`: at a position **inside the composed prompt, past the written entity**, where the
     unwritten bridge could be resolved.
3. **Compare them** as selective-abstention scores against 13 published detectors.
4. **Bound what's possible.** A theorem (`thm:floor`) lower-bounds the risk of any rule that
   reads only written positions.

## What the current data supports

The scope is **two models**. Each passes the pre-registered positive control (cross-estimator
agreement ≥ 0.50, each estimator AUROC ≥ 0.65) on **all 5 seeds** under the corrected entity
split:

| | Gemma 2 9B (layer 8) | Llama 3.1 70B (layer 25, 4-bit) |
|---|---|---|
| Positive-control agreement, seeds 0-4 | 0.50-0.52 | 0.60-0.63 |
| Surface gate predicts composition, AUROC(g_S→M) | 0.31-0.35, **below chance** | 0.32-0.33, **below chance** |
| Latent bridge read, AUROC(gate→K2) at peak offset −1 | 0.657 (p<0.001 vs permutation) | 0.682 (p<0.001) |
| **Bridge-only vs BGR-min** (AUROC for Y, seed 0) | **0.686 vs 0.415**, Δ=+0.271 [0.209, 0.336] | **0.606 vs 0.506**, Δ=+0.100 [0.036, 0.166] |
| Best published baseline | INSIDE 0.538 | Lookback Lens **0.670** (beats bridge-only) |
| Named-bridge gap, control (a) | 0.007 (writing the bridge barely helps) | 0.304 |

What these numbers mean:

- **The surface score is uninformative, so combining it hurts.** The surface reading does not
  predict whether the composed query succeeds. Taking its minimum with the bridge reading
  (BGR-min, the paper's original design) is *worse* than reading the bridge position alone.
  The paper's design rationale has been rewritten accordingly (§4.3).
- **Bridge-only is the best one-pass selector, not the best overall.** On Llama 3.1 70B, Lookback
  Lens beats it at twice the cost.
- **Three models are disclosed nulls.** Gemma 2 2B (0/130 layer×seed combinations pass), Llama 3.1
  8B (0/32 layers) and Qwen3 32B (0/64 layers) have no stable gate layer, after an exhaustive
  search.

## Open issues (read before citing any number)

1. **BGR-lr is contaminated.** The BGR-lr row in `tab:bgr` (0.827 / 0.669) still uses the old
   `g_B`, which was captured from a prompt that *names the gold bridge*. Bridge-only, BGR-min
   and BGR-mean have been recomputed on the genuinely latent read; BGR-lr has not.
2. **The latent bridge-only result is single-seed.** The 5-seed rerun of `skc_run_e3.py` (jobs
   152-155 for seeds 1-4) is not done.
3. **Most of the manuscript still describes the old Gemma 2 2B setup.** That covers the abstract,
   Fig. 1, contributions, §5.1-5.4, and `tab:e0-allmodels`/`gap`/`compound-allmodels`/`power`/`bench`/`map`.
   Only §4.3, `tab:bgr` and `tab:bgr-full` reflect the current two-model scope.
4. **Baseline sign conventions are not fixed.** Several baselines sit far below 0.5 on Gemma 2 9B
   and would beat bridge-only if sign-flipped. Fix the sign on the fit split.
5. **Llama 3.1 70B's K2 labels look suspect.** P(M=0)=0.71 on city→country. The non-vacuous
   Theorem-1 floor (+0.062) depends on these labels. Re-score with an LLM judge from the saved
   generations.
6. **The read position is chosen unfairly.** Bridge-only's offset is swept; the baselines'
   positions are not.
7. **Theorem 1 needs a human author's sign-off.** A from-scratch derivation finds it true:
   `VERSION3/THEOREM1_HAND_DERIVATION_DRAFT.md`.
8. **There is one relation pair**, which Yang et al. (2024) exclude from their own benchmark as
   guessable.

The full list, with costs, is in `docs/SESSION_HISTORY.md` §6. `VERSION3/REDFLAGS.md` is dated
2026-09-13 and partly superseded.

## Setup

```bash
# 1. Environment (Python 3.11, torch 2.4.1 + CUDA 12.4 wheels)
conda env create -f environment.yml            # exact env used for every result
conda activate skc_iclr27
#   or the minimal version: slurm_jobs/00_setup_env.sh (torch first, then requirements.txt)

# 2. Secrets
cp .env.example .env    # add HF_TOKEN; accept the Gemma 2 and Llama 3.1 licences on huggingface.co first

# 3. Public third-party data (Ferrando et al. entity files, MD5-verified)
bash scripts/download_data.sh

# 4. Offline tests (CPU, ~1 min)
cd src && python3 skc_metrics.py --smoke-test && python3 skc_pipeline_test.py   # 51 checks
cd ../VERSION3/code && python3 tests/test_smoke.py
```

Model weights download from the Hugging Face Hub at run time, and SAE dictionaries come via
`sae_lens`. Large saved artifacts (17 GB: activations, raw generations, logs) are on a Hugging
Face dataset; see [`HF_MANIFEST.md`](HF_MANIFEST.md) and `scripts/download_from_hf.sh`.

**Hardware.** NVIDIA A100 (QCRI Panther cluster) for the 9B model and below, plus most 70B runs; H200 for the
final Llama 3.1 70B baseline and E3 runs. The 32B and 70B models run 4-bit (bitsandbytes). Job
budgets total about 923 GPU-hours of requested walltime (716 A100 + 206 H200) for the current
jobs.

## Reproduce the current results

All GPU jobs are Slurm scripts in [`slurm_jobs/`](slurm_jobs/README.md). Submit them from
inside the repo; each finds the repo root itself. Wait for each E0 job to report
`gate_open: true` before submitting the E1 jobs that depend on it.

| Step | What | Jobs (Gemma 2 9B / Llama 3.1 70B) |
|---|---|---|
| 0 | Datasets (already shipped in `data/`; rebuild optional) | `02`, `23` |
| 1 | **E0** positive control, seed 0 and seeds 1-4 | `97`, `116 118 120 122` / `102` (layer sweep), `103`, `124 126 128 130` |
| 2 | **E1** surface gate + labels, seeds 0-4, plus fit split | `106`, `117 119 121 123`, `137` / `107`, `125 127 129 131`, `138` |
| 3 | **E3-latent**: the corrected `g_B`, eval split then fit split | `152`, `154` / `153`, `155` |
| 4 | **E4**: 13 baselines | `139`, `140` / `145`, `146`, `147`, `148`, `149` |
| 5 | **E7**: named-bridge + distractor controls | `150` / `151` |
| 6 | **CPU analysis**: record export, seed merge, `reconcile.py`, `ablations.py`, C4 rebuild | `bash scripts/run_analysis.sh` |

Step 6 was run end to end after the restructure and reproduced every current output
exactly. Its step 4 (`ablations.py`) is legacy: it reads the old, named-bridge `g_B` from
`VERSION3/data/records`. The current C4 numbers come from step 5 of the script,
`results/c4_latent_rebuild/`.

To verify the disclosed nulls: jobs `89`-`93`, `96`, `98`-`101`, `108 110 112 114` and
`132`-`136`. See `results/README.md`.

To skip the GPU steps, run `bash scripts/download_from_hf.sh` and then step 6.

## Repository layout

```
src/                  core library + offline tests
src/driver_scripts/   one CLI per experiment, dataset builders, export/merge
slurm_jobs/           current GPU jobs (archive/: superseded, with reasons)
scripts/              data download, HF upload/download, CPU analysis chain
data/                 query sets and fit splits
results/              one directory per run (large binaries on HF)
VERSION3/             manuscript, re-analysis package, per-item records, audit docs
docs/                 condensed history, full work log, restructure log, original design docs
archive/              earlier paper drafts (local_only/ is gitignored)
```

## Credits and licences

- **Entity data:** Ferrando et al. (2025), *Do I Know This Entity?*
  ([javiferran/sae_entities](https://github.com/javiferran/sae_entities), Apache-2.0).
- **Three-hop chains:** Wikidata (CC0), pulled 2026-08-26.
- **Pageviews:** Wikimedia Pageviews API.
- **Attribution graphs:** [circuit-tracer](https://github.com/safety-research/circuit-tracer).
- **SAEs:** Gemma Scope and Llama Scope, via `sae_lens`.
- **Model weights:** used under their own licences (Gemma, Llama 3.1, Qwen3).
- **This repository's code and data:** no licence chosen yet; add one before making the repo public.

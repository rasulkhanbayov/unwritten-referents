# Project state — read this first, every session

## Repo layout (restructured 2026-10-10 for GitHub)

Start with `README.md`, `INDEX.md` and `docs/SESSION_HISTORY.md` (condensed history,
current numbers, open TODOs). Code is in `src/` (core) and `src/driver_scripts/` (CLIs);
current GPU jobs are in `slurm_jobs/` (superseded ones in `slurm_jobs/archive/`); the CPU
analysis chain is `scripts/run_analysis.sh`. Large binaries are gitignored and on Hugging
Face (`HF_MANIFEST.md`). The move log is `docs/RESTRUCTURE_LOG.md`.

**Critical known bug:** `src/driver_scripts/skc_harness.py::capture_for_queries()`'s third
return value (`acts_hop2`, saved by E1/E4 as `acts_bridge*`, exported as `g_B`) is read
from a prompt that NAMES THE GOLD BRIDGE. It is not a latent read. The correct latent
`g_B` comes from `skc_run_e3.py --save-peak-activations`. See `docs/SESSION_HISTORY.md` §2b.

## SUPERSEDED: the "Tokens, Not Referents" paper is no longer the live document

As of 2026-09-13, the supervisor delivered `VERSION3/`, a full submission
package for a **different paper**: *"Unwritten Referents: Separating
Availability from Scoring in Selective Abstention."* Confirmed with the user
that **VERSION3 replaces** the earlier `archive/iclr2027_tokens_not_referents/skc_main.tex` ("Tokens, Not
Referents: Latent Entities and the Limits of Knowledge-Aware Abstention").

- **`archive/iclr2027_tokens_not_referents/`** is now historical/frozen. Its own extensive completion
  effort (documented in the old portion of `docs/EXECUTION_PLAN_full.md`, all entries
  before 2026-09-13) is no longer the active task, but the underlying
  experimental infrastructure (`src/driver_scripts/`, `src/skc_experiments.py`,
  `src/skc_metrics.py`, `results/`) is still valuable: it is the closest thing
  to a "model-side pipeline" VERSION3's own analysis code expects to be fed
  by, and per-item gate scores (`g_S`, `g_B` in VERSION3's own naming) are
  computed in-memory inside functions like `run_E1()` but were never
  persisted to disk in VERSION3's required schema.
- **`VERSION3/`** is now the live document. Read `VERSION3/START_HERE.md`,
  `VERSION3/REDFLAGS.md`, `VERSION3/EXPERIMENTS.md`,
  `VERSION3/REPRODUCIBILITY.md`, `VERSION3/MANIFEST.md` in that order before
  doing anything. The live manuscript is `VERSION3/paper/skc_main.tex`.

## What VERSION3 is

A pure **re-analysis package**: `VERSION3/code/skc/*.py` and
`VERSION3/code/scripts/*.py` compute every number the paper needs from
per-item JSONL records at `VERSION3/data/records/{model}/{family}.jsonl`
(schema documented at the top of `VERSION3/code/skc/io.py`). **It does not
run models itself** — no torch/transformers dependency for the analysis
path. Someone (us) must export real per-item records from the actual
model-side pipeline (our existing `src/driver_scripts/`) into that exact schema
before any of VERSION3's scripts can run on real data. Every script raises
with a clear message naming the missing file/field rather than
substituting a placeholder — deliberately, so nothing gets fabricated.

VERSION3's own `REDFLAGS.md` lists 12 CRITICAL + 15 MAJOR + 15 MINOR issues,
verdict "NOT SUBMISSION-READY." The single highest-priority item (C4, listed
first in `START_HERE.md`) is the **bridge-only ablation**: nobody has ever
checked whether reading the bridge position alone matches BGR-min's own
performance. If it does, the paper's central design argument (that the
*minimum* of two readings is what matters) is unsupported and Section 4.3
needs rewriting. This is cheap to check once real per-item records exist,
and per both `START_HERE.md` and `REDFLAGS.md`, must be run **before**
writing anything else.

## Model scope, as of 2026-09-18 (supersedes anything said above about "5 models" or "3 in-scope models")

C7 (the split-leakage bug, see below) turned out to be real and
project-wide, and finding an honest positive-control layer per model
took most of this project's GPU budget so far. After exhaustive
per-layer AND per-seed checking (not just per-layer):

- **In scope, confirmed stable across all 5 seeds (0-4):**
  **gemma-2-9b** (layer 8, agreement 0.50-0.52 across every seed) and
  **llama-3.1-70b** (layer 25, agreement 0.60-0.63 across every seed,
  the most robust result in the project — a 19-layer-wide passing band
  at seed 0 alone, see `results/e0_layersweep_llama70b/`).
- **Disclosed negative results (report as limitations, not main-scope
  claims):** **llama-3.1-8b** and **qwen3-32b** (exhaustive by layer —
  every layer of each model fails the honest positive control) and
  **gemma-2-2b** (exhaustive by layer AND by seed — 0/130 layer×seed
  combinations pass; its own earlier-looking "layer 16" pass at seed 0
  was a lucky single-seed outlier that failed 4/4 other seeds, and
  additionally fails the full 3-estimator bar even at that same seed
  once SAE is included — two independent reasons, not one).
- **The paper's main empirical argument should be built on gemma-2-9b
  and llama-3.1-70b only.** Do not reintroduce gemma-2-2b, llama-3.1-8b,
  or qwen3-32b as main-scope models without first re-running their own
  exhaustive checks — their null results are real and expensive to
  produce, not provisional.
- Full numeric trail (every seed, every layer, every job number) is in
  `docs/EXECUTION_PLAN_full.md`; search for "C7" and "Phase 4" to find it.

## Working discipline carried over from the old project (still applies)

- **Never fabricate or approximate a manuscript number.** Same rule as
  before, now enforced mechanically by VERSION3's own `io.py`/`MissingData`
  exceptions — treat a raised `MissingData` as correct behavior to satisfy
  with real data, never to route around.
- **After every edit to `VERSION3/paper/skc_main.tex`**, check brace balance
  and `\begin`/`\end` balance — no compiler available in this environment
  session to catch a mismatch (though `VERSION3/paper/build.sh` exists and
  should be tried; if `pdflatex`/`bibtex` turn out to be available in
  whatever environment actually runs the build, use them and say so).
- **Never `pip install` anything into the shared `skc_iclr27` conda env.**
  Same incident as before still applies if any new tooling is needed for
  VERSION3's own data-export step.
- **Diagnose Slurm job failures to root cause** before resubmitting.
- **This session cannot run `sbatch` directly** — it runs on a Panther GPU
  compute node, not the login node. Any new GPU job must be built and
  handed to the user to submit, same as every job in the old project.
- When multiple valid approaches exist and the choice is the user's to
  make, ask — don't assume. This applies especially now: VERSION3's own
  REDFLAGS.md sometimes offers two legitimate repairs (e.g. C6/M6's
  baseline-fairness fix, C10's positive-control criterion) and picking one
  is an authorial decision.

## Key files (VERSION3)

- `VERSION3/paper/skc_main.tex` — the manuscript (live).
- `VERSION3/REDFLAGS.md` — the full, ordered issue list with actions.
- `VERSION3/EXPERIMENTS.md` — the 11 open experiments, in priority order.
- `VERSION3/code/skc/io.py` — the per-item record schema every analysis
  script depends on; read this before exporting any data.
- `VERSION3/code/scripts/` — `reconcile.py`, `ablations.py`, `controls.py`,
  `figures.py`, `timing.py`.
- `VERSION3/data/records/` — where real per-item JSONL exports must land.
- `src/driver_scripts/`, `src/skc_experiments.py`, `src/skc_metrics.py`, `results/`
  (repo root) — the old pipeline; adapt it to emit VERSION3's schema
  rather than rebuilding model-loading/activation-capture from scratch.

## Key files (old project, historical)

- `archive/iclr2027_tokens_not_referents/skc_main.tex` — the old manuscript, no longer being edited.
- `docs/EXECUTION_PLAN_full.md` — full dated log of the old project; entries before
  2026-09-13 describe finishing the old paper, not VERSION3.

# Restructure log (2026-10-09/10)

The folder `NEW-PREPARATION-ICLR27/4/` was turned in place into this GitHub-ready
layout. Every move was a rename on the same filesystem. Nothing was copied, and only the
items under **Deleted** were removed. No git commands were run.

## Moved

| From | To | Why |
|---|---|---|
| `skc_data.py`, `skc_experiments.py`, `skc_gates.py`, `skc_metrics.py`, `skc_pipeline_test.py` | `src/` | One code location |
| `driver_scripts/` | `src/driver_scripts/` | Same. Drivers' `sys.path.insert(dirname(dirname(__file__)))` now resolves to `src/`, so imports need no edits |
| `skc_README.md`, `skc_REDFLAGS.md`, `skc_runbook.txt` | `docs/original_design/` | Original design docs, still cited by name in code comments |
| `EXECUTION_PLAN.md` | `docs/EXECUTION_PLAN_full.md` | Full dated work log |
| `skc_main.tex`, `skc_main.bbl`, `skc_refs.bib`, `iclr2026_conference.{sty,bst}` | `archive/old_paper_v0/` | First paper draft (pre-iclr2027) |
| `iclr2027/` | `archive/iclr2027_tokens_not_referents/` | Superseded paper ("Tokens, Not Referents") |
| `sae_entities/` (git clone, 1.9 GB) | `archive/local_only/third_party/` (gitignored) | Public; re-fetched by `scripts/download_data.sh` at the pinned commit |
| `circuit-tracer/` (git clone, editable install) | `archive/local_only/third_party/` (gitignored) | Public; pinned in `environment.yml` / `requirements-freeze.txt` |
| `data/ferrando_entities/` | `archive/local_only/third_party/ferrando_entities/` (gitignored) | Byte-identical (MD5-checked) to the upstream release; `scripts/download_data.sh` recreates it at the same path |
| `2411.14257v2.pdf`, `4-iclr-2411.14257v2.pdf` (duplicates), `End Users Guide - Panther ... .pdf` | `archive/local_only/pdfs/` (gitignored) | Ferrando et al. arXiv PDF; internal cluster guide, not for a public repo |
| `slurm_jobs/*.out`, `slurm_jobs/*.err` (440 logs) | `logs/slurm/` (gitignored, on HF) | Moved, not deleted: they are the only record of the timeouts and failures |
| 92 superseded `slurm_jobs/*.sh` | `slurm_jobs/archive/` | Reason per job in `slurm_jobs/archive/README.md` |

Deliberately **not** moved:

- `results/`: every job script's `--out` points at its own subdirectory, and manuscript
  sections still pending rewrite cite many of the superseded ones. They are classified
  in `results/README.md` instead.
- Diagnostic scripts (`src/driver_scripts/skc_diag_*.py`): moving them into a
  subdirectory would break their `sys.path` imports. They are listed in `INDEX.md`.

## Deleted

- Every `__pycache__/` directory outside `archive/`. Regenerated automatically.

## Edited

| File | Change |
|---|---|
| All 155 `slurm_jobs/**/*.sh` | Hard-coded `REPO="/export/home/rakh90100/NEW-PREPARATION-ICLR27/4"` replaced with a walk-up search for `src/skc_data.py` (or `$SKC_REPO`); `cd "$REPO/driver_scripts"` → `cd "$REPO/src/driver_scripts"`; `01`/`00` now `cd "$REPO/src"` |
| `src/driver_scripts/skc_build_c4_fit_queries.py`, `skc_merge_seed_records.py`, `skc_rebuild_c4_latent.py` | Repo root is now three levels up (was two) |
| `src/driver_scripts/skc_export_records.py` | `parents[1]` → `parents[2]` for `VERSION3/code` |
| `CLAUDE.md` | Paths updated to the new layout |

## Added

`README.md`, `INDEX.md`, `HF_MANIFEST.md`, `.gitignore`, `.env.example`, `environment.yml`,
`requirements-freeze.txt`, `scripts/{download_data,upload_to_hf,download_from_hf,run_analysis}.sh`,
`docs/SESSION_HISTORY.md`, `docs/RESTRUCTURE_LOG.md`, `slurm_jobs/README.md`,
`slurm_jobs/archive/README.md`, `results/README.md`.

In the generated env files, `circuit-tracer`'s editable local path was replaced by its pinned
upstream commit, and the PyTorch cu124 index URL was added so `torch==2.4.1+cu124` resolves.

## Verification after the restructure

| Check | Result |
|---|---|
| `src/skc_pipeline_test.py` | 51/51 passed (same as before) |
| `src/skc_metrics.py --smoke-test` | Passed |
| `VERSION3/code/tests/test_smoke.py` | Passed |
| Import of all 34 driver scripts from `src/driver_scripts/` | Clean |
| `bash -n` on all 155 Slurm scripts and 4 helper scripts | Clean |
| `slurm_jobs/01_run_offline_tests.sh`, run for real | Passed. Repo-root detection also tested from `slurm_jobs/`, `slurm_jobs/archive/`, the repo root, and with `$SKC_REPO`; it fails loudly from outside the repo |
| `skc_build_c4_fit_queries.py` re-run | Identical to `data/c4_fit_queries.json` |
| `scripts/run_analysis.sh`, run end to end against backups | Every output reproduced: VERSION3 records (same rows), `reconcile`/`ablations` JSON identical, `c4_latent_rebuild` byte-identical |
| `scripts/download_data.sh` MD5 logic | Checked offline against the archived clone (the clone step itself not run: no git, no internet on the compute node) |
| HF include patterns | Checked with `huggingface_hub.utils.filter_repo_objects` against the real tree: they match exactly the gitignored large files (643 core + 40 graphs) and nothing else |
| Secrets | The real `HF_TOKEN` value appears nowhere outside `.env` (logs and archive included); no generic token patterns in tracked files |

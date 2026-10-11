# Slurm jobs

Every job behind a number the paper currently reports, plus the evidence for the
three disclosed-null models. Superseded jobs are in [`archive/`](archive/README.md).

## How to submit

```bash
cd slurm_jobs
sbatch 97_e0_gemma9b_layer8_splitfix.sh
```

- **Repo root:** each job finds the repo root itself. It walks up from the directory you ran
  `sbatch` in until it finds `src/skc_data.py`, so submit from anywhere inside the repo. To
  submit from outside, `export SKC_REPO=/path/to/repo`. Don't hard-code paths back in: under
  `sbatch` the script runs from a spool copy, so `dirname "$0"` is not the repo.
- **Environment:** jobs `source "$HOME/miniconda3/etc/profile.d/conda.sh"` and
  `conda activate skc_iclr27`. They also source `$REPO/.env` for `HF_TOKEN` (see
  `.env.example`).
- **Cluster settings:** `#SBATCH -p/-A/-q` lines are for QCRI's Panther cluster (A100:
  `gpu-A100`/`a100_hbku`; H200: `gpu-H200`/`h200_hbku`). Edit them for your own cluster.
- **CUDA:** no job loads a CUDA module, on purpose. torch 2.4.1+cu124 ships its own runtime,
  and a system `cuda12.1` module caused `undefined symbol: __nvJitLinkComplete_12_4`.
- **Logs:** `.out`/`.err` logs land in the submit directory and are gitignored. The logs from
  the original runs are in `logs/slurm/` (on HF).
- **Dependencies:** a job that depends on another says so in its header. Do not submit an E1
  job before its own E0 has **passed** (`gate_open: true` in `E0.json`).

Budgets below are the `#SBATCH -t` limits, not measured use (the cluster gave no
`sacct` access from the compute nodes). Total requested walltime across these jobs is
about 923 GPU-hours (716 A100, 206 H200).

## Setup and data

| Job | What it does | Output |
|---|---|---|
| `00_setup_env.sh` | Creates the `skc_iclr27` conda env, torch 2.4.1+cu124 first, then `requirements.txt` | conda env |
| `01_run_offline_tests.sh` | CPU: `skc_metrics.py --smoke-test` + `skc_pipeline_test.py` (51 checks) | stdout |
| `02_build_pilot_dataset.sh` | Builds the two-hop set from Ferrando et al.'s entity files (run `scripts/download_data.sh` first) | `data/twohop_queries.json` |
| `23_build_threehop_dataset.sh` | Live Wikidata SPARQL pull of player→city→country→continent chains | `data/threehop_queries.json` |

## E0: positive control, with the corrected (non-leaking) entity split

**In-scope models: stable on all 5 seeds.**

| Job | Model / layer | Seed | Output |
|---|---|---|---|
| `97` | Gemma 2 9B / 8 | 0 | `results/e0_gemma9b_layer8_splitfix` |
| `116 118 120 122` | Gemma 2 9B / 8 | 1-4 | `results/e0_gemma9b_layer8_seed{1..4}` |
| `102` | Llama 3.1 70B, all 80 layers | 0 | `results/e0_layersweep_llama70b` (found layers 17-35, peak 25) |
| `103` | Llama 3.1 70B / 25 | 0 | `results/e0_llama70b_layer25_splitfix` |
| `124 126 128 130` | Llama 3.1 70B / 25 | 1-4 | `results/e0_llama70b_layer25_seed{1..4}` |

**Disclosed nulls: evidence that no stable layer exists.**

| Job | What it shows |
|---|---|
| `89 90 91 92 93` | Gemma 2 2B at layers 8/12/16/19/7 under the fixed split. Only 16 and 19 pass at seed 0. |
| `108 110 112 114` | Gemma 2 2B / 16, seeds 1-4: all fail. The seed-0 pass was a fluke. |
| `132-136` | Gemma 2 2B, all 26 layers × 5 seeds: 0/130 pass. |
| `96`, `100` | Llama 3.1 8B: layer 8 fails; all 32 layers fail. |
| `98`, `101` | Qwen3 32B: layer 24 fails; all 64 layers fail. |
| `99` | Llama 3.1 70B at its OLD layer 8 fails under the fix, which is why it moved to layer 25. |

## E1: gate vs. composition outcome (surface score g_S, labels Y/K1/K2/M)

| Job | Model | Notes |
|---|---|---|
| `106`, `107` | Gemma 2 9B, Llama 3.1 70B | Seed 0, `--skip-bridge-resolution` (Patchscopes fails at both layers) |
| `117-123` (odd), `125-131` (odd) | same | Seeds 1-4. Also export VERSION3 records via `--export-records`. |
| `137`, `138` | same | Disjoint **fit split** (`data/c4_fit_queries.json`), used to fit the C4 standardizer |
| `104`, `105`, `95` | Gemma 2 9B, Llama 3.1 70B, Gemma 2 2B / 8 | Patchscopes bridge-resolution attempted: **0%** resolved |
| `94` | Gemma 2 2B / 16 | The only setting where Patchscopes resolves bridges (model out of scope) |

> **Known bug:** the `acts_bridge_filtered.npy` / `g_B` these jobs save comes from
> `skc_harness.capture_for_queries()`'s hop-2 prompt, which **names the gold bridge**.
> It is not a latent read. Use E3-latent (below) for `g_B`.

## E3-latent: the corrected bridge read (`g_B`)

| Job | Model | Population |
|---|---|---|
| `152`, `153` | Gemma 2 9B / 8, Llama 3.1 70B / 25 | Eval set. Offset sweep −1/0/+1 inside the composed prompt; peak −1 for both, p<0.001 vs. permutation null. |
| `154`, `155` | same | Fit split, at the peak offset only |

Then run `python3 src/driver_scripts/skc_rebuild_c4_latent.py` (CPU) to get `results/c4_latent_rebuild/`.

## E4: 13 baselines

| Job | Model | Baselines |
|---|---|---|
| `139` | Gemma 2 9B | P(True), SAPLMA, entity-token probe, HaloScope, INSIDE, SEP, self-consistency, SelfCheckGPT, semantic entropy, self-ask, operadic consistency |
| `140` | Gemma 2 9B | Lookback Lens, ICR probe (second, eager-attention model instance) |
| `145`, `146`, `147`, `148` | Llama 3.1 70B | Same set as 139, split 4 ways (HaloScope and SEP alone) after 24h timeouts |
| `149` | Llama 3.1 70B | Lookback Lens, ICR probe, with `--sequential-model-load` (two 4-bit 70B copies do not fit) |

> The BGR-min / BGR-lr rows inside these `E4.json` files use the contaminated `g_B`
> (see E1 note). Bridge-only, BGR-min and BGR-mean are recomputed correctly by the C4
> rebuild; **BGR-lr has not been recomputed yet**.

## E7: named-bridge (a) and length-matched distractor (c) controls

| Job | Model |
|---|---|
| `150` | Gemma 2 9B / 8 |
| `151` | Llama 3.1 70B / 25 |

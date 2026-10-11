# results/

One directory per job run, named `<experiment>_<model>_layer<L>[_variant]`. Small JSON
outputs are in git. Large binaries (`*.npy` activations, `*.pt` graphs,
`*_raw_generations.json`) are on Hugging Face; see [`../HF_MANIFEST.md`](../HF_MANIFEST.md)
and `scripts/download_from_hf.sh`. Nothing here was moved during the restructure, so every
job script's `--out` path still points at its own directory.

Typical files:

| File | Contents |
|---|---|
| `E0.json` | Per-estimator AUROC, cross-estimator agreement, `gate_open` |
| `E1.json` | AUROC(gate→K1/M/Y) |
| `E4.json` | One block per selector: AUROC, AURC, risk at 50% coverage |
| `e3_full_map.json` | (layer, offset) → AUROC(gate→K2), plus permutation null |
| `frozen_gate_*.json` | The fitted direction itself |
| `*_labels_filtered.json` | Per-item Y/K1/K2/M |

## Current: backs a number the paper reports now

| Directory | What |
|---|---|
| `e0_gemma9b_layer8_splitfix`, `e0_gemma9b_layer8_seed{1..4}` | Gemma 2 9B positive control, 5 seeds (agreement 0.50-0.52) |
| `e0_llama70b_layer25_splitfix`, `e0_llama70b_layer25_seed{1..4}` | Llama 3.1 70B positive control, 5 seeds (agreement 0.60-0.63) |
| `e0_layersweep_llama70b` | 80-layer sweep that located layer 25 (passing band 17-35) |
| `e1_gemma9b_layer8_skipbridge_splitfix`, `e1_gemma9b_layer8_seed{1..4}` | E1, Gemma 2 9B: `g_S` and labels are valid; `acts_bridge_*` is the contaminated named-bridge read |
| `e1_llama70b_layer25_skipbridge_splitfix`, `e1_llama70b_layer25_seed{1..4}` | Same for Llama 3.1 70B |
| `e1_gemma9b_layer8_fitsplit_c4`, `e1_llama70b_layer25_fitsplit_c4` | Fit-split `g_S` for the C4 standardizer |
| `e3_gemma9b_layer8_latent_splitfix`, `e3_llama70b_layer25_latent_splitfix` | **Corrected latent `g_B`**, eval set (`peak_activations.npy`, `peak_labels.json`) |
| `e3_gemma9b_layer8_latent_fitsplit`, `e3_llama70b_layer25_latent_fitsplit` | Corrected latent `g_B`, fit split |
| `c4_latent_rebuild` | **Headline C4 result**: bridge-only vs BGR-min/mean, with bootstrap CIs (Section 4.3) |
| `e4_gemma9b_layer8_main`, `e4_gemma9b_layer8_lookback` | 13 baselines, Gemma 2 9B (Table tab:bgr-full) |
| `e4_llama70b_layer25_partA1`, `_partA2`, `_partB1`, `_partB2`, `_lookback_h200` | 13 baselines, Llama 3.1 70B, all on H200 (Table tab:bgr-full) |
| `e7ac_gemma9b_layer8_splitfix`, `e7ac_llama70b_layer25_splitfix` | Named-bridge (a) and distractor (c) controls |

## Disclosed-null evidence: why three models are out of scope

| Directory | What |
|---|---|
| `e0_gemma2b_layer{7,8,12,16,19}_splitfix`, `e0_gemma2b_layer16_seed{1..4}` | Gemma 2 2B: layer 16's seed-0 pass does not survive seeds 1-4 |
| `e0_layersweep_gemma2b_seed{0..4}` | Gemma 2 2B, 26 layers × 5 seeds: 0/130 pass |
| `e0_llama8b_layer8_splitfix`, `e0_layersweep_llama8b` | Llama 3.1 8B: 0/32 layers pass |
| `e0_qwen32b_layer24_splitfix`, `e0_layersweep_qwen32b` | Qwen3 32B: 0/64 layers pass |
| `e0_llama70b_layer8_splitfix` | Llama 3.1 70B's old layer 8 fails after the fix |
| `e1_gemma2b_layer16_splitfix` | The only setting where Patchscopes confirms bridge resolution |
| `e1_{gemma2b_layer8,gemma9b_layer8,llama70b_layer25}_bridgeresolved_splitfix` | Patchscopes resolves 0% of bridges |
| `e4_llama70b_layer25_lookback` | A100 run; differs from H200 by 0.001-0.011 AUROC; superseded by `_lookback_h200` |

## Superseded: pre-C7-fix (leaky split), old layers, or out-of-scope models

These are kept for traceability. Several are still cited by manuscript sections not yet
rewritten for the two-model scope (abstract/intro, §5.1-5.4, `tab:compound-allmodels`,
the controls appendix):

- **E0:** `e0_*_v2`; `e0_gemma2b_layer{8,12,16}`, `_layer8_sae`; the `e0_*_it_*` variants;
  `e0_gemma9b_layer8`, `e0_llama8b_layer8`, `e0_llama70b_layer8`, `e0_qwen32b_layer{8,24}`;
  `e0_layersweep_gemma2b` (leaky split).
- **E1:** `e1_gemma2b_layer{8,16}`, `e1_gemma9b_layer8`, `e1_llama70b_layer8`,
  `e1_llama8b_layer8`, `e1_qwen32b_layer24`.
- **E2-E3:** `e2_*` (causal arms; refusal is 0.000 everywhere); `e3_*` without `latent` in
  the name.
- **E4:** `e4_gemma2b*`, `e4_llama70b_layer8`.
- **E5:** `e5_*` (three-hop compounding).
- **E6:** `e6_*` (chain of thought).
- **E7:** `e7_*`, `e7c_*`, `e7e_*`, `e7g_*`, `e7i_*`, `e7j_*`, `e7n_*` (controls, mostly
  Gemma 2 2B), and `e7_popularity_*`.
- **Other:** `pilot`; `c4_graphs_gemma2b` (attribution graphs, 9.5 GB on HF, optional
  group).

## Empty: the job died before writing output

`e1_gemma2b_layer12_patchscope`, `e4_llama70b_layer25_main`, `e4_llama70b_layer25_partA`,
and `e4_llama70b_layer25_partB`. The last three were 24h timeouts; see
`slurm_jobs/archive/README.md`.

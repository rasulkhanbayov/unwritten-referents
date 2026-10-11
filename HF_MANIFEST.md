# Hugging Face manifest

Large artifacts not stored in git (see `.gitignore`). Upload with `scripts/upload_to_hf.sh`
and restore with `scripts/download_from_hf.sh`; paths inside the dataset repo mirror this repo, so files
land back in place. Sizes are measured from the files on disk at restructure time (2026-10-10).

You only need these to re-run the **CPU** analysis (`scripts/run_analysis.sh`, which reads saved
activations) without re-running GPU jobs, or to inspect raw model generations. Every GPU job
regenerates its own artifacts.

| Group | Patterns | Files | Size |
|---|---|---|---|
| `core` (default) | `results/**/*.npy`, `results/**/*_raw_generations.json`, `logs/slurm/*` | 643 | 7.56 GB |
| `graphs` (`--all`) | `results/**/*.pt` (archived Gemma 2 2B C4 attribution graphs) | 40 | 9.53 GB |
| **total** | | 683 | 17.08 GB |

Contents:

- `*.npy`: captured residual-stream activations (`held_acts`, `acts_surface_filtered`,
  `acts_bridge_filtered`, `peak_activations`, ...). Each has a small `.npy.meta.json` sidecar in git
  recording model and layer.
- `*_raw_generations.json`: every prompt and greedy generation behind the K1/K2/Y labels. Needed to
  re-score labels, e.g. with an LLM judge (README -> Open issues, Llama 3.1 70B K2 labels).
- `logs/slurm/`: stdout/stderr of the original runs (440 files, 3 MB), the only record of e.g. the
  24h timeouts.

## Per-directory sizes

| Directory | Group | Size |
|---|---|---|
| `results/e0_llama70b_layer25_seed3` | core | 368 MB |
| `results/e0_llama70b_layer25_seed2` | core | 366 MB |
| `results/e0_llama70b_layer25_splitfix` | core | 366 MB |
| `results/e0_llama70b_layer8_splitfix` | core | 366 MB |
| `results/e0_llama70b_layer25_seed1` | core | 366 MB |
| `results/e0_llama70b_layer25_seed4` | core | 365 MB |
| `results/e0_llama70b_layer8` | core | 337 MB |
| `results/e0_qwen32b_layer24_splitfix` | core | 232 MB |
| `results/e0_qwen32b_layer8` | core | 214 MB |
| `results/e0_qwen32b_layer24` | core | 214 MB |
| `results/e0_llama8b_layer8_splitfix` | core | 187 MB |
| `results/e0_llama8b_it_layer8` | core | 173 MB |
| `results/e0_llama8b_layer8_v2` | core | 173 MB |
| `results/e0_llama8b_layer8` | core | 173 MB |
| `results/e0_gemma9b_layer8_seed3` | core | 166 MB |
| `results/e0_gemma9b_layer8_seed2` | core | 165 MB |
| `results/e0_gemma9b_layer8_splitfix` | core | 165 MB |
| `results/e0_gemma9b_layer8_seed1` | core | 165 MB |
| `results/e0_gemma9b_layer8_seed4` | core | 164 MB |
| `results/e0_gemma9b_layer8` | core | 152 MB |
| `results/e0_gemma9b_layer8_v2` | core | 152 MB |
| `results/e0_gemma2b9b_it_layer16` | core | 152 MB |
| `results/e0_gemma2b9b_it_layer8` | core | 152 MB |
| `results/e0_gemma2b_it_layer8_n30000` | core | 149 MB |
| `results/e0_gemma2b_layer16_seed3` | core | 110 MB |
| `results/e0_gemma2b_layer16_seed2` | core | 109 MB |
| `results/e0_gemma2b_layer16_splitfix` | core | 109 MB |
| `results/e0_gemma2b_layer7_splitfix` | core | 109 MB |
| `results/e0_gemma2b_layer12_splitfix` | core | 109 MB |
| `results/e0_gemma2b_layer8_splitfix` | core | 109 MB |
| `results/e0_gemma2b_layer19_splitfix` | core | 109 MB |
| `results/e0_gemma2b_layer16_seed1` | core | 109 MB |
| `results/e0_gemma2b_layer16_seed4` | core | 109 MB |
| `results/e0_gemma2b_layer8_sae` | core | 101 MB |
| `results/e0_gemma2b_layer8_v2` | core | 101 MB |
| `results/e0_gemma2b_layer16` | core | 101 MB |
| `results/e0_gemma2b_layer8` | core | 101 MB |
| `results/e0_gemma2b_layer12` | core | 101 MB |
| `results/e0_gemma2b_it_layer16` | core | 101 MB |
| `results/e0_gemma2b_it_layer8` | core | 101 MB |
| `results/e3_llama70b_layer25_latent_fitsplit` | core | 41 MB |
| `results/e1_llama70b_layer25_seed2` | core | 32 MB |
| `results/e1_llama70b_layer25_seed3` | core | 32 MB |
| `results/e1_llama70b_layer25_seed1` | core | 32 MB |
| `results/e1_llama70b_layer25_skipbridge_splitfix` | core | 32 MB |
| `results/e1_llama70b_layer8` | core | 32 MB |
| `results/e1_llama70b_layer25_seed4` | core | 32 MB |
| `results/e1_llama70b_layer25_fitsplit_c4` | core | 19 MB |
| `results/e3_gemma9b_layer8_latent_fitsplit` | core | 18 MB |
| `results/e3_llama70b_layer25_latent_splitfix` | core | 16 MB |
| `results/e1_llama8b_layer8` | core | 11 MB |
| `results/e1_gemma9b_layer8` | core | 10 MB |
| `results/e1_gemma9b_layer8_seed2` | core | 10 MB |
| `results/e1_gemma9b_layer8_skipbridge_splitfix` | core | 10 MB |
| `results/e1_gemma9b_layer8_seed1` | core | 10 MB |
| `results/e1_gemma9b_layer8_seed4` | core | 10 MB |
| `results/e1_gemma9b_layer8_seed3` | core | 10 MB |
| `results/e3_gemma9b_layer8_latent_splitfix` | core | 5 MB |
| `results/e1_gemma2b_layer8` | core | 4 MB |
| `results/e1_qwen32b_layer24` | core | 4 MB |
| `results/e1_gemma2b_layer16_splitfix` | core | 3 MB |
| `results/e1_gemma2b_layer16` | core | 3 MB |
| `results/pilot` | core | 3 MB |
| `results/e1_gemma9b_layer8_fitsplit_c4` | core | 3 MB |
| `results/c4_graphs_gemma2b` | graphs | 9.53 GB |
| `logs/slurm` | core | 3 MB |

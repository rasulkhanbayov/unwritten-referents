# Archived Slurm jobs

Jobs that do **not** back any number the paper currently reports. They are kept so
every historical result in `results/` stays traceable to the exact command that
produced it, and they still run: each job locates the repo root by itself (see
`slurm_jobs/README.md`), and their output directories under `results/` were not moved.

Some of these are still cited by manuscript sections not yet rewritten for the
two-model scope. Those sections are marked *(pending rewrite)* below and listed in
`README.md` -> Open issues.

| Job | Why archived |
|---|---|
| `03_pilot_gemma2b.sh` | Pilot run on Gemma 2 2B (300 queries), first end-to-end smoke test on a real GPU. |
| `04_e0_stopgate_gemma2b.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `05_e1_main_gemma2b.sh` | E1 on the pre-fix gate / old layers. Superseded by 106/107 and seeds 117-131. |
| `06_e3_bridge_sweep_gemma2b.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `07_e2_causal_gemma2b.sh` | E2 causal interventions / steering-coefficient sweep, pre-C7-fix. Refusal rate is 0.000 everywhere, so all arms are inert (sec:res-causal). |
| `08_e4_repair_baselines_gemma2b.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `09_e5_khop_gemma2b.sh` | E5 three-hop compounding, pre-C7-fix, old layers. Still cited in tab:compound-allmodels (pending rewrite). |
| `10_e6_cot_gemma9b_it.sh` | E6 chain-of-thought (F3) on instruction-tuned models, pre-C7-fix. Still cited in sec:res-cot / CoT appendix (pending rewrite). |
| `11_e7_ablations_gemma2b.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `12_e0_scalecheck_llama70b_h200.sh` | Early H200 scale check on Llama 3.1 70B, layer 8, before the C7 split-leakage fix. |
| `13_diag_patchscope_layers.sh` | One-off diagnostic (Patchscopes layer/position, tokenizer span, activation collision). Findings recorded in docs/EXECUTION_PLAN_full.md. |
| `14_diag_bridge_vs_final.sh` | One-off diagnostic (Patchscopes layer/position, tokenizer span, activation collision). Findings recorded in docs/EXECUTION_PLAN_full.md. |
| `15_diag_bridge_during_gen.sh` | One-off diagnostic (Patchscopes layer/position, tokenizer span, activation collision). Findings recorded in docs/EXECUTION_PLAN_full.md. |
| `16_diag_bridge_k1filtered.sh` | One-off diagnostic (Patchscopes layer/position, tokenizer span, activation collision). Findings recorded in docs/EXECUTION_PLAN_full.md. |
| `17_diag_tokenize_mismatch.sh` | One-off diagnostic (Patchscopes layer/position, tokenizer span, activation collision). Findings recorded in docs/EXECUTION_PLAN_full.md. |
| `18_e0_stopgate_gemma2b_it.sh` | E0 on instruction-tuned variants, pre-C7-fix. Not part of the current two-model scope. |
| `19_e0_stopgate_gemma2b_it_layer8.sh` | E0 on instruction-tuned variants, pre-C7-fix. Not part of the current two-model scope. |
| `20_e0_stopgate_gemma2b_it_layer8_n30000.sh` | E0 on instruction-tuned variants, pre-C7-fix. Not part of the current two-model scope. |
| `21_e0_stopgate_gemma2b9b_it.sh` | E0 on instruction-tuned variants, pre-C7-fix. Not part of the current two-model scope. |
| `22_e0_stopgate_gemma9b_it_layer8.sh` | E0 on instruction-tuned variants, pre-C7-fix. Not part of the current two-model scope. |
| `24_diag_activation_collision.sh` | One-off diagnostic (Patchscopes layer/position, tokenizer span, activation collision). Findings recorded in docs/EXECUTION_PLAN_full.md. |
| `25_diag_activation_collision_reproduce.sh` | One-off diagnostic (Patchscopes layer/position, tokenizer span, activation collision). Findings recorded in docs/EXECUTION_PLAN_full.md. |
| `26_e0_stopgate_gemma2b_layer8_v2.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `27_e1_main_gemma2b_layer8_skip_patchscopes.sh` | E1 on the pre-fix gate / old layers. Superseded by 106/107 and seeds 117-131. |
| `28_e3_bridge_sweep_gemma2b_layer8.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `29_e4_repair_baselines_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `30_e5_khop_gemma2b_layer8.sh` | E5 three-hop compounding, pre-C7-fix, old layers. Still cited in tab:compound-allmodels (pending rewrite). |
| `31_e7_ablations_gemma2b_layer8.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `32_e7_popularity_match_gemma2b_layer8.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `33_e2_causal_gemma2b_layer8.sh` | E2 causal interventions / steering-coefficient sweep, pre-C7-fix. Refusal rate is 0.000 everywhere, so all arms are inert (sec:res-causal). |
| `34_e0_stopgate_gemma2b_layer8_sae.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `35_e0_stopgate_gemma9b_layer8.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `36_e0_stopgate_llama8b_layer8.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `37_e0_stopgate_llama70b_layer8.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `38_e1_main_llama70b_layer8.sh` | E1 on the pre-fix gate / old layers. Superseded by 106/107 and seeds 117-131. |
| `39_e3_bridge_sweep_llama70b_layer8.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `40_e5_khop_llama70b_layer8.sh` | E5 three-hop compounding, pre-C7-fix, old layers. Still cited in tab:compound-allmodels (pending rewrite). |
| `41_e4_repair_baselines_llama70b_layer8.sh` | Baseline (E4) suite on Llama 3.1 70B at the old layer 8, pre-C7-fix. Superseded by 145-149 (layer 25, fixed split). |
| `42_e7_ablations_llama70b_layer8.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `43_e0_stopgate_gemma9b_layer8_skipsae.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `44_e0_stopgate_llama8b_layer8_skipsae.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `45_e1_main_gemma9b_layer8.sh` | E1 on the pre-fix gate / old layers. Superseded by 106/107 and seeds 117-131. |
| `46_e3_bridge_sweep_gemma9b_layer8.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `47_e5_khop_gemma9b_layer8.sh` | E5 three-hop compounding, pre-C7-fix, old layers. Still cited in tab:compound-allmodels (pending rewrite). |
| `48_e1_main_llama8b_layer8.sh` | E1 on the pre-fix gate / old layers. Superseded by 106/107 and seeds 117-131. |
| `49_e3_bridge_sweep_llama8b_layer8.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `50_e5_khop_llama8b_layer8.sh` | E5 three-hop compounding, pre-C7-fix, old layers. Still cited in tab:compound-allmodels (pending rewrite). |
| `51_e4_saplma_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `52_e4_entitytoken_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `53_e4_lookbacklens_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `54_e4_haloscope_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `55_e4_inside_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `56_e4_icrprobe_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `57_e4_sep_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `58_e4_selfconsistency_selfcheckgpt_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `59_e4_semanticentropy_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `60_e7g_counterfactual_gemma2b.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `61_e7j_crosstype_gemma2b.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `62_e7c_distractor_gemma2b.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `63_e7i_paraphrase_gemma2b.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `64_e0_stopgate_qwen32b_layer8.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `65_e1_main_qwen32b_layer8.sh` | E1 on the pre-fix gate / old layers. Superseded by 106/107 and seeds 117-131. |
| `66_e3_bridge_sweep_qwen32b_layer8.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `67_e5_khop_qwen32b_layer8.sh` | E5 three-hop compounding, pre-C7-fix, old layers. Still cited in tab:compound-allmodels (pending rewrite). |
| `68_e0_stopgate_qwen32b_layer24.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `69_e1_main_qwen32b_layer24.sh` | E1 on the pre-fix gate / old layers. Superseded by 106/107 and seeds 117-131. |
| `70_e3_bridge_sweep_qwen32b_layer24.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `71_e5_khop_qwen32b_layer24.sh` | E5 three-hop compounding, pre-C7-fix, old layers. Still cited in tab:compound-allmodels (pending rewrite). |
| `72_e0_stopgate_gemma2b_layer12_patchscope.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `73_e1_main_gemma2b_layer12_patchscope.sh` | E1 on the pre-fix gate / old layers. Superseded by 106/107 and seeds 117-131. |
| `74_e3_fulllayersweep_gemma2b.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `75_e0_stopgate_llama8b_it_layer8.sh` | E0 on instruction-tuned variants, pre-C7-fix. Not part of the current two-model scope. |
| `76_e6_cot_llama8b_it.sh` | E6 chain-of-thought (F3) on instruction-tuned models, pre-C7-fix. Still cited in sec:res-cot / CoT appendix (pending rewrite). |
| `78_e7_popularity_match_gemma9b_it.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `79_e3_bridge_sweep_gemma9b_it_layer8.sh` | E3 offset sweep on the pre-fix gate / old layers. Superseded by 152-155 (latent read on the fixed gate). |
| `80_e2_coefficientsweep_gemma9b_it.sh` | E2 causal interventions / steering-coefficient sweep, pre-C7-fix. Refusal rate is 0.000 everywhere, so all arms are inert (sec:res-causal). |
| `81_e7e_attention_gemma2b.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `82_e2_c4_graphtrace_gemma2b.sh` | C4 attribution graphs on Gemma 2 2B (circuit-tracer). Model out of scope; graphs (9.5 GB .pt) are on HF as the optional 'graphs' group. |
| `83_e7n_artifactprobe_gemma2b.sh` | E7 control on the pre-fix gate (mostly Gemma 2 2B). Still cited in the controls appendix (pending rewrite); (a)/(c) redone in 150/151. |
| `84_e0_layersweep_gemma2b.sh` | E0 positive control on the LEAKY entity split (before the C7 fix). Superseded by 89-103 and the multi-seed runs. |
| `85_e4_rho_allbaselines_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `86_e4_ptrue_operadic_oracle_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `87_e4_ptrue_operadic_oracle_gemma2b_layer8_fixed.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `88_e4_inside_rho_gemma2b_layer8.sh` | Baseline (E4) suite on Gemma 2 2B, pre-C7-fix. Model out of scope; superseded by 139/140/145-149 on the in-scope models. |
| `109_e1_gemma2b_layer16_seed1.sh` | Never run: E1 for Gemma 2 2B seeds 1-4, built before those seeds' own E0 (jobs 108/110/112/114) failed the positive control. |
| `111_e1_gemma2b_layer16_seed2.sh` | Never run: E1 for Gemma 2 2B seeds 1-4, built before those seeds' own E0 (jobs 108/110/112/114) failed the positive control. |
| `113_e1_gemma2b_layer16_seed3.sh` | Never run: E1 for Gemma 2 2B seeds 1-4, built before those seeds' own E0 (jobs 108/110/112/114) failed the positive control. |
| `115_e1_gemma2b_layer16_seed4.sh` | Never run: E1 for Gemma 2 2B seeds 1-4, built before those seeds' own E0 (jobs 108/110/112/114) failed the positive control. |
| `141_e4_llama70b_layer25_baselines_main.sh` | Timed out at 24h with nothing saved: fit split relabelled 3x (bug, now fixed). Replaced by 145-148. |
| `142_e4_llama70b_layer25_baselines_lookback.sh` | Succeeded on A100, but surface_gate/BGR differ from the H200 runs by 0.001-0.011 AUROC (hardware). Superseded by 149 (H200). |
| `143_e4_llama70b_layer25_baselines_partA.sh` | Timed out at 24h (HaloScope / Semantic Entropy Probe fit too slow on 70B). Split further into 145-148. |
| `144_e4_llama70b_layer25_baselines_partB.sh` | Timed out at 24h (HaloScope / Semantic Entropy Probe fit too slow on 70B). Split further into 145-148. |

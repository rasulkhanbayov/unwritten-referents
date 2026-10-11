#!/bin/bash -l
#SBATCH -J skc_e0_gemma2b
#SBATCH -o e0_gemma2b_%j.out
#SBATCH -e e0_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# STEP 3: E0, THE STOP GATE.
#
# STATUS NOTE: the pilot (03) actually came back INVERTED
# (AUC(S->K1)=0.819, AUC(S->K2)=0.712, both clearly above chance) on
# n=300, layer 12, diffmeans only, with only 7 items in the decisive
# K1=1,K2=0 cell. Per skc_runbook.txt, INVERTED is a real result ("the
# model does compose self-knowledge") that must not be tuned away -- but
# the pilot explicitly is NOT a result on its own (one small sample, one
# estimator, no held-out-entity split, no cross-estimator agreement check).
# Running E0 now is the right next step BECAUSE OF the INVERTED pilot, not
# despite it: E0 is what actually verifies the gate is a trustworthy
# instrument (proper fit/held-out entity split, >=2/3 estimators agreeing)
# before the INVERTED signal can be treated as anything more than a
# small-sample curiosity. If E0 also comes back informative-but-composing
# on K2, that is the real signal to escalate on skc_runbook.txt's
# FLAG-BACK TRIGGERS -- not the pilot alone.
#
# --skip-sae is passed here DELIBERATELY on the first attempt. skc_run_e0.py
# raises NotImplementedError on the SAE branch until sae_lens's exact
# encode() call is verified against the pinned sae_lens>=3.0 version (see
# that script's comments) -- something that needs a live GPU session to
# check, not something to guess offline. Running with --skip-sae first
# gets you 2 of 3 estimators (diffmeans, probe), which is enough to clear
# E0's own gate (>=2/3 passing) while the SAE wiring is verified separately.
# Once verified, remove --skip-sae and re-run -- the paper needs the SAE
# estimator too (skc_runbook.txt requires all three where available).
#
# PARTITION: gpu-A100, same reasoning as the pilot -- Gemma-2-2B/9B and
# Llama-3.1-8B all fit in bf16 well under 80GB. Reserve H200 for the 70B/32B
# 4-bit scale-check runs (see 09_e0_scalecheck_70b.sh) where 80GB is tight.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here. torch==2.4.1+cu124's wheel
# bundles its own complete CUDA 12.4 runtime (cusparse, nvJitLink, etc.) and
# needs nothing from the system toolkit -- nothing in this pipeline compiles
# a custom CUDA extension. `module load cuda12.1/toolkit/12.1.1` was tried
# here originally and caused a real failure: it prepends an OLDER cusparse/
# nvJitLink onto LD_LIBRARY_PATH ahead of torch's own bundled cu124 libs,
# producing `undefined symbol: __nvJitLinkComplete_12_4` on the first real
# GPU job. Do not re-add a module load line to "fix" a CUDA-looking error
# without checking whether the module itself is the cause first.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }

# google/gemma-2-2b (and every other Gemma checkpoint) is a GATED model on
# HuggingFace -- Google requires an authenticated, license-accepted account
# to download it, or transformers fails with GatedRepoError/401 even though
# the network path itself works. .env holds HF_TOKEN=... (never echo this
# file's contents or commit/zip it into the submission -- see EXECUTION_PLAN.md).
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e0_gemma2b_layer16"
mkdir -p "$OUT"

# --n and --layer history:
#   Attempts at n=4000, 4000 (again), 16000, 24000, all at LAYER 12, gave
#   agreement -0.198, 0.229, 0.467, -0.201 -- both individual AUROCs
#   improved and stabilized above 0.65 (diffmeans up to 0.687, probe up to
#   0.751 by n=24000), but agreement itself did NOT improve monotonically
#   with more data. That ruled out "just needs more samples."
#
#   ROOT CAUSE FOUND by recomputing gate_agreement() OFFLINE from the
#   n=24000 run's saved held_acts.npy + frozen_gate_*.json (no GPU needed):
#   14.16% of held-out layer-12 activations carry a "massive activation"
#   signature -- norm ~9x the typical value, and among the outlier rows
#   there are only 2 DOMINANT near-identical activation vectors (not
#   independent outliers; one feature dimension alone reaches ~584 in an
#   otherwise-normal-scale vector). This is a documented phenomenon in
#   transformer residual streams, not a bug in this bundle's span-finding or
#   capture code (verified directly: entity spans and tokenization for the
#   affected queries are computed correctly). fit_probe standardizes
#   features before fitting and is comparatively robust to this; fit_diffmeans
#   has no normalization (deliberately, per skc_gates.py's own docstring: "no
#   hyperparameters, so nothing to tune away") and is highly sensitive to it.
#   Excluding just these outlier points from the SAME n=24000 held-out data
#   flips the recomputed correlation from -0.20 to +0.17.
#
#   --layer 8 (n=8000, then n=20000) CONFIRMED the layer hypothesis but did
#   NOT resolve it: outlier fraction dropped from 14.16% (layer 12) to
#   7.04% (layer 8), and agreement at n=8000 reached 0.399 (second-best of
#   any attempt). But n=20000 on the SAME layer 8 gave agreement 0.085 --
#   WORSE than n=8000, breaking the "more data helps" pattern a second time,
#   on a second layer. Across all 7 real attempts (2 layers x up to 5
#   sample sizes), agreement bounced between roughly -0.20 and +0.47 with
#   no stable relationship to sample size OR layer. Both individual AUROCs
#   were consistently >= 0.65 across nearly every attempt -- the instability
#   is specifically in cross-estimator AGREEMENT, not in either estimator's
#   own signal.
#
#   FIXED AT THE SOURCE instead of continuing to search sample sizes/layers:
#   fit_diffmeans() in skc_gates.py now standardises each activation
#   dimension by its pooled standard deviation before taking the class-mean
#   difference (folding the scaling back into the returned direction, same
#   pattern fit_probe already used) -- this directly targets the confirmed
#   mechanism: massive-activation outliers on a few dimensions were
#   dominating the UNSTANDARDISED mean, which fit_probe's own
#   standardisation already protected it from. Both offline test suites
#   re-verified passing after the change (50/0 pipeline test, full smoke
#   test), including the diffmeans-specific separation check (AUROC 0.937
#   on synthetic data, unchanged from before the fix).
#
#   layer 8 / n=20000 with the FIXED fit_diffmeans PASSED: diffmeans AUROC
#   0.759, probe AUROC 0.742, agreement 0.620 -- the best result of any
#   attempt, and it stayed passed. Results preserved at
#   results/e0_gemma2b_layer8/ for reference; this is NOT a re-run of a
#   failure.
#
#   BUT: Patchscopes (skc_harness.py::patchscope_decode(),
#   bridge_resolved_patchscope()) needs a layer where the BRIDGE ENTITY
#   ITSELF decodes cleanly when patched into a fresh identity prompt --
#   a DIFFERENT requirement from "this layer's frozen direction separates
#   known/unknown with high AUROC." Diagnosed via
#   slurm_jobs/13_diag_patchscope_layers.sh (3 rounds, see EXECUTION_PLAN.md
#   for the full trail): layer 8 (and 12) decode to template noise or the
#   surface entity's OWN name fragments at every position tested, but
#   layers 16/20/24, at offset +1 (one token past the surface entity, the
#   position with a principled claim to "after the bridge would be
#   resolved"), decode to REAL, MOSTLY-CORRECT COUNTRY NAMES ("Russia" for
#   a player born in Khabarovsk, "Nigeria" for Lagos, "the United States of
#   America" for US-born players) -- direct, positive confirmation the
#   bridge IS resolved internally and IS decodable, just not at layer 8/12.
#
#   THIS RUN refits E0's gate at LAYER 16 (same n=20000, same fit/held-out
#   split logic) so the WHOLE E0->E3->E1 chain operates at a layer where
#   Patchscopes actually works, rather than mixing "gate fit at layer 8"
#   with "Patchscopes decode at layer 16-24" as two disconnected choices.
#   Output goes to a NEW directory (e0_gemma2b_layer16) so the layer-8
#   result is not overwritten -- both are legitimate, independently useful
#   results, not one superseding the other.
cd "$REPO/src/driver_scripts"
python3 skc_run_e0.py \
    --queries "$REPO/data/onehop_queries.json" \
    --named-bridge-queries "$REPO/data/twohop_queries.json" \
    --n 20000 \
    --model gemma-2-2b \
    --layer 16 \
    --fit-frac 0.5 \
    --seed 0 \
    --skip-sae \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E0 PASSED. frozen_gate_*.json files are in $OUT. Proceed to E1."
else
    echo "E0 FAILED (exit $exit_code). Do NOT proceed to E1. See stderr above and" >&2
    echo "skc_runbook.txt's FLAG-BACK TRIGGERS." >&2
fi
exit "$exit_code"

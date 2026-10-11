#!/bin/bash -l
#SBATCH -J skc_e7e_attn_gemma2b
#SBATCH -o e7e_attn_gemma2b_%j.out
#SBATCH -e e7e_attn_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 4:00:00

# GPU item 5 of 9: (e) Attention control, Gemma 2 2B -- NEWLY
# IMPLEMENTED this session (cmd_attention() in skc_run_e7_ablations.py
# previously raised NotImplementedError; real output_attentions=True
# capture code did not exist anywhere in this pipeline before now).
#
# QUESTION THIS CONTROL ANSWERS: does the bridge-resolution position
# (E3's own empirical peak -- the SAME position E3 shows the frozen
# gate reads K2 above chance at) receive substantial attention from the
# model's own final-prompt-token computation? If not, "the gate is not
# routed" collapses into the weaker, different claim "the position is
# simply ignored" -- this control is what licenses the routing
# interpretation the paper's own causal-test framing depends on.
#
# AGGREGATION CHOICE, STATED EXPLICITLY (per the ORIGINAL stub's own
# requirement that this be decided and stated, not left implicit):
# attention weight FROM the final prompt token TO the bridge position
# (E3's own peak: entity-end-token-index + peak offset), MEAN over all
# heads AT the SAME layer E3's own peak was found at -- not averaged
# across layers, not summed over earlier positions. Reported against a
# UNIFORM-attention null (1/sequence_length, what an arbitrary context
# token would receive under a flat distribution) as the comparison
# point, not a bare number with no reference.
#
# REUSES the existing need_attentions=True eager-attention loading path
# (built for Lookback Lens, E4 Phase 2) rather than adding a second,
# parallel attention-capture code path -- the exact "two copies of the
# same logic diverging" bug class this project has repeatedly found and
# fixed elsewhere.
#
# --e3-map points at THIS model's own already-completed E3 run
# (results/e3_gemma2b_layer8_v2/e3_full_map.json) -- the SAME map
# Section~\ref{sec:res-latent}'s own reported peak (layer 8, offset +1,
# AUROC 0.788) comes from.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27
# DELIBERATELY NOT loading a cuda module here -- see other scripts' identical comment.

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
OUT="$REPO/results/e7e_attention_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e7_ablations.py attention \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --e3-map "$REPO/results/e3_gemma2b_layer8_v2/e3_full_map.json" \
    --seed 0 \
    --out "$OUT"

echo "Result (mean attention to bridge position vs. uniform null) is in"
echo "$OUT/e7e_attention.json"

#!/bin/bash -l
#SBATCH -J skc_e7j_ct_gemma2b
#SBATCH -o e7j_ct_gemma2b_%j.out
#SBATCH -e e7j_ct_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00

# Phase 3, item 2 of 2 LOAD-BEARING controls: (j) Cross-type
# generalization, ENTITY-TYPE half, Gemma 2 2B, layer 8. Paper's own
# words (skc_main.tex, Appendix~\ref{app:controls}): "fit the gate on
# one entity type and the bridge readout on one relation type, then
# test on held-out types." Explicitly flagged LOAD-BEARING ("without
# which the readout may be relation-specific").
#
# THIS RUN COVERS THE ENTITY-TYPE HALF ONLY. The RELATION-TYPE half
# ("...and the bridge readout on one relation type... test on held-out
# types") is NOT run here and is disclosed as a genuine data gap in the
# script's own output JSON (relation_type_half_status field): the
# two-hop bridge dataset (data/twohop_queries.json) has exactly ONE
# relation pair (place_birth->country) across all 3503 items -- nothing
# to hold out on the relation side with current data. Confirmed
# directly (skc_build_dataset_from_ferrando.py's own docstring: this IS
# the only buildable two-hop bridge from the released Ferrando data).
# User explicitly chose to implement the entity-type half and disclose
# this gap, rather than build a second relation pair first or skip (j)
# entirely.
#
# PROTOCOL: leave-one-entity-type-out, cycled across all 4 Ferrando
# entity types (player, movie, city, song) present in
# data/onehop_queries.json. For each held-out type: fit diffmeans+probe
# on the POOLED single-hop data of the OTHER 3 types only, score
# AUROC(gate->K1) on the held-out type's own data; ALSO compute an
# in-distribution reference AUROC (ordinary pooled-across-all-4-types
# fit, entity-level split, mirroring skc_run_e0.py's own protocol
# exactly) on the SAME held-out type's own held-out-entity subset, for
# an apples-to-apples comparison point. A leave-type-out AUROC close to
# the in-distribution reference is the evidence this control needs: the
# gate generalizes across entity types rather than being a type-
# specific fit.
#
# Reuses skc_run_e0.py's OWN capture_one_hop()/split_entities()
# (imported directly, not reimplemented) and skc_gates.py's
# fit_diffmeans/fit_probe -- avoids yet another instance of "two copies
# of the same logic diverging". Does NOT touch skc_run_e0.py itself or
# produce any FrozenGate consumed elsewhere -- this is a standalone
# diagnostic script (skc_run_e7j_crosstype.py, new this session).
#
# --n-per-type 2000: the full one-hop pool spans ~7.4k (player, the
# smallest type) to ~17.3k (movie) queries per type -- running all of
# them for a diagnostic control is not necessary; 2000/type is
# comparable in scale to a single-type slice of skc_run_e0.py's own
# usual fit split.
#
# Offline-verified before this run: the core leave-one-type-out fit/
# score arithmetic on synthetic activations -- a shared cross-type
# direction gives AUROC=0.991 on the held-out type (correctly high);
# type-specific, mutually unrelated directions give AUROC=0.089 (far
# from chance in either direction, confirming the test has real
# discriminating power, not a constant ~0.5 regardless of input).
# split_entities() confirmed to handle a small (15-item) entity list
# without crashing. Full end-to-end run on gpt2/CPU was attempted but
# hit an unrelated, expected architecture mismatch (capture_activations
# expects model.model.layers[.], gpt2 uses model.transformer.h[.]) --
# not a bug in this new script, capture_activations itself is already
# verified against the real target model family (Gemma/Llama) elsewhere
# in this pipeline.

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
OUT="$REPO/results/e7j_crosstype_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e7j_crosstype.py \
    --queries "$REPO/data/onehop_queries.json" \
    --model gemma-2-2b \
    --layer 8 \
    --n-per-type 2000 \
    --fit-frac 0.5 \
    --seed 0 \
    --out "$OUT"

echo "Result (per-held-out-type AUROC, leave-type-out vs in-distribution reference) is in"
echo "$OUT/e7j_crosstype.json"

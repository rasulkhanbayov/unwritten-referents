#!/bin/bash -l
#SBATCH -J skc_e7c_dist_gemma2b
#SBATCH -o e7c_dist_gemma2b_%j.out
#SBATCH -e e7c_dist_gemma2b_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 4:00:00

# Phase 3, item 3: (c) Length-matched distractor control, Gemma 2 2B,
# layer 8. Paper's own words (skc_main.tex, Appendix~\ref{app:controls}):
# "(c) Length-matched distractor control." -- no further protocol detail
# exists anywhere in the repo or manuscript beyond this name; the exact
# design below is an EXPLICIT CHOICE confirmed directly with the user.
#
# PROTOCOL: the COMPOSED prompt never names the bridge at all (e.g. "The
# country of the place of birth of the player Jake Bornheimer is") --
# there is no bridge slot to swap a distractor into there. The only
# template that DOES name a city is named_bridge ("The country of the
# city New Brunswick is"), the same template control (a) uses. This
# control builds a DISTRACTOR variant of that SAME template: instead of
# the TRUE bridge, an UNRELATED real city (from this same dataset's own
# pool, same sampling discipline as control (g)'s counterfactual cities
# -- always a real place, never fabricated) is named instead,
# LENGTH-MATCHED toward the TRUE named-bridge prompt's own token count
# via skc_data.length_match_named_bridge() -- built earlier in the
# project but NEVER actually called anywhere until this control
# (confirmed by grepping the whole codebase before using it).
#
# If control (a)'s own AUC(gate->K2) gap reflects the gate genuinely
# reading the TRUE bridge's actual properties, AUC(gate->K2) on these
# DISTRACTOR named-bridge prompts should collapse toward CHANCE (0.5) --
# a gate merely detecting "a city name is present here" (surface/length
# artifact) rather than the true bridge's properties would instead show
# a similarly elevated AUROC on distractor prompts, which this control
# is built to catch. Compare this run's own auroc_K2_distractor_named_
# bridge directly against control (a)'s auroc_K2_named_bridge (job that
# produced e7a_named_bridge.json).
#
# DISCLOSED ASYMMETRY (confirmed directly with the user, not silently
# accepted): length_match_named_bridge() pads ONE-DIRECTIONALLY
# (distractor toward true length only). Checked directly against the
# real dataset: only ~21% of distractor prompts (727/3503 in the full
# pool) actually receive the "well-known" filler at all, since most real
# distractor city names are already >= the true bridge's own name
# length -- reported as an explicit fraction in this run's own output
# JSON (e7c_distractor.json), not hidden.
#
# COST: mirrors control (a)'s own cost profile exactly -- one generation
# (for K2 labeling) + two capture passes (composed, distractor) per
# query, no sampling, same test-set size (239 queries). -t 4:00:00 is
# generous, not tight.

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
OUT="$REPO/results/e7c_distractor_gemma2b"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e7_ablations.py distractor \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-path "$REPO/results/e0_gemma2b_layer8_v2/frozen_gate_diffmeans.json" \
    --seed 0 \
    --out "$OUT"

echo "Result (auroc_K2_composed, auroc_K2_distractor_named_bridge, gap_vs_composed) is in"
echo "$OUT/e7c_distractor.json"

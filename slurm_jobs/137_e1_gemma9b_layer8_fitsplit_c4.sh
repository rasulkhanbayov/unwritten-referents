#!/bin/bash -l
#SBATCH -J skc_e1_gemma9b_l8_fitsplit
#SBATCH -o e1_gemma9b_l8_fitsplit_%j.out
#SBATCH -e e1_gemma9b_l8_fitsplit_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 60000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00

# VERSION3 era, Phase 1 / C4: score the DISJOINT fit-split composed queries
# for gemma-2-9b, to standardize g_S/g_B before scripts/ablations.py can
# run the bridge-only test.
#
# WHY THIS JOB EXISTS: scripts/ablations.py's own Standardizer requires a
# genuinely SEPARATE "fit" split of two-hop composed queries (not just
# excluded entities from E0's single-hop fit, and not the eval set itself,
# which would leak -- see selectors.Standardizer's own docstring: "Using
# evaluation statistics leaks and changes min(.,.) because the minimum is
# not scale invariant"). Checked directly that data/twohop_queries.json's
# own 3503 queries have ZERO available for this (every one of their
# entities is excluded from E0's fit_set BY DESIGN, per the C7 fix) and
# that Ferrando et al.'s own player/city data is fully exhausted at 3503
# usable chains (7487 players, only 3503 have a city bridge). Solution:
# data/c4_fit_queries.json (built by skc_build_c4_fit_queries.py) --
# 1253 REAL, live-Wikidata-verified two-hop chains (the k=2 prefix of
# data/threehop_queries.json's own player->city->country->continent
# chains, built earlier this project, confirmed entity-disjoint from
# data/twohop_queries.json's own eval-side population).
#
# Uses --export-records/--export-split fit (this session's own earlier
# addition to skc_run_e1.py) to write these to VERSION3's schema at
# split="fit" -- the FIRST records this project has ever exported at
# that split. Same K1=1+bridge-resolution filter as the eval-side run
# (gemma-2-9b needs --skip-bridge-resolution, per job 104's own confirmed
# 0/346 Patchscopes resolution at this layer) -- keeping the SAME
# selection criterion on both fit and eval sides, just different,
# disjoint entities, rather than an unmatched population.
#
# Uses e0_gemma9b_layer8_splitfix (the model's own confirmed-stable,
# 5-seed-verified gate) as --gate-dir.

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
OUT="$REPO/results/e1_gemma9b_layer8_fitsplit_c4"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e1.py \
    --queries "$REPO/data/c4_fit_queries.json" \
    --singlehop-correct-queries "$REPO/data/onehop_correct.json" \
    --model gemma-2-9b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma9b_layer8_splitfix" \
    --primary-estimator diffmeans \
    --skip-bridge-resolution \
    --seed 0 \
    --export-records "$REPO/VERSION3/data/records/gemma-2-9b/f1_twohop_fitsplit.jsonl" \
    --export-split fit \
    --out "$OUT"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo "E1 (gemma-2-9b, layer 8, FIT-SPLIT c4_fit_queries.json) completed (exit 0)."
    echo "Records exported to VERSION3/data/records/gemma-2-9b/f1_twohop_fitsplit.jsonl."
    echo "This run's own E1.json (gate_to_M etc.) is NOT the number that matters here --"
    echo "only the exported g_S/g_B records are needed, for ablations.py's Standardizer."
else
    echo "E1 (gemma-2-9b, layer 8, FIT-SPLIT) FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"

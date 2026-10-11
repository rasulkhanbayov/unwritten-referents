#!/bin/bash -l
#SBATCH -J skc_build_threehop
#SBATCH -o build_threehop_%j.out
#SBATCH -e build_threehop_%j.err
#SBATCH -p cpu-all
#SBATCH -c 2
#SBATCH --mem 4000MB
#SBATCH -t 00:15:00

# STEP 8-PREREQUISITE: build a real THREE-hop dataset for E5. No GPU, no
# local model -- CPU-only, network-only (live Wikidata SPARQL query).
#
# WHY: E5 ("compounding across hop depth", skc_runbook.txt Proposition 4)
# needs k=3 chains this bundle never had -- data/twohop_queries.json only
# has k=2 (built from Ferrando et al.'s released data, which was checked
# directly and does NOT support a third hop; see
# skc_build_dataset_threehop.py's own docstring and EXECUTION_PLAN.md for
# the full check). Built and VERIFIED live against Wikidata instead: this
# script's default (etype_qid, r0-r3 PIDs) were each confirmed correct by
# direct SPARQL lookup before use, and a real pull (limit=4000) returned
# 3961 valid, sensible chains (e.g. "Jason Richardson" -> Saginaw ->
# United States -> North America) -- not placeholder/guessed data, unlike
# an earlier, abandoned attempt at this project's wikidata path.
#
# This script was ALREADY RUN ONCE manually (not via sbatch) while
# verifying skc_data.fetch_three_hop() end to end -- data/threehop_queries.json
# already exists with 3961 queries. This Slurm script exists for
# reproducibility/record (so a future re-run does not need a login-node
# manual invocation) and to re-build the dataset if the Wikidata snapshot
# needs refreshing -- NOT something that needs to run again right now.

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate skc_iclr27

# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }

cd "$REPO/src/driver_scripts"
python3 skc_build_dataset_threehop.py \
    --out "$REPO/data/threehop_queries.json" \
    --wikidata-date "$(date +%Y-%m-%d)"

echo "Three-hop dataset saved to $REPO/data/threehop_queries.json"
echo "Ready for 09_e5_khop_gemma2b.sh."

#!/bin/bash -l
#SBATCH -J skc_diag_bridge_k1filtered
#SBATCH -o diag_bridge_k1filtered_%j.out
#SBATCH -e diag_bridge_k1filtered_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 20000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 00:30:00

# DIAGNOSTIC ONLY, not part of the main E0-E7 pipeline.
#
# WHY: 15_diag_bridge_during_gen.sh (capturing activations WHILE the model
# generates its own hop1 answer, not just at static prompt positions) found
# the bridge (city) never decodable at any position -- but reading its own
# printed "model's own greedy completion" showed the REAL explanation:
# the model's hop1 answer was WRONG for nearly all 8 sampled queries (e.g.
# "Garcia Domingos" born Luanda -> model said "the city of Porto, Portugal";
# "Artem Zabelin" born Khabarovsk -> model said "the city of Moscow").
#
# EVERY diagnostic run so far in this investigation (E3's own sweep, both
# rounds of 14_diag_bridge_vs_final.sh, and 15_diag_bridge_during_gen.sh)
# sampled queries by taking the first N in twohop_queries.json, with NO K1
# filter. If the model never correctly resolved hop1 for most of that
# sample, there was no correctly-resolved bridge for Patchscopes to find in
# the first place -- skc_runbook.txt's own precondition ("CONDITION EVERY
# PRIMARY ANALYSIS ON K1 = 1") was silently violated by every diagnostic's
# sampling, not just by the main E1 filter.
#
# THIS RUN fixes the sampling: labels a pool of 200 queries for K1 first
# (same skc_harness.label_queries_batched() the main E1 filter uses), keeps
# only K1=1 queries, and reruns the same during-generation Patchscopes
# capture from round 3 on that correctly-filtered sample. If the bridge now
# decodes cleanly on queries the model demonstrably gets right, the entire
# earlier "0% bridge-resolved" result across three sweeps was a SAMPLING
# ARTIFACT, not evidence about Patchscopes or the model's representations.
# If it still never decodes even here, that is a much stronger, now
# well-controlled negative result.
#
# Labeling 200 queries for K1 costs about the same per-query as E1's own
# 3503-query labeling pass (a couple of minutes there, so a small fraction
# of that here) plus the same cheap ~56-decode-call diagnostic as round 3.

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

cd "$REPO/src/driver_scripts"
python3 skc_diag_bridge_k1filtered.py \
    --queries "$REPO/data/twohop_queries.json" \
    --model gemma-2-2b \
    --layers 16 \
    --max-new-tokens 6 \
    --pool-size 200 \
    --n 8

echo
echo "Read the K1=1 rate first: if it comes back near 6.5% (matching every"
echo "prior E1/E3 run on the full 3503-query set), the pool size was"
echo "adequate. Then read the decodes exactly as in round 3 -- but now on"
echo "queries CONFIRMED correctly resolved. If the city decodes cleanly"
echo "somewhere here, the earlier 'city never decodable' conclusion across"
echo "three sweeps was a sampling artifact (testing queries the model gets"
echo "wrong), not a fact about the model's representations or Patchscopes."
echo "If it STILL never decodes even on K1=1 queries, that is a genuinely"
echo "stronger negative result -- worth reporting as such."

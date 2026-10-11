#!/bin/bash -l
#SBATCH -J skc_e4_inside_rho_gemma2b
#SBATCH -o e4_inside_rho_gemma2b_l8_%j.out
#SBATCH -e e4_inside_rho_gemma2b_l8_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 14:00:00

# Rerun of job 55 (e4_gemma2b_layer8_inside), needed ONLY because an
# internal-consistency audit caught a real, pre-existing bug in
# Table tab:bgr-full: several baselines (SAPLMA, entity-token-probes,
# ICR probe, HaloScope, Lookback Lens) were shown in the "One forward
# pass" ($1\times$) tier, but their own cost_report (this same file)
# says "1 generation + 1 capture pass" (2 forward passes) -- and INSIDE
# was shown at $1\times$ despite its own cost_report saying "k=10
# (one generation batch of k samples per query)". Fixed the table's
# cost-tier grouping to match each baseline's own recorded cost, which
# moved INSIDE into the k=10 tier -- where it turns out to beat
# Semantic entropy on every metric (AUROC 0.596 vs 0.575, AURC 0.116
# vs 0.145, Risk@50% 0.134 vs 0.143), making INSIDE the ACTUAL best
# k=10 baseline, not Semantic entropy as the main-body summary table
# (tab:bgr) previously claimed.
#
# INSIDE's own rho (share of the blind term closed) was never computed
# -- that run (job 55, above) predates this session's own fix to
# run_E4() (skc_experiments.py) that computes recovery_fraction() for
# EVERY selector, not just BGR-min. Raw per-item scores were never
# saved to disk from that run, so rho cannot be recovered post-hoc;
# this is the same situation as the earlier P(True)/Operadic/Oracle
# rho fills, and the same fix (rerun under the already-fixed code)
# applies here.
#
# Otherwise IDENTICAL to job 55: same queries/gate/layer/seed, same
# --run-inside plus the other three baselines that ran alongside it
# originally (--run-self-ask --run-saplma --run-entity-token-probe),
# so AUROC/AURC/Risk@50% for every selector should reproduce exactly
# (deterministic given the same seed), and the new information is the
# `rho` dict now attached to every selector, INSIDE's included.

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
OUT="$REPO/results/e4_gemma2b_layer8_inside_rho"
mkdir -p "$OUT"

cd "$REPO/src/driver_scripts"
python3 skc_run_e4.py \
    --queries "$REPO/results/e1_gemma2b_layer8/queries_filtered.json" \
    --model gemma-2-2b \
    --layer 8 \
    --gate-dir "$REPO/results/e0_gemma2b_layer8_v2" \
    --primary-estimator diffmeans \
    --run-self-ask \
    --run-saplma \
    --run-entity-token-probe \
    --run-inside \
    --saplma-train-queries "$REPO/data/onehop_fit_gemma2b_layer8_v2.json" \
    --coverage 0.5 \
    --seed 0 \
    --out "$OUT"

echo "INSIDE's real rho (plus self_ask/saplma/entity_token_probe's own,"
echo "re-scored under the corrected code) is in $OUT/E4.json's own"
echo "\"selectors\" and \"rho\" dicts. Before trusting it, confirm the"
echo "AUROC/AURC/Risk@50% for every selector match e4_gemma2b_layer8_inside's"
echo "own E4.json exactly (same seed, should reproduce byte-for-byte) --"
echo "if they do not match, investigate before using the new rho value."

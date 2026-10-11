#!/usr/bin/env python3
"""
skc_rank_layersweep_by_seed_stability.py

Ranks layers from a multi-seed set of skc_run_e0_layersweep.py outputs
by CROSS-SEED agreement stability (min and mean agreement across seeds),
not by any single seed's own best value -- built specifically because
Gemma-2-2b/layer-16's own single-seed "best" layer (job 91/92, seed 0
only) turned out to be a lucky outlier that failed 4 of 4 other seeds
(jobs 108-114).

Usage:
    python3 skc_rank_layersweep_by_seed_stability.py \\
        --sweep-dirs ../results/e0_layersweep_gemma2b_seed0 \\
                     ../results/e0_layersweep_gemma2b_seed1 \\
                     ../results/e0_layersweep_gemma2b_seed2 \\
                     ../results/e0_layersweep_gemma2b_seed3 \\
                     ../results/e0_layersweep_gemma2b_seed4
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dirs", nargs="+", required=True,
                    help="one e0_layersweep_*.py output dir per seed")
    args = ap.parse_args()

    per_layer_agreements: dict[int, list[float]] = {}
    per_layer_gate_open: dict[int, list[bool]] = {}
    seeds_seen = []

    for d in args.sweep_dirs:
        path = os.path.join(d, "e0_layersweep_full.json")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found -- did this sweep job actually finish?")
        with open(path) as f:
            data = json.load(f)
        seeds_seen.append(d)
        for layer_str, e0 in data.items():
            layer = int(layer_str)
            agr = e0.get("agreement", {}).get("min_pairwise_spearman")
            per_layer_agreements.setdefault(layer, []).append(agr)
            per_layer_gate_open.setdefault(layer, []).append(bool(e0.get("gate_open")))

    n_seeds = len(args.sweep_dirs)
    rows = []
    for layer, agrs in per_layer_agreements.items():
        if len(agrs) != n_seeds:
            print(f"WARNING: layer {layer} has {len(agrs)}/{n_seeds} seed results -- "
                 "one or more sweep dirs may be missing this layer or still running.")
        valid = [a for a in agrs if a is not None]
        if not valid:
            continue
        n_pass = sum(per_layer_gate_open[layer])
        rows.append((layer, min(valid), sum(valid) / len(valid), max(valid), n_pass, len(valid)))

    rows.sort(key=lambda r: -r[1])  # sort by MIN agreement across seeds, not mean or max

    print(f"Ranked by MIN agreement across {n_seeds} seeds (most conservative -- a layer "
         "that is only good on its best seed ranks low here, correctly):\n")
    print(f"{'layer':>5} {'min_agr':>9} {'mean_agr':>9} {'max_agr':>9} {'n_pass':>7} {'n_seeds':>8}")
    for layer, mn, mean, mx, n_pass, n in rows[:15]:
        print(f"{layer:>5} {mn:>9.3f} {mean:>9.3f} {mx:>9.3f} {n_pass:>4}/{n:<3}{'':>1} {n:>8}")

    stable_passing = [r for r in rows if r[4] == r[5]]  # gate_open true on EVERY seed
    print(f"\nLayers passing gate_open on EVERY seed checked: "
         f"{[r[0] for r in stable_passing] if stable_passing else 'NONE'}")


if __name__ == "__main__":
    main()

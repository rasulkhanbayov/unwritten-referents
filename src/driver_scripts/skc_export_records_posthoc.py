#!/usr/bin/env python3
"""
skc_export_records_posthoc.py

Offline (no GPU) export of VERSION3 per-item records from an E1 run
ALREADY on disk -- reuses the saved acts_surface_filtered.npy /
acts_bridge_filtered.npy / e1_labels_filtered.json / frozen_gate_*.json
files an skc_run_e1.py run already wrote, rather than re-running the
model. Written so records can be produced for runs that finished BEFORE
--export-records existed on skc_run_e1.py itself (see that script's own
--export-records flag for new runs going forward).

Usage:
    python3 skc_export_records_posthoc.py \\
        --e1-dir ../results/e1_gemma2b_layer16_splitfix \\
        --gate-dir ../results/e0_gemma2b_layer16_splitfix \\
        --estimator diffmeans \\
        --out ../VERSION3/data/records/gemma-2-2b/f1_twohop.jsonl \\
        --split eval --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for skc_gates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # driver_scripts, for skc_export_records
from skc_gates import FrozenGate  # noqa: E402
from skc_export_records import write_jsonl_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1-dir", required=True,
                    help="an skc_run_e1.py output directory (has e1_labels_filtered.json, "
                         "acts_surface_filtered.npy, acts_bridge_filtered.npy)")
    ap.add_argument("--gate-dir", required=True,
                    help="the E0 output directory that produced this E1 run's gate "
                         "(has frozen_gate_{estimator}.json)")
    ap.add_argument("--estimator", default="diffmeans", choices=["diffmeans", "probe", "sae"])
    ap.add_argument("--split", default="eval", choices=["fit", "validation", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels_path = os.path.join(args.e1_dir, "e1_labels_filtered.json")
    surf_path = os.path.join(args.e1_dir, "acts_surface_filtered.npy")
    bridge_path = os.path.join(args.e1_dir, "acts_bridge_filtered.npy")
    for p in (labels_path, surf_path, bridge_path):
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found -- is --e1-dir a real skc_run_e1.py output dir?")

    with open(labels_path) as f:
        labels = json.load(f)
    acts_surface = np.load(surf_path)
    acts_bridge = np.load(bridge_path)
    if not (len(labels) == acts_surface.shape[0] == acts_bridge.shape[0]):
        raise SystemExit(f"length mismatch: {len(labels)} labels, "
                         f"{acts_surface.shape[0]} surface acts, "
                         f"{acts_bridge.shape[0]} bridge acts")

    gate_path = os.path.join(args.gate_dir, f"frozen_gate_{args.estimator}.json")
    if not os.path.exists(gate_path):
        raise SystemExit(f"{gate_path} not found. Run skc_run_e0.py for this model/layer "
                         "first, with this estimator not skipped.")
    gate = FrozenGate.load(gate_path)

    gs = gate.score(acts_surface)
    gb = gate.score(acts_bridge)

    write_jsonl_records(
        args.out,
        query_ids=[l["qid"] for l in labels],
        split=args.split,
        seed=args.seed,
        y=[l["Y"] for l in labels],
        k1=[l["K1"] for l in labels],
        k2=[l["K2"] for l in labels],
        g_s=gs.tolist(),
        g_b=gb.tolist(),
        estimator=args.estimator,
        offset=args.offset,
    )
    print(f"Exported {len(labels)} records from {args.e1_dir} (gate: {args.gate_dir}, "
         f"estimator: {args.estimator}) to {args.out}")


if __name__ == "__main__":
    main()

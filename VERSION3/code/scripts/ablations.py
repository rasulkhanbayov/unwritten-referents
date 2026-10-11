#!/usr/bin/env python3
"""
The ablation that decides whether BGR's design rationale survives.

Section 4.3 argues the MINIMUM matters because answering a composed query needs
both hops. Four arms test that on data already collected:

    surface_gate  read at the written entity only        (reference)
    bridge_only   read at the bridge position only       <- the missing arm
    bgr_min       conjunction                            (the paper's method)
    bgr_mean      compensatory control

If bridge_only matches bgr_min within the resolvable difference, the minimum
contributes nothing and Section 4.3 is wrong. This script prints that verdict
rather than leaving it to interpretation.

Usage:  python scripts/ablations.py --config configs/experiments.yaml
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skc import io, metrics, selectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--out", default="results/ablations")
    ap.add_argument("--mde", type=float, default=None,
                    help="resolvable AUROC difference; from scripts/reconcile.py")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); Path(a.out).mkdir(parents=True, exist_ok=True)
    boot = cfg["analysis"]["bootstrap"]

    for m in cfg["data"]["models"]:
        mid = m["id"]
        try:
            recs = io.load(cfg["data"]["root"], mid, "f1_twohop")
            io.require_seeds(recs, cfg["seeds"]["n_required"])
        except io.MissingData as e:
            print(f"[{mid}] SKIPPED -- {e}\n"); continue

        fit, ev = recs.where(split="fit"), recs.where(split="eval")
        std = selectors.Standardizer("fit").fit(g_S=fit.col("g_S"), g_B=fit.col("g_B"))
        z = std.transform(g_S=ev.col("g_S"), g_B=ev.col("g_B"))
        Y, qid = ev.col("Y"), ev.col("query_id")

        per_seed, point = {}, {}
        for name in selectors.ablation_arms:
            s = selectors.SELECTORS[name](z["g_S"], z["g_B"])
            ci = metrics.bootstrap_ci(metrics.auroc, s, Y, groups=qid,
                                      n_resamples=boot["n_resamples"], ci=boot["ci"])
            point[name] = ci["point"]
            # seed spread, which the bootstrap does not capture
            vals = []
            for sd in ev.seeds():
                sub = ev.where(seed=sd)
                zz = std.transform(g_S=sub.col("g_S"), g_B=sub.col("g_B"))
                vals.append(metrics.auroc(
                    selectors.SELECTORS[name](zz["g_S"], zz["g_B"]), sub.col("Y")))
            per_seed[name] = {"mean": float(np.mean(vals)),
                              "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                              "values": [float(v) for v in vals]}
            print(f"[{mid}] {name:14s} AUROC {ci['point']:.3f} "
                  f"[{ci['lo']:.3f},{ci['hi']:.3f}]  seed sd "
                  f"{per_seed[name]['sd'] if per_seed[name]['sd'] is None else round(per_seed[name]['sd'],4)}")

        # paired test on the arm that matters
        pd = metrics.paired_bootstrap_diff(
            metrics.auroc,
            selectors.bgr_min(z["g_S"], z["g_B"]),
            selectors.bridge_only(z["g_S"], z["g_B"]),
            Y, qid, n_resamples=boot["n_resamples"], ci=boot["ci"])
        tol = a.mde if a.mde is not None else abs(pd["hi"] - pd["lo"]) / 2
        verdict = selectors.interpret_ablation(point, tol)
        print(f"[{mid}] bgr_min - bridge_only = {pd['diff']:+.3f} "
              f"[{pd['lo']:+.3f},{pd['hi']:+.3f}] p={pd['p']:.4f}")
        print(f"[{mid}] VERDICT: {verdict}\n")
        json.dump({"model": mid, "auroc": point, "per_seed": per_seed,
                   "bgr_min_vs_bridge_only": pd, "tolerance": tol,
                   "verdict": verdict}, open(f"{a.out}/{mid}.json", "w"), indent=2)


if __name__ == "__main__":
    main()

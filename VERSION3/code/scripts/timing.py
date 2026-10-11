#!/usr/bin/env python3
"""
Per-selector wall-clock and peak memory. The paper's practical selling point is
"one forward pass" and it is currently supported by a cost-tier label, not a
measurement.

This script times the SCORING step only, from the per-item records, which is
what distinguishes the selectors: BGR reads two activations from a pass the
model already performs, Lookback Lens needs a second pass, INSIDE needs ten
samples. Forward-pass counts come from the records; the marginal scoring cost
is measured here.

Usage: python scripts/timing.py --config configs/experiments.yaml --repeats 20
"""
import argparse, json, sys, time, tracemalloc
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skc import io, selectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="results/timing")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); Path(a.out).mkdir(parents=True, exist_ok=True)
    if cfg["hardware"]["gpu"] is None:
        print("WARNING: configs/experiments.yaml hardware.gpu is null. Fill it "
              "before reporting any timing, or the numbers describe nothing.")
    for m in cfg["data"]["models"]:
        mid = m["id"]
        try:
            recs = io.load(cfg["data"]["root"], mid, "f1_twohop")
        except io.MissingData as e:
            print(f"[{mid}] SKIPPED -- {e}"); continue
        ev = recs.where(split="eval")
        fit = recs.where(split="fit")
        std = selectors.Standardizer("fit").fit(g_S=fit.col("g_S"), g_B=fit.col("g_B"))
        z = std.transform(g_S=ev.col("g_S"), g_B=ev.col("g_B"))
        out = {"model": mid, "n_items": len(ev), "hardware": cfg["hardware"]}
        for name, fn in selectors.SELECTORS.items():
            tracemalloc.start()
            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter(); fn(z["g_S"], z["g_B"]); ts.append(time.perf_counter() - t0)
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            out[name] = {"median_us_per_item": float(np.median(ts) / len(ev) * 1e6),
                         "peak_kb": peak / 1024}
        if ev.has("wall_clock_ms"):
            out["model_side_ms_per_item"] = float(np.median(ev.col("wall_clock_ms")))
        else:
            out["model_side_ms_per_item"] = None
            print(f"[{mid}] note: 'wall_clock_ms' absent, so only the scoring "
                  f"step is timed; the forward-pass cost is not measured here.")
        json.dump(out, open(f"{a.out}/{mid}.json", "w"), indent=2)
        print(f"[{mid}] scoring cost per item, microseconds: " +
              ", ".join(f"{k}={out[k]['median_us_per_item']:.2f}"
                        for k in selectors.SELECTORS))


if __name__ == "__main__":
    main()

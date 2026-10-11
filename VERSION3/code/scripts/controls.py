#!/usr/bin/env python3
"""
The three controls the manuscript flags as open.

  zero-overlap   Control (n) does not clear: a text-only detector reaches
                 |AUROC-0.5| = 0.341, which sign-flipped exceeds BGR-min.
                 Re-run every selector on the zero-overlap subset and report
                 whether the margin survives.
  level-shift    Lathkar (2026) show that supplying a confirmed intermediate
                 fact raises confidence on wrong answers, so the named-bridge
                 manipulation moves score LEVEL as well as RANKING. Only the
                 ranking change is availability. Report both.
  cot-paired     The chain-of-thought split rests on 44 and 46 items with no
                 test. Run the paired contrast or drop the claim.

Usage:  python scripts/controls.py --which all --config configs/experiments.yaml
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skc import io, metrics, selectors


def zero_overlap(recs, cfg):
    ev = recs.where(split="eval")
    if not ev.has("text_overlap"):
        raise io.MissingData(
            "field 'text_overlap' absent. Control (n) needs the prompt/target "
            "lexical overlap per item to build the zero-overlap subset.")
    ov, Y, qid = ev.col("text_overlap"), ev.col("Y"), ev.col("query_id")
    keep = ov <= 0
    out = {"n_all": int(Y.size), "n_zero_overlap": int(keep.sum())}
    fit = recs.where(split="fit")
    std = selectors.Standardizer("fit").fit(g_S=fit.col("g_S"), g_B=fit.col("g_B"))
    z = std.transform(g_S=ev.col("g_S"), g_B=ev.col("g_B"))
    for name in ("surface_gate", "bgr_min", "bridge_only"):
        s = selectors.SELECTORS[name](z["g_S"], z["g_B"])
        out[name] = {
            "all": metrics.auroc(s, Y),
            "zero_overlap": metrics.auroc(s[keep], Y[keep]),
        }
    # the text-only detector itself, reported two-sided
    a_txt = metrics.auroc(ov, Y)
    out["text_only"] = {"auroc": a_txt, "distance_from_chance": abs(a_txt - 0.5),
                        "sign_flipped": 1 - a_txt}
    out["margin_survives"] = bool(
        out["bgr_min"]["zero_overlap"] - out["surface_gate"]["zero_overlap"] > 0)
    return out


def level_shift(recs, cfg):
    ev = recs.where(split="eval")
    if not ev.has("g_S_named"):
        raise io.MissingData(
            "field 'g_S_named' absent. The level-shift control needs the gate "
            "score on the matched named-bridge prompt for the same items.")
    comp, named, K2, qid = ev.col("g_S"), ev.col("g_S_named"), ev.col("K2"), ev.col("query_id")
    d = named - comp
    boot = cfg["analysis"]["bootstrap"]
    rank = metrics.paired_bootstrap_diff(metrics.auroc, named, comp, K2, qid,
                                         n_resamples=boot["n_resamples"])
    return {"ranking_change_auroc": rank,
            "level_shift_mean": float(d.mean()),
            "level_shift_sd": float(d.std(ddof=1)),
            "level_shift_t": float(stats.ttest_rel(named, comp).statistic),
            "level_shift_p": float(stats.ttest_rel(named, comp).pvalue),
            "note": ("availability is the RANKING change. A large level shift "
                     "with a small ranking change means the manipulation moved "
                     "confidence, not information (Lathkar 2026).")}


def cot_paired(recs, cfg):
    ev = recs.where(split="eval")
    for f in ("cot_bridge_correct",):
        if not ev.has(f):
            raise io.MissingData(
                f"field '{f}' absent. The chain-of-thought contrast needs a "
                f"per-item flag for whether the self-written bridge is correct.")
    arm, g, Y, qid = ev.col("cot_bridge_correct"), ev.col("g_S"), ev.col("Y"), ev.col("query_id")
    out = {}
    for label, sel in (("correct_arm", arm == 1), ("wrong_arm", arm == 0)):
        n_pos = int(Y[sel].sum())
        out[label] = {"n": int(sel.sum()), "n_pos": n_pos}
        try:
            out[label]["auroc"] = metrics.bootstrap_ci(
                metrics.auroc, g[sel], Y[sel], groups=qid[sel],
                n_resamples=cfg["analysis"]["bootstrap"]["n_resamples"])
        except ValueError as e:
            out[label]["auroc"] = {"error": str(e)}
    return out


HANDLERS = {"zero-overlap": zero_overlap, "level-shift": level_shift,
            "cot-paired": cot_paired}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", *HANDLERS])
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--out", default="results/controls")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); Path(a.out).mkdir(parents=True, exist_ok=True)
    which = list(HANDLERS) if a.which == "all" else [a.which]
    for m in cfg["data"]["models"]:
        mid = m["id"]
        for name in which:
            try:
                recs = io.load(cfg["data"]["root"], mid, "f1_twohop")
                res = HANDLERS[name](recs, cfg)
            except io.MissingData as e:
                print(f"[{mid}] {name}: SKIPPED -- {e}"); continue
            json.dump(res, open(f"{a.out}/{mid}.{name}.json", "w"), indent=2)
            print(f"[{mid}] {name}: {json.dumps(res, default=str)[:220]}")


if __name__ == "__main__":
    main()

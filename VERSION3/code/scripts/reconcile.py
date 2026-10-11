#!/usr/bin/env python3
"""
Recompute every number the paper reports, from per-item records, with intervals.

This is the script that resolves the manuscript's open arithmetic:
  * AUROC(S->M) against AUROC(S->K2). At k=2 these are the SAME variable, so
    the manuscript's 0.616 and 0.777 cannot both be right.
  * the class balance of the evaluation set, which the paper cannot state
  * the slack terms with the one-sided delta, which can only improve the floor
  * the minimum resolvable AUROC difference on this sample

Usage:  python scripts/reconcile.py --config configs/experiments.yaml
Output: results/reconcile/{model}.json  and a summary table on stdout.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skc import io, metrics, slack


def mde(n_pos, n_neg, auroc_ref=0.6, power=0.8, alpha=0.05):
    """Minimum detectable AUROC difference, Hanley-McNeil, paired rho=0.5."""
    A = auroc_ref
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    var = (A * (1 - A) + (n_pos - 1) * (q1 - A * A) + (n_neg - 1) * (q2 - A * A)) / (n_pos * n_neg)
    return float(2.8 * np.sqrt(2 * var * (1 - 0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--out", default="results/reconcile")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    boot, cov = cfg["analysis"]["bootstrap"], cfg["analysis"]["headline_coverage"]

    for m in cfg["data"]["models"]:
        mid = m["id"]
        try:
            recs = io.load(cfg["data"]["root"], mid, "f1_twohop")
            io.require_seeds(recs, cfg["seeds"]["n_required"])
            est = io.require_single_estimator(recs)
        except io.MissingData as e:
            print(f"[{mid}] SKIPPED -- {e}\n"); continue

        ev = recs.where(split="eval")
        Y, K1, M = ev.col("Y"), ev.col("K1"), ev.M(2)
        gS, qid = ev.col("g_S"), ev.col("query_id")

        out = {"model": mid, "estimator": est, "n": len(ev),
               "seeds": ev.seeds(),
               "class_balance_Y": metrics.class_balance(Y),
               "class_balance_M": metrics.class_balance(M)}

        for tgt, lab in (("K1", K1), ("M", M), ("Y", Y)):
            if len(set(lab.tolist())) < 2:
                # K1 is constant (all 1) whenever the exported "eval" split was
                # already filtered to K1=1 upstream (this pipeline's own
                # non-negotiable composition filter, per skc_data.py's
                # filter_bridge_resolved() docstring) -- AUROC(S->K1) is
                # undefined on that population, the SAME issue this project's
                # own skc_run_e1.py already solves by requiring a SEPARATE
                # unfiltered array (see run_E1()'s own "SUBSET DISCIPLINE"
                # docstring). VERSION3's schema has no unfiltered-K1 field to
                # fall back to, so this is skipped here with a stated reason
                # rather than raising (which would kill every model's
                # analysis) or fabricating a value.
                out[f"auroc_gate_{tgt}"] = None
                print(f"[{mid}] AUROC(S->{tgt}) UNDEFINED: {tgt} is constant on the "
                     f"exported eval split (this model's export is already "
                     f"K1=1-filtered upstream). Not computed; not a missing "
                     f"measurement, a structural non-applicability.")
                continue
            out[f"auroc_gate_{tgt}"] = metrics.bootstrap_ci(
                metrics.auroc, gS, lab, groups=qid,
                n_resamples=boot["n_resamples"], ci=boot["ci"])

        # M and K2 are the same variable at k=2. Prove it or fail loudly.
        if not np.array_equal(M, ev.col("K2")):
            raise AssertionError(
                "M != K2 at k=2, which contradicts the definition in Section 3. "
                "One of the two columns is wrong; the manuscript's 0.616 vs "
                "0.777 discrepancy lives here.")
        out["M_equals_K2"] = True

        sl = slack.floor(gS, Y, M, clip=False,
                         one_sided=cfg["analysis"]["slack"]["one_sided_delta"],
                         min_count=cfg["analysis"]["slack"]["min_bin_count"],
                         quantile=cfg["analysis"]["slack"]["quantile"])
        out["slack"] = sl
        cb = out["class_balance_Y"]
        out["mde_auroc"] = mde(cb["n_pos"], cb["n_neg"])

        json.dump(out, open(f"{a.out}/{mid}.json", "w"), indent=2)
        g = out["auroc_gate_K1"]; y = out["auroc_gate_Y"]
        g_str = f"{g['point']:.3f} [{g['lo']:.3f},{g['hi']:.3f}]" if g is not None \
            else "UNDEFINED (K1 constant on this eval split)"
        print(f"[{mid}] n={out['n']}  pos_rate={cb['pos_rate']:.3f}  est={est}")
        print(f"   AUROC(S->K1) {g_str}"
              f"   AUROC(S->Y) {y['point']:.3f} [{y['lo']:.3f},{y['hi']:.3f}]")
        print(f"   floor {sl['floor_raw']:+.3f} "
              f"({'VACUOUS' if sl['vacuous'] else 'binding'}) "
              f"= P(M=0) {sl['p_m0']:.3f} - delta {sl['delta']:.3f} - v {sl['v']:.3f}")
        print(f"   minimum resolvable AUROC difference: {out['mde_auroc']:.3f}\n")


if __name__ == "__main__":
    main()

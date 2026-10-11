#!/usr/bin/env python3
"""
skc_rebuild_c4_latent.py

Rebuilds the C4 bridge-only-vs-BGR-min ablation using the CORRECTED,
genuinely latent g_B (from skc_run_e3.py's own composed-prompt offset
capture, --save-peak-activations) instead of the original, contaminated
g_B (skc_harness.capture_for_queries()'s own acts_hop2, which names the
gold bridge directly -- see EXECUTION_PLAN.md's "CRITICAL" entry for the
full account of this bug and how it was found).

Reuses VERSION3's own selectors.Standardizer (fit-split mean/std, not
eval-set leakage) and metrics.bootstrap_ci/paired_bootstrap_diff (real
significance testing), matching the rigor of the original ablation this
replaces.

Fit-split alignment note: skc_run_e3.py does not apply the K1=1 filter
the way skc_run_e1.py's own fit-split export did (job 137/e1_gemma9b_
layer8_fitsplit_c4 etc.) -- it captures at ALL 1253 disjoint fit queries.
This script intersects on qid to recover the same K1=1-filtered fit
subset (92 gemma-2-9b, 287 llama-3.1-70b) the original ablation's own
Standardizer used, rather than assuming order/length match.

Usage: run directly, no arguments (paths are hard-coded to this
project's own real result directories for the 2 in-scope models).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # <repo>/src/driver_scripts/x.py -> <repo>
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "driver_scripts"))
sys.path.insert(0, os.path.join(REPO, "VERSION3", "code"))

from skc_gates import FrozenGate                        # noqa: E402
from skc import selectors, metrics as M                 # noqa: E402

CONFIGS = [
    {
        "model": "gemma-2-9b",
        "e1_eval_dir": "results/e1_gemma9b_layer8_skipbridge_splitfix",
        "e0_gate_dir": "results/e0_gemma9b_layer8_splitfix",
        "e3_eval_dir": "results/e3_gemma9b_layer8_latent_splitfix",
        "e3_fit_dir": "results/e3_gemma9b_layer8_latent_fitsplit",
        "e1_fitsplit_dir": "results/e1_gemma9b_layer8_fitsplit_c4",
    },
    {
        "model": "llama-3.1-70b",
        "e1_eval_dir": "results/e1_llama70b_layer25_skipbridge_splitfix",
        "e0_gate_dir": "results/e0_llama70b_layer25_splitfix",
        "e3_eval_dir": "results/e3_llama70b_layer25_latent_splitfix",
        "e3_fit_dir": "results/e3_llama70b_layer25_latent_fitsplit",
        "e1_fitsplit_dir": "results/e1_llama70b_layer25_fitsplit_c4",
    },
]


def _p(rel):
    return os.path.join(REPO, rel)


def main():
    out_dir = _p("results/c4_latent_rebuild")
    os.makedirs(out_dir, exist_ok=True)

    for cfg in CONFIGS:
        mid = cfg["model"]
        print(f"=== {mid} ===")

        e1_labels = json.load(open(_p(f"{cfg['e1_eval_dir']}/e1_labels_filtered.json")))
        acts_surface_eval = np.load(_p(f"{cfg['e1_eval_dir']}/acts_surface_filtered.npy"))
        e3_eval = json.load(open(_p(f"{cfg['e3_eval_dir']}/peak_labels.json")))
        acts_bridge_eval = np.load(_p(f"{cfg['e3_eval_dir']}/peak_activations.npy"))
        assert [l["qid"] for l in e1_labels] == [l["qid"] for l in e3_eval["labels"]], \
            "eval-set qid order mismatch between E1 surface and E3 latent-bridge captures"

        e3_fit = json.load(open(_p(f"{cfg['e3_fit_dir']}/peak_labels.json")))
        acts_bridge_fit_full = np.load(_p(f"{cfg['e3_fit_dir']}/peak_activations.npy"))
        e3_fit_qids = [l["qid"] for l in e3_fit["labels"]]
        e3_fit_qid_to_idx = {q: i for i, q in enumerate(e3_fit_qids)}

        fit_labels_orig = json.load(open(_p(f"{cfg['e1_fitsplit_dir']}/e1_labels_filtered.json")))
        acts_surface_fit_full = np.load(_p(f"{cfg['e1_fitsplit_dir']}/acts_surface_filtered.npy"))
        fit_qids = [l["qid"] for l in fit_labels_orig]

        keep_idx_fit_orig = [i for i, q in enumerate(fit_qids) if q in e3_fit_qid_to_idx]
        if len(keep_idx_fit_orig) != len(fit_qids):
            missing = len(fit_qids) - len(keep_idx_fit_orig)
            raise SystemExit(f"{mid}: {missing} K1=1-filtered fit qids not found in the "
                             "E3 fit-split capture -- check the two runs used the same "
                             "--queries file.")
        acts_surface_fit = acts_surface_fit_full[keep_idx_fit_orig]
        matched_qids = [fit_qids[i] for i in keep_idx_fit_orig]
        e3_fit_idx = [e3_fit_qid_to_idx[q] for q in matched_qids]
        acts_bridge_fit = acts_bridge_fit_full[e3_fit_idx]

        gate = FrozenGate.load(_p(f"{cfg['e0_gate_dir']}/frozen_gate_diffmeans.json"))
        gs_eval = gate.score(acts_surface_eval)
        gb_eval = gate.score(acts_bridge_eval)
        gs_fit = gate.score(acts_surface_fit)
        gb_fit = gate.score(acts_bridge_fit)

        std = selectors.Standardizer("fit").fit(g_S=gs_fit, g_B=gb_fit)
        z = std.transform(g_S=gs_eval, g_B=gb_eval)

        y = np.array([l["Y"] for l in e1_labels])
        qid = np.array([l["qid"] for l in e1_labels])

        surface_gate = z["g_S"]
        bridge_only = z["g_B"]
        bgr_min = selectors.bgr_min(z["g_S"], z["g_B"])
        bgr_mean = (z["g_S"] + z["g_B"]) / 2

        point = {}
        for name, s in [("surface_gate", surface_gate), ("bridge_only", bridge_only),
                        ("bgr_min", bgr_min), ("bgr_mean", bgr_mean)]:
            ci = M.bootstrap_ci(M.auroc, s, y, groups=qid, n_resamples=2000, ci=0.95)
            point[name] = ci
            print(f"  {name}: AUROC={ci['point']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]")

        pd = M.paired_bootstrap_diff(M.auroc, bridge_only, bgr_min, y, qid,
                                     n_resamples=2000, ci=0.95)
        print(f"  bridge_only - bgr_min = {pd['diff']:+.3f} "
             f"[{pd['lo']:+.3f},{pd['hi']:+.3f}] p={pd['p']:.4f}")

        out = {
            "model": mid,
            "n_eval": len(e1_labels),
            "n_fit_matched": len(matched_qids),
            "peak_offset_eval": e3_eval["peak_key"],
            "peak_offset_fit": e3_fit["peak_key"],
            "auroc": point,
            "bridge_only_vs_bgr_min": pd,
            "note": "g_B here is the CORRECTED latent read (skc_run_e3.py's own composed-"
                   "prompt offset capture), replacing the earlier contaminated "
                   "named-gold-bridge read. See EXECUTION_PLAN.md's CRITICAL entry.",
        }
        with open(_p(f"results/c4_latent_rebuild/{mid}.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"  saved to results/c4_latent_rebuild/{mid}.json\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
skc_pipeline_test.py
End-to-end offline test of the experiment pipeline.

WHAT THIS PROVES AND WHAT IT DOES NOT
  It runs the entire driver sequence E0 -> E6 against a MOCK model whose ground
  truth is known by construction, so it proves the PLUMBING is correct: labels,
  filters, the E0 stop gate, the frozen-gate guarantee, result schemas, and
  every guard that is supposed to raise.

  It does not prove anything about language models. Every number it prints is
  synthetic and belongs in no table. The parts that touch a real transformer
  (capture_activations, steer_at_position, patch_bridge) cannot be exercised
  without a GPU and remain UNVERIFIED. Treat your first real run as a debugging
  run and keep the shape assertions in place.

Usage:  python3 skc_pipeline_test.py
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

import skc_data as D
import skc_experiments as E
import skc_gates as G

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# =====================================================================
# A mock world with known ground truth
# =====================================================================

def build_mock_world(n=1200, d=64, seed=0):
    """
    A synthetic model in which the paper's hypothesis is TRUE by construction:
      - the surface activation encodes K1 and carries no information about M;
      - the bridge activation encodes M ALONG THE SAME DIRECTION the gate reads.

    That last clause is the point, and getting it wrong the first time made two
    tests fail. The paper's claim is not merely that the bridge state exists,
    but that it is expressed in the format the refusal mechanism already
    consumes, which is what lets a FROZEN direction read it. A mock that encoded
    M along some unrelated direction would make E3 fail by construction and
    would be testing nothing.

    A correct pipeline must recover exactly this structure.
    """
    rng = np.random.default_rng(seed)
    k1 = (rng.random(n) < 0.75).astype(int)
    m = (rng.random(n) < 0.55).astype(int)
    y = k1 * m

    u_known = rng.normal(size=d); u_known /= np.linalg.norm(u_known)
    u_bridge = rng.normal(size=d); u_bridge /= np.linalg.norm(u_bridge)

    acts_surface = rng.normal(scale=1.0, size=(n, d)) + 2.2 * k1[:, None] * u_known
    # M is encoded along u_known: same format, different position.
    acts_bridge = rng.normal(scale=1.0, size=(n, d)) + 2.0 * m[:, None] * u_known

    n_fit = 500
    fit_known = rng.normal(scale=1.0, size=(n_fit, d)) + 2.2 * u_known
    fit_unknown = rng.normal(scale=1.0, size=(n_fit, d))
    ho_lab = (rng.random(400) < 0.5).astype(int)
    ho = rng.normal(scale=1.0, size=(400, d)) + 2.2 * ho_lab[:, None] * u_known

    return dict(n=n, d=d, k1=k1, m=m, y=y, u_known=u_known, u_bridge=u_bridge,
                acts_surface=acts_surface, acts_bridge=acts_bridge,
                fit_known=fit_known, fit_unknown=fit_unknown,
                acts_heldout=ho, known_heldout=ho_lab, rng=rng)


# =====================================================================
# 1. Data layer
# =====================================================================

def test_data():
    print("\n[1] Data: matching, templates, labels")

    check("exact match", D.alias_match("John Lennon", ["John Lennon"]))
    check("case and accent folded", D.alias_match("joHN LENNON", ["Jóhn Lennon"]))
    check("alias matched", D.alias_match("Lennon", ["John Lennon", "Lennon"]))
    check("prefix accepted", D.alias_match("John Lennon, the musician", ["John Lennon"]))
    check("unrelated rejected", not D.alias_match("Paul McCartney", ["John Lennon"]))
    check("containment rejected by default",
          not D.alias_match("not John Lennon but someone else", ["John Lennon"]),
          "a list of candidates must not score as correct for each")
    check("disambiguator preserved",
          not D.alias_match("Michael Jordan", ["Michael Jordan (footballer)"]),
          "collapsing these would corrupt labels in our own favour")

    check("hedged answer is not a refusal",
          not D.is_refusal("I'm not sure, but I think it's John Lennon", ["John Lennon"]))
    check("true refusal detected",
          D.is_refusal("I don't know who that is.", ["John Lennon"]))
    check("plain answer is not a refusal",
          not D.is_refusal("John Lennon", ["John Lennon"]))

    q = D.ComposedQuery(
        qid="q1", entity_type="song",
        hops=[D.Hop("Imagine", "song", "performer", ["John Lennon"]),
              D.Hop("John Lennon", "musician", "mother", ["Julia Lennon"])])
    p = D.render_prompts(q)
    check("composed prompt names e1 not bridge",
          "Imagine" in p["composed"] and "John Lennon" not in p["composed"],
          "this is the manipulation the paper turns on")
    check("named-bridge control writes the bridge", "John Lennon" in p["named_bridge"])
    # entity="John Lennon" (TWO words): REAL BUG FOUND on the first actual
    # GPU use of this function (job 389064, E7 control (c) distractor):
    # the original signature took no `entity` argument at all and guessed
    # the insertion point by word count alone ("before the last 2 words"),
    # which silently assumed the entity is always ONE word -- for a
    # two-word entity like "John Lennon" this split the entity itself in
    # half ('John well-known well-known Lennon'), corrupting it. This
    # exact test case COULD have caught that (it already used a two-word
    # bridge entity), but the assertion below only checked the padded
    # length, never that the entity survived as an intact substring -- a
    # real gap in this test's own coverage, not just in the code, fixed
    # here alongside the signature change.
    padded = D.length_match_named_bridge(p["composed"], p["named_bridge"], entity="John Lennon")
    check("length matching does not shrink", len(padded.split()) >= len(p["named_bridge"].split()))
    check("length matching keeps a multi-word entity intact as one substring",
         "John Lennon" in padded,
         "a real bug (job 389064) split a 2-word entity in half when the old "
         "insertion logic guessed the entity was always exactly one word")

    try:
        D.ComposedQuery(qid="bad", entity_type="song",
                        hops=[D.Hop("Imagine", "song", "performer", ["John Lennon"]),
                              D.Hop("Paul McCartney", "musician", "mother", ["Mary"])]).validate()
        check("broken chain refused", False, "did not raise")
    except ValueError:
        check("broken chain refused", True)

    lab = D.compute_labels(q, {"composed": "Julia Lennon", "hop1": "John Lennon",
                               "hop2": "Julia Lennon", "named_bridge": "Julia Lennon"})
    check("labels correct", lab["Y"] == 1 and lab["K1"] == 1 and lab["K2"] == 1 and lab["M"] == 1)
    try:
        D.compute_labels(q, {"composed": "x"})
        check("partial generations refused", False, "did not raise")
    except KeyError:
        check("partial generations refused", True)

    kept = D.filter_bridge_resolved([{"qid": "a", "K1": 1}, {"qid": "b", "K1": 0}])
    check("K1=0 filtered out", len(kept) == 1 and kept[0]["qid"] == "a")
    kept2 = D.filter_bridge_resolved([{"qid": "a", "K1": 1}], {"a": False})
    check("unresolved bridge filtered out", len(kept2) == 0)

    dec = D.assign_popularity_deciles({f"e{i}": float(i) for i in range(100)})
    check("deciles span 0..9", min(dec.values()) == 0 and max(dec.values()) == 9)
    sid = D.snapshot_id([q], "2026-08-01")
    check("snapshot id is deterministic", sid == D.snapshot_id([q], "2026-08-01"))
    try:
        D.snapshot_id([q], "August 2026"); check("bad date refused", False, "did not raise")
    except ValueError:
        check("bad date refused", True)


# =====================================================================
# 2. Gates
# =====================================================================

def test_gates(W):
    print("\n[2] Gates: fitting, freezing, agreement")

    dm = G.fit_diffmeans(W["fit_known"], W["fit_unknown"], layer=12, model_name="mock")
    pr = G.fit_probe(W["fit_known"], W["fit_unknown"], layer=12, model_name="mock")
    n_lat = 40
    rng = W["rng"]
    dec = rng.normal(size=(n_lat, W["d"]))
    dec[7] = W["u_known"]
    def sae_acts(a):
        z = a @ dec.T
        return np.maximum(z - 0.5, 0.0)
    sae = G.fit_sae_gate(
        {"typeA": (sae_acts(W["fit_known"][:250]), sae_acts(W["fit_unknown"][:250])),
         "typeB": (sae_acts(W["fit_known"][250:]), sae_acts(W["fit_unknown"][250:]))},
        dec, layer=12, model_name="mock")
    check("SAE picks the planted latent", sae.sae_latent_id == 7,
          f"selected {sae.sae_latent_id}")

    gates = {"diffmeans": dm, "probe": pr, "sae": sae}
    import skc_metrics as M
    for name, g in gates.items():
        auc = M.auroc(W["known_heldout"], g.score(W["acts_heldout"]))
        check(f"{name} separates on held-out entities", auc > 0.8, f"AUROC={auc:.3f}")

    agree = G.gate_agreement(gates, W["acts_heldout"])
    check("estimators agree", agree["min_pairwise_spearman"] > 0.5,
          f"min rho={agree['min_pairwise_spearman']:.3f}")

    try:
        dm.assert_not_fit_on(np.vstack([W["fit_known"], W["fit_unknown"]]))
        check("refit-on-fitting-data detected", False, "did not raise")
    except RuntimeError:
        check("refit-on-fitting-data detected", True)
    dm.assert_not_fit_on(W["acts_heldout"])
    check("held-out data passes the freeze check", True)

    try:
        dm.score(np.zeros((5, W["d"] + 3)))
        check("dim mismatch refused", False, "did not raise")
    except ValueError:
        check("dim mismatch refused", True)

    import os, tempfile as tf
    p = os.path.join(tf.mkdtemp(), "g.json")
    dm.save(p)
    check("save/load round-trips",
          np.allclose(G.FrozenGate.load(p).direction, dm.direction))
    return gates


# =====================================================================
# 3. Drivers
# =====================================================================

def test_drivers(W, gates):
    print("\n[3] Drivers: E0 stop gate and E1-E6")
    outdir = tempfile.mkdtemp()
    try:
        rng = W["rng"]
        n = W["n"]
        named_k2 = (rng.random(400) < 0.5).astype(int)
        acts_named = rng.normal(size=(400, W["d"])) + 2.0 * named_k2[:, None] * W["u_known"]
        k2_named = named_k2

        labels = [{"qid": f"q{i}", "k": 2, "Y": int(W["y"][i]), "K1": int(W["k1"][i]),
                   "K2": int(W["m"][i]), "M": int(W["m"][i])} for i in range(n)]

        inputs = {
            "gates": gates, "primary_estimator": "diffmeans",
            "acts_heldout": W["acts_heldout"], "known_heldout": W["known_heldout"],
            "acts_named_bridge": acts_named, "k2_named": k2_named,
            "labels": labels,
            "acts_surface": W["acts_surface"], "acts_bridge": W["acts_bridge"],
            "acts_singlehop_correct": W["acts_heldout"][W["known_heldout"] == 1],
        }
        man = E.run_all(inputs, outdir, "2026-08-01+mock", "mock-model", seed=0)

        e0 = man.entries["E0"]
        check("E0 gate opens on a working instrument", e0["gate_open"])

        e1 = man.entries["E1"]
        check("gate is informative about K1", e1["auroc"]["gate_to_K1"] > 0.75,
              f"{e1['auroc']['gate_to_K1']:.3f}")
        check("gate is at chance on M (H1 recovered)",
              abs(e1["auroc"]["gate_to_M"] - 0.5) < 0.06,
              f"{e1['auroc']['gate_to_M']:.3f}")
        check("floor is not contradicted", not e1["floor"]["violated"],
              f"floor={e1['floor']['floor']:.3f}")
        d50 = e1["decomposition"]["0.5"]
        check("blind term is large when the bridge is unread", d50["blind"] > 0.02,
              f"blind={d50['blind']:.3f}")
        check("decomposition identity is exact", abs(d50["identity_residual"]) < 1e-9)
        check("silence check runs", "ks_statistic" in e1["silence"])

        e3 = E.run_E3(gates["diffmeans"],
                      {"L12_bridge": W["acts_bridge"], "L12_surface": W["acts_surface"]},
                      W["m"], man, n_perm=200)
        check("E3 peaks at the bridge position",
              e3["peak"]["key"] == "L12_bridge", e3["peak"]["key"])
        check("E3 peak beats the permutation null", e3["permutation_null"]["p_value"] < 0.05)

        import skc_metrics as M
        gs = gates["diffmeans"].score(W["acts_surface"])
        gb = gates["diffmeans"].score(W["acts_bridge"])
        e4 = E.run_E4(W["y"], {"surface gate": gs}, gs, gb, man)
        check("BGR-min beats the surface gate",
              e4["selectors"]["BGR-min"]["auroc"] > e4["selectors"]["surface gate"]["auroc"],
              f"{e4['selectors']['BGR-min']['auroc']:.3f} vs "
              f"{e4['selectors']['surface gate']['auroc']:.3f}")
        check("rho is reported", "rho" in e4["rho_BGR_min"])

        dr = E.dissociation_ratio(rng.normal(0.8, 0.4, 200), rng.normal(0.02, 0.4, 200))
        check("dissociation ratio >> 1 when only recall moves", dr["ratio"] > 3,
              f"ratio={dr['ratio']:.1f}")

        labels3 = [{"Y": int(W["y"][i] * W["m"][(i + 7) % n]), "K2": int(W["m"][i]),
                    "K3": int(W["m"][(i + 7) % n])} for i in range(n)]
        e5 = E.run_E5(labels, labels3, gs, gs, man)
        check("uncovered fraction does not shrink with k", e5["leading_term_grows"],
              f"k2={e5['k2']['p_M_zero']:.3f} k3={e5['k3']['p_M_zero']:.3f}")

        import os
        check("manifest written", os.path.exists(os.path.join(outdir, "manifest.json")))
        check("per-experiment JSON written",
              all(os.path.exists(os.path.join(outdir, f"{e}.json"))
                  for e in ("E0", "E1", "E3", "E4", "E5")))
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


# =====================================================================
# 4. The guards must fire
# =====================================================================

def test_guards(W, gates):
    print("\n[4] Guards: failure must be loud")
    outdir = tempfile.mkdtemp()
    try:
        man = E.RunManifest(outdir=outdir, snapshot_id="x", model_name="mock")
        rng = W["rng"]
        junk = rng.normal(size=(400, W["d"]))
        broken = {k: gates[k] for k in gates}
        e0 = E.run_E0(broken, junk, (rng.random(400) < 0.5).astype(int),
                      junk, (rng.random(400) < 0.5).astype(int), man)
        check("E0 closes on an uninformative instrument", not e0["gate_open"])
        try:
            E.require_E0(e0); check("require_E0 raises when closed", False, "did not raise")
        except E.E0Failure:
            check("require_E0 raises when closed", True)

        try:
            E.run_all({"gates": gates, "acts_heldout": junk,
                       "known_heldout": (rng.random(400) < 0.5).astype(int),
                       "acts_named_bridge": junk,
                       "k2_named": (rng.random(400) < 0.5).astype(int),
                       "labels": [], "acts_surface": junk, "acts_bridge": junk,
                       "acts_singlehop_correct": junk},
                      outdir, "x", "mock")
            check("run_all halts at a failed E0", False, "did not raise")
        except E.E0Failure:
            check("run_all halts at a failed E0", True)

        try:
            E.dissociation_ratio(np.ones(50), np.ones(50))
            check("degenerate dissociation refused", False, "did not raise")
        except RuntimeError:
            check("degenerate dissociation refused", True)

        try:
            E.run_E6(np.array([1, 0]), np.array([1.0, 0.0]), np.array([1, 0]),
                     np.array([1.0, 0.0]), np.array([True, True]),
                     E.RunManifest(outdir=outdir, snapshot_id="x", model_name="m"))
            check("E6 refuses to pool a one-sided split", False, "did not raise")
        except ValueError:
            check("E6 refuses to pool a one-sided split", True)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def main():
    print("=" * 74)
    print("PIPELINE TEST. Mock model, synthetic data. These numbers are NOT results.")
    print("=" * 74)
    W = build_mock_world()
    test_data()
    gates = test_gates(W)
    test_drivers(W, gates)
    test_guards(W, gates)
    print("\n" + "=" * 74)
    print(f"PIPELINE TEST {'PASSED' if not FAIL else 'FAILED'}   "
          f"({len(PASS)} passed, {len(FAIL)} failed)")
    if FAIL:
        print("Failures:", ", ".join(FAIL))
    print("Verified: plumbing only. Model-facing code (activation capture, steering,")
    print("patching) needs a GPU and remains UNVERIFIED. See skc_REDFLAGS.md.")
    print("=" * 74)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())

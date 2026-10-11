#!/usr/bin/env python3
"""
skc_run_e2_c4_graphs.py
C4: graph trace. Builds real attribution graphs over transcoders
(circuit-tracer, Ameisen et al. 2025 / Lindsey et al. 2025) on matched
composed and named-bridge prompts, per skc_main.tex's own Appendix
control-suite spec ("graph trace reads routing off the computational
graph ... every graph-derived claim is validated by intervention").

SCOPE, STATED HONESTLY (do not present this script's own output as a
complete C4 result): circuit-tracer's installation and attribution-
graph CONSTRUCTION are genuinely turnkey for Gemma-2-2b (confirmed
first-class support: "gemma" transcoder shortcut, GemmaScope/CLT
dictionaries on HuggingFace, works within ~15GB VRAM with CPU/disk
offload). What this script does NOT do, and what circuit-tracer itself
does NOT provide, is IDENTIFY which specific feature(s) in the saved
graph correspond to "the refusal features" the paper's own causal
design (Figure~\\ref{fig:causal}, control (m)) needs edge weight
INTO -- there is no pre-labeled refusal feature in circuit-tracer's own
released dictionaries; that is separate, original interpretability
research (inspecting top-activating examples for candidate features,
cross-checking against Neuronpedia where indexed, or a dedicated
feature-search pass), not a library function to call. THIS IS MADE
WORSE, not just theoretically absent, by an already-confirmed real
finding this project made independently: Gemma-2-2b's refusal rate on
this exact composed-prompt template is EXACTLY 0.000 at every steering
coefficient tried (C1/C2/C3 all agree, job 383331) -- there is no
observed refusal behavior on this template to correlate candidate
features against in the first place.

WHAT THIS SCRIPT DOES: for each query, builds and saves the RAW
attribution graph (circuit_tracer.attribute()'s own output, via
graph.to_pt()) for both the composed prompt (bridge latent) and the
matched named-bridge prompt (bridge written) -- the "positive
comparison" control (m)'s own text calls for ("there the entity is
written, so a path should appear"). Saves every graph to disk for
LATER feature-level analysis, once a refusal-feature identification
pass exists, rather than computing a premature "summed edge weight into
refusal features" number that would need to invent which nodes count as
"refusal features" without real evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skc_data import load_queries, render_prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True,
                    help="two-hop ComposedQuery JSON (composed + named_bridge prompts "
                         "rendered from each item)")
    ap.add_argument("--model", default="gemma-2-2b", choices=["gemma-2-2b"],
                    help="circuit-tracer's own first-class support this session was "
                         "confirmed ONLY for Gemma-2-2b ('gemma' transcoder shortcut); "
                         "do not pass another model key without first confirming "
                         "circuit-tracer has a transcoder dictionary for it.")
    ap.add_argument("--n", type=int, default=20,
                    help="number of queries to build graphs for (a full-dataset pass is "
                         "expensive; this is a first real batch, not the full test set)")
    ap.add_argument("--max-feature-nodes", type=int, default=8192)
    ap.add_argument("--offload", default="cpu", choices=["cpu", "disk", "none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import torch
        from circuit_tracer import ReplacementModel, attribute
    except ImportError as e:
        raise SystemExit(
            f"circuit-tracer is not installed or not importable ({e}). Install with:\n"
            "  git clone https://github.com/safety-research/circuit-tracer\n"
            "  pip install -e circuit-tracer/\n"
            "This is a SEPARATE toolchain from the rest of this pipeline's own "
            "requirements.txt, confirmed genuinely absent from this environment before "
            "this script was written -- see the Slurm job's own install step."
        )

    import random
    rng = random.Random(args.seed)
    queries = load_queries(args.queries)
    if len(queries) > args.n:
        queries = rng.sample(queries, args.n)
    print(f"Building attribution graphs for {len(queries)} queries "
         f"(composed + named-bridge, {2 * len(queries)} graphs total) ...", flush=True)

    print(f"Loading {args.model} as a ReplacementModel (transcoders='gemma') ...",
         flush=True)
    offload = None if args.offload == "none" else args.offload
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", "gemma", dtype=torch.bfloat16, backend="transformerlens")

    manifest = []
    for i, q in enumerate(queries):
        prompts = render_prompts(q)
        for kind in ("composed", "named_bridge"):
            prompt = prompts[kind]
            graph_path = os.path.join(args.out, f"{q.qid}_{kind}.pt")
            print(f"  [{i+1}/{len(queries)}] {kind}: {q.qid}", flush=True)
            graph = attribute(
                prompt=prompt, model=model, max_n_logits=10, desired_logit_prob=0.95,
                batch_size=256, max_feature_nodes=args.max_feature_nodes,
                offload=offload, verbose=False,
            )
            graph.to_pt(graph_path)
            manifest.append({"qid": q.qid, "kind": kind, "prompt": prompt,
                            "graph_path": graph_path,
                            "n_active_features": len(graph.active_features)})

    with open(os.path.join(args.out, "c4_graph_manifest.json"), "w") as f:
        json.dump({
            "n_queries": len(queries),
            "n_graphs": len(manifest),
            "graphs": manifest,
            "NOT_DONE": "Refusal-feature identification and summed-edge-weight scoring "
                       "are NOT computed by this script -- see this file's own module "
                       "docstring for why (no pre-labeled refusal feature exists in "
                       "circuit-tracer's own dictionaries, and this model/template has "
                       "0.000 observed refusal rate to correlate candidate features "
                       "against in the first place). This manifest only records that the "
                       "raw graphs were built and saved; C4/control (m) remain open "
                       "pending a real feature-identification pass.",
        }, f, indent=1)
    print(f"\nSaved {len(manifest)} raw attribution graphs and a manifest to {args.out}. "
         "Refusal-feature identification is NOT done by this script -- see the "
         "manifest's own NOT_DONE field.")


if __name__ == "__main__":
    main()

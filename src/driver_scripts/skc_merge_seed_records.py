#!/usr/bin/env python3
"""
skc_merge_seed_records.py

Merges per-seed VERSION3 record files (f1_twohop_seed0.jsonl,
f1_twohop_seed1.jsonl, ...) into the single f1_twohop.jsonl VERSION3's
skc/io.py expects (one file per model/family, containing all seeds --
io.Records.seeds() reads the union of the 'seed' field across all rows in
that one file, not one file per seed).

Also exports seed 0's own already-existing single-seed file (the
pre-Phase-4 "*_splitfix" runs did not use --export-records, since that
flag did not exist yet when they ran) if a seed-0-specific file is not
already present, by looking for it under the model's own directory
without a _seed suffix (the posthoc exporter's own naming convention).

Usage:
    python3 skc_merge_seed_records.py --model gemma-2-2b --family f1_twohop
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records-root", default=None,
                    help="defaults to <repo>/VERSION3/data/records")
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="f1_twohop")
    ap.add_argument("--overwrite", action="store_true",
                    help="write directly to <family>.jsonl instead of <family>.jsonl.merged")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # <repo>/src/driver_scripts/x.py -> <repo>
    root = args.records_root or os.path.join(repo, "VERSION3", "data", "records")
    model_dir = os.path.join(root, args.model)

    seed_files = sorted(glob.glob(os.path.join(model_dir, f"{args.family}_seed*.jsonl")))
    fitsplit_file = os.path.join(model_dir, f"{args.family}_fitsplit.jsonl")
    base_file = os.path.join(model_dir, f"{args.family}.jsonl")

    seen_qids_by_seed: dict[int, set] = {}

    # The seed-0 export (from skc_export_records_posthoc.py, before
    # --export-records existed) landed at the base filename with no _seedN
    # suffix. Include it first if present, then every f1_twohop_seedN.jsonl,
    # then the fit-split file (C4's own Standardizer population, split="fit"
    # -- a DIFFERENT split label from every seed file's own split="eval", so
    # merging it in does not create any split-leakage: io.py's own
    # _check_split_leakage() guards on query_id appearing in >1 split, and
    # the fit-split queries are a disjoint query population by construction
    # (see skc_build_c4_fit_queries.py), not the same items relabeled).
    sources = []
    if os.path.exists(base_file):
        sources.append(base_file)
    sources.extend(seed_files)
    if os.path.exists(fitsplit_file):
        sources.append(fitsplit_file)

    if not sources:
        raise SystemExit(f"no record files found for {args.model}/{args.family} under {model_dir}")

    # De-duplicate by (query_id, seed, split): base_file may ALREADY be the
    # output of a previous merge (this script overwrites it in --overwrite
    # mode, and even without --overwrite a user can `mv` the .merged file
    # onto it) -- re-including base_file alongside the individual seed/
    # fitsplit files would then double-count every row already folded in.
    # Keying on (query_id, seed, split) rather than just query_id: the SAME
    # query_id can legitimately appear once per seed (each seed re-scores
    # the same eval queries under a different gate) but must appear at most
    # once per (seed, split) pair. Last write wins, so re-running an export
    # for one seed and re-merging naturally picks up the newer file.
    by_key: dict[tuple, dict] = {}
    for path in sources:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["query_id"], row["seed"], row["split"])
                by_key[key] = row
    all_rows = list(by_key.values())
    for row in all_rows:
        seen_qids_by_seed.setdefault(row["seed"], set()).add(row["query_id"])

    seeds_found = sorted(seen_qids_by_seed)
    print(f"Merging {len(sources)} file(s) for {args.model}/{args.family}: "
         f"{[os.path.basename(s) for s in sources]}")
    print(f"Seeds present after merge: {seeds_found} ({len(all_rows)} total rows)")

    out_path = base_file if args.overwrite else base_file + ".merged"
    with open(out_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote merged file to {out_path}.")
    if not args.overwrite:
        print(f"Once you've confirmed this looks right, replace {base_file} with it "
             f"(e.g. `mv {out_path} {base_file}`), OR re-run this script with --overwrite.")


if __name__ == "__main__":
    main()

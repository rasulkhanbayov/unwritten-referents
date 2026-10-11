"""
skc_export_records.py

Writes real per-item records from this project's own pipeline
(driver_scripts/, skc_experiments.py) into the exact JSONL schema
VERSION3/code/skc/io.py requires, so VERSION3's analysis scripts
(reconcile.py, ablations.py, controls.py, figures.py, timing.py) can run
on real data instead of raising MissingData.

One function, write_jsonl_records(), imports io.py's own REQUIRED/BINARY/
SPLITS/ESTIMATORS constants directly (not a re-typed copy) so the two
schemas can never drift apart. Every record is validated against them
BEFORE being written, so a malformed export fails loudly here rather than
being discovered later as an opaque MissingData deep inside some
analysis script.

SPLIT NAMING: this project's own E0/E1 pipeline uses "fit"/"held-out" for
the entity-level split (skc_run_e0.py's split_entities()). VERSION3's
io.py wants "fit" | "validation" | "eval" (three-way, not two). The
mapping used here: this project's "fit" entities -> io.py's "fit"
(standardization/gate-fitting data, same role); this project's
"held-out" entities, WHICH THEMSELVES SUPPLY BOTH E1's held-out gate
evaluation AND the composed-query test set -- since there is no existing
third partition in this pipeline, held-out queries are exported as
"eval" (the split every analysis in VERSION3 actually reads by default,
per io.py's own module docstring: "Analyses read 'eval'"). No records
are exported as "validation" from this pipeline; VERSION3's offset-sweep
analysis (which reads "validation") will need its own dedicated
held-out-within-held-out slice if that becomes necessary -- not invented
here without being asked for.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

VERSION3_SKC = str(Path(__file__).resolve().parents[2] / "VERSION3" / "code")  # <repo>/src/driver_scripts/x.py -> <repo>
if VERSION3_SKC not in sys.path:
    sys.path.insert(0, VERSION3_SKC)

from skc.io import REQUIRED, BINARY, SPLITS, ESTIMATORS  # noqa: E402


def write_jsonl_records(path, query_ids, split, seed, y, k1, k2, g_s, g_b,
                         estimator, offset, **optional_fields):
    """
    Write one JSONL record per item to `path` (parent directories created as
    needed), validated against io.py's own schema before any line is written.

    Positional args are parallel sequences, one entry per item (query_ids,
    y, k1, k2, g_s, g_b) except split/seed/estimator/offset, which are
    either a single value applied to every row or a parallel sequence (to
    support, e.g., a run that mixes estimators across items -- though
    io.require_single_estimator() will still require exactly one estimator
    per FILE, so mixing within one write_jsonl_records() call to the same
    path is very likely a mistake, not a feature to rely on).

    optional_fields: keyword args, each a parallel sequence, for any of
    io.py's REQUIRED-FOR-SPECIFIC-ANALYSES fields (g_S_named, g_S_decoy,
    text_overlap, relation, baseline_scores, wall_clock_ms). Omit entirely
    for records that don't have them -- do not pass None-filled lists,
    since io.py's own loader treats a present-but-None field as missing
    anyway (col()'s own check), but a field ABSENT from the dict is cleaner
    and matches how optional fields are described in io.py's docstring.
    """
    n = len(query_ids)
    for name, seq in [("y", y), ("k1", k1), ("k2", k2), ("g_s", g_s), ("g_b", g_b)]:
        if len(seq) != n:
            raise ValueError(f"{name} has length {len(seq)}, expected {n} (len(query_ids))")

    def _broadcast(v, name):
        if isinstance(v, (list, tuple)):
            if len(v) != n:
                raise ValueError(f"{name} has length {len(v)}, expected {n}")
            return list(v)
        return [v] * n

    splits = _broadcast(split, "split")
    seeds = _broadcast(seed, "seed")
    estimators = _broadcast(estimator, "estimator")
    offsets = _broadcast(offset, "offset")

    for name, seq in optional_fields.items():
        if len(seq) != n:
            raise ValueError(f"optional field '{name}' has length {len(seq)}, expected {n}")

    rows = []
    for i in range(n):
        row = {
            "query_id": query_ids[i],
            "split": splits[i],
            "seed": int(seeds[i]),
            "Y": int(y[i]),
            "K1": int(k1[i]),
            "K2": int(k2[i]),
            "g_S": float(g_s[i]),
            "g_B": float(g_b[i]),
            "estimator": estimators[i],
            "offset": int(offsets[i]),
        }
        for name, seq in optional_fields.items():
            v = seq[i]
            if v is not None:
                row[name] = v
        rows.append(row)

    # Validate against io.py's own constants before writing anything --
    # catches a schema mismatch here, at export time, rather than as a
    # MissingData raised later inside some analysis script far from the
    # code that produced the bad row.
    for row in rows:
        for k in REQUIRED:
            if k not in row or row[k] is None:
                raise ValueError(f"record for query_id={row.get('query_id')!r} "
                                 f"is missing required field '{k}'")
        for k in BINARY:
            if row[k] not in (0, 1):
                raise ValueError(f"record for query_id={row['query_id']!r} field "
                                 f"'{k}' = {row[k]!r}, want 0 or 1")
        if row["split"] not in SPLITS:
            raise ValueError(f"record for query_id={row['query_id']!r} split "
                             f"'{row['split']}' not in {SPLITS}")
        if row["estimator"] not in ESTIMATORS:
            raise ValueError(f"record for query_id={row['query_id']!r} estimator "
                             f"'{row['estimator']}' not in {ESTIMATORS}")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} records to {out_path}", file=sys.stderr)
    return out_path

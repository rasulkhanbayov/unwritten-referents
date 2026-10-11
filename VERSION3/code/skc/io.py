"""
The data contract between the model-side pipeline and every analysis here.

Export one JSONL file per (model, family) to `data/records/{model}/{family}.jsonl`.
One line per evaluation item. The loader validates every field and refuses to
substitute a default for a missing one, because a silent default is how a
number ends up in a paper without anyone having measured it.

REQUIRED FIELDS
  query_id   str    stable id. The bootstrap resamples THIS, so a composed item
                    and its decompositions must share one id.
  split      str    "fit" | "validation" | "eval". Analyses read "eval"; the
                    offset sweep reads "validation"; standardization reads "fit".
  seed       int    which pipeline seed produced this record.
  Y          0/1    composed prompt answered correctly (alias-aware exact match)
  K1         0/1    hop 1 correct in isolation
  K2         0/1    hop 2 correct in isolation, asked of the GOLD bridge
  g_S        float  gate score at the final token of the written entity
  g_B        float  same frozen direction at the bridge-resolution position
  estimator  str    which of diffmeans | probe | sae produced g_S and g_B.
                    The paper never says; this field forces the answer.
  offset     int    read offset used for g_B, relative to the written entity

REQUIRED FOR SPECIFIC ANALYSES (absent is fine if you are not running them)
  g_S_named    float  gate score on the length- and popularity-matched
                      named-bridge prompt          -> named-bridge comparison
  g_S_decoy    float  same on the length-matched decoy prompt -> distractor
  text_overlap float  prompt/target lexical overlap -> control (n) subset
  relation     str    composable pair id           -> relation-wise breakdown
  baseline_scores  {name: float}  one entry per published baseline
  wall_clock_ms    float          per-item scoring cost -> efficiency table

The relation field matters more than it looks. Every availability number in
the paper currently comes from place-of-birth -> country, which Yang et al.
(2024) exclude from their own two-hop data because the bridge is inferable
from the head entity. Recording it lets scripts/reconcile.py report per
relation as soon as the inventory widens.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED = ("query_id", "split", "seed", "Y", "K1", "K2", "g_S", "g_B",
            "estimator", "offset")
BINARY = ("Y", "K1", "K2")
SPLITS = ("fit", "validation", "eval")
ESTIMATORS = ("diffmeans", "probe", "sae")


class MissingData(RuntimeError):
    """Raised instead of guessing. Carries what to export and why."""


@dataclass
class Records:
    rows: list[dict]
    model: str
    family: str
    _arrays: dict = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.rows)

    def col(self, name: str) -> np.ndarray:
        if name in self._arrays:
            return self._arrays[name]
        missing = [r["query_id"] for r in self.rows if name not in r or r[name] is None]
        if missing:
            raise MissingData(
                f"field '{name}' is absent on {len(missing)} of {len(self.rows)} "
                f"records for {self.model}/{self.family} (e.g. {missing[:3]}). "
                f"See skc/io.py for what to export. Nothing is substituted.")
        arr = np.array([r[name] for r in self.rows])
        self._arrays[name] = arr
        return arr

    def has(self, name: str) -> bool:
        return all(name in r and r[name] is not None for r in self.rows)

    def where(self, **eq) -> "Records":
        rows = [r for r in self.rows if all(r.get(k) == v for k, v in eq.items())]
        return Records(rows, self.model, self.family)

    def seeds(self) -> list[int]:
        return sorted({r["seed"] for r in self.rows})

    def M(self, k: int = 2) -> np.ndarray:
        """M = prod_{j>=2} K_j. At k=2 this IS K2, by definition."""
        if k == 2:
            return self.col("K2")
        cols = [self.col(f"K{j}") for j in range(2, k + 1)]
        return np.prod(np.column_stack(cols), axis=1)


def load(root, model: str, family: str) -> Records:
    path = Path(root) / model / f"{family}.jsonl"
    if not path.exists():
        raise MissingData(
            f"{path} does not exist. Export it from the pipeline; the schema is "
            f"documented at the top of skc/io.py. No analysis in this package "
            f"will run on partial data.")
    rows = []
    with path.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise MissingData(f"{path}:{i} is not valid JSON: {e}") from e
            for k in REQUIRED:
                if k not in r or r[k] is None:
                    raise MissingData(f"{path}:{i} is missing required field '{k}'")
            for k in BINARY:
                if r[k] not in (0, 1):
                    raise MissingData(f"{path}:{i} field '{k}' = {r[k]!r}, want 0 or 1")
            if r["split"] not in SPLITS:
                raise MissingData(f"{path}:{i} split '{r['split']}' not in {SPLITS}")
            if r["estimator"] not in ESTIMATORS:
                raise MissingData(f"{path}:{i} estimator '{r['estimator']}' not in {ESTIMATORS}")
            rows.append(r)
    if not rows:
        raise MissingData(f"{path} is empty")
    _check_split_leakage(rows, path)
    return Records(rows, model, family)


def _check_split_leakage(rows, path):
    """
    A query id appearing in two splits is the leakage vector that would explain
    the headline result, so this is a hard failure rather than a warning.
    """
    by_id: dict[str, set] = {}
    for r in rows:
        by_id.setdefault(r["query_id"], set()).add(r["split"])
    bad = {q: s for q, s in by_id.items() if len(s) > 1}
    if bad:
        raise MissingData(
            f"{path}: {len(bad)} query ids appear in more than one split "
            f"(e.g. {list(bad)[:3]}). The partition must be over the union of "
            f"head entity, bridge and target; if a test item's BRIDGE appeared "
            f"as a written entity in a fit-split prompt, the direction was fit "
            f"on that referent and the transfer result is partly memorization.")


def require_seeds(recs: Records, n_required: int) -> list[int]:
    seeds = recs.seeds()
    if len(seeds) < n_required:
        raise MissingData(
            f"{recs.model}/{recs.family} has {len(seeds)} seed(s), needs "
            f"{n_required}. A bootstrap over queries estimates sampling noise "
            f"in the evaluation set and says nothing about variance from probe "
            f"initialization, penalty-selection folds, cross-fitting folds, or "
            f"sparse-autoencoder latent selection. Re-run the pipeline under "
            f"seeds {list(range(n_required))}.")
    return seeds


def require_single_estimator(recs: Records) -> str:
    """The paper reports one gate number per item but fits three directions."""
    est = sorted({r["estimator"] for r in recs.rows})
    if len(est) != 1:
        raise MissingData(
            f"{recs.model}/{recs.family} mixes estimators {est}. Export one "
            f"file per estimator and state in the paper which one produces the "
            f"reported gate score, or give the combination rule.")
    return est[0]

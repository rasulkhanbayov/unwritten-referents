# Analysis code

No module here contains a number from the paper. Everything is computed from
per-item records (schema at the top of `skc/io.py`), and every function raises
rather than substituting a default for data that is absent.

```
skc/metrics.py    AUROC, DeLong, AURC, selective risk, R*, query-level bootstrap, BH
skc/slack.py      delta (one-sided), v, the written-position floor, the decomposition
skc/selectors.py  surface gate, bridge-only, BGR min/mean/max/lr, fit-split standardizer
skc/io.py         record schema, loader, split-leakage and seed checks

scripts/reconcile.py  recompute every reported number with intervals
scripts/ablations.py  the bridge-only arm and its verdict
scripts/controls.py   zero-overlap, level shift, CoT paired
scripts/figures.py    risk-coverage curve and offset sweep as TikZ
scripts/timing.py     per-selector cost

tests/test_smoke.py   synthetic-data checks, including the two estimation traps
```

Start with `python tests/test_smoke.py`. Then `../EXPERIMENTS.md` for what to
run and why, and `../REPRODUCIBILITY.md` for the data contract.

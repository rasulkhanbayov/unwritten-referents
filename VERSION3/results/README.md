# Results

Empty until you run something. Scripts create their own subdirectories.

| Subdirectory | Written by | Contents |
|---|---|---|
| `reconcile/` | `code/scripts/reconcile.py` | every reported number with intervals, class balance, slack terms, minimum resolvable difference |
| `ablations/` | `code/scripts/ablations.py` | the four ablation arms, the paired test, and a plain-words verdict |
| `controls/` | `code/scripts/controls.py` | zero-overlap, level shift, chain-of-thought paired |
| `figures/` | `code/scripts/figures.py` | TikZ the paper can `\input`, so figures and records cannot drift apart |
| `timing/` | `code/scripts/timing.py` | per-selector scoring cost |

Everything is JSON except the figures. Nothing here is under version control by
default; commit what you want to keep.

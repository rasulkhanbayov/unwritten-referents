# Manifest

```
START_HERE.md          read this first if you are picking the project up
README.md              what the paper claims and does not claim
REDFLAGS.md            42 issues by severity, each with the exact action
EXPERIMENTS.md         the eleven open experiments, in priority order
REPRODUCIBILITY.md     dependencies, data contract, seeds, hardware
MANIFEST.md            this file

paper/
  skc_main.tex         manuscript source; figures are native TikZ, no image files
  skc_refs.bib         56 entries, all cited, none uncited
  skc_main.bbl         precompiled, so two pdflatex passes suffice without BibTeX
  build.sh             builds both variants into pdf/
  iclr2027_conference.{sty,bst}, natbib.sty, fancyhdr.sty, math_commands.tex
                       venue style files, unmodified, kept alongside the source
                       so plain pdflatex works with no TEXINPUTS fiddling

pdf/
  skc_main_ANNOTATED.pdf   all 44 red flags visible; the working copy
  skc_main_CLEAN.pdf       submission shape, main text 10 pages

code/
  README.md            what each module does
  requirements.txt     numpy, scipy, pyyaml; torch is commented out and not needed
  configs/experiments.yaml
  skc/                 metrics, slack, selectors, io
  scripts/             reconcile, ablations, controls, figures, timing
  tests/test_smoke.py  20 checks on synthetic data
  verify_skc.py        theory harness and manuscript arithmetic

data/records/          empty, with one directory per model; see its README
results/               empty; scripts create their own subdirectories
```

## Build

```
cd paper && ./build.sh
```

Writes both PDFs into `pdf/`. `\redflagstrue` on line 104 of `skc_main.tex` is
the default and produces the annotated build. Both compile with no errors, no
undefined or multiply-defined citations, and no overfull boxes.

## Deliberately absent

**No per-item records.** The analysis code consumes them in a documented schema
and the authors must export them. Shipping placeholders would let every script
run and print numbers that mean nothing, which is what this package exists to
prevent.

**No results.** Every script writes to `results/` on first run.

**No fabricated outputs anywhere.** Where an experiment has not run, the code
raises with what to supply and `EXPERIMENTS.md` says what running it would show.

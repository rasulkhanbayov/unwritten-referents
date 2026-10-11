#!/usr/bin/env bash
# Build both variants. Run from this directory.
set -e
build () {  # $1 = redflag state, $2 = output name
  sed -i.bak "s/^\\\\redflags\(true\|false\)$/\\\\redflags$1/" skc_main.tex && rm -f skc_main.tex.bak
  pdflatex -interaction=nonstopmode skc_main.tex >/dev/null
  bibtex skc_main >/dev/null
  pdflatex -interaction=nonstopmode skc_main.tex >/dev/null
  pdflatex -interaction=nonstopmode skc_main.tex >/dev/null
  cp skc_main.pdf "../pdf/$2"
  echo "  ../pdf/$2  ($(pdfinfo skc_main.pdf | awk '/Pages/{print $2}') pages)"
}
echo "building:"
build true  skc_main_ANNOTATED.pdf
build false skc_main_CLEAN.pdf
sed -i.bak 's/^\\redflagsfalse$/\\redflagstrue/' skc_main.tex && rm -f skc_main.tex.bak
echo "done. annotated build is the default in skc_main.tex"

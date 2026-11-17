#!/bin/bash
# Builds the paper, the literature review, the code appendix and the combined PDF.
#   GROUP=7 ./tools/build_pdf.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GROUP="${GROUP:-XX}"
NAME="INN-RC"
sed -i "s/\\\\newcommand{\\\\groupno}{[^}]*}/\\\\newcommand{\\\\groupno}{$GROUP}/" "$ROOT/paper/main.tex" "$ROOT/docs/header.tex"

cd "$ROOT/paper"
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
echo "paper: $(pdfinfo main.pdf | grep Pages)"

cd "$ROOT/docs"
pdflatex -interaction=nonstopmode literature_review.tex > /dev/null
: > code_list.tex
for f in "$ROOT"/matlab/*.m; do
  b=$(basename "$f")
  echo "\\subsection*{\\texttt{${b//_/\\_}}}" >> code_list.tex
  echo "\\lstinputlisting{../matlab/$b}" >> code_list.tex
done
pdflatex -interaction=nonstopmode code_appendix.tex > /dev/null
pdflatex -interaction=nonstopmode code_appendix.tex > /dev/null
echo "code appendix: $(pdfinfo code_appendix.pdf | grep Pages)"

mkdir -p "$ROOT/submission"
cp "$ROOT/paper/main.pdf" "$ROOT/submission/Group${GROUP}_${NAME}_IEEE_paper.pdf"
pdfunite "$ROOT/paper/main.pdf" "$ROOT/docs/literature_review.pdf" "$ROOT/docs/code_appendix.pdf" \
         "$ROOT/submission/Group${GROUP}_${NAME}.pdf"
echo "combined: $(pdfinfo "$ROOT/submission/Group${GROUP}_${NAME}.pdf" | grep Pages)"
rm -f "$ROOT"/paper/*.aux "$ROOT"/paper/*.log "$ROOT"/paper/*.blg "$ROOT"/paper/*.out "$ROOT"/docs/*.aux "$ROOT"/docs/*.log "$ROOT"/docs/*.out

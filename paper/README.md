# paper/

IEEE two-column versions of the thesis.

    journal/      report.tex  -> report.pdf (33 pages)  IEEEtran [journal], the version to submit
                  REVIEW_report.md lists the fixes already applied to report.tex
    conference/   paper.tex   -> paper.pdf (16 pages)   IEEEtran [conference], shorter version

Both read their figures from `../../thesis/images/` (set by `\graphicspath`), so a figure
redrawn for the thesis is picked up here on the next compile. Both use `thebibliography`
inside the .tex file, so no biber step is needed:

    pdflatex report.tex
    pdflatex report.tex        (second pass for cross-references)

Keep these in sync with the thesis text: they were condensed from it.

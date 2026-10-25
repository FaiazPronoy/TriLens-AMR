# thesis/

Final thesis LaTeX, exactly as submitted in August 2026, plus the submitted PDF.

    thesis.pdf          the submitted version (126 pages)
    main.tex            entry point
    chapters/           chapter_1.tex ... chapter_6.tex
    core/               title page, declaration, approval, abstract, acknowledgement, ethics
    bibliography/       references.bib (biblatex + biber)
    images/             every figure the thesis includes: PDFs drawn by code/scripts/figures,
                        TikZ diagrams as .tex
    *.png, *.jpg        signature images used by core/declaration.tex and core/approval.tex

Compile (TeX Live or MiKTeX with biber):

    latexmk main.tex

`.latexmkrc` runs biber and the nomenclature step. Only files the thesis actually uses are
here; working notes and older chapter copies were left out (see ../docs/file_map.md).

Figures are redrawn from code with `python scripts/figures/make_all_figures.py` (run inside
`code/`), which writes straight into `images/`. See `../code/docs/FIGURES.md` first: two
figures must not be redrawn yet.

# thesis/

The undergraduate thesis this work grew out of, submitted at BRAC University in August 2026. It is
kept here because the paper in `paper/` reuses its figures and much of its method description.

The thesis reports the earlier numbers. Where the two differ, the paper is the corrected version:
the presence/absence matrix was rebuilt, the lineage holdout was added, and the pre-registered
experiments W6 to W14 came later.

## Layout

| Path | What it holds |
|---|---|
| `main.tex` | the entry point |
| `chapters/` | `chapter_1.tex` to `chapter_6.tex` |
| `core/` | title page, declaration, approval, abstract, acknowledgement and ethics statement |
| `bibliography/references.bib` | the bibliography, for biblatex and biber |
| `images/` | figures as PDF, and the diagrams as TikZ `.tex` files |

## Building it

```bash
cd thesis
latexmk main.tex
```

`.latexmkrc` adds the nomenclature step and runs biber. The build needs a full TeX Live, including
`biblatex`, `biber`, `ulem`, `nomencl` and TikZ.

The scanned signature images, used by `core/declaration.tex` and `core/approval.tex`, are not in
this repository. The thesis still builds without them: LaTeX reports each missing file and leaves
the space blank.

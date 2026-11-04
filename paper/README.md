# paper/

The manuscript, written for NAR Genomics and Bioinformatics.

| Path | What it holds |
|---|---|
| `main.tex` | the manuscript |
| `sections/` | abstract, introduction, methods, results, discussion, back matter and tables |
| `supplement.tex` | the supplementary methods and tables, compiles on its own |
| `refs.bib` | the bibliography |

## Building it

```bash
cd paper
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

The plain `article` class is used so that both files compile anywhere. For submission the sections
are pasted into the journal's own template.

Figures are placeholder boxes. Each box names the figure and the result file that feeds it, so
every figure can be drawn from `results/` without guessing which numbers belong to it. Anything
still open is marked in red by the `\todo` command in `main.tex`.

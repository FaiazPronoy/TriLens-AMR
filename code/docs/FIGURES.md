# Figures

Every figure in the thesis, the script that draws it, and what it reads. All figure
scripts write straight into `../../thesis/images/` (the thesis LaTeX folder), so nothing has
to be copied by hand.

Run everything from inside `code/`.

---

## The one command

```bash
python scripts/figures/make_all_figures.py
```

This redraws 20 of the 21 data figures in one palette. It calls `make_thesis_figures.py`
for the 17 figures whose numbers are locked in a thesis table, then draws the ones that
need real arrays. Use `--skip-genespace` on a machine with less than about 8 GB of free RAM
(that figure loads the 3.7 GB SNP matrix).

It deliberately leaves two figures alone. See "Do not redraw yet" below.

---

## Data figures (PDF, in `thesis/images/`)

| File | Drawn by | Reads |
|---|---|---|
| `fig_label_overview` | `make_thesis_figures.py` | values from `tab:labels` |
| `fig_population_structure` | `make_thesis_figures.py` | values from `tab:phylogroups` |
| `fig_split_composition` | `make_thesis_figures.py` | values from `tab:splits` |
| `fig_split_prevalence_shift` | `make_thesis_figures.py` | values from the split diagnostics and `tab:previnv` |
| `fig_gate_weights` | `make_thesis_figures.py` | values from `tab:gate` |
| `fig_ci_forest` | `make_thesis_figures.py` | values from `tab:ci` |
| `fig_clinical_heatmap` | `make_thesis_figures.py` | values from `tab:clinical` and `tab:holdoutclin` |
| `fig_giessen_overview` | `make_thesis_figures.py` | values from `tab:giessen` |
| `fig_giessen_retention` | `make_thesis_figures.py` | values from `tab:giessen_transfer` |
| `fig_lineage_marker` | `make_thesis_figures.py` | values from `tab:umark` |
| `fig_coresistance` | `make_thesis_figures.py` | phi coefficients |
| `fig_tabnet_vs_trilens` | `make_thesis_figures.py` | values from `tab:baselines` |
| `fig_baseline_ladder` | `make_thesis_figures.py` | values from `tab:baselines` |
| `fig_progression` | `make_thesis_figures.py` | values from `tab:progression` |
| `fig_layer_ablation` | `make_thesis_figures.py` | values from `tab:ablation` |
| `fig_requirements_status` | `make_thesis_figures.py` | Chapter 3 requirement outcomes |
| `fig_mechanism_gap` | `make_thesis_figures.py` | per-drug holdout drop |
| `fig_st_distribution` | `make_all_figures.py` | `data/processed/mlst_lineage_holdout_split.pkl` |
| `fig_gene_feature_space` | `make_all_figures.py` | `data/processed/arcadia_v2_snp_matrix.pkl`, `arcadia_v2_gene_filtered.pkl` |
| `fig_pr_curves` | `make_results_figures.py` (last drawn); `make_all_figures.py --rebuild-prcurves` | `results/trilens_attnfix/` and the post-fix holdout predictions |
| `fig_attention_concentration` | `make_results_figures.py` | standard Lens B (`results/standard/trilens_attnfix/` from the re-run, else `results/trilens_v10/`) vs `results/trilens_attnfix/attention_B_topgenes.csv` |

`make_all_figures.py` also draws `fig_threshold_sweep`, which the thesis does not include.

**When a table changes, change the matching constant in `make_thesis_figures.py` too.**
Those 17 figures read the numbers from constants at the top of that file, not from the
results folders, so they cannot drift from the typeset tables by accident, but they also do
not update by themselves.

---

## Do not redraw yet

| File | Why |
|---|---|
| `fig_pr_curves` | The holdout curve needs the post-fix holdout predictions (`KNOWN_ISSUES.md`, issue 1). Redrawing now would plot the pre-fix 0.6300 model. The `analysis` notebook of the Kaggle re-run redraws it. |
| `fig_attention_concentration` | Needs the standard-model run; the `analysis` notebook redraws it from `results/standard/`. |

`make_results_figures.py` is the older script. It also draws `fig_gate_weights`,
`fig_tabnet_vs_trilens` and `fig_threshold_sweep` in the old palette, so running it overwrites
the current versions of those files. Run `make_all_figures.py` after it to restore them.

---

## Diagrams (TikZ, drawn by LaTeX at compile time)

No script is needed. They live in `thesis/images/` as `.tex` files:

`pipeline`, `pnn_overall`, `pnn_expertA`, `pnn_expertB`, `pnn_expertC`, `pnn_moegate`,
`pnn_stacking`, `xgboost_flow`, `lightgbm`, `logistic-regression`, `random-forest`,
`fig_lineage_drop` (values from `tab:previnv`), `fig_evalsplit_landscape`.

---

## The IEEE paper

`paper/journal/report.tex` and `paper/conference/paper.tex` reuse the thesis figures from
`thesis/images/` through `\graphicspath`. Redrawing a thesis figure updates both papers on
their next compile.

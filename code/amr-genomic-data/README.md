# amr-genomic-data

Data pipeline for an undergraduate thesis on ML prediction of antimicrobial
resistance (AMR) in *E. coli* from genomic data (Arcadia v2 SNP/pan-genome
data + Giessen external validation set). Refactored out of five Kaggle
notebooks into a tested, config-driven package. See `docs/AUDIT_FINDINGS.md`
for exactly what was wrong with the notebook version and what changed.

## Quick start

```bash
conda env create -f environment.yml
conda activate amr-genomic-data
pytest tests/ -v
```

Everything reads paths from `config/paths.yaml` (auto-detects Kaggle / Colab /
local) and locked parameters from `config/params.yaml`. Nothing in `src/` or
`scripts/` contains a hardcoded path.

## What is and is not here

This package builds **from the raw Arcadia/Giessen source files forward**. If
you already have the processed `.pkl` files (from the original Kaggle
notebooks or from the existing thesis's `data/processed/`), you can skip
straight to using `src/amr_data/qa.py` to verify them and `src/amr_data/splits.py`
to build the new `lineage_holdout_split`.

If you are starting from raw source data, run in this order (each script is
also a thin CLI, see `scripts/`):

1. `scripts/build_snp_matrix.py` — VCF → SNP matrix (Phase A logic)
2. `scripts/make_splits.py` — labels, phylogroups, all three split regimes
3. `scripts/build_gene_pa.py` — PLINK → gene presence/absence matrix
4. Stage 1.5 gene filtering — **not reimplemented as a script here**; the
   existing, already-correct `src/experiments/run_xgb_gene_importance.py` +
   `build_gene_filtered_dataset.py` in the main thesis repo do this, and
   `src/amr_data/gene_filter.py` in this package is the from-scratch,
   per-split-regime-aware version for use going forward (see B3 in
   `docs/AUDIT_FINDINGS.md` for why "per split regime" matters).
5. `scripts/build_views.py` — assemble the three-view (SNP / gene-agg / P-A)
   Pillar-2-ready bundle.
6. `scripts/port_cluster_labels.py` — **done.** `amr_metax_dataset.pkl` was
   supplied and ported at a 100.0% match rate; output at
   `data/processed/ported_cluster_labels_v2.pkl`. These 30 real, disjoint
   clusters are now the recommended grouping variable for
   `lineage_holdout_split` (see the split table below and
   `docs/CHANGELOG_NUMBERS.md`).

## The three splits

| Name | What it actually does | Group-exclusive? |
|---|---|---|
| `random_split` | Stratified on the AMP label | No (never claimed to be) |
| `phylogroup_stratified_split` | Stratified on the 8 broad phylogroup labels — **preserves** each phylogroup's proportion on both sides | **No** — this is the pipeline's former "phylogeny split", renamed honestly. It does not hold any lineage out. See `docs/AUDIT_FINDINGS.md` B1. |
| `lineage_holdout_split` | `GroupShuffleSplit` on a chosen grouping variable — genuinely disjoint groups across train/val/test, asserted at runtime | **Yes** |

If your thesis text says "lineages were held out", `lineage_holdout_split` is
the split that claim is actually about. `phylogroup_stratified_split` is not.

## Repository layout

See `docs/PIPELINE.md` for a stage-by-stage diagram. Short version:

```
config/        paths.yaml + params.yaml -- the only place paths/params live
src/amr_data/  all logic, importable, unit-tested
notebooks/     thin -- import from src/amr_data, no business logic in cells
scripts/       thin CLI wrappers around src/amr_data functions
tests/         pytest -- includes the mandatory group-leak and orientation checks
results/       small CSV/MD outputs, committed to git
data/          gitignored -- raw/interim/processed pkl and npz artifacts
archive/       superseded code, kept for provenance only
docs/          provenance, changelog, open questions, pipeline diagram
```

## Data provenance (short version — full table in docs/DATA_PROVENANCE.md)

This pipeline draws from **two separate Zenodo records**, not one:

- Zenodo `12692732` (`10.57844/arcadia-d2cf-ebe5`) — VCF variant calls,
  phenotype matrix, strain metadata.
- Zenodo `14364732` (`10.57844/arcadia-8391-465f`) — the
  `presence_absence_data/` PLINK files (32,441 pan-genome genes). This is
  the record the *original* pipeline's citation missed for roughly a third
  of the features actually used (Lens B / gene presence-absence).
- Giessen: Ren et al. 2022, `10.1093/bioinformatics/btab681` — 809 isolates,
  60,936 SNP features, 4 antibiotics.

## Rules this refactor followed

1. **Never silently change a number.** Every value that differs from the
   original notebooks is in `docs/CHANGELOG_NUMBERS.md` with old vs. new and
   why.
2. **Never invent a value.** Anything that could not be determined from
   available code or data is `# TODO(faiaz): unresolved — see
   docs/OPEN_QUESTIONS.md`, not a plausible-looking guess.

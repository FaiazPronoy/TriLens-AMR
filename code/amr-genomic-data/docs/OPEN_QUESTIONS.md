# Open Questions

Every item here is something this refactor could not determine or complete,
per Rule 2 ("never invent a value"). Nothing below has been guessed or
filled in with a plausible-looking default.

## 1. ~~`amr_metax_dataset.pkl` is not present~~ RESOLVED

Supplied (2.85 GB, at the parent thesis repository's root;
`config/paths.yaml`'s `legacy_files.amr_metax_dataset_pkl` points directly at
it rather than duplicating a multi-GB file, same pattern as
`processed_dir`/`giessen_files`). `scripts/port_cluster_labels.py` was run
for real:

```
Matched 6962/6962 (100.0%) v2 samples to a legacy cluster label by genome_names.
Saved: .../data/processed/ported_cluster_labels_v2.pkl
```

**100% match rate** -- every genome in the current pipeline's sample order
was found by name in the legacy dataset. `lineage_holdout_split()` (already
written, no code changes needed -- it accepts any grouping array) applied to
these real 30 clusters gives sizes **4912/830/1220** (20/5/5 clusters per
split), confirmed group-disjoint. See `docs/CHANGELOG_NUMBERS.md` for the
full comparison against the 8-phylogroup proxy grouping used before this
file arrived -- the real clusters show a **much larger** train/test
prevalence shift (up to 41.2 percentage points on CIP) than either the old
stratified split or the 8-phylogroup approximation, which is the strongest
evidence yet in this refactor for why a genuine lineage holdout matters.
Locked in as `tests/test_splits.py::test_lineage_holdout_split_on_real_30_cluster_grouping`
and three tests in `tests/test_port_cluster_labels.py`.

## 2. Derivation method for the 30 clusters -- STILL OPEN (circumstantial evidence only)

Even with the file in hand, there is **no explicit field** in
`amr_metax_dataset.pkl` naming a clustering algorithm. Inspected every
non-array key: `preprocess_meta` (`{'filters_applied_on': 'train_only',
'train_split_source': 'random_split', 'split_built_before_filters': True}`)
describes feature filtering, not clustering. `aggregate_info` describes 5
variant-count aggregate features, also not clustering. `reduction_method`
(`'TruncatedSVD'`) and `reduction_model`
(`TruncatedSVD(n_components=50, random_state=42)`) describe a 50-dimensional
PCA-style reduction (`X_pca`, shape `(6962, 50)`) that sits in the *same*
dict as `cluster_labels` (30 clusters) -- **circumstantial evidence** that
clustering was performed on this 50-dim reduced space (SVD then some
clustering algorithm, with `n_clusters=30` looking like a pre-specified
parameter, consistent with e.g. `KMeans(n_clusters=30)`), but this is an
inference from file structure, not a documented fact, and is reported here
as exactly that -- not as a resolved answer. **Action needed:** find the
notebook/script that produced `cluster_labels` (a sibling of whatever
produced this `TruncatedSVD` reduction), or ask whoever ran that Kaggle
session, before treating "genuine phylogenetic lineages" as a confirmed
description of what these 30 groups represent.

## 3. ~~Raw phenotype token audit could not be completed~~ RESOLVED

`phenotype_matrix_08302024.csv` was supplied (copied into
`data/raw/phenotype_matrix/` to match `config/paths.yaml`'s expected
layout). Ran `labels.py::audit_raw_phenotype_values()` against the real
file: **the only tokens that occur, across all 7 antibiotic columns, are
`{"resistant", "susceptible", "intermediate", NaN}`** -- none of cell 3's
extra abbreviated tokens (`"res"`, `"sus"`, `"int"`) ever appear. Further
cross-checked by building `Y`/`label_mask` from this raw file with this
refactor's encoder and diffing against the locked
`arcadia_v2_master_features.pkl` (cell 5's encoder), aligned by
`Genome.Name`: **zero mismatches across all 6,962 genomes and 35,516
comparable label cells.** See `docs/CHANGELOG_NUMBERS.md` for the full
writeup and `tests/test_labels.py` for the three tests that lock this in
permanently. `arcadia_v2_full_dataset.pkl` (cell 3's own output) is still
not available, but the question this item was tracking -- "did the two
encoders ever actually disagree on real data" -- now has a real-data-backed
answer: no.

## 4. Gene-selection leakage into `phylogroup_stratified_split` (new finding, B3)

Not one of the pre-identified bugs in the refactor spec, but discovered
while auditing B3: Stage 1.5's 5,000-gene selection was fit on
`random_split`'s `train_idx`, and the resulting single gene set is reused
for `phylogroup_stratified_split` evaluation. **Verified directly**: 73.1%
(764/1045) of `phylogroup_stratified_split`'s test genomes were part of
`random_split`'s training set whose labels picked those genes. This is not
something to "resolve" as an open question so much as something to
**re-run**: `gene_filter.select_genes_for_split()` in this refactor is
already written to take `train_idx` per split regime, so producing a
corrected, leakage-free `phylogroup_stratified_split`-specific (and, once
(1)/(2) are resolved, `lineage_holdout_split`-specific) gene selection is a
compute task (~tens of minutes on CPU per split regime per the timing note
in `gene_filter.py`), not a code task. **Action needed:** run
`select_genes_for_split()` once per split regime and regenerate every
downstream T7/phylogeny-split result with the corresponding regime-specific
gene set before reporting a "corrected" T7 number.

## 5. TriLens v10 training code

`arcadia-v2-phase-c.ipynb` ends at a pre-flight data-integrity check ("Ready
for TriLens v10"); the architecture and training code itself lives
elsewhere (the existing thesis's own `src/experiments/run_trilens_v10.py`,
outside the five notebooks this refactor was scoped to). Not touched by
this refactor.

## 6. The notebook producing thesis Table 5.1 (eight preliminary models,
80,984-feature space)

Not among the five notebooks supplied. If it exists, it likely shares the
`amr_metax_dataset.pkl` lineage (item 1) -- worth checking once that file is
located.

## 7. The Giessen CascadeStack v9.1 script with `n_test = 81`

`giessen-cross-validation.ipynb` produces a 566/121/**122** split (verified
directly, see manifest E8). The thesis's own Table 5.10 (per the refactor
spec) reports Giessen numbers with `n_test = 81`, from a *different* script
in a different repository:

| Source | CIP | CTX | CAZ/CTZ | GEN |
|---|---|---|---|---|
| `giessen-cross-validation.ipynb` (fusion strategies grid, per-antibiotic best) | 0.964 | 0.850 | 0.769 | 0.630 |
| thesis Table 5.10 (`n_test=81`, source script not in this repo) | 0.9868 | 0.7876 | 0.7775 | 0.5480 |

Per the spec's own instruction, this refactor does **not** attempt to
reconcile these -- they are evaluated on different test partitions of
different sizes and were very likely produced by genuinely different code
paths (the `n_test=81` numbers match the split sizes used by
`src/experiments/run_pillar4_giessen_transfer.py` in the existing, separate
thesis codebase -- a `StratifiedShuffleSplit(test_size=0.10)`, not this
notebook's 70/15/15 -- confirmed by inspecting that script directly).
**Action needed:** decide which of the two Giessen evaluations (a from-
scratch fusion-strategy comparison trained only on Giessen, vs. a trained-
on-Arcadia model transferred to Giessen) is the one the thesis actually
means to report, and cite the correct source script for it. These are not
interchangeable numbers.

## 8. Citation/derivation for the 57-entry ST->phylogroup lookup table

Used both for the phylogroup baseline and to derive `phylogroup_final`
wherever `Other.Typing` lacks an explicit tag. No citation, paper reference,
or generation script was found anywhere in the five notebooks. Diffed the
two copies (Phase B cells 3 and 5) programmatically -- they are identical,
so there is nothing to reconcile *between* them, but the table's own
provenance (ClermonTyping? EnteroBase? hand-curated from a specific paper?)
remains unknown. **Action needed:** ask whoever originally built this
table, or reverse-engineer it against a published Achtman-ST/Clermont-
phylogroup reference table (e.g. EnteroBase's own ST->phylogroup calls) to
confirm/attribute it.

## 9. Two Zenodo DOIs not independently confirmed

See `docs/DATA_PROVENANCE.md` -- both DOIs are taken as given from the
refactor specification; this refactor did not verify them against the live
Zenodo records (no network access used).

## 10. AMR-EnsembleNet reproduction not executed in this environment

`src/amr_data/amr_ensemblenet.py` is a faithful code transcription (see its
own module docstring) but this environment has no `tensorflow` installed,
so it was not run, and no number attributed to it should be assumed to come
from this file's execution. See `docs/CHANGELOG_NUMBERS.md`.

## 11. Full 5-strategy Giessen fusion-comparison training code not fully ported

`giessen.py::honest_headline_strategy()` and `full_strategy_grid()`
implement the **statistical fix** for the selection-bias bug (Part C.4) as
generic functions over an already-computed strategy x antibiotic x split
results table -- this is fully implemented and tested (synthetic data
reproducing the exact bias pattern described in the spec). What is **not**
ported is the training code for the five fusion strategies themselves
(Early/Joint/Decision/Hybrid/Proximity-Adaptive -- each its own PCA/
autoencoder/classifier pipeline in the original notebook). Porting five
distinct, untested-in-this-environment training pipelines line-by-line
without being able to run and verify them risked violating Rule 2 more than
it helped; the statistical fix is the part of Part C.4 that is a genuine
bug (biased comparison), and that part is done and tested. **Action
needed:** if the full per-strategy training code is wanted in this
refactored form too, port it into a new `fusion_strategies.py` following the
same pattern as `amr_ensemblenet.py` (local imports for heavy deps,
docstring stating what was/wasn't executed here).

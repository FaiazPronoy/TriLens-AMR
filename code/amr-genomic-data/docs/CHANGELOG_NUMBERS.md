# Changelog: Numbers (Rule 1)

Every value that could differ from the original notebooks, with old vs. new
and why. Anything not listed here that looks different is a bug -- report it.

## Values confirmed UNCHANGED (verified directly against real local data)

| Value | Old (notebook) | New (this refactor) | How verified |
|---|---|---|---|
| `random_split` train/val/test indices | locked in `arcadia_v2_master_features.pkl` | `splits.random_split()` output | **Byte-identical**, `tests/test_splits.py::test_random_split_matches_locked_reference`, run against the real pkl |
| B7 nan-cast fix effect on stratification array | `np.where(np.isnan(...), -1, Y.astype(int))` (warns) | `np.where(np.isnan(...), -1, np.nan_to_num(...)).astype(int)` (no warning) | Directly diffed both arrays element-wise on real `Y`/`label_mask` from `arcadia_v2_master_features.pkl`: **identical everywhere**. The warning is cosmetic; behaviour is unchanged. |
| Gene grouping stats (Stage 1.5 input) | `n_genes=18975, total_snps=535439, median=12.0, max=717, n>=5 SNPs=13799` (docs manifest E5, from notebook log) | `gene_filter.build_gene_to_snp_ranges()` + `get_gene_size_stats()` output | **Exact match**, computed fresh from the real `arcadia_v2_snp_matrix.pkl`'s `snp_positions` array, `tests/test_gene_filter.py::test_build_gene_to_snp_ranges_reproduces_real_stage1_5_input_stats` |
| Gene-aggregated feature shape | 5000 genes x 4 stats = 20,000 columns (observed when the existing pipeline's own code was run in-session) | `views.compute_gene_aggregated_features()` output shape | Matches on a 50-gene subset of the real `arcadia_v2_gene_filtered.pkl`, `tests/test_views.py` |
| PCA-1024 variance retained | `0.9465089` (locked in `arcadia_v2_master_features.pkl`, from notebook cell 5) | `0.94650644` -- `baselines.build_pca_features()` (sparse TruncatedSVD, B8 fix), full scale, real `random_split` train partition | **Confirmed, diff = 0.0000025** (well inside the ±0.002 manifest tolerance). Full-scale run: sparse build 154.9s + TruncatedSVD(1024) fit 2796.6s (~46.6 min, CPU) = ~49 min total. See entry below. |
| Giessen dataset shape/labels | `(809, 60936)`, CIP 366/443, CTX 358/451, CTZ 276/533, GEN 188/621 | `giessen.load_giessen_raw()` output | **Exact match**, loaded fresh from the real CSVs, `tests/test_giessen.py::test_load_giessen_raw_matches_manifest_e8` |
| Giessen split sizes | 566/121/122, stratified on CIP | `giessen.build_giessen_split()` output | **Exact match**, `tests/test_giessen.py::test_giessen_split_sizes_match_notebook` |
| Gene P/A shape/stats | `(6962, 32441)` uint8, mean 3708.6/median 3331.0 genes/strain, 6949/6962 matched, 5000/5000 gene-ID overlap | `gene_pa.py` logic (not re-run end-to-end, no raw PLINK files locally) cross-checked via `qa.verify_gene_pa()` against the real, already-built `arcadia_v2_gene_presence_absence.pkl` | **Exact match**, `tests/test_gene_pa.py::test_real_gene_pa_artifact_passes_orientation_and_overlap_checks` |
| Label matrix `Y`/`label_mask` (B2) | `arcadia_v2_master_features.pkl`'s `Y` (cell 5's stricter encoder) | `labels.build_label_matrix()` applied to the real `phenotype_matrix_08302024.csv`, aligned by `Genome.Name` | **Zero mismatches across 35,516 comparable (non-missing in both) label cells, across all 6,962 genomes.** `tests/test_labels.py::test_real_phenotype_matrix_matches_locked_master_features_y`. See B2 resolution below. |

## Values that ARE new (did not exist before this refactor)

| Value | What it is | Why it's new, not a "changed" number |
|---|---|---|
| `lineage_holdout_split` sizes, 8-phylogroup grouping (2841/3859/262, seed 42) | Genuine group holdout using the only grouping variable available *before* `amr_metax_dataset.pkl` was supplied | This split did not exist before (B1) -- there is no "old" number to compare against. Verified group-disjoint, `tests/test_splits.py`. Superseded as the recommended grouping by the entry below (kept as a fallback/comparison). |
| `lineage_holdout_split` sizes, **real 30-cluster grouping** (4912/830/1220, 20/5/5 clusters, seed 42) | Genuine group holdout using the ported 30 disjoint clusters from `amr_metax_dataset.pkl` (100% match rate -- see OPEN_QUESTIONS #1, resolved) | Also new -- no prior number exists. `tests/test_splits.py::test_lineage_holdout_split_on_real_30_cluster_grouping`. This is now the **recommended** grouping variable. |
| `results/splits/split_diagnostics.csv` | Per-split, per-antibiotic train/test prevalence delta | New diagnostic table (was "missing entirely" per the spec). Real output, max absolute prevalence delta across all 7 antibiotics per split: `random_split` 2.5pp, `phylogroup_stratified_split` 1.5pp, `lineage_holdout_split` (8-phylogroup) 23.2pp, **`lineage_holdout_split` (real 30-cluster) 41.2pp (CIP: 15.7% train -> 56.8% test)**. This is the strongest quantitative evidence in this refactor that the old "phylogeny split" tested nothing resembling genuine generalisation, and that the real ported clusters expose population heterogeneity the coarse 8-phylogroup proxy only partially captured. |
| `results/qa/unmatched_strains.csv` | The 13 genome names with no gene P/A data, by name | Existed as a count (`6949/6962`) before; the actual 13 names were not written down anywhere. New export, not a changed number. |
| `data/processed/ported_cluster_labels_v2.pkl` | 30-cluster labels ported onto the current 6,962-sample order | New artifact, `scripts/port_cluster_labels.py` run for real. Match rate **100.0%** (6962/6962) -- see OPEN_QUESTIONS #1. |

## Full-scale PCA-1024 reproduction (B8) -- CONFIRMED

Run to completion against the real `arcadia_v2_snp_matrix.pkl` and the real
`random_split` train partition from `arcadia_v2_master_features.pkl`:

```
sparse build time:        154.9 s   (6962, 535439) int8 -> (6962, 535439) float32 CSR, 4.99 GB
TruncatedSVD(1024) fit:  2796.6 s   (~46.6 min, CPU, random_state=42)
variance_retained (new):   0.94650644
variance_retained (locked): 0.9465089
absolute difference:       0.0000025
```

This is the concrete resolution of B8: the original `X.astype(np.float32)`
on the full dense matrix `MemoryError`s outright in this environment (see
`docs/AUDIT_FINDINGS.md` B8); the sparse, row-chunked replacement completes,
uses ~3x less peak memory (4.99 GB vs. 14.9 GB), and reproduces the locked
PCA-1024 variance-retained figure to 5 decimal places.

## B2 resolution -- raw phenotype token audit (was OPEN_QUESTIONS #3, now resolved)

`phenotype_matrix_08302024.csv` was supplied after the initial audit. Ran
`labels.audit_raw_phenotype_values()` against the real file (6,983 rows,
7 antibiotic columns, matches the notebook's own printed shape exactly):

```
Every antibiotic column's distinct raw tokens (lowercased/stripped):
    {"resistant", "susceptible", "intermediate", NaN}
```

**None of the abbreviated tokens** ("res", "sus", "int", "r", "s", "i") that
made Phase B cell 3's encoder more permissive than cell 5's ever occur in
the real data. Cross-checked further: built `Y`/`label_mask` from this raw
file with this refactor's single encoder, aligned by `Genome.Name` to the
locked `arcadia_v2_master_features.pkl` (built by cell 5's stricter
encoder) -- **all 6,962 genomes matched by name, zero mismatches across
35,516 comparable non-missing label cells, zero mask disagreements.**

**Conclusion: B2's two-encoder divergence existed in the code but never
altered a single label in practice.** `arcadia_v2_full_dataset.pkl` (cell
3's output, still not available locally) is, on this evidence, extremely
likely to have an identical `Y` to `arcadia_v2_master_features.pkl` -- the
question is no longer open, just not literally re-derived from that one
specific missing file.

## B1 upgrade -- real 30-cluster grouping ported (was OPEN_QUESTIONS #1, now resolved)

`amr_metax_dataset.pkl` (2.85 GB) was supplied after the initial audit.
Inspected directly before running anything:

```
n_clusters: 30, cluster_labels shape (6962,), 30 unique values
phylogeny_split.{train,val,test}_clusters: 21/4/5 clusters, pairwise disjoint, union = all 30
genome_names format: "Escherichia coli strain ..." -- same convention as the current pipeline
```

Pre-check: 6962/6962 (100%) of the current pipeline's genome names were
found in this legacy file's genome names, before even running the port
script. Then ran `scripts/port_cluster_labels.py` for real:

```
Matched 6962/6962 (100.0%) v2 samples to a legacy cluster label by genome_names.
Saved: data/processed/ported_cluster_labels_v2.pkl
```

Built `lineage_holdout_split()` (unchanged code -- it already accepted any
grouping array) on these real clusters:

```
sizes: 4912 / 830 / 1220   (20 / 5 / 5 clusters per split, pairwise disjoint)
```

**This is now the recommended grouping variable for `lineage_holdout_split`**,
superseding the 8-phylogroup proxy used before this file arrived (still
available via `groups_from_phylogroup()` for comparison). See the
`split_diagnostics.csv` entry above for the resulting prevalence-shift
numbers, which are larger (more informative) than the 8-phylogroup version.

**Not resolved by this file:** the clustering algorithm/derivation method
itself (OPEN_QUESTIONS #2 remains open -- circumstantial evidence only,
found while inspecting this file: a 50-dim `TruncatedSVD` reduction sits in
the same dict as `cluster_labels`, suggesting but not confirming clustering
was done on that reduced space).

## Values NOT reproduced (compute tasks, not refactor tasks -- see docs/OPEN_QUESTIONS.md #4, #10, #11)

- Stage 1.5 gene selection re-run per split regime (would produce a
  corrected `phylogroup_stratified_split`/`lineage_holdout_split` T7-style
  number, currently contaminated by gene-selection leakage -- see
  OPEN_QUESTIONS #4). Not run: tens of minutes per regime.
- AMR-EnsembleNet CNN reproduction (no TensorFlow in this environment).
- The five Giessen fusion strategies' actual training.

None of these produce a number that appears anywhere in this refactor's
code or docs as if it had been verified -- they are absent, not guessed.

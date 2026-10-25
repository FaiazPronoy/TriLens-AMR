# Pipeline, Stage by Stage

```
                         ┌─────────────────────────────┐
                         │  Zenodo 12692732              │
                         │  VCF + phenotype + metadata    │
                         └──────────────┬──────────────┘
                                        │  vcf_to_snp.py
                                        ▼
                         arcadia_v2_snp_matrix.pkl
                         (6962, 535439) int8, Arcadia convention
                         (missing genotype -> ref allele 0)
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │ labels.py               │ phylogroup.py            │ splits.py
              │ encode_phenotype()      │ derive_phylogroup()      │ random_split()
              │ (ONE encoder, B2)       │ ST_TO_PHYLOGROUP (B10)    │ phylogroup_stratified_split() (B1, honest rename)
              │                         │                           │ lineage_holdout_split() (B1, NEW)
              └─────────────────────────┴─────────────────────────┘
                                        │
                                        ▼
                         arcadia_v2_master_features.pkl
                         Y, label_mask, cluster_labels, phylogroup_onehot,
                         random_split, phylogroup_stratified_split
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                           │
              ▼                         ▼                           ▼
   baselines.py (PCA-1024,   gene_filter.py (Stage 1.5,   gene_pa.py (PLINK -> P/A,
   sparse, B8 fix)           per-split-regime, B3 fix)     S2 encoding + chrom IDs,
                                        │                   B4/B5/B6 fixes)
                                        │                           │
                                        │        Zenodo 14364732 ───┘
                                        │        presence_absence_data/
                                        ▼
                             views.py: assemble_pillar2_bundle()
                             Lens A (per-gene SNP blocks)
                             Lens B (gene P/A)
                             Lens C (gene-aggregated tabular)
                                        │
                                        ▼
                          Pillar-2-ready bundle, PER SPLIT REGIME
                          (random / phylogroup_stratified / lineage_holdout)
```

```
                         ┌─────────────────────────────┐
                         │  Ren et al. 2022               │
                         │  Giessen: 809 isolates,         │
                         │  60,936 SNP features            │
                         └──────────────┬──────────────┘
                                        │  giessen.py::load_giessen_raw()
                                        │  (CTZ -> CAZ renamed here)
                                        ▼
                         giessen.py::build_giessen_split()
                         (566/121/122, stratified on CIP)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                                       ▼
        amr_ensemblenet.py                    (5 fusion strategies --
        (authors' code, verbatim,               training code NOT ported,
        NOT executed here --                    see OPEN_QUESTIONS #11;
        no tensorflow in this env)               statistical fix IS ported:
                    │                            giessen.py::honest_headline_
                    └──────────────┬──────────────strategy())
                                   ▼
                         metrics.py::bootstrap_ci() etc.
                         (1000 resamples, seed 42, percentile method)
```

## Split regimes: what each one actually is

| | `random_split` | `phylogroup_stratified_split` | `lineage_holdout_split` |
|---|---|---|---|
| Stratifies on | AMP label | 8 broad phylogroups | n/a -- groups whole |
| Holds out whole groups? | No | **No** (this is the fix -- see B1) | **Yes**, asserted at runtime |
| Sizes | 4872/1045/1045 (fixed) | 4872/1045/1045 (fixed -- proof it isn't a holdout) | uneven, depends on group sizes |
| Grouping variable available today | n/a | `cluster_labels` (8 phylogroups, in `arcadia_v2_master_features.pkl`) via `groups_from_phylogroup()` | **Recommended:** `data/processed/ported_cluster_labels_v2.pkl` (30 real disjoint clusters, ported from `amr_metax_dataset.pkl` at 100% match rate -- see OPEN_QUESTIONS #1). `groups_from_phylogroup()` (8 groups) remains available as a coarser fallback/comparison. `groups_from_mlst()` needs raw per-strain MLST strings not currently retained in any local artifact. |

## What is a code task vs. a compute task in this refactor

Everything in `src/amr_data/` is written, unit-tested, and (where real local
data exists) cross-checked against it. Three things are explicitly **not**
re-run end-to-end at full scale as part of this refactor, because they are
GPU/CPU-hours-scale compute jobs, not refactoring:

1. Re-running Stage 1.5 gene selection once per split regime (fixes the
   leakage in OPEN_QUESTIONS #4) -- tens of minutes on CPU per regime.
2. AMR-EnsembleNet's CNN half (needs TensorFlow, not installed here).
3. The five Giessen fusion strategies' actual training (only the
   statistical fix for comparing them honestly was ported, see
   OPEN_QUESTIONS #11).

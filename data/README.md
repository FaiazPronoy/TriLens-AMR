# data/

This folder is not in git. Nothing here is redistributed: every file comes from a public release
and is rebuilt by the pipeline or downloaded from its source.

## Where the data come from

| Source | What it gives | Where to get it |
|---|---|---|
| Arcadia Science E. coli AMR release | the SNP VCF, the phenotype matrix, strain metadata, the pan-genome and the PLINK presence/absence files | the public Zenodo record of the release; `step04 --part download` fetches the presence/absence part on its own |
| Giessen collection (Ren et al., 2022) | a second E. coli collection used for the transfer test | the authors' public release |
| NCBI Pathogen Detection | the external cohort, its determinant calls and its MIC values | downloaded by `step25`, `step26`, `step28` and `step29` |

## Layout

```
data/
  raw/
    phenotype_matrix_08302024.csv      resistance phenotypes as published
    SRA_to_genome_name.csv             SRA accession to genome name
    strains_metadata.csv               country, year, study, sequence type, assembly statistics
    pangenome/                         pangenome_cds.fa, gene_presence_absence.csv and the rest
  processed/                           everything the pipeline builds
  Giessen_dataset/                     cip_ctx_ctz_gen_multi_data.csv, cip_ctx_ctz_gen_pheno.csv
```

## Files the pipeline builds

| File | Built by | Holds |
|---|---|---|
| `arcadia_v2_snp_matrix.pkl` | `step01` | the SNP matrix after the frequency and missingness filters |
| `arcadia_v2_master_features.pkl` | `step02` | labels, the label mask, PCA features, phylogroups and the random and phylogroup splits |
| `mlst_lineage_holdout_split.pkl` | `step03` | the lineage holdout split, no sequence type shared between train and test |
| `arcadia_v2_gene_presence_absence.pkl` | `step04` | the presence/absence matrix in the correct orientation |
| `arcadia_v2_gene_filtered.pkl` | `step05` | the SNPs of the selected genes, random split |
| `arcadia_v2_gene_filtered__phylogeny_clean.pkl` | `step05` | the same, ranked inside the phylogroup split |
| `arcadia_v2_gene_filtered__mlst_holdout.pkl` | `step05` | the same, ranked inside the lineage holdout |
| `country_*_split.pkl`, `time_norway_2013_split.pkl` | `step08` | the country and time shift splits |
| `cds_eggNog.tsv`, `genes.gff`, `ported_cluster_labels_v2.pkl` | from the release | gene annotation, gene coordinates and cluster labels |

Gene selection is label-aware, so it is run once inside each split. That is why there are three
gene-filtered files and not one.

## Running without the raw data

Most analysis steps read only `results/`, which is in git. The steps that train a model need
`data/processed/`. On Kaggle the processed files come from the `trilens-data` dataset, so nothing
is rebuilt there; see `kaggle/notebooks/`.

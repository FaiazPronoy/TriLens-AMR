# Data and large files

Large files are not in git. This file says what each one is, where it lives, and how to get
it.

> **TODO before publishing:** upload the files in sections 2 and 3 to a shared Drive (or
> Zenodo) folder, replace every `PASTE_LINK_HERE`, and set sharing to "anyone with the link".

---

## 1. Layout inside `code/`

```
code/
├── data/
│   ├── processed/          the inputs every experiment reads (section 2)
│   ├── Giessen_dataset/    external collection (section 4)
│   └── raw/                small raw files kept for the next experiments (section 4)
└── results/                .csv/.md/.json/.yaml in git; .npz from section 3
```

On the authors' machine `setup_data.ps1` (in the project folder above `code/`) fills all of
this from the old working folders. On Kaggle the same files come from private datasets; see
`../../kaggle/README.md`.

---

## 2. Processed datasets → `data/processed/`

| File | Size | What it is | Link |
|---|---|---|---|
| `arcadia_v2_snp_matrix.pkl` | 3.9 GB | 6,962 × 535,439 SNP matrix (int8, 0 = ref), pre-filter | `PASTE_LINK_HERE` |
| `arcadia_v2_master_features.pkl` | 31 MB | labels, label mask, phylogroups, PCA-1024 features, random and phylogroup splits | `PASTE_LINK_HERE` |
| `arcadia_v2_gene_presence_absence.pkl` | 229 MB | 6,962 × 32,441 binary pan-genome profile (18,494 CDS + 13,947 intergenic clusters) | `PASTE_LINK_HERE` |
| `arcadia_v2_gene_filtered.pkl` | 1.9 GB | after the Stage 1.5 filter (5,000 genes / 262,225 SNPs), random split | `PASTE_LINK_HERE` |
| `arcadia_v2_gene_filtered__phylogeny_clean.pkl` | 1.9 GB | same filter refitted on the phylogroup split | `PASTE_LINK_HERE` |
| `arcadia_v2_gene_filtered__mlst_holdout.pkl` | 1.7 GB | same filter refitted on the lineage holdout | `PASTE_LINK_HERE` |
| `mlst_lineage_holdout_split.pkl` | 96 KB | the sequence-type partition, with ST per genome | `PASTE_LINK_HERE` |
| `ported_cluster_labels_v2.pkl` | 63 KB | cluster labels used by the split diagnostics | `PASTE_LINK_HERE` |
| `genes.gff`, `cds_eggNog.tsv` | 21 MB, 7 MB | reference gene coordinates and eggNOG annotation | `PASTE_LINK_HERE` |

**Why the gene filter appears three times:** the 5,000-gene selection is refitted inside each
split's own training partition, so test genomes never influence which features exist.

---

## 3. Saved predictions → `results/<run>/`

Only small text outputs are committed. The arrays:

| File | Belongs in | Needed for |
|---|---|---|
| `predictions_saved.npz` | `results/trilens_attnfix/` | PR curves, clinical metrics, any re-scoring |
| `gate_weights_test.npz` | `results/trilens_attnfix/` | gate routing table and figure |
| `attention_full.npz` | `results/trilens_attnfix/` | attention analyses |
| `trilens_kfold_oof.npz` | `results/trilens_attnfix/` | stacking meta-learner inputs |
| `predictions_saved.npz` | `results/trilens_v10/` | the before/after attention comparison |
| `predictions_saved.npz` | `results/pillar_cascadestack_classical_mlst/`, `results/pillar_cascadestack_classical_phylogeny_clean/` | the holdout and phylogroup stacks |
| `predictions_saved.npz` | `results/pillar_mlst_final_comparison/`, `results/pillar_trilens_mlst/` | everything under the lineage holdout (after the re-run) |
| all other `*.npz` | their own run folders | listed in the Drive folder |

Link: `PASTE_LINK_HERE`

---

## 4. Raw source data

We do not host the raw releases. Download them from the original publishers.

| Dataset | What | Where |
|---|---|---|
| Arcadia release 1 | variant calls (VCF), phenotype matrix, strain metadata, pan-genome FASTA | Zenodo, Morin (2024), reference [40] in the thesis |
| Arcadia release 2 | gene presence/absence (PLINK `.bed/.bim/.fam`) | Zenodo, York and Sandler (2025), reference [57] |
| Giessen | `cip_ctx_ctz_gen_multi_data.csv`, `cip_ctx_ctz_gen_pheno.csv` | Ren et al. (2022), reference [28] |

The two Arcadia releases are **separate deposits with separate DOIs**. The SNP views come
from release 1 and the gene presence view from release 2.

A few small raw files are copied into `data/raw/` because the next experiments need them:

| File | Why |
|---|---|
| `raw/pangenome/pangenome_cds.fa` | sequences of the 18,494 CDS columns of the P/A matrix; input for AMRFinderPlus |
| `raw/pangenome/pangenome_igr.fasta`, `gene_presence_absence.csv` | the intergenic clusters and the ECOR72 pan-genome table |
| `raw/strains_metadata.csv` | MLST, typing and source metadata per strain |
| `raw/phenotype_matrix_08302024.csv`, `raw/SRA_to_genome_name.csv` | labels and the SRA ↔ genome-name join |

---

## 5. If a link is dead

Open an issue on the repository, or contact the authors at the addresses in the thesis.

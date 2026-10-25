# Data Provenance

This pipeline draws from **three independent sources**. The original
notebooks' own `preprocess_meta` field (saved inside every processed pkl)
cited a single source string, `"Arcadia v2 VCF (Zenodo 14364732)"`, for
*both* the VCF-derived data and the presence/absence PLINK data -- but these
are two separate Zenodo records, and the citation used throughout the
notebooks does not distinguish them. This table is the corrected version.

| Record | DOI | What this pipeline uses from it | Used by |
|---|---|---|---|
| Zenodo `12692732` | `10.57844/arcadia-d2cf-ebe5` | VCF variant calls (`filtered_output.vcf.gz`), phenotype matrix (`phenotype_matrix_08302024.csv`), strain metadata (`strains_metadata.csv`), SRA-to-genome mapping | `vcf_to_snp.py`, `labels.py`, `phylogroup.py` |
| Zenodo `14364732` | `10.57844/arcadia-8391-465f` | `presence_absence_data/` PLINK triple (`.bed`/`.bim`/`.fam`), 32,441 pan-genome genes | `gene_pa.py` |
| Ren et al. 2022 | `10.1093/bioinformatics/btab681` | Giessen: 809 *E. coli* isolates, 60,936 integer-encoded SNP features, 4 antibiotics (CIP, CTX, CTZ/CAZ, GEN) | `giessen.py` |

## Why this distinction matters

Roughly a third of the features this pipeline's models actually use (Lens B
/ gene presence-absence, 32,441 genes) come from the *second* Zenodo record,
not the one the notebooks' own saved metadata names. A reader trying to
locate the raw presence/absence data from the thesis's existing citation
(pointing only at `14364732` or whichever single ID was written down) would
be looking in the wrong place for roughly a third of the feature space.

## What was and wasn't independently verified

- **VCF -> SNP matrix**: not re-run (no VCF file present in this repository
  in any form -- it lives only in the Kaggle/Zenodo source). Every number in
  `docs/AUDIT_FINDINGS.md`'s VCF section (2,451,184 variants streamed, etc.)
  was re-derived directly from `arcadia_v2_snp_matrix.pkl`'s own saved
  arrays, which *is* present locally, not merely copied from the notebook's
  printed log.
- **Presence/absence PLINK triple**: not present locally in any form (no
  `.bed`/`.bim`/`.fam`), so `gene_pa.py`'s `build_gene_presence_absence()`
  path could not be executed end-to-end here either -- but its output,
  `arcadia_v2_gene_presence_absence.pkl`, *is* present locally and was
  independently re-verified (see `docs/AUDIT_FINDINGS.md` B4/B5/B6).
- **Giessen CSVs**: present locally in full, loaded and verified directly
  by `tests/test_giessen.py` against real data (see manifest E8).

## Citations for the thesis text

```
Arcadia v2 E. coli AMR genotype-phenotype dataset (VCF, phenotype, metadata):
    Zenodo 12692732, https://doi.org/10.57844/arcadia-d2cf-ebe5

Arcadia v2 pan-genome presence/absence data (PLINK):
    Zenodo 14364732, https://doi.org/10.57844/arcadia-8391-465f

Giessen external validation set:
    Ren Y. et al. (2022). "Machine learning-based prediction of
    antimicrobial resistance from whole-genome sequences." Bioinformatics.
    https://doi.org/10.1093/bioinformatics/btab681

AMR-EnsembleNet reproduction (Part C.3):
    Siddiqui & Tarannum (2025), arXiv:2509.23552.
    https://github.com/Saiful185/AMR-EnsembleNet
```

# TODO(faiaz): unresolved
Neither Zenodo DOI above has been independently confirmed against the live
Zenodo record by this refactor (no network access was used to verify) --
they are taken as given from the refactor specification. Before final
submission, cross-check both DOIs resolve to records containing the files
named in the table above.

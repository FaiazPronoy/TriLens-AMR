# Gene Annotation - Top Marker Genes (eggNOG)

Source: `data/processed/cds_eggNog.tsv` (15056 annotated locus tags, 72 distinct Prokka prefixes). `data/processed/genes.gff` was inspected but not used for annotation (no functional description column, only coordinates) -- see script docstring for the full investigation.

**Overall coverage: 836/1143 rows matched (73.1%).**

**Canonical AMR gene name pattern hits: 36/1143 rows (3.1%).**

## Coverage by source

| Source | Total rows | Matched | Coverage % | Canonical AMR hits |
|---|---|---|---|---|
| TriLens_LensA_attention | 140 | 127 | 90.7% | 0 |
| TriLens_LensB_attention | 140 | 84 | 60.0% | 0 |
| TriLens_LensC_attention | 140 | 98 | 70.0% | 0 |
| full_stack_SHAP_gene_based | 560 | 389 | 69.5% | 16 |
| full_stack_SHAP_raw_chi2 | 140 | 116 | 82.9% | 15 |
| shap_xgb_standalone | 15 | 14 | 93.3% | 3 |
| gain_xgb_standalone | 8 | 8 | 100.0% | 2 |

## Canonical AMR gene matches found

| Source | Gene ID | Matched locus tag | Preferred name | Description |
|---|---|---|---|---|
| full_stack_SHAP_gene_based | MFCAOJAD_04226 | MFCAOJAD_04226 | - | Beta-lactamase |
| full_stack_SHAP_gene_based | LMHECDEF_01124 | LMHECDEF_01124 | parC | Topoisomerase IV is essential for chromosome segregation. It relaxes supercoiled |
| full_stack_SHAP_gene_based | LMHECDEF_04343 | LMHECDEF_04343 | gyrA | A type II topoisomerase that negatively supercoils closed circular double-strand |
| full_stack_SHAP_gene_based | ONLNHHCL_01578 | ONLNHHCL_01578 | mdtD | Major facilitator superfamily |
| full_stack_SHAP_gene_based | LMHECDEF_02447 | LMHECDEF_02447 | mdtC | The MdtABC tripartite complex confers resistance against novobiocin and deoxycho |
| full_stack_SHAP_gene_based | LMHECDEF_00510 | LMHECDEF_00510 | emrB | Part of the tripartite efflux system EmrAB-TolC, which confers resistance to ant |
| full_stack_SHAP_gene_based | PKACBNFE_01616 | PKACBNFE_01616 | cusA | AcrB/AcrD/AcrF family |
| full_stack_SHAP_raw_chi2 | LMHECDEF_03770 | LMHECDEF_03770 | mdtO | Could be involved in resistance to puromycin, acriflavine and tetraphenylarsoniu |

## Caveat

A `no match` result means the eggNOG file (72 genome prefixes) did not contain that specific locus tag -- it does NOT mean the gene is unannotatable in general, only that this particular reference genome's annotation wasn't in the supplied eggNOG file for that specific pan-genome cluster representative. No gene name was inferred or guessed for unmatched rows.
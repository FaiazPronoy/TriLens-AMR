# W0 addition: strains with unusually many loci: description and pre-registration

**Status:** description done (CPU, label-free); the handling rule below is pre-registered and
committed **before** these strains are excluded or flagged in any analysis. Script:
`code/scripts/inventory/high_locus_strains.py`. Files: `code/results/data_inventory/high_locus_*`.

## What the rebuilt matrix shows

The rebuilt presence/absence matrix (KNOWN_ISSUES 12, release `outputs-pa_fix`) has 6,949 strains
with a profile; 13 have none (all zeros, `matched_mask` false).

- Loci per strain: median 11,802, robust SD (1.4826 × MAD) 676, range 9,569–18,620.
- **Outlier rule:** loci > median + 4 robust SD = **14,506**. This flags **61 strains** (31 above 5 SD, 17
  above 6 SD; none below median − 4 SD). The rule reads no phenotype, so it cannot leak labels.

| collection (PMID) | strains | mean loci | flagged |
|---|---:|---:|---:|
| Norway (no PMID in the metadata) | 2,961 | 11,723 | **48** |
| 30127495 | 164 | 13,134 | **10** |
| 28877757, 30550564 | 139 | 11,951 | 2 |
| 27965630 | 26 | 12,349 | 1 |
| 28720578, 34485958, 32205351 (UK), USA, others | 3,659 | 11,575–12,194 | 0 |

Where they come from:

- **Norway, one sequencing block:**
  - 27 of the 48 Norway strains are in the SRA runs ERR4034361–ERR4034705, a block of 296 Norway
    strains. That is 9.1% flagged, against 0.8% in the rest of Norway.
  - 37 of the 61 strains were collected in 2007–2009.
- **PMID 30127495** (flagged: India 3, Kenya 2, Mali 2, Gambia 1, Mozambique 1, one unknown): its
  mean is high overall (13,134), so the collection as a whole sits higher.

The assemblies look normal:

| | flagged (61) | other strains |
|---|---:|---:|
| median assembly size (metadata) | 5.107 Mb | 5.088 Mb |
| median annotated CDS | 5,102 | 5,080 |
| median contigs | 119 | 113 |
| flagged with assembly > 5.8 Mb | 0 | — |

The excess is in the **shell accessory genome**
(`high_locus_excess_by_prevalence.csv`; prevalence measured in the other strains):

| loci carried by … of the other strains | CDS loci present, flagged | other |
|---|---:|---:|
| > 95% (core) | 3,912 | 3,900 |
| 50–95% | 1,883 | 1,468 |
| 15–50% | **2,304** | **1,115** |
| 5–15% | 566 | 190 |
| ≤ 5% | 341 | 91 |

The intergenic loci show the same pattern. Flagged strains carry on average 3,637 more loci, about
twice the other strains' shell genes, and the same core.

### Could they be mixed samples?

**Possibly.** Two strains sequenced together would give the union of two accessory genomes and the
same core. That is what we see. The assemblies, however, are of normal size and gene count.

- A reads-level mixture with one dominant strain could explain both, since the minor strain's genes
  are called present from reads but barely assembled.
- So could a P/A calling artefact of one sequencing batch (the Norway SRA block).

The metadata cannot separate these. CheckM contamination is recorded for only 4 of the 61. We do not call them mixed samples until the check below has been run.

## Check (Kaggle, CPU; added to `w0_encoding`, which streams the VCF anyway)

Per strain, from Arcadia's `filtered_output.vcf.gz`:

1. share of heterozygous / mixed genotype calls (`0/1`-type);
2. if `AD` is present, the median minor-allele read fraction at variable sites.

Compare the 61 flagged strains with the rest (Mann–Whitney, one test, no correction needed).

**Decision rule, fixed now:** a flagged strain is called a **probable mixed sample** if its
heterozygous-call share or its median minor-allele fraction exceeds the 99th percentile of the
unflagged strains. Everything else stays "high-locus, cause unknown".

## Pre-registered handling (applies to W1–W6)

1. **Primary analyses keep all strains.** The flagged strains stay in every split, as they are in the
   delivered model and the re-run. Dropping strains after seeing a property of the data, on one
   arm only, would be an untracked analysis choice.
2. **Sensitivity analysis, for every headline table (W1, W2, W3, W5):** recompute the test metrics with
   the 61 flagged strains removed from the test sides (models unchanged). For the lookups (cheap),
   also refit without them in training.
   - Report Δ next to the primary number.
   - A conclusion that changes (any of P1–P5 of `w1_five_axis.md` switching between pass and fail) is
     reported as **not robust** to these strains. The primary result is not replaced.
3. **If the VCF check calls some of them probable mixed samples:**
   - the sensitivity analysis is repeated with only those removed;
   - the paper's Methods describe them;
   - the primary analysis still keeps them, unless the owner decides otherwise **before** the
     W1/W3 GPU runs.
4. **The 13 strains without a profile** are left out of every P/A-based analysis (kNN on P/A, Lens B
   readings, W4 mapping). They stay in SNP-only analyses.
5. The flag is recomputed only if the matrix changes. Any change of the rule is logged under
   "Deviations".

## Deviations

- **26 October 2026, same session, before any use of the flag (addition, not a rule change).**
  Arcadia's own pipeline removes "outlier samples" before analysis: samples with more than 500
  synonymous quadrupleton alleles (`workflow/rules/filtering.smk` and `scripts/quadrupleton_stats.R`
  in Arcadia-Science/2024-predicting-antimicrobial-resistance). Our pipeline never did this.
  - Added to the `w0_encoding` check: recompute Arcadia's list from `annotated_output.vcf.gz`
    (biallelic synonymous SNPs with AC = 4; per-sample count of alternate alleles > 500).
  - Report its overlap with our 61 strains (count, Jaccard, Fisher test).
  - The handling rules above are unchanged. The same sensitivity analysis is run a second time with
    Arcadia's outlier list in place of ours.

# W13. Is there a second species, and MIC values, to test the findings outside *E. coli*?

**Status:** pre-registered 2026-10-29, before the probe has run. The probe
(`src.mechanism.species_probe`) only counts; no model is fitted and no phenotype is related to any
feature until the counts are read and the gate below is met.

## Why

Every external review named "one species, mostly two countries, no MIC" as the limit that caps the
paper. Two of the paper's findings need no pan-genome catalogue and can be tested on any organism
in NCBI Pathogen Detection, which publishes for each isolate the AMRFinderPlus calls
(`AMR_genotypes`), submitted susceptibilities (`AST_phenotypes`), assembly statistics and a SNP
cluster (`PDS_acc`), the same source and software as W8:

- **W7/W12:** assembly statistics predict resistance *within lineage* (here: within SNP cluster),
  and the signal is carried by size/CDS, not N50.
- **W10-B:** on held-out lineages a model on the AMRFinderPlus calls beats a fixed determinant rule.

W6 (the catalogue-versus-ARG rescue) needs a pan-genome catalogue like Arcadia's, which no other
species here has. **W6 is not replicated** and the paper says so.

Separately, the Pathogen Detection AST browser holds MIC values with the testing standard for many
isolates (NCBI AST browser documentation). These are not in the metadata table the probe reads, so
the MIC question is asked by a second, manual route (below).

## The probe

For every organism folder under `pathogen/Results/` it reads the newest
`latest_snps/Metadata/*.metadata.tsv` in chunks and counts, among isolates with
`AST_phenotypes`:

- isolates in total, isolates called with AMRFinderPlus 4.2.7, and the database versions;
- per drug: resistant, susceptible, intermediate, not determined;
- the share with a SNP cluster (`PDS_acc`), the number of distinct clusters, and the share with
  assembly length, contig count and contig N50;
- country and isolation-type spread, and the metadata columns present.

*E. coli* is skipped (W8 already counted it). Nothing else is computed.

## Gate, per species (decided by the probe, not by us)

A species is used for W13 only if **all** hold:

1. ≥ 2,000 isolates with AST and an AMRFinderPlus 4.2.7 call;
2. ≥ 3 drugs with ≥ 100 resistant **and** ≥ 100 susceptible;
3. ≥ 80% of those isolates with a SNP cluster, and ≥ 50 distinct clusters (lineage grouping needs
   enough groups for 5 folds);
4. ≥ 80% with all three assembly statistics.

At most **two** species are taken: the two passing species with the most AST isolates, decided by
that count alone so the choice cannot follow any result. If none passes, W13 is reported as
"attempted: no second species in this source meets the gate", with the counts.

## What runs if the gate is met (fixed now, written in full in an amendment before any fit)

- **W12 replication:** within-SNP-cluster concordance of assembly size, CDS-free (CDS is not in the
  table) — size, contig count and N50 only — against each passing drug's phenotype, and size's
  correlation with the acquired-gene count. Pass: size's mean within-cluster concordance over the
  passing drugs ≥ 0.55.
- **W10-B replication:** XGBoost (the W6 settings) on the AMRFinderPlus calls (acquired + point),
  SNP-cluster-grouped 5-fold, against a determinant rule. The rule for a second species is
  written in the amendment, from the drug's known determinant classes, **before** any score is
  computed. Pass: model − rule ≥ 0.05 mean nAP with the cluster-clustered CI above zero.

## MIC route (manual, no code runs until the file is in hand)

The AST browser's MIC download is a browser export, not an FTP file. The user exports
*E. coli/Shigella* rows with the MIC, measurement-sign and testing-standard columns; the export is
then matched to the W8 cohort by BioSample. The question, fixed now: **are the model's and the
rule's errors on the W8 cohort concentrated within one doubling dilution of the EUCAST breakpoint?**
If the export covers fewer than 1,000 W8 isolates for a drug, that drug is not analysed.

## Stated limits

- The second species' cohort is Pathogen Detection only: no pan-genome, no independent external
  cohort, and the AST labels are submitter calls under mixed standards.
- A SNP cluster is NCBI's 50-SNP clustering, not an MLST sequence type; it is finer than an ST, so
  "held-out cluster" is a stricter split than "held-out ST", and results are not numerically
  comparable with the *E. coli* ST-grouped numbers.
- The isolates are surveillance-heavy, which W8 showed is where the determinant rule is strongest.

## Amendment 2026-10-29, a data-plumbing fix, before any gate is read

The first probe run (commit 20736fa) reported a SNP-cluster share of 0% for every organism,
including Acinetobacter and Campylobacter, which Pathogen Detection does cluster. The cause is in
the probe, not the data: the metadata table has no `PDS_acc` column; cluster membership is
published in a separate file, `latest_snps/Clusters/*.cluster_list.tsv`. That run was stopped and
its gate is not used.

The probe now reads the newest file in `Clusters/`, joins it to the metadata on `target_acc`, and
records the file name and its column names per organism (`cluster_source.json`), so a reader can
check the join. An organism with no `Clusters/` file gets a cluster share of 0 and fails gate 3,
as before. The gate, its thresholds and the species-choice rule are unchanged.

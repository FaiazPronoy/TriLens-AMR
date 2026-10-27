# W4. Mechanism validation: pre-registration

**Status:** pre-registered. Step 1 (AMRFinderPlus on the pan-genome) is label-free and may run before
the rest; nothing in steps 2–5 is computed before this file is committed.

## Inputs

- `code/data/raw/pangenome/pangenome_cds.fa`: 18,494 CDS of the ECOR72 pan-genome. The FASTA
  header's first token is the locus tag, which is also the presence/absence column name.
- The **rebuilt** presence/absence matrix (`outputs-pa_fix`, KNOWN_ISSUES 12); the 13 strains without
  a profile are left out.
- `processed/cds_eggNog.tsv` (eggNOG annotation); `raw/pangenome/gene_presence_absence.csv` (ECOR gene
  names).
- The SNP matrix for step 4 (Kaggle only).
- Attention and SHAP rankings from W3 / the re-run, fixed Lens B, seed 42, rebuilt matrix.

## Step 1. AMRFinderPlus

- Command: `amrfinder -n pangenome_cds.fa --organism Escherichia --plus --threads 4`.
- The software version and database version are written to the output.
- Output: `code/results/w4_amrfinder/amrfinder_hits.tsv`, plus `determinants.csv`, one row per
  (locus, drug) after the drug mapping below.
- Answers KNOWN_ISSUES 7 (is blaCTX-M in the pan-genome?).

### Drug mapping

Fixed now, from AMRFinderPlus' `Class` / `Subclass` columns:

| drug | determinant if |
|---|---|
| AMP | Class = BETA-LACTAM |
| AMC | Class = BETA-LACTAM (coarse: the database does not say which enzymes clavulanate inhibits) |
| CTX, CAZ | Class = BETA-LACTAM and Subclass contains CEPHALOSPORIN |
| CIP | Class = QUINOLONE (genes such as qnr, aac(6')-Ib-cr; point mutations in step 4) |
| GEN | Class = AMINOGLYCOSIDE and Subclass contains GENTAMICIN |
| SXT | Class = TRIMETHOPRIM, or Class = SULFONAMIDE |

Hits with < 90% identity or < 80% coverage are reported but not counted.

### Limitation, stated up front

The pan-genome was built from 72 ECOR reference strains. Genes absent from those 72 (possibly the main
ESBL family) are absent from the feature space, the mapping and the W5 rules alike.

## Step 2. Known determinants among the model's genes

- Per drug: the share of the top-k genes (k = 20 and 50) that are determinants for that drug, for
  (a) Lens B attention and (b) SHAP of the classical learners.
- **Null:** 10,000 random gene sets of the same size, matched on prevalence decile (rebuilt matrix).
  Empirical one-sided p; Holm across 7 drugs × 2 rankings × 2 k.
- **Claim:** "the model's genes are enriched for known determinants" for a ranking if Holm p < 0.05 for
  ≥ 4 of 7 drugs at k = 20.

## Step 3. SXT and the aminoglycoside signal

- Genes:
  - aadA, dfrA, sul1/sul2: from AMRFinderPlus;
  - intI1: from AMRFinderPlus if reported, else by ECOR gene name or eggNOG annotation "class 1 integron
    integrase". The choice is recorded.
- **Co-occurrence** (strains with a profile):
  - phi coefficient and Fisher exact test for aadA × {dfrA, sul1, intI1}, overall;
  - within ST by Cochran–Mantel–Haenszel over STs with ≥ 20 strains;
  - Holm over the 3 pairs.
- **Linkage test:** logistic regression of SXT-R on aadA, dfrA, sul1 (presence), with phylogroup fixed
  effects, on strains tested for SXT. The aadA–SXT association counts as **explained by linkage** if
  aadA's 95% CI includes 1 (odds ratio) once dfrA and sul1 are in the model, while it excludes 1 alone.
- **Attention coverage:** whether aadA, dfrA, sul1 and intI1 are in Lens B's top-50 for SXT (reported).

## Step 4. CIP: gyrA × parC (Kaggle, CPU; needs the SNP matrix)

- **Genotypes.** SNP names are `LOCUS:POS:REF>ALT`; POS is the position in the pan-genome CDS.
  - Codon = ⌈POS/3⌉. The reference codon is checked against the translated CDS of `pangenome_cds.fa`:
    gyrA codon 83 must be Ser and 87 Asp; parC codon 80 Ser.
  - If a check fails, step 4 stops and reports. Positions are not "fixed" by hand.
  - Calls: gyrA S83L; gyrA D87N/H/Y/G; parC S80I/R. Missing calls are coded reference (KNOWN_ISSUES 16);
    gyrA and parC are core genes, so absence is rare and is reported.
- **Interaction.** Logistic regression CIP-R ~ gyrA83 + gyrA87 + parC80 + gyrA(any) × parC80 +
  phylogroup, on CIP-tested strains. Report:
  - the interaction odds ratio with 95% CI;
  - the likelihood-ratio test against the model without it.
  - Arcadia reported the interaction as not significant; we report whatever comes out.
- **Model view.** Mean predicted probability of TriLens (S1 test strains) per genotype combination,
  with bootstrap CIs.

## Step 5. Candidate genes

Genes in a ranking's top-50 that are not determinants, listed with eggNOG description and COG
category, U_ST and U_res. They are labelled **hypotheses**; no claim is made about them.

## Compute

AMRFinderPlus: CPU, minutes (here if it installs; otherwise the Kaggle notebook `w4_amrfinder`).
Step 4: Kaggle, CPU, about 0.5 h.

## Deviations

- **26 October 2026, before any data were scored (correction).** NCBI's own test output shows
  AMRFinderPlus reporting `blaEC`, the chromosomal ampC every *E. coli* carries, with Class
  BETA-LACTAM. Under the mapping above it would count as an AMP/AMC determinant in every strain, so
  the W5 "any determinant" call would be resistant for all strains.
  - Rule added: a **gene** determinant carried by ≥ 95% of strains (rebuilt P/A matrix) is excluded
    from the drug counts, as intrinsic. It is still listed in `determinants.csv`, flagged `intrinsic`.
  - Point mutations (for example ampC promoter variants) are not affected by this rule.
  - Decided from the tool's documented behaviour, not from any of our results.

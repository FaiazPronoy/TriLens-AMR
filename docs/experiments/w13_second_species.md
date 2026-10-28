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

## Amendment 2026-10-29, the MIC analysis, fixed before any MIC is related to any score

The AST-browser export (*E. coli* and *Shigella*, all rows) has 171,750 rows for **11,100**
isolates, exactly the number of AST isolates `ext_probe` counted, so no download cap cut it. It
matches **all 10,232** isolates of the W8 primary cohort by PDT accession (`Isolate` =
`target_acc`). Isolates with an MIC, per drug: AMP 9,166, CIP 8,228, AMC 8,512, SXT 8,548,
GEN 9,346, CAZ 5,405, **CTX 950**. By the rule above, **CTX is not analysed** (under 1,000).
Testing standards on the matched rows: CLSI 137,349, EUCAST 16,050, NARMS 2,784, SFM 415,
missing 2,903. Only counts and value lists were read to write this; no MIC has been set against
any model score or rule call.

**Harmonisation, fixed now.**

1. *Breakpoints:* EUCAST Clinical Breakpoint Tables **v16.1** (valid from 2026-06-24),
   Enterobacterales, systemic/iv lines, transcribed into
   `results/data_inventory/eucast_v16_1_enterobacterales.csv` and committed before the analysis
   runs. Read from the official PDF so far: ampicillin S≤8 R>8; amoxicillin–clavulanic acid iv
   S≤8 R>8 (clavulanate fixed at 2 mg/L); cefotaxime S≤1 R>2; ceftazidime S≤1 R>4.
   Ciprofloxacin, gentamicin and trimethoprim–sulfamethoxazole are transcribed from the same
   file when it is in hand; the analysis does not run with a value taken from memory.
2. *Censored values:* `==v` is [v, v]; `<=v` is (0, v]; `<v` is (0, v/2]; `>=v` is [v, ∞);
   `>v` is [2v, ∞). A harmonised label is R if the whole interval is above the R breakpoint, S if
   the whole interval is at or below the S breakpoint, I if it lies wholly between them, and
   **undetermined** (excluded) if the interval straddles a breakpoint. I is counted as resistant,
   as everywhere in the paper.
3. *Trimethoprim–sulfamethoxazole:* EUCAST expresses the breakpoint as the trimethoprim
   concentration (1:19). The export mixes conventions: Vitek reports the sulfamethoxazole
   component (9.5, 19, 38, 76, 152, 304 = 19 × a trimethoprim dilution). Those six values are
   divided by 19; any other value above 32 mg/L, and the single `>26.95`, are excluded.
4. *One doubling dilution:* with S≤s and R>r, an isolate is **near** the breakpoint if its MIC
   interval lies within [s/2, 2r], **far** if it lies wholly outside that range, and is excluded
   from the near/far comparison otherwise.

**Questions and rules.** All use the frozen W8 scores (`scores.csv.gz`), so nothing is refitted.

- **M1, endpoints under harmonised labels.** W8's AP − M and AP − R, mean nAP over AMP, CIP, AMC,
  SXT, GEN, CAZ, BioProject-clustered, recomputed with EUCAST labels in place of the submitter's.
  Reported, no new pass rule: W8's verdict is restated, and any change in sign or in whether a CI
  excludes zero is stated.
- **M2, where the errors are.** The model is binarised at 0.5 (as in the paper's clinical tables)
  and the rule is already binary. Over all analysed drugs pooled, the error rate among **near**
  isolates against the error rate among **far** isolates, as a ratio with a BioProject-clustered
  bootstrap CI, for each arm. **Concentrated** if the ratio is ≥ 2 with the CI's lower bound > 1.
  Per drug: reported without a rule.
- **M3, label agreement.** Per drug, agreement of the submitter's call with the EUCAST call, and
  the count of undetermined isolates. Descriptive.

**Limit.** MICs come from different platforms and vendors, and a broth-microdilution panel's
lowest and highest wells censor the value, which is why the interval rule above is used.
EUCAST is applied to isolates most of which were interpreted under CLSI; M1 is therefore a
harmonisation, not a correction of anyone's call.

## Result of the probe, and amendment 2026-10-29: the Campylobacter analysis, before any score

**Gate (probe run at commit 70a5188, log of the Kaggle run):** of 107 organism folders, **one**
passes: *Campylobacter* (4,426 AST isolates on AMRFinderPlus 4.2.7; SNP clusters on 88%, 1,206
clusters; assembly statistics on 100%; four drugs with ≥ 100 R and ≥ 100 S). The two nearest
misses, stated so nobody has to guess: *Klebsiella* (2,634 isolates, 21 drugs) fails gate 3 with
SNP clusters on only 55%; *Salmonella* has 9,977 AST isolates but only 469 called with 4.2.7.
The gate is not relaxed for either. So W13 has one second species, and the paper says so.

The cohort: *C. jejuni* 2,995, *C. coli* 1,424, *C. lari* 7. Database 2026-08-07.1 for 4,398, 2026-03-24.1 for 28;
because training and testing are both inside this cohort, all 4,426 are used.
Drugs passing (R/S): ciprofloxacin 977/3,449, nalidixic acid 950/3,309, tetracycline 2,172/2,253,
clindamycin 120/4,139. Only these counts and the list of call symbols were read to write this.

**Fixed now:**

1. **Calls.** `AMR_genotypes` parsed with the W6/W8 rules: a symbol without suffix is a gene,
   `=POINT` a point mutation, `PARTIAL`, `PARTIAL_END_OF_CONTIG`, `MISTRANSLATION`, `HMM` are not
   counted. Features: every gene and point mutation present in ≥ 5 isolates, binary.
2. **Lineage folds.** SNP-cluster-grouped 5-fold, built by the same procedure as the *E. coli*
   ST-grouped folds (largest cluster first to the emptiest fold). Isolates with no cluster are in
   training in every fold and never scored. BioProject is not used for folds.
3. **Model.** XGBoost with the W6 settings, no tuning, one model per drug.
4. **Determinant rule** (from the mechanisms in the literature, not from these data):
   - ciprofloxacin, nalidixic acid: any `gyrA` point mutation — Thr-86-Ile in the QRDR gives
     high-level resistance to both (Iovine 2013, *Virulence* 4:230);
   - tetracycline: any `tet(` gene — TetO ribosomal protection is the main mechanism (same source);
   - clindamycin: any `23S` point mutation, `erm(`, `lnu(` or `cfr(` gene. **Pending verification**
     that the 23S A2074/A2075 mutations are reported as clindamycin determinants in *Campylobacter*
     (Zhao et al. 2016, *Appl Environ Microbiol* 82:459, which could not be opened here). If the
     source does not support it, the clindamycin rule is `erm(`, `lnu(`, `cfr(` only. The analysis
     does not run until this is settled, and the choice is made from the paper, not from the data.
5. **Endpoints.**
   - **W10-B replication (primary):** model − rule, mean nAP over the four drugs, cluster-clustered
     CI. Pass: ≥ 0.05 with the CI above zero. Ciprofloxacin and nalidixic acid share one mechanism,
     so the mean is also reported over the three distinct classes (quinolone = the mean of the two);
     that figure is secondary.
   - **W12 replication:** within-SNP-cluster concordance of assembly length, contig count and N50
     with each drug's phenotype; pass if length's mean over the four drugs is ≥ 0.55. Cluster cells
     with no resistant–susceptible pair contribute nothing.
   - Also reported: a metadata arm on the three assembly statistics alone (HistGradientBoosting),
     same folds, as in W8.

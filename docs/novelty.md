# Where the novelty is, and what is only a hypothesis

Written 28 October 2026, during the re-run. Every number marked **verified** was computed
today from the project's own data and the script is named; every number marked
**hypothesis** has not been computed yet and may fail.

The purpose of this memo: the re-run reproduces the thesis, but reproducing is not a
contribution. This says what, in this dataset, is new and worth a Q1 paper, and what the
honest risks are.

---

## 1. What the field already has (so we do not repeat it)

| Paper | What it established | What it leaves open |
|---|---|---|
| Yu, Wheeler & Barquist, *PLOS Biology* 2025 | Population structure and biased sampling inflate AMR ML scores; 5 species, LightGBM. Asks for lineage-aware models and a phylogenetic-placement baseline. | Diagnosis only. No fix, no mechanism, no baseline below which a score is meaningless. |
| Cheon, Nguyen & Unno, *Briefings in Bioinformatics* 2026 | Phylogeny-blocked CV (F1 0.876 → 0.753), 20k genomes, 5 species; expression/promoter features; genotype-phenotype discordance classes. | Features built from detected ARGs, so the "what if the causal gene is not in the feature space" question cannot arise. |
| Fistarol, Gervasio & Szöllősi, *npj Antimicrobials and Resistance* 2025 | In *S. aureus*, gene copy-number features survive lineage holdout far better than core SNPs (F1 ≈ 0.89 vs ≈ 0.60). | One species; explanation is proposed, not tested. |
| Ahmad & Busair, bioRxiv 2026 (preprint) | Leave-one-country-out in African *K. pneumoniae*: AUROC 0.85 → 0.66–0.69. | Pilot size (≤20 test genomes/fold); CIs overlap. |
| Arcadia (York & Sandler 2025) | Linear genomic prediction (BSLMM) on this exact dataset; heritability ≈ 0.94–0.99. | Only linear models. Their own open questions: the SXT signal sits on aminoglycoside genes; gyrA × parC epistasis not significant; nonlinear models listed as future work. |

**Conclusion:** "lineage-aware evaluation" is now a crowded field. Another split-and-measure
paper adds little. What is missing is *why* models fall back on lineage, and *what a score
should be compared against*. Both are answerable with the data in this repository.

---

## 2. Three findings, all verified today

### 2.1 A confounder floor: no sequence at all reaches most of the score

`scripts/diagnostics/confounder_floor.py` trains gradient boosting on strain metadata only —
assembly statistics (contigs, assembly size, GC, N50, CDS count, tRNA count), collection
year, isolation country, sequencing platform, source publication. **No SNP, no gene
presence/absence, no k-mer: nothing is read from the genome sequence.**

**Verified** (`results/diagnostics/confounder_floor.csv`, mean over the 7 antibiotics,
test-part PR-AUC):

| Split | Metadata only | Assembly stats only | Country + year | TriLens full stack |
|---|---|---|---|---|
| Random | **0.710** | 0.596 | 0.516 | 0.850 |
| Phylogroup | **0.712** | 0.617 | 0.491 | 0.853 (paper) |
| ST (lineage) holdout | **0.280** | 0.225 | 0.303 | 0.634 (paper) |

Read that again: on the random split a model that never sees the genome reaches 84% of the
delivered model's PR-AUC, and the phylogroup split — the paper's "leakage-corrected"
split — has exactly the same floor. Only the lineage holdout brings the floor down, and
there the model clears it by a wide margin (0.634 against 0.280).

**Honest reading, which must be in the paper.** Assembly size and CDS count are not pure
artefacts: a larger accessory genome means more acquired genes, so part of this is real
biology reached through a proxy. The claim is therefore *not* "the dataset is all batch
effect". The claim is: **this is the performance attainable without variant- or gene-level
features, and every reported score should be read against it.** That is a floor, not an
accusation, and it is exactly the comparison a reader cannot currently make in any AMR ML
paper we found.

**Why it is novel.** We found no AMR ML paper that reports a metadata-only baseline. Yu et
al. show lineage confounding, Ahmad & Busair show country effects; nobody shows that
database covariates alone get this far. It costs minutes of CPU and applies to every public
AMR dataset, so it is a recommendation other groups can adopt immediately.

**Risk:** low. The numbers are computed and reproducible. The only work left is framing it
fairly and running it on the remaining splits.

### 2.2 The feature space contains the causes for some drugs and not for others — and that ordering predicts which drugs generalise

The SNP and presence/absence views are cut from a pan-genome built from the **72 ECOR
reference strains**. AMRFinderPlus 4.2.7 (database 2026-08-07.1, `--organism Escherichia
--plus`) on all 18,494 pan-genome CDS (notebook `w4_amrfinder`, results branch
`results/w4_amrfinder`) gives the curated determinant panel. **Verified:**

| Drug | Main acquired mechanism in *E. coli* | In the feature space (prevalence in our strains) | Classical arm, lineage holdout PR-AUC |
|---|---|---|---|
| CIP | gyrA / parC point mutations (chromosomal) | **yes** — the loci are there, variants are SNP columns | **0.916** |
| AMP | TEM-1 | **yes** — *bla*TEM-1 (49%) | **0.901** |
| SXT | *sul1/2* + *dfrA* | **half** — *sul1* (28%), *sul2* (36%); **no *dfrA*** | 0.774 |
| CTX | CTX-M (ESBL) | **no** — no CTX-M, no ESBL of any kind | 0.602 |
| AMC | OXA-1, TEM hyperproduction, *ampC* promoter | **mostly no** — only TEM-1 and intrinsic *bla*EC | 0.476 |
| CAZ | CTX-M / other ESBL | **no** | 0.464 |
| GEN | *aac(3)-II*, *aac(3)-IV*, *ant(2'')-Ia* | **no** — no gentamicin determinant at all | **0.222** |

Other determinants present: *aadA1*, *aadA7*, *aph(3'')-Ib*, *aph(6)-Id*, *aph(3')-Ia*
(streptomycin/kanamycin, not gentamicin), *tet(A/B/C)*, *catA1*, *fosA7.5*. No CTX-M was found
anywhere (`ctx_m_hits: []`).

**The ordering is the finding.** The four drugs whose causal determinant is in the feature
space generalise to unseen lineages (0.77–0.92). The three whose determinant is absent
collapse (0.22–0.60), with gentamicin — the drug with no determinant at all — worst. The model
is not failing on hard drugs; it is failing exactly where it was never shown the cause, and
on those drugs it can only reach resistance through lineage, which is why the lineage
holdout breaks it.

**This turns the field's lineage problem into an explanation.** Yu et al. (2025) show that
models lean on lineage; this shows *when they must*: whenever the feature space is built
from reference genomes and the causal element is mobile and absent from them. That is a
statement about study design that applies to any reference-panel or reference-mapping
feature space, not only to this dataset.

**It also answers one of Arcadia's open questions (hypothesis, one check away).** Arcadia
found the SXT signal on aminoglycoside markers (*aadA*) and called it puzzling. In the ECOR
pan-genome, *aadA1*, *sul1* and *catA1* sit side by side on one reference genome
(`DHJNCGMO_04398`–`04406`), the arrangement of a class 1 integron, and no *dfrA* exists in the
feature space at all. So the trimethoprim half of SXT resistance has no direct marker, and the
nearest available proxy is the cassette neighbour *aadA1* that travels with *sul1* on the same
element. Check needed: that Arcadia's top SXT marker is this *aadA1* locus or its contig.

**Correction to an earlier draft.** A first pass searched annotation *text* and reported no
TEM and no *sul*. That was wrong: the annotations did not name them. AMRFinderPlus is the
authority here, and the table above replaces that pass.

**What would make it conclusive (hypothesis until then):** with 7 drugs the ordering is
suggestive, not a test. Two steps turn it into one: (1) extend to the 16 drugs with enough
labels (§4.2) and test whether "determinant present in feature space" predicts the holdout
gap; (2) the intervention — add each strain's own determinant calls as features and measure
whether the gap closes for CTX, CAZ and GEN. Step 2 needs per-strain ARG calls (§4.1).

### 2.3 A metric that separates resistance from lineage: within-lineage discrimination

Every current paper measures generalization by *splitting* — train on some lineages, test on
others. A split tells you the score drops; it does not tell you whether the model can tell a
resistant strain from a susceptible one *of the same lineage*, which is the only question a
clinician cares about and the only one lineage memorisation cannot answer.

The dataset supports this directly. **Verified** today: strains whose ST is known = 6,487,
and per antibiotic, the sequence types that contain both resistant and susceptible strains:

| Antibiotic | STs with both R and S | Strains in those STs | Resistant among them |
|---|---|---|---|
| AMP | 121 | 4,060 | 2,023 |
| CIP | 56 | 4,132 | 948 |
| AMC | 82 | 2,961 | 1,030 |
| SXT | 83 | 2,739 | 925 |
| GEN | 66 | 4,532 | 554 |
| CTX | 49 | 3,700 | 579 |
| CAZ | 58 | 4,047 | 606 |

So thousands of strains per drug sit in phenotype-discordant lineages. The metric: restrict
scoring to those strains, or score ranking within each ST (a stratified concordance /
within-group AUC). A model that only reads lineage scores at chance there, whatever its
pooled PR-AUC says. Reported next to the pooled number and the §2.1 floor, it gives three
honest layers instead of one headline.

**Risk:** low to compute (CPU, from the saved `predictions_saved.npz`), but the *result* may
be unflattering — the model may do poorly within lineages. That is still a finding worth
publishing, and it is the kind of result that makes reviewers trust the rest of the paper.

---

## 3. What this makes the paper

Not "another AMR model". The claim becomes:

> Reported genomic AMR prediction scores are not comparable to zero. On the standard random
> and phylogroup splits, strain metadata alone reaches 84% of a strong model's PR-AUC, and
> the feature spaces built from reference pan-genomes do not contain the acquired genes that
> cause resistance for most drugs — so models predict lineage by necessity, not by choice.
> We give three measurements any such study can report (a confounder floor, a lineage
> holdout, and within-lineage discrimination), show what each one does to a
> state-of-the-art stacked model, and show which drugs survive.

TriLens stops being the contribution and becomes the instrument: a strong, honestly
evaluated model used to measure what the field's benchmarks actually test. Its lineage-aware
Lens B and the gate then answer a sharper question — does conditioning on strain help a
model clear the floor and discriminate within lineages?

That claim does not depend on Lens B winning. If the strain-conditioning ablation turns out
to be noise (a real possibility — the earlier gap was 0.8462 vs 0.8486), the paper still
stands, which is why this framing is safer than the one we had.

---

## 4. What we cannot do with what is in the repository

Stated plainly so nobody pretends otherwise.

1. **Per-strain ARG calls.** The decisive experiment of §2.2 needs the determinants each
   strain actually carries. The repository has the pan-genome CDS (one copy of each
   reference locus), not per-strain assemblies. Options, in order of cost:
   - BV-BRC AMR gene tables for these genomes — `Genome.ID` is in
     `strains_metadata.csv` for all 6,985 strains (format `562.xxxx`), so they can be
     requested in bulk from BV-BRC. **This is the cheapest decisive route.**
   - Arcadia's own `resistance_output.non_silent.vcf.gz` (in the Zenodo release) — variants
     in loci they considered resistance-related; worth reading before downloading anything.
   - Assemblies from BV-BRC or ENA (≈35 GB for 6,962 genomes) and AMRFinderPlus per strain.
     Feasible on Kaggle in chunks, but it is days of CPU.
2. **Phenotype breadth.** The paper uses 7 antibiotics. **Verified:** the phenotype matrix
   holds 50 drugs, of which **16** have ≥500 tested strains with ≥100 resistant and ≥100
   susceptible (adds piperacillin/tazobactam, cefuroxime, amoxicillin, tobramycin,
   trimethoprim, tetracycline, ceftriaxone, cefoxitin, aztreonam, cefepime, nalidixic acid,
   amikacin and others). Going to 16 drugs turns §2.2 into a proper test: does the floor-
   clearing margin track whether a drug's mechanism is chromosomal or acquired? That is the
   analysis that makes the mechanism claim quantitative instead of anecdotal, and it needs
   no new data at all.
3. **A second species.** Everything here is *E. coli*. Any claim about "the field" would be
   stronger with one more species, which means a new dataset.

---

## 5. Risks and how they would show up

| Risk | How it would appear | What we do then |
|---|---|---|
| The floor is dismissed as "assembly size is biology" | Reviewer says the baseline is not a confounder | Already handled: report the blocks separately (assembly stats 0.596, country+year 0.516, publication 0.397) and frame it as a floor, not an artefact |
| Within-lineage discrimination is near chance for every model | §2.3 table is flat | Publish it. It is the strongest possible support for §2.1 and §2.2, and it is the honest answer |
| Lens B strain conditioning is noise | multi-seed CIs overlap | The §3 framing survives; the ablation becomes a negative result reported with CIs |
| BV-BRC ARG tables unavailable for these genome IDs | §2.2 stays correlational | Fall back on the Arcadia resistance VCF, and state the limitation |
| 16-drug expansion changes the headline | some drugs do much worse | That is the point of the analysis; report per-drug, grouped by mechanism class |

---

## 6. Order of work

CPU-only and no new data, so these can start now:

1. `confounder_floor.py` on every split (S4a, S4b, S5 included) — **done for random,
   phylogroup, ST holdout**.
2. Within-lineage discrimination (§2.3) from the saved predictions of the re-run.
3. The 16-drug expansion: labels and splits exist; the classical arm and the lookup
   baselines run per drug.
4. `w4_amrfinder` → the curated determinant panel for §2.2 — **done**.

Needs a decision from the owner:

5. Request BV-BRC AMR gene tables by `Genome.ID` (§4.1). This is the one thing that turns
   §2.2 from a strong argument into a demonstration.

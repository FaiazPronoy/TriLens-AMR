# Task for you: find the strongest publishable contribution in this dataset, and try to break ours

You are advising on a genomic antimicrobial-resistance (AMR) machine-learning paper being
prepared for a Q1 journal (target: *Briefings in Bioinformatics*, *Bioinformatics*, *NAR
Genomics and Bioinformatics*, or *Genome Medicine*). The model and the re-run are finished.
What is missing is a contribution that a reviewer will call new. I want your independent
judgement, not agreement.

Read the whole brief before answering. It is long on purpose: every number below was
computed from the actual data, and I want your reasoning anchored to those numbers rather
than to what you assume such a dataset looks like.

---

## Rules you must follow

These are not style preferences. An answer that breaks them is useless to me.

1. **Never invent a number about this dataset.** Every quantity about our data that you use
   must be quoted from this brief. If you need a quantity that is not here, write
   "NEEDS MEASUREMENT: <exact quantity, how to compute it>" and move on.
2. **Only cite papers you have actually opened in this session.** For each citation give
   title, authors, venue, year, DOI or URL, and one sentence on what it did. If you cannot
   open a paper, list it under "found but not read" and do not build an argument on it. I
   will check the citations. A fabricated or misattributed citation makes the whole answer
   worthless.
3. **Label every claim** as `[VERIFIED — from the brief]`, `[LITERATURE — cite]`, or
   `[SPECULATIVE — would need test X]`. Mixing these is the failure mode I am trying to
   prevent.
4. **Every idea must come with a falsifiable pass/fail rule written before the experiment**,
   in the form: "I predict metric M on data D will be above/below value V; if it is not, the
   idea is dead." An idea without a way to fail is not an idea.
5. **State compute cost** for each idea: CPU-only or GPU, and roughly how many hours on one
   NVIDIA T4. Say if it needs data we do not have.
6. **Do not flatter the existing plan.** Section 6 asks you to attack it. If our three
   proposed contributions are weak or already published, say so and name the paper.
7. If you are uncertain, say you are uncertain and say what would settle it. "I don't know"
   is an acceptable and useful answer.

---

## 1. The data (fixed, cannot be changed)

Source: Arcadia Science's public *E. coli* AMR genotype–phenotype release
(dataset pub DOI 10.57844/arcadia-d2cf-ebe5, Zenodo 12692732; analysis pub DOI
10.57844/arcadia-8391-465f, Zenodo 14364732). CC BY 4.0. Strains were found through BV-BRC
by intersecting "has SRA reads" with "has AMR phenotype".

- **6,962 strains** used (6,985 in the metadata; 7,057 including 72 ECOR references).
- Countries: 14; mostly United Kingdom (3,387 including "England") and Norway (2,964); 611
  strains from other countries. Collection year available for 4,910 strains, 2001–2017.
  Host available for 3,922, almost all human — only 24 non-human, so host is unusable as an
  analysis axis.
- **Phenotypes for 50 antibiotics** (susceptible / intermediate / resistant; intermediate is
  recoded as resistant). The paper currently uses **7**: AMP, CIP, AMC, SXT, GEN, CTX, CAZ.
  **16 antibiotics** have ≥500 tested strains with ≥100 resistant and ≥100 susceptible.
- **Features**, all cut from a pan-genome built from the **72 ECOR reference strains**:
  - SNP matrix 6,962 × 535,439 (genotype codes 0–3; missing calls are coded as reference,
    which is Arcadia's convention).
  - Gene presence/absence 6,962 × 32,441 loci (18,494 CDS + 13,947 intergenic clusters).
  - A 5,000-gene / 262,225-SNP filtered view, the gene ranking refitted inside each split's
    own training part.
- Rich per-strain metadata from BV-BRC: assembly statistics (contigs, assembly size, GC,
  N50, CDS count, tRNA count), sequencing platform, source publication (PubMed ID, present
  for 3,779 strains), `Genome.ID` for all 6,985, SRA accession for all 6,985, MLST sequence
  type for 6,487.
- **We do NOT have:** per-strain assemblies, per-strain AMR-gene (ARG) calls, MIC values,
  plasmid reconstructions, long reads, or any second species. The owner is willing to add
  datasets if you justify it concretely.

## 2. The model (fixed; it is the instrument, not the question)

"TriLens-MoE-Stack": three neural lenses — a per-gene SNP CNN, a gene presence/absence gated
attention lens whose attention can be conditioned on the strain, and a structured
cross-antibiotic head — combined by a mixture-of-experts gate, then stacked with five
classical learners (XGBoost on gene features, LightGBM, random forest, logistic regression,
XGBoost on χ²-selected SNPs) through a per-antibiotic logistic-regression meta-learner.
Five-fold out-of-fold stacking. Multi-label over the 7 antibiotics.

## 3. Results actually measured (use these; do not invent others)

Mean PR-AUC over the 7 antibiotics, on each split's held-out test part:

| | Random split | Phylogroup split | Sequence-type (lineage) holdout |
|---|---|---|---|
| TriLens full stack | **0.850** | 0.853 | 0.634 |
| Classical arm alone | 0.846 | — | 0.622 |
| Neural lenses alone | 0.780 | — | 0.474 |
| + MoE gate, no stacking | 0.796 | — | — |
| kNN on SNP Hamming distance (relatives only) | ~0.69 | ~0.70 | ~0.29 |
| Phylogroup label only | 0.291 | — | — |
| **Metadata only, no sequence at all** | **0.710** | **0.712** | **0.280** |

Per-drug, random split, full stack: AMP 0.943, CIP 0.975, AMC 0.754, SXT 0.842, GEN 0.709,
CTX 0.904, CAZ 0.823. Under the lineage holdout the neural lenses alone give GEN 0.137,
CTX 0.307, CAZ 0.212 — i.e. they nearly collapse for the drugs whose resistance is carried
on mobile elements.

**The metadata-only row** is gradient boosting on assembly statistics + collection year +
isolation country + sequencing platform + source publication. No SNP, no gene presence, no
k-mer: nothing read from the genome sequence. Broken into blocks on the random split:
assembly statistics alone 0.596, country + year 0.516, publication alone 0.397, all together
0.710.

**Feature-space audit** (AMRFinderPlus 4.2.7, database 2026-08-07.1, on all 18,494 pan-genome
CDS). Present: *bla*TEM-1 (in 49% of strains), *sul1* (28%), *sul2* (36%), *aadA1*, *aadA7*,
*aph(3'')-Ib*, *aph(6)-Id*, *aph(3')-Ia*, *tet(A/B/C)*, *catA1*, *fosA7.5*, intrinsic *bla*EC;
gyrA/parC loci (CIP mutations are SNP columns). **Absent: any CTX-M or other ESBL, any
gentamicin determinant (*aac(3)-II/IV*, *ant(2'')-Ia*), any *dfrA*.** Against the classical
arm's lineage-holdout PR-AUC: CIP 0.916, AMP 0.901, SXT 0.774 (determinant present or half
present) versus CTX 0.602, AMC 0.476, CAZ 0.464, GEN 0.222 (determinant absent or mostly
absent). *aadA1*, *sul1* and *catA1* sit side by side on one reference genome, a class 1
integron arrangement.

**Within-lineage phenotype discordance** (strains with a known ST = 6,487): sequence types
containing both resistant and susceptible strains — AMP 121 STs / 4,060 strains, CIP 56 /
4,132, AMC 82 / 2,961, SXT 83 / 2,739, GEN 66 / 4,532, CTX 49 / 3,700, CAZ 58 / 4,047.

**Other measured facts.** A position-shuffling test on a plain 1D CNN over the whole
pan-genome: Arcadia ordered 0.775 → shuffled 0.632 (drop 0.143), Giessen (an independent
*E. coli* collection, single-reference SNPs) ordered 0.840 → shuffled 0.660 (drop 0.180).
The gene presence/absence matrix was found to encode each locus's **minority** state rather
than presence (2,642 of 2,652 reference core genes appeared "present" in <1% of strains;
gyrA and parC in none); it has been rebuilt from the PLINK allele letters, after which
gyrA/parC are present in 100% of strains and 10,511 of 32,441 loci were flipped.

## 4. Literature we have read (verify and extend; do not take our summaries on trust)

- Yu, Wheeler & Barquist, *PLOS Biology* 2025 — biased sampling driven by population
  structure confounds AMR ML; 5 species, LightGBM; asks for lineage-aware models and a
  phylogenetic-placement baseline.
- Cheon, Nguyen & Unno, *Briefings in Bioinformatics* 2026 (bbag451) — 20,296 genomes, 5
  species, 107 tasks; promoter/expression features; genotype–phenotype discordance classes;
  phylogeny-blocked CV F1 0.876 → 0.753.
- Fistarol, Gervasio & Szöllősi, *npj Antimicrobials and Resistance* 2025 — in *S. aureus*,
  gene copy-number features survive lineage holdout much better than core SNPs.
- Ahmad & Busair, bioRxiv 2026 — leave-one-country-out in African *K. pneumoniae*, AUROC
  0.85 → 0.66–0.69; pilot-scale, overlapping CIs.
- Arcadia (York & Sandler 2025) — linear BSLMM on this same dataset; heritability 0.94–0.99;
  their open questions: the trimethoprim/sulfamethoxazole signal lands on aminoglycoside
  genes (*aadA*), gyrA × parC epistasis not significant, nonlinear models left as future
  work.
- A NeurIPS 2026 workshop paper (Anojulu & Puttagunta) reports that phylogeny-weighted
  conformal prediction fails under clade shift in *M. tuberculosis*, so we are treating
  conformal prediction as a cautionary note rather than a contribution.

## 5. Resources

Three Kaggle accounts (≈30 GPU-hours per account per week, T4; CPU effectively unlimited),
roughly 2–3 months of calendar time, and willingness to add public datasets. No local GPU.
Everything must be reproducible from a public GitHub repository.

---

## 6. What I want from you

Answer these in order. Be concrete. Length is fine; padding is not.

**(a) Attack our three proposed contributions.** For each, tell me whether it is genuinely
new, already published (name the paper), or too weak to carry a paper:
 1. a **confounder floor** — report the metadata-only baseline alongside every split, and
    argue that scores on the random and phylogroup splits are not comparable to zero;
 2. a **feature-space explanation** — models fall back on lineage because the pan-genome
    feature space does not contain the acquired determinants, which is a study-design
    failure rather than a modelling one;
 3. a **within-lineage discrimination metric** — score only strains in phenotype-discordant
    sequence types, so lineage memorisation cannot help.

**(b) Propose 3–6 contributions of your own**, ranked by (novelty × feasibility) with our
resources. For each: the question, the exact experiment, the metric, the pre-registered
pass/fail rule, the compute cost, and what data is needed. Prefer ideas that work with what
we have. Say explicitly which are only possible with new data, and name the dataset.

**(c) Tell me which single experiment would most change what we believe.** One experiment.
Justify why it beats the alternatives.

**(d) Design the decisive test of contribution 2.** We think the test is: obtain the ARGs
each strain actually carries, add them as features, and measure whether lineage reliance
falls. We have `Genome.ID` (BV-BRC, format `562.xxxx`) and SRA accessions for all strains,
but no assemblies. Give the concrete route — which resource, which identifier, which bulk
endpoint or download, what the output looks like, and what could go wrong. If you know this
route does not work, say so; that saves us days.

**(e) Say what would make a reviewer reject this paper** even if every experiment succeeds,
and what we should do about each objection now rather than after review.

**(f) Choose the target journal and the title**, and say in two sentences what the abstract's
central claim should be. If you think the honest version of this work is a benchmark /
negative-results paper rather than a methods paper, say that plainly — we would rather
publish a true benchmark paper than oversell a model.

---

## 7. One more thing

If anything in §3 looks wrong, impossible, or internally inconsistent to you, say so and say
which measurement would settle it. We would much rather find an error now than have a
reviewer find it. We have already thrown away two finished GPU runs over data problems we
discovered this way, and we will throw away more if needed.

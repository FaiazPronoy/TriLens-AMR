# Literature for the introduction and related work (2025–2026, plus the anchors)

**Status (session 2, 26 October 2026): incomplete.** The rule is to cite only papers we have opened.
This session's network policy refused every journal, preprint and literature-API host:

- journals: `journals.plos.org`, `academic.oup.com`, `www.biorxiv.org`, `arxiv.org`,
  `pmc.ncbi.nlm.nih.gov`;
- the Arcadia pub site: `research.arcadiascience.com`;
- literature APIs: `www.ebi.ac.uk` (Europe PMC), `api.crossref.org`, `api.openalex.org`,
  `api.semanticscholar.org`.

Only GitHub was reachable. So:

- **§1** lists what was actually opened: the source repositories of papers, which describe their methods.
- **§2** lists papers found by web search, whose text was **not** opened. Nothing in §2 may be cited
  until someone has read it; each row says what to check.

To finish this file: open the §2 papers from a normal browser, or allow these hosts in the
environment's network settings. Move each paper to §1 with a line written from the paper itself.

---

## 1. Opened

| work | venue, year | link | what they did (from the opened source) | what it leaves open for us |
|---|---|---|---|---|
| Hu, Meyer, Deng, Asgari, Kuo, Münch, McHardy. *Assessing computational predictions of antimicrobial resistance phenotypes from microbial genomes* (opened: the benchmark repository README) | *Briefings in Bioinformatics* 25(3) bbae206, 2024 (venue from the search result; the README's citation field is empty) | https://github.com/hzi-bifo/AMR_benchmarking | Benchmarks Kover 2.0, PhenotypeSeeker 0.7.3, Seq2Geno2Pheno and Aytan-Aktug against ResFinder 4.0 (the rule baseline) and a "majority" baseline. Data: PATRIC genomes of 11 species including *E. coli*. Design: 10-fold nested CV with random, phylogeny-aware and homology-aware folds | It compares existing tools on PATRIC and proposes no lineage-aware model. It has no country or time axes, and no lineage-lookup baseline beyond "majority". *Their abstract's conclusion (ML better on close relatives, ResFinder on divergent genomes) is known to us only from the search snippet; read it before citing it* |
| Arcadia Science (Bell, McGeever, Patton, Sandler, York). *Predicting antimicrobial resistance phenotypes across 7,000 E. coli genomes* (opened: the analysis repository: README, `workflow/rules/filtering.smk`, `scripts/quadrupleton_stats.R`) | Arcadia pub, 2025; data on Zenodo 12692732 and 14364732 | https://github.com/Arcadia-Science/2024-predicting-antimicrobial-resistance | Genomic prediction / GWAS with GEMMA (linear) on three AMR phenotypes of the 7,000-genome dataset. Their pipeline: biallelic SNPs on ECOR-shared contigs; **removes "outlier samples" with > 500 synonymous quadrupleton alleles** before analysis; turns missing genotype calls into reference (`sed 's/\.\:\./0\:255\,0/g'`). The README names nonlinear models as their future work | Nonlinear models (their stated next step). The outlier-sample rule matters for us: we never removed those samples (see `docs/experiments/w0_high_locus_strains.md`). The SXT/aadA and gyrA × parC findings must be read in the pub itself (site blocked here) |
| NCBI. *AMRFinderPlus* (opened: README) | software; GitHub | https://github.com/ncbi/amr | Finds acquired resistance genes and point mutations in protein or assembled nucleotide sequences; the "plus" set adds stress, heat, biocide and virulence genes for curated organisms | The tool of W4/W5. Cite the paper given on its wiki ("How to cite") once opened |

## 2. Found by search, not opened (do not cite yet)

| work | venue, year | link | why it matters | check when reading |
|---|---|---|---|---|
| Yu, Wheeler, Barquist. *Biased sampling driven by bacterial population structure confounds machine learning prediction of antimicrobial resistance* | *PLOS Biology*, Dec 2025, doi 10.1371/journal.pbio.3003539 | https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003539 | The direct predecessor: lineage confounding inflates ML AMR scores; > 24,000 genomes, 5 pathogens, 6,740 models (per press coverage) | Which species, which model (task text says LightGBM only), their sampling designs, and exactly what they recommend (lineage-aware models? phylogenetic-placement baseline?) |
| Arcadia Science (York, Sandler et al.). *Creating a 7,000-strain E. coli genotype dataset with antimicrobial resistance phenotypes* | Arcadia pub, Aug 2024 | https://research.arcadiascience.com/pub/dataset-ecoli-amr-genotype-phenotype | Our data source (release 1) | How presence/absence was called (reads or assemblies), so we can judge the mixed-sample hypothesis; source collections |
| (authors to read) *Genomic and socioeconomic drivers of antimicrobial resistance forecast to 2050* | *Cell Genomics*, 2026 | https://pmc.ncbi.nlm.nih.gov/articles/PMC13347946/ | Uses group k-fold CV with folds grouped by PopPUNK cluster to stop population-structure leakage | Species, models, how much the grouped CV lowered scores |
| (authors to read) *Computational paradigms for antimicrobial resistance prediction: integrating multi-omics, structural modeling, and foundation AI systems* | *Briefings in Bioinformatics* 27(3) bbag219, 2026 | https://academic.oup.com/bib/article/27/3/bbag219/8675356 | Review; per the search snippet, it lists robustness under shift, calibration and uncertainty reporting as open barriers | Whether it covers lineage-aware evaluation; use it for the "state of the field" paragraph only |
| (authors to read) *Prediction of antimicrobial minimum inhibitory concentration from bacterial genomes using a scalable and interpretable machine learning approach* | *npj Antimicrobials and Resistance*, 2026 | https://pmc.ncbi.nlm.nih.gov/articles/PMC13530227/ | External-data test: F1 0.85 vs 0.82 for annotated determinants (search snippet) | Whether the external set is lineage-disjoint |
| (authors to read) *resLens: genomic language models to enhance antibiotic resistance gene detection* | *npj Antimicrobials and Resistance*, 2026 | https://pubmed.ncbi.nlm.nih.gov/42168341/ | Genomic language models for resistance-gene detection | Whether any held-out-lineage evaluation is reported |
| (authors to read) *KG-TRACE: a neuro-symbolic framework for mechanistic grounding in AMR prediction* | arXiv 2606.26179, 2026 | https://arxiv.org/abs/2606.26179 | Mechanism-grounded AMR prediction; relevant to W4 | Species, evaluation splits |
| Siddiqui, Tarannum. *Fusing sequence motifs and pan-genomic features: AMR prediction using an explainable lightweight 1D CNN–XGBoost ensemble* | arXiv 2509.23552, 2025 (already cited in the thesis) | https://arxiv.org/abs/2509.23552 | Giessen collection, 4 drugs, stratified 80/20 | Whether any lineage-aware split is used |
| Moat et al. *Machine learning methods to identify markers and predict antimicrobial resistance in Escherichia coli* | *Canadian Journal of Microbiology*, 2025 | https://cdnsciencepub.com/doi/full/10.1139/cjm-2024-0208 | 11-mer SVM / GBT / NN, 34 drugs; random 20% validation; per the snippet, the discussion names genomic context as a limit to generalisation | Their split design |
| Moradigaravand et al. *Prediction of antibiotic resistance in Escherichia coli from large-scale pan-genome data* | *PLOS Computational Biology*, 2018 | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006258 | Older anchor: 1,936 *E. coli*, GBT best, a held-out clade test | The clade-holdout result per drug |
| Anojulu, Puttagunta. Conformal prediction under clade shift (TB) | NeurIPS 2026 workshop (per the task) | not found by two searches | The task says it shows conformal prediction failing under clade shift; we must not make conformal prediction a main claim | Find the paper (title, workshop) before citing |

## 3. What the related-work section will argue (draft, to check against §2 once read)

1. Lineage confounding in genomic AMR prediction is diagnosed: Yu et al. 2025 across species; Hu et
   al. 2024 with phylogeny-aware folds. What is missing is a model and an evaluation built around it.
2. Benchmarks so far use random, phylogeny- or homology-aware folds. None adds country, time and
   study axes on one collection, or reports a lineage-lookup baseline that every model must beat.
   *Check against Hu et al. and Yu et al. before claiming.*
3. Rule-based catalogues (ResFinder, AMRFinderPlus) are the honest reference under shift (Hu et al.).
   We report them on every axis (W5).
4. Arcadia's own analysis is linear and lists nonlinear models as future work. We are that
   nonlinear model on the same data, and we answer their open mechanism questions (W4).

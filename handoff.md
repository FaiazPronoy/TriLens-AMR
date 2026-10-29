# TriLens-AMR — full handoff for the next agent

Read this whole file before doing anything. It replaces the chat history. Last updated 2026-10-29.

---

## 0. Who you are working with and how

- User: Faiaz Abrar Zaman Pronoy ("Zishan"), BRAC University CSE graduate (2026). This paper is
  built on his undergraduate thesis "TriLens-MoE-Stack". The supervisor will read the paper, so the
  thesis content **must not be dropped** — it is merged into the paper (see section 3).
- **Language:** reply in Banglish (Bangla in Latin script), technical terms in English. When he
  says "shortly bolo", keep it short.
- **File names:** short and plain. No numbers, no dates, not his name.
- **Rules he set for the work (keep all of them):**
  1. No hallucination. Every number in the paper must come from a result file. Every reference
     must be checked against a publisher/PubMed record.
  2. Every new analysis is pre-registered: write `docs/experiments/wN_*.md` (question, design,
     endpoint, pass rule) and **commit it before reading any data**. Amendments are dated and
     committed before the data they concern.
  3. Every Kaggle notebook is simulated/tested locally before he runs it.
  4. If you cannot open a link or file, **tell him** — he will forward it. Never skip a task
     because a link is inaccessible.
  5. Report failed outcomes honestly. Do not move a threshold or endpoint after seeing results.
- **Security / operations:**
  - Never print or share tokens.
  - Never run the same notebook from two Kaggle accounts at the same time.
  - Do not push to `main` carelessly; no force-push without his OK.
  - Do not delete his old folders.
  - Every notebook needs a `GITHUB_TOKEN` (Kaggle secret) that works for his repo.

## 1. Where everything is

- Repo: `https://github.com/FaiazPronoy/TriLens-AMR` (old URL `.../trilens-amr` redirects).
  Local clone in this session: `/home/claude/trilens-amr`.
- Working branch: **`extension`**. Latest commit: `77428c1` (merged final paper), pushed.
- Result branches (each Kaggle run uploads to its own branch): `results/analysis` (re-run of all
  thesis numbers — the source of truth for thesis numbers), `results/trilens_random`,
  `results/trilens_splits`, `results/standard_random`, `results/standard_splits`,
  `results/classical`, `results/shortcut_floor` (metadata floor), `results/w8_external`,
  `results/w4_amrfinder`, `results/ext_probe`, `results/species_probe`, `results/lineage_seeds`,
  `results/pa_fix`, `results/shuffle_test`, `results/arg_probe`, `results/w1_gene_sets`,
  `results/w2_snp_knn`, `results/data_check`.
- Paper (final): `paper/nargab/`
  - `main.tex` → `\input`s `sections/abstract, intro, methods_data, methods_models,
    results_internal, results_prereg, results_arch, discussion, backmatter, tables`.
  - `supplement.tex` (compiles on its own; Supplementary Methods + Supplementary Results).
  - `refs.bib` (all verified; entries with `note = {check ...}` need the author list confirmed on
    the publisher page before submission).
  - Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (same for
    `supplement`). Current: main.pdf 31 pages, supplement.pdf 7 pages, 0 undefined refs.
  - `.gitignore` ignores aux/log/pdf.
- Old thesis paper (IEEE format, ~33k words): `paper/journal/report.tex`. Thesis bib:
  `thesis/bibliography/references.bib`. **Its numbers are outdated** — use the re-run values
  below, never the thesis text.
- Pre-registration docs: `docs/experiments/w6 … w14_*.md` (each has a dated Result section).
  Earlier exploratory diagnostics: `w0–w5`, `s6`, `data_inventory.md`.
- Code: `code/scripts/diagnostics/` (coverage_and_rule.py, external_robustness.py,
  shortcut_mechanism.py, mic_check.py, campylobacter.py, clinical_and_linear.py),
  `code/src/mechanism/species_probe.py`, `code/src/experiments/run_lineage_seeds.py`.
  Tests: 341 passing (`pytest` in `code/`).
- Committed inputs: `code/results/data_inventory/ncbi_ast_ecoli.tsv.gz` (MIC export),
  `eucast_v16_1_enterobacterales.csv`, `strain_table.csv`, `labels16.csv`.
- Kaggle: `kaggle/make_nbs.py` generates notebooks (STAGES with needs/writes/step markers).
  **If markers change, the user must re-import the notebook** — the cells embed the skip markers.
  Periodic upload only happens when a line is printed.

## 2. Kaggle accounts

- A: the user's own — CPU only (GPU quota used up).
- B: Sinthia.
- C: Purnata — has GPU quota; uses the user's fine-grained token as a Kaggle secret.
- **All experiments are finished.** No notebook needs to run now.

## 3. What the paper is (story + structure)

**Target journal: NAR Genomics and Bioinformatics (NARGAB, OUP).** Backup: Microbial Genomics.
Another option: Computers in Biology and Medicine (Elsevier). For NARGAB submission, paste the
sections into the OUP Overleaf template ("Modern Large"); abstract must be ≤ 200 words (check);
optional graphical abstract 5:2, ≥ 127×50 mm.

**Title:** "Genomic antimicrobial resistance prediction under lineage and cohort shift: a three-view
model, the shortcuts it reveals, and when a learned model beats a determinant rule". TriLens is
named in the body, not the title (his choice).

**One-line story:** We built a strong three-view stacked model (TriLens) on 6,962 *E. coli*
genomes, then used it with simpler models and a determinant rule to show what random-split AMR
benchmarks actually measure. Three regimes: (1) determinants nearly sufficient → a rule is enough;
(2) determinants many/diffuse → a sparse linear model on AMRFinderPlus calls beats the rule;
(3) determinants missing from the features → models lean on proxies (lineage, assembly size)
that don't transfer.

**Sections:** Abstract; Introduction (7 contributions); Related work (MoE/stacking/attention,
evaluation & population structure, 16-study table); Materials and methods (data, features incl.
P/A orientation repair, integration checks, TriLens summary, simpler models, splits, Giessen,
metrics, near-duplicate audit, external cohort, MIC harmonisation, second species, clinical error
rates, pre-registration); Results (random split + metadata floor; components & baselines; lineage
holdout with prevalence correction; clinical cost; Giessen; determinant rescue; assembly proxy;
external transfer; linear vs rule; very major errors; architecture: seed stability, gate,
attention, strain-conditioning ablation, attribution disagreement); Discussion + Limitations;
Conclusion; Data availability; tables. Supplement: full architecture equations, P/A repair, SNP
damaged-block repair, variant filtering, threshold table, dedup per drug, split integrity,
phylogroup composition.

**Figures:** all are blank placeholder boxes (`\figbox{label}{height}{what}{data file}`); the user
will draw them in LaTeX himself later. Each box says what to plot and which file feeds it.
Main: Fig 1 study design, 2 metadata floor, 3 two groups, 4 rescue, 5 assembly proxy, 6 external,
7 model vs rule, 8 clinical. Supplement: Fig S1 co-resistance.

## 4. Verified numbers (use these; all are in the paper)

Data: 6,962 strains; 7 drugs AMP CIP AMC SXT GEN CTX CAZ; intermediate → resistant. Coverage
49.1–86.7%, prevalence 0.118 (GEN)–0.501 (AMP). 91% UK+Norway, years 2001–2017. ST typed for
6,487 (93.2%), 458 STs, ST73 = 884. Variants 2,451,184 → 535,439; 5,000 genes / 262,225 SNPs.
P/A: 32,441 loci, 10,511 flipped in the repair; gyrA/parC/rpoB/recA prevalence 1.0 after.
Determinant calls for 5,072/6,962 (72.9%).

Three splits (mean AP, re-run): random 0.850 (classical stack alone 0.846), phylogroup-stratified
0.856, ST holdout 0.645. Ablation: branches 0.780 → +gate 0.796 → full 0.850. Baselines: phylogroup
only 0.291; TabNet 0.667 / 0.341; PCA-XGB 0.803; RF 0.781/0.493; LR 0.792/0.440; LGBM 0.833/0.552;
XGB gene 0.838/0.563; XGB χ² 0.842/0.622. Neural alone holdout 0.520; classical stack holdout 0.622.
Prevalence-matched holdout 0.732 (true gap 0.118 vs raw 0.205); 3 drugs lose 0.014–0.026,
4 drugs (CTX CAZ AMC GEN) lose 0.152–0.239. VME random 0.241 → holdout 0.559. Threshold tuning:
F2 → VME 0.133 / ME 0.141; ME-cap → VME 0.308. Near-dup: 47 strains, −0.005. Giessen classical
retrain mean 0.769 vs 0.849, retention 89.6% (prevalence-inflated). Gate: Lens B 0.553–0.807;
CIP Lens A 0.081 (lowest). Attention recovered 0 known AMR genes; SHAP recovered 8 (gyrA, parC,
emrB, mdtC, mdtD, mdtO, cusA, a β-lactamase). Annotation 836/1,143 (73.1%). U_ST attention 0.430,
SHAP 0.401, random 0.539; U_res attention 0.021 vs SHAP 0.048. Strain conditioning (re-run):
neural holdout +0.009, interpretability worse → reported as a trade, and as a correction of the
thesis (+0.035). Shuffle test: CIP 0.775 → 0.632 (Arcadia), drop 0.180 on Giessen.

Metadata floor (`results/shortcut_floor`): random all-metadata AP 0.705, assembly-only 0.601,
epi-only 0.516; holdout 0.297; epi nAP across country/time 0.000–0.056; assembly nAP 0.097–0.232.

Pre-registered (nAP, cluster bootstrap):
- W6 rescue +0.276 [0.178, 0.328] (0.450 → 0.726), CIP −0.004 — pass
- W7 within-ST ρ 0.325, concordance 0.552–0.676 — pass
- W8 external all-calls − assembly arm +0.762 [0.676, 0.804], 10,232 isolates / 62 BioProjects — pass
- W9 seeds +0.028, +0.020, +0.018 (mean 0.022, SD 0.005) — pass
- W10-A coverage check: 0.641 uncalled vs 0.449 called — **fail**
- W10-B model − rule +0.122 [0.075, 0.197] — pass; W10-C UK +0.256, Norway +0.308
- W11 leave-one-BioProject-out 0.723–0.777, min CI 0.643 — pass
- W12 study conditioning 0.613 → 0.592; N50 0.383–0.490; ρ −0.185 — pass
- W13-a Campylobacter model − rule −0.006 [−0.017, 0.004] — **fail**; W13-b length concordance
  0.595 (TET 0.832) — pass; W13-c breakpoint ratio 1.64/1.87 vs threshold 2 — **fail**;
  MIC/EUCAST margin +0.123 [0.081, 0.174]; same-isolate +0.064 → +0.101 (exploratory)
- W14-B linear vs XGB internal −0.011 [−0.026, 0.007] — pass (linear suffices); external linear
  beats XGB by 0.038 (inconclusive by margin); linear − rule +0.104 [0.060, 0.166]
- W14-A clinical (EUCAST, thr 0.5): CA 0.914/0.938/0.942, ME 2.7/1.7/2.7%, VME 25.1/18.7/14.4%
  (XGB/linear/rule); no arm meets FDA VME < 7.5%

## 5. What is still left (in order)

1. **Fill the remaining `\todo{}` markers** (grep `todo{` in `paper/nargab`):
   authors/affiliations/ORCID; acknowledgements; CRediT roles; funding; conflict of interest;
   Zenodo DOI for a tagged release of the repo (and cite it in Data availability and
   Pre-registration); Campylobacter cohort row (geography, source type, years — from the
   species_probe output `per_species.json`) and Giessen years in Table "Cohorts"; all 9 figure
   captions; supplement per-drug tables for W6–W14 (or cite repo paths) and attach the
   pre-registration documents.
2. **Check bib entries marked `note = {check ...}`** (author lists: cheon2026, bell2025arcadia,
   feldgarden2021, ren2022giessen, orcales2024, siddiqui2025 name spelling, wang2025hpylori,
   paredes2025mtb) and remove the notes. Confirm ST assignment method in the Arcadia release.
3. **Count abstract words** (must be ≤ 200 for NARGAB; currently close — trim if over).
4. User draws figures in LaTeX → replace each `\figbox` with the real figure.
5. Port into the OUP NARGAB Overleaf template; write cover letter; suggest reviewers.
6. Optional, only if a reviewer asks: multi-seed repeat of the whole protocol; masked (not
   reference-imputed) missing calls; ClermonTyping phylogroups. Any new analysis must be
   pre-registered first (rule 2).

## 6. Things not to break

- Do not reintroduce thesis-only numbers (old 0.849, 0.634, 0.723, 0.472, +0.035, U_ST 0.357,
  U_res 0.0419, VME 0.249). The re-run values above replace them.
- The thesis-era analyses (three splits, ablations, Giessen) are labelled **exploratory, not
  pre-registered** in the paper. Keep that distinction.
- Keep negative results in the paper (W10-A, W13-a, W13-c, gate routing, attention, VME).
- The paper avoids repeated sentences on purpose; check for repetition after edits.

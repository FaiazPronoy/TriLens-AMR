# Progress of the journal extension

## Session 2 (26 October 2026, working alone; no Kaggle runs)

Branch `extension-session2` (from `extension` at `7b2a398`), one pull request into `extension`.
`main`, `results/*` and `outputs-*` were not touched. Tests: `pytest code/tests`, **140 passed**
(90 on the base commit; 50 new this session).

### Done

| step | what | where |
|---|---|---|
| 0 | `extension` checked out; the zip parts are the same files as session 1 (sha256 `4ed65c69…5b7` checked then), data in `code/data/`; rebuilt P/A matrix from `outputs-pa_fix` installed in place of the old one (old kept as `*.minority_state.pkl`, not committed) | — |
| 1 | **W2 P/A pilot redone on the rebuilt matrix**; the 13 strains without a profile dropped from every partition; what changed is reported | `docs/experiments/w2_lineage_lookup.md` (last section), `code/results/w2_lineage_lookup_pa_pilot/` (old files in `superseded_minority_state_matrix/`) |
| 2 | **W0 addition: high-locus strains** described, with the handling pre-registered before any use | `docs/experiments/w0_high_locus_strains.md`, `code/scripts/inventory/high_locus_strains.py`, `code/results/data_inventory/high_locus_*` |
| 3 | **Pre-registrations** of S6 (leave one study out), the S5 cut-off check, W3, W4, W5, committed before any of them runs | `docs/experiments/s6_leave_one_study_out.md`, `w3_lineage_vs_resistance.md`, `w4_mechanism.md`, `w5_rules.md` |
| 4 | **Literature**, incomplete: only GitHub sources could be opened (below) | `docs/literature.md` |
| 5 | **W4 step 1 code**: AMRFinderPlus run, drug mapping, P/A mapping, tested on NCBI's own example output. AMRFinderPlus 4.2.7 installs here, but its database cannot be downloaded (NCBI FTP refused), so it runs as a Kaggle notebook | `code/src/mechanism/amrfinder.py`, `code/scripts/install_amrfinder.sh`, `code/tests/test_amrfinder_mapping.py` |
| 6 | **Kaggle stages** `w1_gene_sets`, `w1_classical`, `w2_snp_knn`, `w4_amrfinder` in `kaggle/make_nbs.py`, notebooks regenerated. Splits are rebuilt in the notebook from `code/results/splits/*.json`. The stages clone `extension`. Supporting code: `src/splits/from_json.py`, `src/experiments/run_split_pipeline.py`, `src/baselines/snp_distance.py` | `kaggle/notebooks/w1_*.ipynb`, `w2_snp_knn.ipynb`, `w4_amrfinder.ipynb`, `kaggle/README.md`, `code/tests/test_extension_notebooks.py` |
| 7 | **W3 support**: TriLens returns `z_A`, `z_B`, `z_C`, `z_fused`; every TriLens run writes `representations.npz` (last fold's model, trained and untrained, on train/val/test). Checked by a test and by a 1-epoch CPU smoke run of the real K-fold code | `code/src/models/trilens.py`, `code/src/experiments/run_trilens_v10.py` (+ `run_trilens_mlst.py`, `run_trilens_t7_clean_gene_set.py`), `code/tests/test_representations.py` |

### Results this session

1. **W2 on the rebuilt matrix.**
   - Distances between strains with a profile are identical (all pairs checked), as expected.
   - The 13 all-zero strains had been **near neighbours** in the old matrix: closer to everyone than
     real strains are to each other. On the ST holdout, 10% of CTX test strains had one among their
     10 nearest.
   - kNN changed by more than 0.01 in 3 of 38 cells (S3 CTX 0.212 → 0.340); ST lookup and phylogroup
     changed by at most 0.006.
   - **Conclusions unchanged:** kNN averages 0.684 on the random split and 0.310 on the ST holdout. The
     classical arm beats the best lookup on all 14 cells of S2/S3 (Holm p ≤ 0.009).
   - **S3 CTX is unstable in k:** validation is flat, test ranges 0.159–0.385. It is marked as noisy.
2. **High-locus strains.**
   - 61 strains have more than 14,506 loci (median + 4 robust SD).
   - 48 are Norway, 27 of them in one SRA block (9.1% of that block flagged, against 0.8% elsewhere
     in Norway), mostly collected 2007–2009. 10 come from PMID 30127495.
   - Assemblies are of normal size (5.11 vs 5.09 Mb). The excess is shell accessory genes (2,304 vs
     1,115 CDS loci at 15–50% prevalence).
   - This is consistent with read-level mixed samples, but not proven.
   - Pre-registered handling: keep them in the primary analyses, and report a sensitivity analysis
     without them.
3. **Arcadia removes outlier samples; we never did.** Their pipeline drops samples with > 500
   synonymous quadrupleton alleles before every analysis (`filtering.smk`). Recomputing their list
   and comparing it with our 61 is added to the W0 check (logged as a deviation, before any use).

### Blocked

| what | why | what is needed |
|---|---|---|
| Literature: opening papers | this environment's network policy refuses every journal, preprint and literature-API host (PLOS, OUP, bioRxiv, arXiv, PMC, Europe PMC, Crossref, OpenAlex, Semantic Scholar, the Arcadia pub site). Only GitHub works | read the papers in `docs/literature.md` §2 from a browser, or allow those hosts in the environment's network settings; then I move each to §1 |
| AMRFinderPlus here | database host `ftp.ncbi.nlm.nih.gov` refused | runs on Kaggle (`w4_amrfinder`), or allow the host |
| End-to-end test of `w1_gene_sets` / `w1_classical` / `w2_snp_knn` | they need the SNP matrix (Kaggle only) | their first Kaggle run is the test; each step prints and fails loudly |

### Next: Kaggle, in this order

None of these needs the re-run to have finished, except `w1_classical` (it needs a GPU slot).

| order | account | notebook | accelerator | est. time | before running |
|---|---|---|---|---|---|
| 1 | B | `w2_snp_knn` | CPU | about 1 h | nothing |
| 2 | B | `w1_gene_sets` | CPU | about 12 h (resumes across sessions) | nothing |
| 3 | B | `w4_amrfinder` | CPU, Internet on | about 0.5 h | add `raw/pangenome/pangenome_cds.fa` (from `small_data.zip`) to `trilens-data` |
| 4 | A | `w1_classical` | GPU T4 x2 | about 4.5 h | `w1_gene_sets` finished; a free GPU slot after the re-run's GPU notebooks |

CPU notebooks cost no GPU quota and can run while the re-run uses the GPUs.

Then, in code (next session):

1. `w0_encoding` (VCF part: missing calls, heterozygous calls, Arcadia's outlier list);
2. `w1_trilens` (the neural arm on S4a, S4b, S5);
3. the S6 split builder and its notebooks;
4. W4 steps 2–5 (they need the re-run's attention/SHAP and `w4_amrfinder`);
5. W5.

### Open questions for the owner

1. **Literature access.** Can you open the §2 papers of `docs/literature.md` (or allow those hosts)? I
   will not cite any of them before they are read.
2. **High-locus strains.** The pre-registered default keeps them in the primary analysis. If the VCF
   check calls some of them mixed samples, should they be **removed before the W1/W3 GPU runs**?
   It is your decision, and it must be made before those runs.
3. **S3 CTX kNN instability.** Keep the pre-registered "smallest k with the best validation PR-AUC",
   or pre-register a tie rule now? For example: when validation PR-AUC varies by < 0.01 across k,
   take k = 20. Either way the cell stays marked as noisy.
4. **AMRFinderPlus intrinsic-gene rule.** Before any data were scored, I added "gene determinants
   carried by ≥ 95% of strains are not counted" (the chromosomal ampC `blaEC` would otherwise call
   every strain AMP-resistant). OK?
5. **Gene ranking per split uses only that split's usable antibiotics** (the lineage holdout ranked
   on all 7). S4b has no AMC training labels anyway; S4a's 12 UK SXT strains would add noise. OK?
6. Should `w0_encoding` (needs the raw-download dataset with the VCF) be the next notebook I build?

### Decisions I made alone, and why

| decision | why |
|---|---|
| The 13 strains without a P/A profile are **dropped** (not flagged) from every partition of P/A-based analyses, including the ST-lookup and phylogroup rows | keeps every comparison on identical strains; they have no P/A data to be scored on |
| The old pilot kept as `superseded_minority_state_matrix/` | the history and the difference stay checkable |
| High-locus rule: robust z > 4 (61 strains) | a common, label-free outlier rule; 5 and 6 SD counts reported too |
| Literature file split into "opened" and "found, not opened" | the task's rule is to cite only opened papers, and only GitHub sources could be opened |
| `blaEC`-type intrinsic genes not counted (W4/W5) | found in NCBI's documented output before any scoring; logged under Deviations in both docs |
| W3 probe on the **last fold's** model; untrained reference on a forked RNG; train rows from the fold's own datasets | fold models do not share a representation space; attention already uses the last fold; the forked RNG leaves training bit-for-bit unchanged; no extra 2 GB copy |
| New splits run through a thin wrapper around the reviewed ST-holdout scripts (`run_split_pipeline.py`) | same logic as the reviewed code. Two constants were added to those scripts (`SPLIT_KEY`, `REQUIRE_ST_DISJOINT`); their defaults leave the ST-holdout run unchanged |
| Gene ranking per split on its usable antibiotics only | the minimum-count rule; question 5 above |
| `load_strain_table` falls back to the committed `strain_table.csv` when the raw metadata is absent, checked label for label | the Kaggle dataset has no raw metadata; this way nothing extra has to be uploaded |
| The setup cell (all notebooks on `extension`) links raw files if a dataset has them, and sets `TRILENS_REPO_RESULTS` | needed by W4 and the split rebuild; harmless where the files are absent |
| The four new stages clone `extension` | their code is not on `main`; the re-run notebooks still clone `main` |
| SNP distance = binary non-reference indicator | as pre-registered. My first draft compared code values instead; I caught the mismatch before anything ran |

---


# Session 1 (history)

Last update: 26 October 2026, night run (working alone; the owner was asleep). Branch
`extension-w0-w2`, pull request into `extension`. `main`, `results/*` and `outputs-*` were not touched.

## Session 1: done

| step | what | where |
|---|---|---|
| 1 | `extension` checked out; `small_data.zip` joined, sha256 checked (`4ed65c69…5b7`, matches), unzipped into `code/data/` (git-ignored, not committed); §0 read | — |
| 2 | **W0 data inventory**, complete | `docs/experiments/data_inventory.md`, `code/results/data_inventory/`, `code/scripts/inventory/data_inventory.py`, `code/src/data/metadata.py` |
| 3 | **Plan** for W1–W6: splits, models, metrics, pass/fail, notebooks, accounts, hours, order | `docs/PLAN.md` |
| 4 | **Pre-registration** of W1 and W2, committed before any run on the new splits | `docs/experiments/w1_five_axis.md`, `docs/experiments/w2_lineage_lookup.md` |
| 5 | **Country and time split builders** (S4a UK → Norway, S4b Norway → UK, S5 Norway ≤ 2013 → 2014–2017) | `code/src/splits/country_time.py`; indices in `code/results/splits/*.json` |
| 5 | **Lineage-lookup baselines** (ST lookup, phylogroup, kNN on P/A) and the shared statistics helpers (bootstrap CI, paired bootstrap, Holm, prevalence matching, VME/ME) | `code/src/baselines/lineage_lookup.py`, `code/src/evaluation/shift.py`, `code/src/experiments/run_lineage_lookup.py` |
| 5 | **Tests**: 15 new (63 in total, all pass, 9 s) | `code/tests/test_country_time_splits.py`, `code/tests/test_lineage_lookup.py` |
| 5 | **W2 P/A-only pilot** on all six splits, and the classical arm against the best lookup on S2/S3 | `code/results/w2_lineage_lookup_pa_pilot/`, results section of `docs/experiments/w2_lineage_lookup.md` |
| — | Two new known issues (12, 13) for the encoding findings | `code/docs/KNOWN_ISSUES.md` |

## Session 1: main findings

1. **The presence/absence matrix marks the minority state, not presence** (KNOWN_ISSUES 12).
   - No column is "present" in more than 50.3% of strains.
   - 2,642 of 2,652 ECOR core genes are "present" in under 1% of strains; gyrA and parC in none.
   - Predictive numbers are not affected (trees, LogReg, Hamming kNN do not care). Lens B and every
     "attended gene" interpretation are.
   - Strong hypothesis; needs the raw `.bim`/`.bed` on Kaggle to confirm.
2. **SNP calls code missing as reference** (Arcadia convention, KNOWN_ISSUES 16; renumbered when merged with main). In accessory
   genes, gene absence and reference allele are the same value.
3. **Country = study; host is unusable.** UK (3,387) and Norway (2,964) hold 91% of strains.
   - Norway has no AMC, the UK almost no SXT, so the country split works for AMP, CIP, GEN, CTX, CAZ.
   - 94% of Norway strains have an ST also seen in the UK, so the country axis tests geography and
     study, not new lineages.
4. **Time works only inside Norway:** train 2002–2013, test 2014–2017. This passes for AMP, CIP, SXT, GEN;
   CTX and CAZ fall in the secondary tier. A pooled time split would mostly be a country split.
5. **W2 pilot (P/A distance):**
   - Relatives alone (kNN) reach mean PR-AUC 0.682 on the random split, but 0.294 on the ST holdout.
   - The classical arm beats the best lookup on every antibiotic of S2 and S3 (14 of 14, Holm p ≤ 0.009).
     On S3, CIP gains +0.674 and GEN +0.139 [+0.028, +0.268].
   - Pilot only: P/A distance, and pre-re-run classical predictions.

## Session 1: not done, and why

| item | why | next |
|---|---|---|
| W1 pilot (classical XGB-χ² on S5, CIP) | needs the SNP matrix (Kaggle only); not part of tonight's step list | first step of `w1_gene_sets` / `w1_classical` |
| New notebooks in `kaggle/make_nbs.py` | tonight's steps did not include them; PLAN says notebooks are added once their pre-registration is committed. W1 and W2 now are, so they can be added next | add `w0_encoding`, `w1_gene_sets`, `w2_snp_knn` first |
| Literature search for the introduction | not part of tonight's steps | before writing §6 of the task |
| W3–W6 pre-registrations | after W1/W2, per the task | — |

## Session 1: next (in order)

1. **`w0_encoding` on Kaggle** (CPU, account B, minutes for the orientation part, about 3 h with the VCF).
   It settles finding 1, which must be settled before any new TriLens GPU run.
2. Owner uploads to the Kaggle dataset:
   - `raw/strains_metadata.csv`, `raw/pangenome/pangenome_cds.fa`, `raw/pangenome/gene_presence_absence.csv`;
   - `processed/cds_eggNog.tsv`, `processed/genes.gff`;
   - the three new split files: `country_uk_to_norway_split.pkl`, `country_norway_to_uk_split.pkl`,
     `time_norway_2013_split.pkl`. They are built by `python -m src.splits.country_time`, or rebuilt
     from `code/results/splits/*.json`.
3. When the re-run is finished: `w1_gene_sets` (CPU, B) → `w1_classical` (GPU, A) → `w1_trilens`
   (GPU, A); `w2_snp_knn` (CPU, B). Full order and hours: `docs/PLAN.md`.

## Review of this PR (26 October 2026, reviewing session)

Merged into `extension`. Checked: the split builders (disjoint, country/year rules asserted, no
test strain in any fitting step), the lookup baselines (k and the best lookup chosen on validation
only, test touched once), the metadata join (by genome name, checked against the SRA accession;
the phylogroup index order matches `phylogroup_to_int`). No leakage found.

Since this PR was written, on `main` (merged into `extension`):
- **Finding 1 is confirmed and fixed** by the Kaggle notebook `pa_fix` (`scripts/repair_gene_pa.py`):
  `.bim` letter `2` = present, 10,511 loci flipped, gyrA/parC/rpoB/recA now 100% present. The TriLens
  notebooks and `analysis` take the rebuilt matrix from the release `outputs-pa_fix`. `w0_encoding`
  is no longer needed for the orientation; the VCF part (missing calls, issue 16) still is.
- The rebuilt matrix has 9,569 to 18,620 loci per strain (mean 11,846). Look at the strains at the top
  of that range in W0 (possible mixed samples) and report how many there are.
- The P/A Hamming distance of the W2 pilot is unchanged by the flip for all matched strains (both
  sides of every pair flip the same columns). The 13 unmatched strains (all zeros) are not; recompute
  the pilot once on the rebuilt matrix and drop or flag those 13.
- The shuffle test now saves each of its four runs separately (`EXP3_ONLY`).

### Answers to the open questions (defaults set in review; the owner can change any of them)

1. Done: re-oriented on `main` before any new TriLens run (above).
2. Keep the minimum counts. Say in the paper that they were set after seeing the counts and before
   any model ran.
3. Keep Norway, cut-off 2013. Add 2012 and 2014 as a pre-registered sensitivity check (lookup and
   classical only, CPU).
4. Yes; report the 611 strains left out and their countries.
5. Keep 15% (the same rule on every split). Mark S5 thresholds and k as noisy where validation has
   fewer than 20 resistant strains; do not tune thresholds there.
6. Not known to the owner. Look up the source publications (PMIDs in the metadata) and describe the
   sampling of each collection in the paper; flag outbreak or ESBL-enriched sampling where stated.
7. Yes, as a secondary axis S6 (leave one study out): pre-register it, run lookup + classical (cheap);
   TriLens only if GPU budget is left after W1/W3.
8. Yes, on `extension`.

## Session 1: open questions (answered in the review above)

1. **P/A orientation (finding 1).** After the Kaggle check confirms it, should `X_pa` be re-oriented
   (1 = present) on `main`?
   - This changes Lens B, so the current re-run would describe the old encoding, and the delivered
     model would need one more run.
   - Alternative: keep the model as it is, re-orient only for interpretation (W3/W4), and state it in
     the paper.
   - My recommendation: re-orient before the W1/W3 GPU runs, because W3's "what does Lens B attend
     to" is the main contribution and cannot be read on the flipped matrix.
2. **Minimum counts for a split side:**
   - test ≥ 50 R and ≥ 50 S; train ≥ 100 R and ≥ 100 S;
   - a secondary tier when train R is 50–99.
   - I set these after seeing the counts but before any model ran. OK, or other numbers?
3. **Time split:** Norway only, cut-off 2013. OK? (2012 and 2014 give the same antibiotics; I took the middle.)
4. **Country split** leaves out the 611 strains of other countries entirely. OK?
5. **Validation sets on S5 are small** (e.g. CAZ 7 resistant, CIP 16 in validation). k and the tuned
   thresholds will be noisy there. Keep 15%, raise it to 25%, or use 5-fold CV inside the training years?
6. **Norway 2012 has 39% GEN-R** (78/201; other years 0–10%), and PMID 34485958 (England, 1,015
   strains) has 33% CTX-R. Do you know these collections (outbreak, ESBL-enriched sampling)? It
   matters for the country axis and for the paper's description of the data.
7. **Leave-one-study-out** as an extra axis (Norway / 28720578 / 34485958 for CIP, GEN, CTX, CAZ)?
   It is not in the task; I only noted it as possible.
8. W3 needs the TriLens runs to save each lens' representation `z`. This is a small change to
   `run_trilens_v10.py`, on a branch before W3. OK to make it on `extension`?

## Session 1: decisions I made alone, and why

| decision | why |
|---|---|
| "England" merged into "United Kingdom" | England is part of the UK; keeping it apart would split one country in two |
| Host dropped as an axis | 24 non-human strains |
| Pooled and UK-only time splits rejected | pooled = mostly country (recent side is Norway/USA/Thailand); UK-only has 1,439 strains and passes for 2 drugs |
| Validation = 15%, seed 42, stratified on AMP with "not tested" as a stratum | the convention of the existing random split |
| ST lookup smoothing α = 1; fallback ST → phylogroup → prior; k grid {1, 3, 5, 10, 20, 50}, smallest k on ties | fixed in the pre-registration before running, not tuned |
| kNN on P/A uses Hamming distance | invariant to the orientation problem, so valid before it is fixed |
| Pilot comparison uses the classical predictions of the `outputs-classical` release | S2/S3 gene sets were not damaged; the comparison is labelled as a pilot and is replaced by the re-run |
| `code/results/data_inventory/strain_table.csv` committed | per-strain country/year/host/study/ST/labels derived from public Arcadia metadata; genome names and SRA accessions dropped; 468 KB |
| Work on branch `extension-w0-w2`, one PR into `extension` | the task asks for a PR into `extension`; `extension` itself is left for the owner to merge into |
| Fixed one wording error in the W2 pre-registration before any run (the phylogroup-LR note) | it was a factual error, not a rule change; committed separately and visible in the history |

## Session 1: notes on honesty

- Every number in the docs comes from a file in `code/results/` (`data_inventory/`, `splits/`,
  `w2_lineage_lookup_pa_pilot/`).
- The W2 pilot ran with the pre-registered settings and was not re-run with other settings. The one
  code change between the CIP step and the full run (NaN instead of a warning when prevalence matching
  has an empty side) is listed in the W2 doc.
- kNN was best on validation for CIP on S3, but scored below the ST lookup's fallback on test
  (0.242 vs 0.309). It is reported as is: "best lookup" is chosen on validation, as pre-registered.

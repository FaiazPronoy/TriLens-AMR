# Progress of the journal extension

Last update: 26 October 2026, night run (working alone; the owner was asleep). Branch
`extension-w0-w2`, pull request into `extension`. `main`, `results/*` and `outputs-*` were not touched.

## Done tonight

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

## Main findings tonight

1. **The presence/absence matrix marks the minority state, not presence** (KNOWN_ISSUES 12).
   - No column is "present" in more than 50.3% of strains.
   - 2,642 of 2,652 ECOR core genes are "present" in under 1% of strains; gyrA and parC in none.
   - Predictive numbers are not affected (trees, LogReg, Hamming kNN do not care). Lens B and every
     "attended gene" interpretation are.
   - Strong hypothesis; needs the raw `.bim`/`.bed` on Kaggle to confirm.
2. **SNP calls code missing as reference** (Arcadia convention, KNOWN_ISSUES 13). In accessory
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

## Not done, and why

| item | why | next |
|---|---|---|
| W1 pilot (classical XGB-χ² on S5, CIP) | needs the SNP matrix (Kaggle only); not part of tonight's step list | first step of `w1_gene_sets` / `w1_classical` |
| New notebooks in `kaggle/make_nbs.py` | tonight's steps did not include them; PLAN says notebooks are added once their pre-registration is committed. W1 and W2 now are, so they can be added next | add `w0_encoding`, `w1_gene_sets`, `w2_snp_knn` first |
| Literature search for the introduction | not part of tonight's steps | before writing §6 of the task |
| W3–W6 pre-registrations | after W1/W2, per the task | — |

## Next (in order)

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
  is no longer needed for the orientation; the VCF part (missing calls, issue 13) still is.
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

## Open questions for the owner

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

## Decisions I made alone, and why

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

## Notes on honesty

- Every number in the docs comes from a file in `code/results/` (`data_inventory/`, `splits/`,
  `w2_lineage_lookup_pa_pilot/`).
- The W2 pilot ran with the pre-registered settings and was not re-run with other settings. The one
  code change between the CIP step and the full run (NaN instead of a warning when prevalence matching
  has an empty side) is listed in the W2 doc.
- kNN was best on validation for CIP on S3, but scored below the ST lookup's fallback on test
  (0.242 vs 0.309). It is reported as is: "best lookup" is chosen on validation, as pre-registered.

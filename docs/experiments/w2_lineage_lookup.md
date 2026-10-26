# W2. Lineage-lookup baselines: pre-registration

**Status:** pre-registered. Implementation (`code/src/baselines/lineage_lookup.py`) and the P/A-only CPU
pilot follow this commit. The SNP-distance kNN runs on Kaggle (`w2_snp_knn`).

## Question

How much of each model's score could be obtained by knowing only **who a strain's relatives are**? On
every split of W1, what does TriLens (and each of its parts) add above the best of these lookups?

## Baselines

All three use only training-side strains labelled for the antibiotic. None sees a test label.

### (a) ST lookup

Score of a test strain = resistant fraction of the training strains with the same Achtman ST,
smoothed towards the training prevalence p:

  score = (r_ST + α·p) / (n_ST + α), with α = 1.

- If the ST is unknown, or absent from training, the strain falls back to the same formula over its
  phylogroup, and then to p.
- The binary call ("majority label of the ST") is score ≥ 0.5.
- α = 1 is fixed here, not tuned.
- On S3 every test ST is absent from training by construction, so (a) reduces to the phylogroup
  fallback there. This is expected and reported as such.

### (b) kNN on genetic distance

Score = resistant fraction among the k nearest training strains (Hamming distance, unweighted, ties
broken by the lower strain index).

- **k is chosen per split and antibiotic on validation PR-AUC** from {1, 3, 5, 10, 20, 50}; the
  smallest k wins a tie. For the reported test score, the neighbours are searched among the training
  strains only (validation strains are not added), so test is scored exactly like validation.
- Two distance variants:
  - **P/A:** Hamming distance on the 32,441 gene presence/absence columns
    (`arcadia_v2_gene_presence_absence.pkl`). Hamming is unchanged by the column-orientation problem
    of W0 §4, so this variant is valid before that is fixed. CPU, here. **This is a stand-in.**
  - **SNP:** Hamming distance on the 535,439 SNP columns with every non-zero code counted as
    "differs from reference". Kaggle (`w2_snp_knn`). This is the variant used in the paper's
    headline table; the P/A variant is reported as a robustness check.

### (c) Phylogroup only

The existing `run_phylogroup_only_baseline.py` (logistic regression on the phylogroup one-hot) is the
paper's number on S1. For the other splits, the same idea is computed as the smoothed resistant
fraction of the phylogroup (formula of (a), α = 1). An unregularised one-hot logistic regression
converges to the unsmoothed fraction, so the two differ only for very small phylogroups. On S1
both are reported.

## Splits and antibiotics

S1–S5 with the antibiotics of `w1_five_axis.md`. All three baselines are cheap, so they also run on
S4/S5 antibiotics below the minimum. Those cells are marked and kept out of every test.

## Metrics and statistics

- As W1: PR-AUC with 95% bootstrap CI, ROC-AUC, prevalence-matched PR-AUC, VME/ME at 0.5.
- **Key table:** ΔPR-AUC = model − best lookup, per split × antibiotic. "Best lookup" is the baseline
  with the highest **validation** PR-AUC for that split and antibiotic.
- Paired bootstrap CI and Holm-adjusted p, as in W1.

## Pass/fail and sanity rules

These are for the baselines themselves. The model-level claims (P1–P3) are in `w1_five_axis.md`.

| # | rule | expected | if it fails |
|---|---|---|---|
| L1 | on S1, kNN (P/A) PR-AUC > phylogroup-only PR-AUC for ≥ 6 of 7 antibiotics | yes: relatives are informative within one population | implementation suspected; stop and debug before using any number |
| L2 | on S3, ST lookup = phylogroup fallback for 100% of test strains | yes, by construction | the split or the lookup is wrong (test in `code/tests/`) |
| L3 | every baseline's test PR-AUC is computed only from training labels | asserted in code and tests | blocker |

**Reported whatever the outcome:** how close the lookups come to TriLens on S1, i.e. how much of the
random-split score lineage alone explains.

## Pilot

- First: kNN (P/A) on S1, CIP only, seed-free (the method is deterministic). Pass = L1 holds for CIP.
- Then: all splits and antibiotics, P/A only.
- Comparison with models: where saved test predictions of the classical arm exist (S2 and S3 from
  the `outputs-classical` release, made on the clean phylogroup / holdout gene sets), the pilot
  also reports Δ = classical arm − best lookup. **These classical predictions are from a run made before
  the data repair** (they did not read the repaired file) and on a GPU XGBoost whose results drift
  slightly between runs. They are a pilot, replaced by the re-run.

Output: `code/results/w2_lineage_lookup_pa_pilot/`, every file marked "P/A-only pilot".

## Compute

- P/A variant: CPU, minutes, here.
- `w2_snp_knn`: CPU, Kaggle account B, about 1 h. It caches the 6,962 × 6,962 distance matrix once.

## Deviations

(none yet)

---

## Pilot results (26 October 2026, P/A-only; added after the run, rules above unchanged)

Ran exactly as pre-registered: `python -m src.experiments.run_lineage_lookup --pilot`, then the full
run without `--pilot`, then `python scripts/diagnostics/w2_pilot_vs_classical.py`. Files:
`code/results/w2_lineage_lookup_pa_pilot/`. No setting was changed after a result was seen. One
code change between the two runs did not alter any reported number: prevalence matching returns
NaN when a side has fewer than one strain to keep, instead of an sklearn warning. It only touches
the below-minimum S4b/SXT cell.

**Step 1 (CIP on S1).** kNN (P/A) test PR-AUC 0.889 [0.849, 0.924] (k = 20) against phylogroup-only
0.248. Pass, so the full run went ahead. The phylogroup value agrees with the existing
`run_phylogroup_only_baseline` result for CIP (0.245 with logistic regression on a one-hot), which
checks the implementation independently.

**Sanity rules.**

- L1: on S1, kNN > phylogroup for 7 of 7 antibiotics. Pass.
- L2: on S3, 0% of test strains received an ST-level lookup. Pass.
- L3: asserted in code; tested in `code/tests/test_lineage_lookup.py`.

kNN on P/A was the best lookup on validation for every split and antibiotic.

Test PR-AUC [95% CI] of the lookup chosen on validation (always kNN on P/A here), with the chosen k. '—' = antibiotic not usable on that split; † = secondary tier; ‡ = below the minimum counts.

| split | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---|---|---|---|---|---|---|
| S1_random | 0.820 [0.781, 0.856] k=50 | 0.889 [0.849, 0.924] k=20 | 0.607 [0.539, 0.673] k=5 | 0.632 [0.553, 0.707] k=50 | 0.458 [0.362, 0.566] k=3 | 0.723 [0.625, 0.809] k=10 | 0.644 [0.553, 0.726] k=10 |
| S2_phylogroup | 0.808 [0.766, 0.845] k=50 | 0.870 [0.822, 0.910] k=20 | 0.617 [0.552, 0.681] k=10 | 0.666 [0.584, 0.750] k=20 | 0.568 [0.470, 0.658] k=10 | 0.776 [0.709, 0.834] k=10 | 0.695 [0.619, 0.766] k=10 |
| S3_st_holdout | 0.582 [0.529, 0.639] k=50 | 0.242 [0.194, 0.295] k=5 | 0.313 [0.261, 0.381] k=50 | 0.429 [0.362, 0.506] k=20 | 0.083 [0.048, 0.166] k=20 | 0.212 [0.151, 0.297] k=10 | 0.198 [0.108, 0.327] k=20 |
| S4a_uk_to_norway | 0.670 [0.641, 0.696] k=10 | 0.576 [0.523, 0.631] k=3 | — | — | 0.172 [0.133, 0.218] k=5 | 0.231 [0.166, 0.304] k=10 | 0.148 [0.106, 0.202] k=5 |
| S4b_norway_to_uk | 0.817 [0.787, 0.847] k=20 | 0.816 [0.793, 0.838] k=20 | — | 0.932 [0.743, 0.966] k=5‡ | 0.369 [0.312, 0.430] k=10 | 0.524 [0.478, 0.569] k=50 | 0.423 [0.378, 0.474] k=50 |
| S5_time_norway | 0.741 [0.699, 0.780] k=20 | 0.608 [0.534, 0.679] k=5 | — | 0.529 [0.477, 0.586] k=50 | 0.314 [0.231, 0.413] k=50 | 0.337 [0.247, 0.440] k=5† | 0.354 [0.253, 0.473] k=20† |

Mean test PR-AUC over the usable (primary + secondary) antibiotics:

| split | ST lookup | phylogroup | kNN (P/A) | mean test prevalence |
|---|---:|---:|---:|---:|
| S1_random | 0.541 | 0.289 | 0.682 | 0.242 |
| S2_phylogroup | 0.564 | 0.287 | 0.715 | 0.247 |
| S3_st_holdout | 0.230 | 0.230 | 0.294 | 0.185 |
| S4a_uk_to_norway | 0.324 | 0.153 | 0.359 | 0.138 |
| S4b_norway_to_uk | 0.563 | 0.262 | 0.590 | 0.259 |
| S5_time_norway | 0.394 | 0.215 | 0.480 | 0.187 |

Classical arm (5 learners stacked) minus the best lookup, paired bootstrap, Holm across the 7 antibiotics of each split:

| split | antibiotic | classical | best lookup | Δ [95% CI] | Holm p |
|---|---|---:|---:|---:|---:|
| S2_phylogroup | AMP | 0.933 | 0.808 | +0.124 [+0.095, +0.157] | < 0.001 |
| S2_phylogroup | CIP | 0.961 | 0.870 | +0.091 [+0.061, +0.125] | < 0.001 |
| S2_phylogroup | AMC | 0.797 | 0.617 | +0.179 [+0.136, +0.221] | < 0.001 |
| S2_phylogroup | SXT | 0.828 | 0.666 | +0.162 [+0.086, +0.245] | < 0.001 |
| S2_phylogroup | GEN | 0.730 | 0.568 | +0.161 [+0.109, +0.216] | < 0.001 |
| S2_phylogroup | CTX | 0.878 | 0.776 | +0.101 [+0.069, +0.141] | < 0.001 |
| S2_phylogroup | CAZ | 0.840 | 0.695 | +0.145 [+0.105, +0.189] | < 0.001 |
| S3_st_holdout | AMP | 0.901 | 0.582 | +0.319 [+0.262, +0.368] | < 0.001 |
| S3_st_holdout | CIP | 0.916 | 0.242 | +0.674 [+0.611, +0.735] | < 0.001 |
| S3_st_holdout | AMC | 0.476 | 0.313 | +0.163 [+0.092, +0.239] | < 0.001 |
| S3_st_holdout | SXT | 0.774 | 0.429 | +0.345 [+0.274, +0.410] | < 0.001 |
| S3_st_holdout | GEN | 0.222 | 0.083 | +0.139 [+0.028, +0.268] | 0.009 |
| S3_st_holdout | CTX | 0.602 | 0.212 | +0.390 [+0.276, +0.492] | < 0.001 |
| S3_st_holdout | CAZ | 0.464 | 0.198 | +0.266 [+0.125, +0.387] | < 0.001 |

**Reading (pilot only):**

1. **Relatives alone go a long way on the random split.** kNN averages 0.682 over 7 drugs, and 0.889
   for CIP. Every random-split score must be read against this.
2. **On a new lineage they collapse.** On S3 kNN averages 0.294; CIP falls to 0.242, below the ST
   lookup's phylogroup fallback (0.309).
3. **The classical arm adds a lot above lineage on every antibiotic of S2 and S3.** All 14 Δ are
   positive with Holm p ≤ 0.009. The largest is CIP on S3 (+0.674), the drug whose determinants are
   point mutations the SNP view sees directly. The smallest is GEN on S3 (+0.139, CI [+0.028, +0.268]).
4. **Country and time sit between the two.** S4a (UK → Norway) is hard for lookups (0.359 mean), partly
   because Norway's prevalence is low (0.138). S4b is easier (0.590).

**What this pilot is not:**

- It is not the SNP-distance kNN of the paper's table (Kaggle, `w2_snp_knn`).
- It is not a comparison with TriLens.
- The classical predictions are from the `outputs-classical` release made before the data repair and
  are replaced by the re-run (the S2/S3 gene sets they read were not damaged).
- P1–P3 of `w1_five_axis.md` are about TriLens and are **not** evaluated here.

---

## Pilot redone on the rebuilt presence/absence matrix (session 2, 26 October 2026)

**Why:** the matrix used above marked the minority state of each locus (KNOWN_ISSUES 12). The rebuilt
matrix (release `outputs-pa_fix`) flips 10,511 of 32,441 loci and leaves 21,930 unchanged.

**Rule for the 13 strains without a profile** (all zeros in both matrices): they are **dropped from
every partition** for all three baselines, so every comparison stays on identical strains. This rule
also binds W1/W3/W4 (`w0_high_locus_strains.md`, point 4). Each split loses 0–5 labelled test strains
per antibiotic. Settings are otherwise unchanged; the runner now refuses the old matrix.

**What changed:**

1. **Distances between strains with a profile are identical**, checked on all 6,949² pairs. A flipped
   column changes both strains of a pair, so Hamming distance cannot change. This is as expected.
2. **The 13 all-zero strains were not harmless in the superseded run.**
   - In the minority-state matrix an all-zero row is *closer* to every strain than real strains are to
     each other: median distance 3,712 against 5,543 for S3 test → train pairs.
   - On the ST holdout, where test strains have no close relatives, 10% of CTX test strains had one of
     them among their 10 nearest neighbours; 3 of the 5 in CTX training are resistant.
   - In the rebuilt matrix the same rows are about 11,800 away, and they are now dropped.
   - Overall, 0.7% of strains had one among their 20 nearest (`unmatched_as_neighbours.json`).
3. **Effect on the numbers:**
   - ST lookup and phylogroup change by at most 0.006 PR-AUC; this only reflects the dropped strains.
   - kNN changes by more than 0.01 in 3 of 38 cells: S3 CTX 0.212 → 0.340, S3 AMP 0.582 → 0.556,
     S4a AMP 0.670 → 0.695. In 5 of 38 cells the k chosen on validation changed.
4. **S3 CTX is unstable in k, and is marked as noisy.**
   - Validation PR-AUC is flat across k (0.228–0.231), while test PR-AUC ranges from 0.159 (k = 5)
     to 0.385 (k = 50).
   - Its "best lookup" value should not be read to two decimals. The pre-registered choice rule stays.
5. **Unchanged conclusions:** L1 and L2 hold. The classical arm beats the best lookup on all 14
   antibiotic × split cells of S2/S3 (Holm p ≤ 0.009). S3 CTX's margin shrinks from +0.390 to
   +0.262 [+0.159, +0.354] because the lookup rose.

Mean test PR-AUC over the usable antibiotics (rebuilt matrix; last column = the superseded run):

| split | ST lookup | phylogroup | kNN (P/A) | kNN, superseded run |
|---|---:|---:|---:|---:|
| S1_random | 0.541 | 0.288 | 0.684 | 0.682 |
| S2_phylogroup | 0.564 | 0.286 | 0.716 | 0.715 |
| S3_st_holdout | 0.230 | 0.230 | 0.310 | 0.294 |
| S4a_uk_to_norway | 0.325 | 0.154 | 0.365 | 0.359 |
| S4b_norway_to_uk | 0.563 | 0.262 | 0.590 | 0.590 |
| S5_time_norway | 0.395 | 0.216 | 0.481 | 0.480 |

Classical arm minus the best lookup on the rebuilt matrix (Holm across the 7 antibiotics of each split):

| split | antibiotic | classical | best lookup | Δ [95% CI] | Holm p |
|---|---|---:|---:|---:|---:|
| S2_phylogroup | AMP | 0.932 | 0.808 | +0.124 [+0.093, +0.158] | < 0.001 |
| S2_phylogroup | CIP | 0.961 | 0.870 | +0.091 [+0.062, +0.124] | < 0.001 |
| S2_phylogroup | AMC | 0.797 | 0.621 | +0.176 [+0.134, +0.221] | < 0.001 |
| S2_phylogroup | SXT | 0.828 | 0.665 | +0.163 [+0.088, +0.245] | < 0.001 |
| S2_phylogroup | GEN | 0.730 | 0.570 | +0.160 [+0.111, +0.219] | < 0.001 |
| S2_phylogroup | CTX | 0.877 | 0.778 | +0.098 [+0.065, +0.137] | < 0.001 |
| S2_phylogroup | CAZ | 0.842 | 0.699 | +0.143 [+0.103, +0.189] | < 0.001 |
| S3_st_holdout | AMP | 0.901 | 0.556 | +0.345 [+0.291, +0.394] | < 0.001 |
| S3_st_holdout | CIP | 0.916 | 0.239 | +0.677 [+0.617, +0.733] | < 0.001 |
| S3_st_holdout | AMC | 0.476 | 0.322 | +0.154 [+0.080, +0.226] | < 0.001 |
| S3_st_holdout | SXT | 0.774 | 0.430 | +0.344 [+0.273, +0.408] | < 0.001 |
| S3_st_holdout | GEN | 0.222 | 0.082 | +0.140 [+0.027, +0.268] | 0.009 |
| S3_st_holdout | CTX | 0.602 | 0.340 | +0.262 [+0.159, +0.354] | < 0.001 |
| S3_st_holdout | CAZ | 0.464 | 0.200 | +0.264 [+0.125, +0.385] | < 0.001 |

The superseded files are in `code/results/w2_lineage_lookup_pa_pilot/superseded_minority_state_matrix/`.

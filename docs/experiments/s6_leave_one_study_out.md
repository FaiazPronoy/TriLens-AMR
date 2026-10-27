# S6 (leave one study out) and the S5 cut-off sensitivity check: pre-registration

**Status:** pre-registered, not run. Committed before any split file of S6, or of the 2012/2014
cut-offs, exists and before any model is fitted on them. The owner's review asked for both (answers 3
and 7 in `docs/PROGRESS.md`).

---

## Part A. S6: leave one study out (secondary axis)

### Question

Does performance survive when the test strains come from a collection never seen in training, with
its own sampling, time and (probably) susceptibility-testing method? S4 asks this for countries.
S6 asks it for studies, so it also covers the UK's three collections separately.

### Folds (fixed now)

Every collection with at least 900 strains is a fold. Its strains form the test side; all other
strains (every other collection, including the small ones) form the training side.

| fold | collection | strains | antibiotics passing the minimum counts |
|---|---|---:|---|
| S6a | Norway (no PMID in the metadata) | 2,963 | AMP, CIP, SXT, GEN, CTX, CAZ |
| S6b | PMID 28720578 (UK) | 1,439 | AMP, CIP, AMC, GEN, CTX, CAZ |
| S6c | PMID 34485958 (England) | 1,015 | CIP, AMC, GEN, CTX, CAZ |
| S6d | PMID 32205351 (UK) | 921 | AMP, AMC |

Counts are from `code/results/data_inventory/strain_table.csv`, using the minimum of
`data_inventory.md` (test ≥ 50 R and ≥ 50 S, train ≥ 100 R and ≥ 100 S).

- For SXT on S6a the training side has 288 R / 167 S, almost all from small resistance-selected
  collections. It passes the rule, but is marked in every table.

### Construction

- **Validation:** 15% of the training side, seed 42, stratified on AMP with "not tested" as a stratum.
  This is the same rule as S4/S5.
- **The 13 strains without a P/A profile:** they stay in S6, and are left out only of P/A-based
  analyses (as in `w0_high_locus_strains.md`).
- **Code and files:** `code/src/splits/study.py`, written after this file is committed. It writes
  `data/processed/study_<fold>_split.pkl` and `code/results/splits/study_<fold>.json`.
- **Tests** assert:
  - every test strain belongs to the fold's study, and no training or validation strain does;
  - the partitions are disjoint and deterministic.

### Models and cost

- **Run:** W2 lookups (CPU), and the classical arm with its 5 solo learners, on a gene set refitted on
  each fold's training + validation side (`w1_gene_sets` pattern: 4 folds × about 4 CPU h; classical
  about 1.5 GPU h per fold).
- **TriLens:** only if GPU budget is left after W1 and W3 (owner's answer 7).

### Metrics, statistics, pass/fail

Metrics and statistics as W1 (`w1_five_axis.md`). Holm runs across the antibiotics of each fold.

| # | claim | passes if |
|---|---|---|
| S6-1 | the classical arm learns more than lineage on an unseen collection | classical > best lookup (chosen on validation), Holm p < 0.05, for ≥ half of the fold's antibiotics, in ≥ 3 of 4 folds |
| S6-2 | collections are a different shift than countries | reported descriptively: PR-AUC of S6a against S4a (both test on Norway; S4a trains on UK only, S6a on everything else) |

S6 is secondary. It is reported in the supplement, and a failure is reported as a limitation, not
fixed by re-tuning.

---

## Part B. S5 cut-off sensitivity (2012 and 2014)

### Question

Are the S5 (time) conclusions specific to the cut-off 2013?

### Splits (fixed now)

These are S5 with cut-offs 2012 and 2014: Norway only, train ≤ T, test > T, the same validation rule.

- Names: `time_norway_2012`, `time_norway_2014`, built by the existing `src/splits/country_time.py`
  with a new `SPLITS` entry each.
- Antibiotics: those passing the minimum at that cut-off (`split_feasibility_time.csv`):
  - T = 2012: AMP, CIP, SXT, GEN primary; CTX, CAZ below 100 training R.
  - T = 2014: the same.
  - Secondary-tier rules as S5.

### Models

W2 lookups and the classical arm (with a gene set refitted per cut-off), CPU or one short GPU run.
No TriLens (owner's answer 3).

### Rule

The S5 conclusions count as **robust to the cut-off** if, for the primary antibiotics, the sign of
classical − best lookup and its Holm significance at 0.05 agree with T = 2013 at both other cut-offs
for ≥ 3 of 4 antibiotics. Otherwise the paper reports S5 as cut-off dependent, with all three
cut-offs in the table. Either way, all three are reported.

## Deviations

(none)

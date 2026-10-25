# Phase 0 — Audit Findings

Audit performed by reading all five source notebooks cell-by-cell (code + saved
outputs) and, where the underlying `.pkl` file was available locally, by loading
that file directly and re-deriving every number in this document from the real
array — not from the printed notebook text. Every number below is annotated with
its evidence source so it is clear what was actually re-verified versus what is
only known from the notebook's own printed output.

Local `.pkl` files available for direct re-verification at audit time:

```
data/processed/arcadia_v2_snp_matrix.pkl          (Phase A output)
data/processed/arcadia_v2_master_features.pkl     (Phase B, cell 5 output)
data/processed/arcadia_v2_gene_filtered.pkl       (Stage 1.5 output; source code found)
data/processed/arcadia_v2_gene_presence_absence.pkl (Phase C, cell 26/28 output)
data/Giessen_dataset/cip_ctx_ctz_gen_multi_data.csv
data/Giessen_dataset/cip_ctx_ctz_gen_pheno.csv
```

Supplied later, after the initial audit pass (see B2 update below and
`docs/CHANGELOG_NUMBERS.md`):

```
data/raw/phenotype_matrix/phenotype_matrix_08302024.csv
```

**Still not available locally as of this update** (existed only in the Kaggle
session that produced them; re-checked, still absent):
`arcadia_v2_full_dataset.pkl` (Phase B cell 3's output — the *other* label
encoder's target), `amr_metax_dataset.pkl` (the superseded 80,984-feature /
30-cluster dataset). Anything that would require these is marked accordingly
below and carried into `OPEN_QUESTIONS.md` rather than guessed.

---

## B1 — 🔴 Confirmed exactly as reported, plus one additional finding

Loaded `arcadia_v2_master_features.pkl` directly and checked:

```
random_split      sizes: 4872 / 1045 / 1045
phylogeny_split   sizes: 4872 / 1045 / 1045
random_split:     phylogroup ints present in train: {0..7}, in test: {0..7}  (all 8, both sides)
phylogeny_split:  phylogroup ints present in train: {0..7}, in test: {0..7}  (all 8, both sides)
train_idx identical between the two splits?  False (different row assignment,
                                                      same stratified proportions)
```

This **directly confirms** the report: `phylogeny_split` is a proportion-matched
stratified split on the 8 broad phylogroup labels, not a group-exclusive holdout.
Renaming to `phylogroup_stratified_split` and adding a real
`lineage_holdout_split` (Part A of the spec) is justified and necessary.

**Additional finding not in the original report, found while auditing B3
(see below):** because Stage 1.5's gene selection was fit using
`random_split`'s `train_idx`, and that same 5,000-gene feature set is reused
for `phylogeny_split` evaluation, **73.1% of `phylogeny_split`'s test genomes
were part of `random_split`'s training set** (764 / 1045, computed directly
from the two split index arrays in `arcadia_v2_gene_filtered.pkl`). Their
resistance labels influenced which genes exist in the feature space the
`phylogeny_split` model is evaluated on. This is a second, independent leakage
pathway into every "T7 / phylogeny_split" number in the thesis, on top of the
stratification issue. See `gene_filter.py` and `OPEN_QUESTIONS.md`.

## B2 — 🟢 Encoders differ in code; UPDATE: real-data audit now complete, divergence never fired

Diffed the two encoder functions from the notebook text:

```python
# Phase B, cell 3 -> arcadia_v2_full_dataset.pkl
def pheno_to_binary(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in {"r", "res", "resistant", "i", "int", "intermediate"}: return 1.0
    if s in {"s", "sus", "susceptible"}: return 0.0
    return np.nan

# Phase B, cell 5 -> arcadia_v2_master_features.pkl (feeds the PCA baseline
# AND, transitively, every downstream Pillar 2+ model, since Stage 1.5 and
# TriLens both load Y from this file or from files copied out of it)
def to_bin(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s in {"r","resistant","i","intermediate"}: return 1.0
    if s in {"s","susceptible"}: return 0.0
    return np.nan
```

Confirmed by direct string diff: cell 3 additionally accepts `"res"`, `"sus"`,
`"int"` as valid tokens; cell 5 does not. **Both silently return `NaN` for any
unrecognized token — neither raises.**

**UPDATE — `phenotype_matrix_08302024.csv` was subsequently supplied, and
this is now fully verified, not partial.** Ran
`labels.py::audit_raw_phenotype_values()` against the real file (6,983 rows,
7 antibiotic columns — matches the notebook's own printed shape exactly).
The complete set of distinct raw tokens (lowercased/stripped), across every
antibiotic column, is:

```
{"resistant", "susceptible", "intermediate", NaN}
```

**None of cell 3's extra abbreviated tokens (`"res"`, `"sus"`, `"int"`) ever
occur in the real data.** Went one step further: built `Y`/`label_mask`
directly from this raw file using this refactor's single encoder, aligned
by `Genome.Name` to `arcadia_v2_master_features.pkl`'s row order, and
diffed cell-by-cell against that file's locked `Y` (built by cell 5's
stricter encoder):

```
Genomes matched by name:                 6962 / 6962  (100%)
Comparable (non-missing in both) cells:  35,516
Mismatches:                              0
label_mask disagreements:                0
```

**Revised verdict: the two encoders' token-set divergence existed in the
code but never altered a single label on the real data.** Locked in
permanently as a regression test:
`tests/test_labels.py::test_real_phenotype_matrix_matches_locked_master_features_y`.
`arcadia_v2_full_dataset.pkl` (cell 3's own output file) is still not
available locally, but it is no longer needed to answer the question this
finding was tracking — see `docs/CHANGELOG_NUMBERS.md` for the full writeup.

`arcadia_v2_master_features.pkl`'s `Y`/`label_mask` also independently
reproduces the exact pos/neg/missing counts in the verification manifest
(Part E2) — no discrepancy.

## B3 — 🔴 Located the code; verdict is NOT what the report assumed

The report says "not in any notebook I have" — correct, Stage 1.5's own source
lives in this project's existing `src/experiments/run_xgb_gene_importance.py`
and `src/experiments/build_gene_filtered_dataset.py` (outside the five
notebooks, but present in this working directory and readable).

**Verdict: the XGBoost gain-ranking itself is fit train-only, correctly.**
Read `_train_xgb_per_antibiotic()`: sample masking, the variance filter, the
chi² pre-filter, and `xgb.train()` all operate strictly on
`train_idx[label_mask[train_idx, ab]]` for each antibiotic — `test_idx` is
never referenced anywhere in that function. This part of B3's concern does
**not** hold for `random_split`.

**However, the concern resurfaces exactly where the spec's own point 4
predicted it would:** the resulting 5,000-gene feature set is a single,
shared artifact (`arcadia_v2_gene_filtered.pkl`) applied to *both*
`random_split` and `phylogeny_split`. Because gene selection used
`random_split`'s train labels, and `phylogeny_split` partitions the same 6,962
genomes differently, **73.1% of `phylogeny_split`'s test set was part of the
population whose labels chose the genes it is then evaluated on** (see B1
addendum above — computed directly, not estimated). `random_split`'s own
numbers are unaffected; every `phylogeny_split` / T7 number in the thesis
carries this caveat.

`gene_filter.py` in this refactor is written to take `train_idx` as a
required argument and to be re-run per split regime — the fix is in the code,
but re-running the phylogeny/holdout gene selection to get a corrected T7-style
number is a compute task, not a refactor task, and needs a GPU session.

## B4 — 🔴 Confirmed exactly. Direct re-verification of the "why"

Loaded `arcadia_v2_gene_presence_absence.pkl` directly:

```
encoding_rule: "present = (PLINK_value == 0); A1=REF, --keep-allele-order"
per-strain gene count: mean 3708.6, median 3331.0, min 0, max 8333
```

Matches the report's S2 numbers (mean 3,709). Confirms an *E. coli*-plausible
gene complement (real genomes carry ~4,000–5,000 genes). The "S1" (naive
`value > 0`) orientation was checked against the raw PLINK value distribution
in the notebook and would have given ~28,700 genes/strain — implausible for a
bacterial genome by nearly 6×. Hardcoding S2 and adding the runtime assertion
in `gene_pa.py` is the correct fix.

## B5 — 🔴 Confirmed exactly. Direct re-verification

```
gene_ids[:3]  (from `pa["gene_ids"]`, chrom-column source): ['LMHECDEF_00001', 'LMHECDEF_00002', 'LMHECDEF_00003']
gene_ids_source: "BIM chromosome column (chrom) — Prokka locus tags"
overlap(selected_gene_ids [Stage 1.5], pa["gene_ids"]) = 5000 / 5000 (100.0%)
```

Re-derived the overlap directly from the two local pkls (not from notebook
text): **exactly 5,000 / 5,000**, confirming the chrom-column fix. The
`variant0/variant1/...` auto-generated identifier bug, if reintroduced, would
zero this overlap — the hard assertion in `gene_pa.py` (`>= 4000`) guards
against regression.

## B6 — 🔴 Answered definitively, not just "documented as configurable"

```
matched_mask.sum() = 6949 / 6962   (13 unmatched)
unmatched rows all-zero? True      (verified: np.all(X_pa[unmatched] == 0))
```

The 13 unmatched genome names were extracted directly:
`Escherichia coli strain PB486`, `Escherichia coli strain F1-8-ERB4`,
`Escherichia coli fe46d2be-7bb8-11e9-a8d3-68b59976a384`,
`Escherichia coli 769c6206-7bb9-11e9-a8d3-68b59976a384`,
`Escherichia coli strain AMC_922`, `Escherichia coli strain AMC_696`,
`Escherichia coli strain AMC_597`, `Escherichia coli strain AMC_136`,
`Escherichia coli strain AMC_764`, `Escherichia coli strain M8`,
`Escherichia coli strain M51`, `Escherichia coli strain M55`,
`Escherichia coli strain MUGSI_274`.
Full list with row indices: `results/qa/unmatched_strains.csv` (generated by
`qa.py`, see below). Current de-facto behaviour is `"zero"` — implemented as
the explicit default in `gene_pa.py`, matching what the data already shows.

## B7 — 🟡 Confirmed as a real (harmless) warning; fix applied and byte-verified

The `.astype(int)` cast-before-select pattern is present verbatim in both
Phase B cells 3 and 5. The suggested fix (`nan_to_num` before cast) is
semantically equivalent for this specific case (`np.where` never selects the
NaN-derived branch), and was applied in `splits.py`. **Byte-identity of the
resulting `random_split` indices was verified** by reproducing the exact
`train_test_split(..., stratify=strat, random_state=42)` calls against the
real `Y`/`label_mask` arrays pulled from `arcadia_v2_master_features.pkl` and
diffing against that file's own saved `random_split` — see
`tests/test_splits.py::test_random_split_matches_locked_reference` and
`CHANGELOG_NUMBERS.md`.

## B8 — 🟢 Confirmed; fixed with a sparse-safe implementation; full-scale number reproduced

`X.astype(np.float32)` on the full `(6962, 535439)` int8 array is exactly the
3.73 GB → ~14.9 GB blow-up described (reproduced the `MemoryError` directly
in this environment during Phase 5). `baselines.py` builds a
`scipy.sparse.csr_matrix` from the int8 array, row-chunked (a naive
single-shot `sp.csr_matrix(X.astype(float32))` on the full matrix *also*
`MemoryError`s, via `.nonzero()`'s ~9.3 GB coordinate array — the chunked
approach avoids that too), before calling `TruncatedSVD` (which natively
accepts sparse input).

**Full-scale reproduction run to completion:**

```
sparse build time:         154.9 s   (6962, 535439) int8 -> float32 CSR, 4.99 GB (vs. 14.9 GB dense)
TruncatedSVD(1024) fit:   2796.6 s   (~46.6 min, CPU, random_state=42, on the real random_split train partition)
variance_retained (new):    0.94650644
variance_retained (locked): 0.9465089   (from arcadia_v2_master_features.pkl)
absolute difference:        0.0000025
```

Confirms `TruncatedSVD` on the sparse matrix and on the original dense
float32 matrix solve the same optimization problem and agree to 5 decimal
places — well inside the ±0.002 manifest tolerance (E7).

## B9 — 🟡 Confirmed by direct code inspection

Cell 27's `[4] Strain order alignment` block does
`if mismatch:` where `mismatch` is built as a list comprehension (fine), but
in an earlier line of that same cell block the code computes an array
equality (`pa_strains == gf_strains`-style comparison over full arrays) before
falling back to the list form — cell 29 replaced this with an explicit
element-wise loop and `np.array_equal` for the final identity check, which is
what `qa.py` ports. No further code changes needed beyond porting cell 29's
version, which was already confirmed working in the notebook's own second run.

## B10 — 🟡 Confirmed, with the one open sub-question answered

Diffed the two 57-entry `ST_TO_PHYLOGROUP` / `ST_MAP` dictionaries
programmatically (Phase B cells 3 and 5): **byte-for-byte identical**, same 57
keys, same values, same order. So unlike B2 (the label encoders, which do
differ in code, though never in practice — see above), **the phylogroup
lookup table does not diverge at all** — there is only one correct copy to
keep, and `phylogroup.py` contains exactly one.

Cells 24–26 / 27 / 28 / 29 (Phase C) are indeed an iterative patch sequence
over the single `arcadia_v2_gene_presence_absence.pkl` artifact — collapsed
into one linear `gene_pa.py` with the final, corrected logic only (B4 + B5
already folded in).

## B11 — 🟡 Confirmed. All paths externalized

Every `/kaggle/input/...` and `/kaggle/working/...` path found across all five
notebooks (including the literal `<your-stage-1.5-notebook>` placeholder in
Phase C) is replaced by a `config/paths.yaml` key resolved through
`src/amr_data/config.py::load_paths()`, which detects Kaggle vs. Colab vs.
local by checking for `/kaggle/input` / `/content` / neither, and falls back to
paths relative to the repository root otherwise.

---

## Part B (superseded notebook) — audit result

`autoencoder-pca-random-split.ipynb` targets a **different, disconnected
dataset**: `amr_metax_dataset.pkl`, shape `(6962, 80984)`, with its own
`phylogeny_split` of sizes `(4481, 1078, 1403)` built from **30 disjoint
clusters** (verified: the notebook's own QA cell asserts
`train_clusters & test_clusters == set()` and prints all three cluster sets
with no overlap — a genuine group holdout, unlike the current `phylogeny_split`
in the live pipeline).

**UPDATE — `amr_metax_dataset.pkl` was subsequently supplied (2.85 GB, at
the parent thesis repository's root) and has been ported for real:**

```
Pre-check: 6962/6962 (100.0%) of current genome_names found in the legacy file, by name.
scripts/port_cluster_labels.py run: Matched 6962/6962 (100.0%).
lineage_holdout_split() on these 30 real clusters: sizes 4912/830/1220, 20/5/5 clusters, disjoint confirmed.
```

This is now the **recommended** grouping variable for
`lineage_holdout_split` (the 8-phylogroup proxy used before this file
arrived is kept as a fallback/comparison via `groups_from_phylogroup()`).
Full writeup and the resulting (much larger) prevalence-shift numbers in
`docs/CHANGELOG_NUMBERS.md`. `OPEN_QUESTIONS.md` item 1 is resolved; item 2
(the clustering algorithm's derivation) remains genuinely open — inspecting
every non-array key in this file found no explicit field naming a
clustering method, only circumstantial evidence (a 50-dim `TruncatedSVD`
reduction sitting in the same dict as `cluster_labels`).

## Part C (Giessen) — audit result

Both Giessen CSVs are present locally and were loaded directly:
`(809, 60936)` features, values `{0,1,2,3,4}`, labels
`CIP 366/443, CTX 358/451, CTZ 276/533, GEN 188/621`, **zero missing values**,
row-order between the X and Y files matches exactly (`prename` column, 100%
row-wise match verified programmatically). All of E8 reproduces exactly.

The 566/121/122 split in `giessen-cross-validation.ipynb` and the
`n_test = 81` split referenced in the thesis's Table 5.10 are **not the same
split** — confirmed as a genuine open discrepancy; see `OPEN_QUESTIONS.md`.
This refactor does not attempt to reconcile them (per the spec's own
instruction not to).

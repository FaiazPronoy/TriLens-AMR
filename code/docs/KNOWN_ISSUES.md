# Known issues

What is wrong, missing or fragile in this repository, as of October 2026. Each item says
what it affects and what fixes it. Items marked **fixed** were corrected while organising
the project and are listed so the change is traceable.

---

## Open

### 1. Several results on disk are not from the delivered model: re-run everything

The thesis reports the strain-conditioned (post-fix) model everywhere. On disk, only the
random-split run (`results/trilens_attnfix/`) is post-fix. The lineage-holdout and
phylogroup runs were done on a second machine that was lost, and several later files were
computed from the pre-fix run:

| Number in the paper | Paper value | Value in the surviving files | Why |
|---|---|---|---|
| Lineage holdout, full stack | 0.6337 | 0.6300 | post-fix run lost |
| Lineage holdout, neural arm alone | 0.472 | 0.4369 | post-fix run lost |
| Phylogroup split, full stack | 0.8531 | 0.8519 | post-fix run lost |
| Layer ablation, stage 1 / stage 2 (`tab:ablation`) | 0.7453 / 0.7807 | 0.7324 / 0.7547 | the ablation read the pre-fix run (`trilens_v10`); its stage 3 was then taken from the delivered run, so the saved table mixed two models |
| `tab:clinical`, `tab:threshold` | VME 0.109 for AMP | pre-fix | `threshold_tuning` ran before the delivered run existed |
| Near-duplicate cost, full metric suite | 0.0041 | 0.0042 | read the pre-fix run |

Everything computed from the holdout runs inherits the problem: `matched_prevalence_check.py`
(0.7226), `holdout_clinical_metrics.py` and `fig_pr_curves`. The pre-fix outputs are in
`results/pre_fix/` so they cannot be mistaken for the delivered model.

**Fix:** the full re-run in `../../kaggle/` recomputes every paper number from the
processed data, starting from an empty `results/` folder. Its last notebook writes
`results/diagnostics/paper_numbers.csv` (paper value against new value for every headline
number).

**The one-line check:** AMP on the random split is **0.9440** for the delivered model and
**0.9408** for the pre-fix one. Any file that says 0.9408 describes the old model.

### 2. Scripts that used to read the pre-fix run

The pre-fix and post-fix runs wrote files with identical names, and several scripts were
written before the fix and read `results/trilens_v10/` (pre-fix). They now read the main
run, named by the environment variable `TRILENS_MAIN_RUN` (default `trilens_attnfix`):
`run_pillar2_layer_ablation` (and its cache path in `pillar2_layer_ablation.yaml`),
`run_pillar_gene_annotation`. `run_pillar_full_metric_suite`, `run_pillar_dedup_eval`,
`diagnose_prevalence_invariant_metrics` and the two `diagnose_*_final_comparison` scripts
prefer `trilens_attnfix` and fall back to `trilens_v10`.

Not used by the paper and left as they were: `run_pillar_cascadestack_phylogeny`,
`run_pillar_master_summary`, `run_pillar_t6_multiseed`, `run_trilens_t7_with_predictions`.

`posthoc_analysis/clinical_metrics.csv` (June 2026) was computed from CascadeStack v9, not
from TriLens. `tab:clinical` comes from `scripts/diagnostics/threshold_tuning.py`.

### 3. The standard (pre-fix) model is now a switch, not a lost file

`ExpertB` in `src/models/trilens.py` takes `strain_conditioned`. Unset, it reads the
environment variable `TRILENS_LENSB`: `fixed` (default) is the delivered model, `standard`
drops the two context projections (`attn_Vc`, `attn_Uc`) and nothing else. With `standard`
those two layers are not created at all, so parameters are initialised in the same order as
in the original standard model. This makes `tab:attnfix` a real ablation that anyone can
re-run (Kaggle notebooks `standard_random` and `standard_splits`); before, only the old
`trilens_v10` files existed. `scripts/diagnostics/strain_ablation_table.py` builds the table.

### 4. Figures that need the re-run

`fig_pr_curves` needs the post-fix holdout predictions. `fig_attention_concentration` needs
the standard-model Lens B attention; it now reads `results/standard/trilens_attnfix/` (from
the re-run) and falls back to `results/trilens_v10/`. The `analysis` notebook redraws both.
Figures drawn from typed-in table values (`make_thesis_figures.py`) are redrawn after the
paper numbers are updated. See `FIGURES.md`.

### 5. Two Giessen splits exist; the paper's one is the right one

The reported Giessen results use **647 / 81 / 81** (stratified 80/10/10, seed 42), which is
what `run_pillar4_giessen_transfer.py` builds itself. `src/data/giessen_loader.py` has its
own 70/15/15 default (566 / 121 / 122), used only by the CNN experiments. The re-run of
`pillar4_giessen_transfer` therefore reproduces the paper's split.

### 6. The data-QA package recomputes a different lineage holdout

`amr-genomic-data/` has its own split code. It lands on 4,912 / 830 / 1,220 instead of the
reported 4,836 / 1,081 / 1,045. The reported partition is the one in
`data/processed/mlst_lineage_holdout_split.pkl`, built by
`src/experiments/build_mlst_lineage_holdout_split.py`. Never take split figures from the
QA package.

### 7. The pan-genome feature space may not contain the main ESBL gene

Found in October 2026. The P/A and SNP views come from a pan-genome built from the 72 ECOR
reference strains. A motif search of all 18,494 pan-genome CDS found TEM (present in 49% of
isolates, P(AMP-R | present) = 0.84) but **no CTX-M**, the dominant ESBL behind cefotaxime
and ceftazidime resistance. If confirmed, CTX and CAZ models cannot see their main cause and
must lean on lineage, which fits their large holdout drop.

**Status:** hypothesis. Confirm with AMRFinderPlus on `data/raw/pangenome/pangenome_cds.fa`
before writing it into the paper. See `../../docs/next_steps.md`.

### 8. A number in `tab:baselines` does not match its file

"XGBoost, chi-squared-filtered SNPs, random split" is **0.8400** in the paper, but
`results/stage3_cascadestack_v9/v9_results.csv` (`solo_XGB-raw-chi2_test_pr_auc`) averages
**0.8430**. The other four component rows match the same file exactly. Take the value from
the re-run; `collect_paper_numbers.py` flags it.

### 9. TabNet under the holdout uses the random-split gene ranking

`run_tabnet_baseline_mlst.py` keeps the gene set ranked on the random split's training
partition, on purpose (same features across the three TabNet runs). That ranking saw the
labels of some holdout test strains, so the TabNet holdout number (0.3255) is if anything
optimistic. It favours the baseline, not the proposed model, so it does not weaken any
claim; say so in one sentence in the paper.

### 10. 46 corrupted columns in the random-split dataset (found and repaired, October 2026)

`kaggle/notebooks/data_check.ipynb` read every processed file. `arcadia_v2_gene_filtered.pkl`
(random split, 6,962 × 262,225) has **46 columns with values from -128 to 127**, in 6,925 of
the 6,962 strains, all inside one block of columns (169,213 to 169,307); SNP codes are 0 to 3. The SNP matrix it was cut from
(`arcadia_v2_snp_matrix.pkl`) and the phylogroup and lineage-holdout files are clean (0 to 3).

Effect: chi-squared feature selection refuses the file, so the classical stack, SHAP and gain
importance on the random split cannot run on it, and every gene-feature model on the random
split (XGBoost, LightGBM, RF, LogReg, TabNet, TriLens Lens A and C) would read the 42 columns
as large genotype values. It is not known which copy of this file the original thesis runs
read; this code cannot have run the classical stack on this one.

**Fix:** `scripts/repair_gene_filtered.py` re-reads the damaged block from the SNP matrix by
their position names (two columns in the block were damaged with values that happen to lie in 0 to 3). It refuses unless every column outside the block is identical to the
matrix, so nothing else changes. Every Kaggle notebook runs it in its setup cell and writes
`results/diagnostics/data_repair.json`; `check_leakage.py` fails if any gene-filtered file has
values outside 0 to 3. Outputs made before the repair from runs that read this file are
thrown away by the notebooks and recomputed.

For the paper: the random-split numbers come from the repaired file. State in Methods that
SNP codes are 0 to 3 and that the data files are checked for this.

### 11. Download links are placeholders

`docs/DATA.md` and `CITATION.cff` still contain `PASTE_LINK_HERE` / `PASTE_REPO_URL_HERE`.

### 12. The presence/absence matrix marks the minority state, not presence (found October 2026, W0; fixed)

`arcadia_v2_gene_presence_absence.pkl` was built with `present = (PLINK value == 0)`. PLINK
picks the minor state of each locus as allele A1, so the meaning of 0 flips from locus to
locus. Evidence from the file itself:

| Check | Value |
|---|---|
| highest prevalence of any locus | 50.3% (a true presence matrix has core loci near 100%) |
| loci in all 72 ECOR genomes (2,652 CDS) | 99.6% of them "present" in under 1% of our strains |
| gyrA, parC, rpoB, recA (in all 72 ECOR genomes) | "present" in 0 of 6,962 strains |

The .bim file carries the allele letters (`1`/`2`, in either order per locus); they say which
state is presence.

**Not affected:** tree models, logistic regression and distance-based baselines (a column flip
does not change them), the classical arm, the SNP lenses, `shuffle_test`.
**Affected:** Lens B (it weights gene embeddings by `X_pa` and attends only to "present"
genes, so it never saw core genes such as gyrA), the gate and stack built on it, and every
presence-based reading: attended genes, U_res / U_ST, prevalence figures, tab:attnfix.
`amr-genomic-data/src/amr_data/gene_pa.py` carries the same wrong rule; no run uses it.

**Fix:** the Kaggle notebook `pa_fix` (CPU) rebuilds the matrix from the PLINK files with
`scripts/repair_gene_pa.py`: presence = the allele letter carried by the ECOR-core loci. It
refuses unless every locus is either unchanged or exactly flipped on all called cells, the
ECOR-core loci are mostly present, and prevalence follows ECOR frequency
(`scripts/reference/ecor72_frequency.csv`). The four TriLens notebooks and `analysis` need
`pa_fix` and use its copy; their zips carry `_PA_FIXED.txt`, and a notebook stops if an
upstream TriLens output lacks it.

**Result (pa_fix, code 03af45d, 26 October 2026):** repaired. Letter `2` = present (ECOR-core
loci: median prevalence 1.0 under `2`, 0.0 under `1`); 10,511 of 32,441 loci flipped, 21,930
unchanged, 0 inconsistent; no missing or heterozygous calls; 2,924 loci are in one state in
every strain (PLINK allele `0`). gyrA, parC, rpoB, recA now present in 100% of strains;
prevalence follows ECOR frequency (Spearman 0.69). Loci per strain (CDS + intergenic):
mean 11,846, range 9,569 to 18,620 (the old file said 3,709). The strains near the top of that
range are worth a look (possible mixed samples).

For the paper: every Lens B number and figure comes from the rebuilt matrix. State the
orientation rule in Methods.


### 13. The re-run's shuffle test does not support the paper's locality claim (October 2026)

The paper (Methods, Motif Pattern Encoder; Pillar 1) says the plain 1D CNN on the whole
pan-genome "performed equally well with the shuffled input". The re-run (notebook
`shuffle_test`, code 3261258, CIP, seed 42, one fixed permutation) says otherwise:

| Dataset | Ordered | Shuffled | Drop |
|---|---|---|---|
| Arcadia (pan-genome) | 0.775 [0.719, 0.829] | 0.632 [0.572, 0.691] | **0.143** |
| Giessen (one reference) | 0.840 [0.731, 0.936] | 0.660 [0.523, 0.786] | 0.180 |

The pre-registered rule (Giessen drop >= 0.05 and Arcadia drop <= 0.02) fails on the
Arcadia side; the script calls it inconclusive. The CIs of the Arcadia pair do not overlap.
A likely reason: the pan-genome columns are ordered by ECOR contig, so neighbouring columns
often come from the same contig and are in linkage; order is not meaningless there.

For the paper: drop the sentence that shuffling did not hurt, and the table row that says
variant order "carries no information". Report both drops. The case for convolving only
inside gene blocks must rest on the layer ablation and on the within-gene design, not on this
test. One seed and one permutation only; more seeds/permutations would make it firmer.

### 14. TriLens T7 inside `trilens_v10` removed (October 2026)

`trilens_v10` ran T6 (random split) and T7 (phylogeny split). T7 stacks with the
random-split classical arm, whose strains differ from the phylogeny split, so the index
check added in PR #4 now refuses it (it failed in the re-run after 3.5 GPU hours). The
paper's phylogroup number comes from `run_trilens_t7_clean_gene_set` (notebook
`trilens_splits`). `configs/experiment/trilens_v10.yaml` now runs T6 only.

### 15. First `trilens_splits` re-run used the old presence/absence matrix (October 2026)

The notebook was imported before `pa_fix` existed, so its cells did not take the rebuilt
matrix (its zip has no `_PA_FIXED.txt`; `analysis` refuses it). It must run again from a
freshly imported notebook. The setup cell now discards a TriLens notebook's own release when
it was made on the old matrix, and saved folds carry a fingerprint of the matrix, so nothing
from that run is reused. Its numbers (phylogroup T7 0.7776; holdout neural lenses 0.4740)
are not for the paper.

### 16. SNP calls: missing = reference

`notebooks/snp_matrix_from_vcf.ipynb` codes a missing genotype as the reference allele (Arcadia's
convention), and the MAF filter counts missing as reference. In accessory genes, absence of the gene
and the reference allele therefore have the same code. The share of missing calls was not recorded.
Check: `../../docs/experiments/data_inventory.md` §4. State it as a limitation in the paper.

---

## Leakage audit (October 2026)

Every script behind a paper number was read for the usual ways test information reaches
training. `scripts/diagnostics/check_leakage.py` re-checks the data-level items on every
re-run (the `analysis` notebook runs it; a FAIL stops the run).

| Where test data could leak | What the code does | Verdict |
|---|---|---|
| Splits | train / val / test disjoint and complete for all three splits (checked on the real files) | clean |
| Lineage holdout | no sequence type shared between train, val and test; the 475 strains without a sequence type are all in train | clean |
| Gene filter (feature selection) | ranked by XGBoost importance on train + val of that split only; test never used; refitted per split | clean |
| PCA-1024 (baseline) | fitted on train, applied to val and test | clean |
| Scaling (LogReg, TabNet, GeneAttentionNet, TriLens features) | fitted on training data only | clean |
| χ²-filtered SNPs | χ² computed inside each training fold | clean |
| Early stopping | always on a part of the training data (or val); never on test | clean |
| Stacking | meta-learner fitted on out-of-fold training predictions only | clean |
| Decision threshold | 0.5 for every reported MCC / F1 / clinical number; tuned thresholds are tuned on val | clean, after the fix below |
| Classical and neural arms combined | same strains in the same order: asserted in the holdout and phylogroup stacks, and now also in the random-split stack | clean |
| Near-duplicate genomes | 47 of 1,045 random-split test strains within 0.5% of a training strain; reported, with the score after removing them; none under the lineage holdout | disclosed |

### Fixed in this audit

| Problem | Effect | Fix |
|---|---|---|
| The main run saved TriLens-only validation probabilities next to stacked test probabilities | `tab:threshold`: thresholds were tuned on a different model than the one they were applied to | `run_meta_stack` applies the same fitted meta-learner to validation; the file records `val_is_stack`; `threshold_tuning.py` refuses old files for the tuned columns |
| A missing classical arm only gave a warning | the headline could silently become TriLens-only | now an error |
| The random-split stack did not check that both arms list the same strains | a wrong file would pair different strains without any message | asserted |
| `threshold_tuning.py` fell back to other runs if the main one was missing | the table could silently describe another model | only the main run is accepted |

### Methods the paper must state

These are not errors, but a Q1 reviewer will look for them. One sentence each in Methods:

1. The minor-allele-frequency filter on the SNP matrix (1%) was applied once to all 6,962
   genomes, without labels (unsupervised).
2. Labels: resistant and intermediate are both coded 1.
3. Development (model versions v7 to v10) was iterated on the random-split test set. The
   lineage holdout and the Giessen collection are the independent tests; the random-split
   number should be read as in-distribution performance, and differences of a few
   thousandths there are within the bootstrap interval (the paper already says this).
4. Giessen is retrained on Giessen, not zero-shot transfer.
5. The 47 near-duplicates (last row of the audit table) and the score without them.
6. The TabNet holdout baseline keeps the random-split gene ranking (issue 9), which can only
   favour TabNet.
7. The PCA dimension of the PCA baseline was chosen while looking at test scores; this can
   only favour the baseline.
8. Strains without a sequence type were kept in training under the lineage holdout.

### What no audit can promise

No code review proves a result "100% correct". What this audit gives: no path from test
labels or test features into training was found in any script the paper uses, and the
data-level properties are re-checked automatically on every run. The one remaining
statistical weakness is item 3: one random split was used both to develop and to report.
The clean answer is to re-run the final, frozen pipeline on several new random splits and
report mean and spread. That is the first experiment in `../../docs/next_steps.md`.

---

## Pre-submission review (October 2026)

An independent review (branch `review/pre-submission`, file `REVIEW_pre_submission.md`)
found 23 items. Its verdict: no test label or test feature reaches training, the lineage
holdout is ST-disjoint and the stacking arms are aligned by index; the paper is not ready
until the full re-run has produced every number.

### Fixed in the code

| Review item | Fix |
|---|---|
| 3: switching strain conditioning on changed the initialisation of every later parameter and the random stream | `attn_Vc`/`attn_Uc` are created last on a forked RNG; both variants now share every other weight (`tests/test_expertb_init.py`) |
| 4: pre-repair standard-model outputs survived under `standard/` | prefix stripped before the check (`tests/test_notebook_unpack.py`) |
| 5: a resumed notebook's old copies overwrote newer upstream outputs; `analysis` could reuse stale derived files | own release unpacked first and never overwrites; `analysis` does not resume from its own release (`tests/test_notebook_resume_order.py`) |
| 6: missing split indices counted as matching | the classical arm must carry its indices or the run stops; `check_leakage` fails on missing indices (`tests/test_classical_arm_loading.py`, `tests/test_check_leakage.py`) |
| 7: unrecorded, unpinned environment | `pytorch-tabnet` and `entmax` pinned; every zip carries `_ENV.txt` (pip freeze, torch, CUDA, GPU) |
| 8: a killed upload left an empty release; downstream could start on a partial upstream | upload under a temporary name, then swap; every finished notebook writes `_COMPLETE.txt` and later notebooks refuse an upstream without it |
| 15: the TriLens early-stopping set was the last 10% of sorted indices | random 10% per fold (same in the stage-1 ablation) |
| 18: the repair trusted clean-looking files and the first of two duplicated names | clean files are compared with the matrix too; duplicated names are refused only when their columns differ (`tests/test_repair_gene_filtered.py`) |
| 10: VME limit | 1.5% for very major errors, 3% for major errors, in every script |
| 19: fallbacks to pre-fix folders | removed from `holdout_clinical_metrics`, `matched_prevalence_check`, `diagnose_prevalence_invariant_metrics` |
| 20: token in `.git/config` and URLs | passed as a one-command HTTP header only |
| 21: CLOSE allowed 0.01 | 0.002 in `collect_paper_numbers.py` |
| 2: Giessen reference | `collect_paper_numbers.py` reports retention against the classical stack (like for like) and, separately, against TriLens (as printed) |
| 14: fold comment in `trilens_v10.yaml` | corrected |

Run `python -m pytest tests` from `code/` (48 tests, CPU, seconds).

### To change in the paper after the re-run

1. All random-split numbers come from the re-run (provenance of the originals is unknown, issue 10).
2. `tab:giessen_transfer`: the Giessen model is the classical stack; divide by the classical
   stack on Arcadia (like for like) and fix the caption.
3. `tab:attnfix`: "everything else identical" is now true for the initialisation, not for the
   training noise; report it with seeds (review §4.3) or soften the claim.
4. CLSI limits: VME ≤ 1.5%, ME ≤ 3% (cite the source). No drug passes either way.
5. One sentence each: TabNet holdout uses the random-split gene ranking; PCA size chosen on
   test; gene ranking used validation labels for early stopping; the two meta-learners differ
   (`class_weight="balanced"` only in the classical arm); classical-file MCC/F2 use a
   validation-tuned threshold, TriLens MCC uses 0.5.
6. "can only raise our apparent very major error rate" → "is expected to raise".
7. Prevalence matching: add the gap at the random-split test prevalence (review item 11) and
   call the interval a resampling range.
8. The holdout "+0.0038 over the best component" is smaller than the run-to-run drift of GPU
   XGBoost; report it with spread or drop it.

---

## Fixed while organising (October 2026)

| What | Before | Now |
|---|---|---|
| Project root in `configs/config.yaml` | hardcoded to another computer (`C:\Users\WINDOWS 11 PRO\Downloads\...`) | `${oc.env:TRILENS_ROOT,.}`, i.e. the folder you run from |
| Giessen paths in `config.yaml` and `data/giessen.yaml` | `<root>/Giessen_dataset/` (folder never existed there) | `<root>/data/Giessen_dataset/` |
| W&B mode | `online` (asks for a login, fails on Kaggle) | `offline` unless `WANDB_MODE` says otherwise |
| `trilens_v10.yaml` output folder | `results/trilens_v10`, which would overwrite the pre-fix comparison run | `results/${TRILENS_MAIN_RUN}`, default `trilens_attnfix_rerun` (the Kaggle re-run sets `trilens_attnfix`) |
| `diagnose_t7_clean_final_comparison.py` | read the random-split column from the pre-fix run | prefers `trilens_attnfix` |
| `run_pillar_full_metric_suite`, `run_pillar_dedup_eval`, `diagnose_prevalence_invariant_metrics` | read the pre-fix run | prefer `trilens_attnfix` |
| Diagnostics and figure scripts | assumed they sat in the project root; wrote CSVs into the root; figure scripts pointed at `Desktop\Thesis` and `..\Latex\images` | live in `scripts/`, find the root themselves, write to `results/diagnostics/` and `TriLens-AMR/thesis/images/` |
| `verify_env.py`, `verify_giessen.py` | hardcoded `D:\T2510649\...` | relative to the code folder |
| `requirements.txt` | listed `pysam-binary` (not a real package); missed `pytorch-tabnet` and `entmax` | corrected |
| IEEE paper sources | looked for figures in `../Latex/images/` | `../../thesis/images/`; both versions compile |

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

### 10. Download links are placeholders

`docs/DATA.md` and `CITATION.cff` still contain `PASTE_LINK_HERE` / `PASTE_REPO_URL_HERE`.

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

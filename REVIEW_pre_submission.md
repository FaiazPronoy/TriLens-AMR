# Pre-submission review

Scope: `main` at `45883b6`, branches `results/classical` (`65ed490`) and `results/data_check` (`f52f0e1`), release `outputs-classical` (`classical_outputs.zip`, 1.6 MB). No Kaggle notebook was run and nothing on Kaggle was opened. Line numbers refer to `main` at `45883b6`. Code fixes are in five pull requests, one per problem, each with a test that fails on `main` and passes on its branch.

## 1. Verdict

The method is sound in the places that matter most:

- No test label or test feature reaches any model behind a headline number.
- The lineage holdout is ST-disjoint.
- The stacking arms are aligned by index.

The paper is not ready to submit. Every random-split headline number (0.8486, `tab:main`, `tab:ci`, `tab:gate`, `tab:clinical`, `tab:ablation`, the random column of `tab:baselines`) has unknown provenance:

- The random-split input had 46 corrupted columns, and nobody knows which copy the original runs read.
- The post-repair re-run that would replace these numbers has not produced them: 42 of 51 rows in `paper_numbers.csv` are MISSING, and the classical arm (`stage3_cascadestack_v9`) has no results at all.

Before submission, four things need fixing:

- the Kaggle-chain bugs that can silently mix pre- and post-repair or old and new outputs (PRs [#2](https://github.com/FaiazPronoy/TriLens-AMR/pull/2), [#3](https://github.com/FaiazPronoy/TriLens-AMR/pull/3));
- the confounded initialisation behind `tab:attnfix` ([#1](https://github.com/FaiazPronoy/TriLens-AMR/pull/1));
- the mislabelled Giessen comparison;
- a single, pinned and recorded re-run that produces every number.

## 2. Findings, most severe first

| # | Sev. | Area | Where | What is wrong | Evidence | Fix |
|---|---|---|---|---|---|---|
| 1 | **blocker** | H, I, C | `code/docs/KNOWN_ISSUES.md` §10; `results/classical:diagnostics/paper_numbers.csv` | The random-split headline numbers have no trustworthy source. KNOWN_ISSUES §10: "It is not known which copy of this file the original thesis runs read". The re-run that should replace them has not produced them. | `paper_numbers.csv`: 51 rows, **42 MISSING**, 4 SAME, 2 CLOSE, 3 CHANGED. `classical_outputs.zip` has only `stage3_cascadestack_v9/config_used.yaml` (no `v9_results.csv`, no `predictions_saved.npz`) and **no `_DATA_REPAIRED.txt`**. It was built from `afb16fc`, which no longer exists on `main` and predates `repair_gene_filtered.py`. So every random-split file in it will be discarded by the setup cell. `data_check.json`: `X_snp_filtered` min −128, max 127, 42 columns with negatives in 6,925 rows. The values span almost the whole int8 range, which looks like corrupted bytes rather than a coding error. | Re-run `classical` → `trilens_random`/`trilens_splits` → `standard_*` → `analysis` with all PRs below merged. Submit only when `paper_numbers.csv` has no MISSING row and every CHANGED row is explained in the text. Do the local md5 check in §4.1. |
| 2 | major | H, E | `paper/journal/report.tex:2852-2868`; `code/src/experiments/run_pillar4_giessen_transfer.py:69-75`; `code/scripts/diagnostics/collect_paper_numbers.py:78-85` | The caption of `tab:giessen_transfer` says *"The classical stacking half of the pipeline … Neither column is the full TriLens model"*. But the Arcadia column (0.9773, 0.8299, 0.9019, 0.7037) is the **full TriLens model** (`tab:main`), so the 90.0% compares a classical-only Giessen model with the full hybrid on Arcadia. The script itself compares against a hard-coded v9 dict (`ARCADIA_V9_REFERENCE = {"CIP": {"pr_auc": 0.9774 …`), and `collect_paper_numbers.py` compares against TriLens. The two pipelines disagree. | Mean of per-drug ratios with the paper's Arcadia column: **0.8996** (the paper's 90.0%). Against the script's v9 dict: **0.9042**. On the re-run (`pillar4_generalization_gap.csv`): **0.8964**. The hard-coded dict will not follow the repaired v9 re-run. | Use the classical stack's Arcadia numbers from the re-run `stage3_cascadestack_v9/v9_results.csv` (not a literal). Report like-for-like retention, fix the caption, and make `collect_paper_numbers.py` use the same reference. |
| 3 | major | D | `code/src/models/trilens.py:245-248` (on main) | With `strain_conditioned` on, `attn_Vc` and `attn_Uc` are created between `attn_U` and `attn_w`. Every later parameter, all of Expert C, the gate, and the RNG stream for dropout and batch order therefore differ between the arms. `report.tex:3004-3006` ("same seeds … changing only this single term") and the `tab:attnfix` caption ("Everything but this one term is identical") are false. The +0.035 rests on one such draw. | `code/tests/test_expertb_init.py`: only `E_pa`, `attn_V` and `attn_U` are equal on main (fails); everything is equal on the PR branch. | **PR [#1](https://github.com/FaiazPronoy/TriLens-AMR/pull/1)**: create them last under `torch.random.fork_rng`. Then run ≥3 seeds per arm for the holdout neural-arm row (§4.3). |
| 4 | major | C | every notebook's setup cell: `if stage in _pre_repair and t.relative_to(RES).parts[0] in USES_RANDOM_GF:` | In fixed notebooks (`analysis`), `standard_*` members stay under `standard/`. `parts[0] == "standard"` is never in `USES_RANDOM_GF`, so a **pre-repair** `standard/trilens_attnfix/` is kept and becomes the "Standard" column of `tab:attnfix`. | `code/tests/test_notebook_unpack.py`: 7 failures on main (all fixed notebooks), 14/14 pass on the branch. | **PR [#2](https://github.com/FaiazPronoy/TriLens-AMR/pull/2)**. |
| 5 | major | F | setup cell: `for stage in [s for s in NEEDS + [NAME] if s not in have]` | The notebook's **own** release is unpacked last and `_take` overwrites unconditionally. `analysis` (`WRITES = ["*"]`) keeps copies of all upstream folders, so after an upstream re-run its old copies replace the fresh ones. The same happens with attached inputs. | `code/tests/test_notebook_resume_order.py`: on main, `assert 'old' == 'new'` in all 14 cases. | **PR [#3](https://github.com/FaiazPronoy/TriLens-AMR/pull/3)**. Still open: skip markers are existence-only (`run(cmd, marker)`), so derived files (`diagnostics/*.csv`) restored from an old release are not recomputed after an upstream change. Delete the `outputs-analysis` release after any upstream re-run, or key markers on upstream `_CODE_VERSION`. |
| 6 | major | B | `code/src/experiments/run_trilens_v10.py:965-974`; `code/scripts/diagnostics/check_leakage.py:143,152-153` | `if key in v9.files and not np.array_equal(...)` skips the row-order check when the arrays are absent. `except Exception as exc: log.warning(f"Cannot load v9.1 predictions ({exc}). Stacking TriLens only.")` turns a present-but-broken classical file into a TriLens-only headline. In `check_leakage`: `all(... for k in (...) if k in za.files and k in zb.files)` gives **PASS** over an empty generator, and check 5 adds no row for a missing index. | Code reading; `tests/test_classical_arm_loading.py` and `tests/test_check_leakage.py`. | **PR [#4](https://github.com/FaiazPronoy/TriLens-AMR/pull/4)**. |
| 7 | major | G, I | `code/requirements-kaggle.txt`; `code/src/experiments/_shared.py:210` (`deterministic=False`); XGBoost `device="cuda"` (`run_cascadestack_classical_mlst.py:56,67`) | The Kaggle environment is unpinned: `lightning`, `pytorch-tabnet` and `entmax` have no version. `torch`, `xgboost`, `lightgbm`, `sklearn` and `numpy` come from whatever image Kaggle serves. `requirements.txt` pins `xgboost==3.0.2` etc. but is never installed there, and no notebook records versions. The holdout re-run moved by more than the effects the paper discusses. | `results/classical`, holdout, against the paper: RF 0.4931→0.4931, LGBM 0.5516→0.5516, LogReg 0.4408→0.4402, but **XGB-gene 0.5507→0.5630 (+0.0123)** and **XGB-χ² 0.6299→0.6220 (−0.0079)**. These read the clean MLST file, so the repair cannot explain it. Only the GPU XGBoost learners moved: version or GPU nondeterminism, not a code bug. The +0.0038 "full model vs best component" on the holdout is smaller than this drift. | Pin the Kaggle extras. Write `pip freeze` plus `torch.version.cuda` and the GPU name into every zip (`_ENV.txt`). Run the classical MLST step twice on one image (§4.6). Report the holdout comparison with seed/rerun spread or drop the "+0.0038" claim. |
| 8 | major | F | `code/scripts/github_sync.py:63-69`; setup/`run()` cells | `upload_asset` deletes the old asset, then uploads. A kill (12 h limit) or a failed upload in between leaves the release empty, so resume restarts from zero and downstream sees "NOT FOUND". Zips are uploaded after **every step** with no completion marker, so a downstream notebook started early (the two-account plan relies on "after `classical` is finished", `kaggle/README.md:71-74`) consumes a partial upstream. | Code reading. | Upload as `<name>.part`, then delete the old asset and rename (PATCH `name`). Write `_COMPLETE.txt` only when `FAILED` is empty, and have downstream refuse a stage without it. |
| 9 | major | H | `paper/journal/report.tex:2326-2340` (`tab:ablation`), text "scored 0.781 on the random split"; `code/docs/PAPER_MAP.md:42` | Stage 1/2 values 0.7453 / 0.7807 "are not in any surviving file" (PAPER_MAP). The arithmetic is consistent (row means verified), but they have no source. | PAPER_MAP row `tab:ablation`. | Take them from the re-run `pillar2_layer_ablation/layer_ablation_results.csv`. |
| 10 | minor | E | `paper/journal/report.tex:2161`; `threshold_tuning.py:179`; `holdout_clinical_metrics.py:169` | "The CLSI limits require both the VME and ME rates to stay under three percent". The usual AST acceptance limits are VME ≤ 1.5% and ME ≤ 3%. The conclusion (no drug passes) is unchanged. | `print(... (df.VME_F2 < 0.03) & (df.ME_F2 < 0.03) ...)` | State 1.5% / 3% with the source, and use 0.015 for VME in both scripts. |
| 11 | minor | E | `code/scripts/diagnostics/matched_prevalence_check.py:161-176`; `report.tex:2638-2641, 2659-2663` | The holdout test set is matched to the **holdout training** prevalence, which is inflated because the big resistant clones were forced into training. It is then compared with the random split at its own test prevalence. The paper concedes "slightly … generous" without a number. It also says "no resistant strain was abandoned", but the `else` branch at `:163-170` subsamples positives. The interval reflects resampling only, not test-set sampling. | Code reading. | Add a column matched to the **random-split test** prevalence per drug and report that gap. Report whether the "subsampled positives" note occurs. Call the interval a resampling range. |
| 12 | minor | A | `code/src/experiments/run_tabnet_baseline_mlst.py:8-17,49`; `run_tabnet_baseline_phylogeny.py` | TabNet under the holdout and the phylogroup split uses the random-split-ranked gene set. That ranking saw labels of test strains of those splits (random train ⊃ part of their test). It is disclosed in KNOWN_ISSUES §9, **but not in the paper**. It favours the baseline. | `GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered.pkl"` | One sentence in the paper, or re-run TabNet on the split's own gene set (§4.5). |
| 13 | minor | B | `run_cascadestack_v9.py:519-520`; `run_cascadestack_classical_mlst.py:167-168`; KNOWN_ISSUES table line 159 | Classical `ensemble_mcc` / `ensemble_f2` use a val-tuned threshold (`thr, _ = _best_threshold_mcc(y_va, val_meta)`). KNOWN_ISSUES says "0.5 for every reported MCC". TriLens and `diagnose_mlst_final_comparison.py:55` use 0.5. | Code reading. | Never quote classical-file MCC/F2 next to TriLens MCC; or compute both at 0.5. |
| 14 | minor | B | `run_trilens_v10.py:899` vs `run_cascadestack_v9.py:510`; `configs/experiment/trilens_v10.yaml:82` | The two meta-learners differ (`LogisticRegression(C=C, …)` with no `class_weight` vs `class_weight="balanced"`). The yaml comment "MUST stay 42 — identical to v9.1 fold splits" is false: v9 uses `StratifiedKFold` per antibiotic on the labelled subset (`:427`), TriLens a `KFold` on all train (`:764`). The OOF folds differ. | Code reading. | State both meta-learners in Methods and correct the comment. |
| 15 | minor | A | `run_trilens_v10.py:784-787`; `build_mlst_lineage_holdout_split.py:240-242` | The ES set is the **last** 10% of each fold's sorted positions (`es_l = fit_es_local[n_fe - n_es:]`). On the MLST split `train_idx` is sorted by genome index, so ES = the highest genome indices, not a random sample (v9 uses `StratifiedShuffleSplit`). Not leakage, but possible selection bias. | Code reading. | `rng.permutation(fit_es_local)` before slicing, or report §4.7. |
| 16 | minor | A | `notebooks/master_features_and_splits.ipynb` cell 4 | The PCA dimension (1,024) was chosen by test PR-AUC: `print("PCA SEARCH RESULTS (TEST set)")`. This favours the baseline. | Notebook source. | Say so, or pick the dimension on val. |
| 17 | minor | A, E | `run_xgb_gene_importance.py:236-243` | Gene ranking uses **val** labels for early stopping, so val is not fully independent when it is used to tune thresholds (`threshold_tuning.py`). The effect is small: ranking only, summed over 7 drugs. | `evals=[(dtrain, "train"), (dval, "val")], early_stopping_rounds=early_stop` | Disclose in one sentence, or rank with an inner split of train. |
| 18 | minor | C | `code/scripts/repair_gene_filtered.py:61-62, 75-78` | A file with every value in 0..3 exits "clean" **without being compared** with the matrix, although damage inside 0..3 is now known to occur (45883b6). `where.setdefault(p, i)` silently maps a duplicated position name to its first column. | `tests/test_repair_gene_filtered.py`: 2 failures on main. | **PR [#5](https://github.com/FaiazPronoy/TriLens-AMR/pull/5)**. Then verify the phylogroup and holdout files too (§4.10). |
| 19 | minor | E | `holdout_clinical_metrics.py:121-147, 48-51`; `matched_prevalence_check.py:49-52`; `diagnose_prevalence_invariant_metrics.py:57` | `matched_VME` / `matched_ME` are no-ops: they are within-class ratios, so subsampling negatives leaves VME identical. The paper's caption says so correctly; the script's note "removes the prevalence shift" is wrong. Three scripts still fall back to pre-fix folders (`trilens_v10`, `*_attnfix`), against the "no fallback" rule in `threshold_tuning.py:53-54`. `matched_prevalence_check.py:218-219` prints stale random-split values. | Code reading. | Drop the matched clinical columns. Use only `TRILENS_MAIN_RUN` and fail when it is missing. |
| 20 | minor | F | notebook setup cell (`git clone https://x-access-token:{TOKEN}@…`); `github_sync.py:126,139` | The token is never printed (all errors pass through `_redact`) and never packaged. But it is stored in `/tmp/trilens/repo/.git/config` and passed on `git push` argv. `kaggle/README.md:3,23` still says the repository is private (it is public). | Code reading. | Use `git -c http.extraheader="Authorization: Basic …"` or a credential helper. Update the README. |
| 21 | minor | H | `code/scripts/diagnostics/collect_paper_numbers.py:174-176`; `code/docs/PAPER_MAP.md:45,87,90` | CLOSE allows up to 0.01, larger than the effects the paper reads (+0.0024, +0.0037, +0.0038). Paper values are typed in, not parsed from `report.tex`. PAPER_MAP gives two sources for 0.8033 (`baselines/xgb_phylogroup_results.csv` vs `pillar1/pillar1_combined.csv`) and for 0.8400 ("v9 file reads 0.8430" vs "trilens_attnfix base-learner columns, verified"). | File reading. | Tighten CLOSE to 0.002 for PR-AUC. Keep one source per number. |
| 22 | minor | E | `report.tex:2176-2181` | "Collapsing … can only raise our apparent very major error rate". Moving intermediates into R also enlarges the VME denominator, so the rate can fall if those strains are mostly called R. | Definition: VME = FN/(TP+FN). | "is expected to raise". |
| 23 | minor | D | `docs/history/code_audit.md:33-41` | Unknown whether the delivered `trilens_attnfix` used the concatenated `2*d_token` design or the current separate layers. They have different default inits (fan_in). | — | §4.2. |

## 3. Checked and found clean

**A. Leakage.** No test label or test feature enters training anywhere along a headline path.

- `run_xgb_gene_importance.py:159-245`: variance and χ² are computed on train rows only; XGB fits on train with early stopping on val; test is never loaded into the function. The `_phylogeny` and `_mlst` variants rank on their own split; `_mlst` asserts no train/test or train/val overlap (lines 78-80).
- `build_gene_filtered_dataset*.py`: the gene filter is "≥ 5 SNPs" (label-free) ∩ that ranking. No index is used.
- PCA (`TruncatedSVD`) and `StandardScaler` are fit on `idx_tr` only (`master_features_and_splits.ipynb` cell 5).
- `pergene_snp_blocks.py:231`: column choice uses train-only variance (label-free). `gene_aggregated_features.py` uses per-row statistics only.
- v9 and the classical arms fit χ² and variance on the fit fold only (`_train_xgb_raw_chi2`), use early stopping on a `StratifiedShuffleSplit` of the fit fold, and fit the scaler on the fit fold.
- Expert C standardisation is fit on train.
- The TriLens meta-learner is fit on OOF train predictions only. Thresholds are tuned on val only (`threshold_tuning.py:141-146`).

**Splits.**

- Random: `train_test_split(..., random_state=42, stratify=amp_strat)`. Phylogroup: stratified on the phylogroup label (documented as not a holdout).
- Lineage holdout (`build_mlst_lineage_holdout_split.py`): whole STs, seeded walk, hard asserts at 260-291. Strains without an ST go to train. `check_leakage.py` checks train/test and val/test ST disjointness and unknown-ST-in-test.
- `run_cascadestack_classical_mlst.py:109-117` asserts ST disjointness.

**B. Stacking.**

- Within each antibiotic, `tr_pos`/`va_pos`/`te_pos` (`run_cascadestack_v9.py`, `_run_one_antibiotic`) index the per-AB labelled subsets consistently across OOF, val and test.
- The saved `train/val/test_indices` follow the split order, and `antibiotic_names` follows `exp_cfg.antibiotics` = `[AMP, CIP, AMC, SXT, GEN, CTX, CAZ]`, which equals `AB_NAMES`.
- `run_meta_stack(return_val=…)` leaves `test_meta` unchanged. `val_is_stack` is saved and `threshold_tuning.py:122-125` refuses tuned columns without it.
- `diagnose_mlst_final_comparison.py:103-111` asserts that train and test indices are equal across the arms.
- The missing classical file raises `FileNotFoundError` (`run_trilens_v10.py:977-983`), and `_intentionally_absent` is used only by `run_trilens_mlst.py`.

**D. Switch.** The standard forward pass equals the original (`docs/history/code_audit.md:15-19`), and the creation order of `attn_V`/`attn_U`/`attn_w` is unchanged for the standard model. `TRILENS_LENSB` is read once, in `ExpertB.__init__` (`trilens.py:230`). The standard notebooks set it for every step and never unpack fixed TriLens stages (`_target` returns `None`). The analysis notebook builds the standard diagnostics in a separate project folder, with the shared classical runs symlinked read-only.

**C. Repair (as changed in 45883b6).** For the corruption it targets, the block rule is sound:

- every column outside `[lo, hi]` must equal the SNP matrix, which over ≈262k columns proves that row order and column mapping are correct;
- the rebuilt values must be in 0..3.

`USES_RANDOM_GF` lists every folder whose script reads `arcadia_v2_gene_filtered.pkl` (checked script by script): v9, the three TabNet runs, `pillar3_xgb_importance`, `pillar3_shap`, `pillar_shap_t6`, `trilens_attnfix`, `pillar2_layer_ablation`, `pillar_dedup_eval` (also covers `_mlst`) and `pillar_metrics_full`. The only gap was the `standard/` prefix (finding 4).

**E. Evaluation.**

- `_bootstrap_ci` (`run_trilens_v10.py:638-659`): a percentile bootstrap with 1,000 resamples over test strains, skipping degenerate resamples. The paper correctly gives no interval for the average and states the single-seed limit (`report.tex:2264-2286`).
- VME/ME/CA definitions are correct.
- Paper arithmetic checked:
  - `tab:main` means: PR-AUC 0.8486, MCC 0.7153, F1 0.7786, F2 0.7615, accuracy 0.9085.
  - `tab:clinical` mean VME 0.2489.
  - `tab:holdoutclin`: GEN 33/38 = 0.868, CTX 65/71 = 0.915, CAZ 36/43 = 0.837; mean 0.559.
  - `tab:previnv`: means 0.6337 / 0.7226 / 0.8439, every per-drug gap, and "two-fifths" (0.089/0.215 = 0.41).
  - `tab:ablation` row means 0.7453 / 0.7807.
  - The v9 reference in `run_trilens_v10.py:67-69` averages to 0.8469.

**F. Token handling.** `GITHUB_TOKEN` comes only from Kaggle secrets or the environment. No `print` of it exists in any notebook or in `github_sync.py`; clone and push errors are redacted (`replace(TOKEN or "\0", "***")`, `_redact`). It is never written into `results/` or a zip. Each notebook writes only to its own release and `results/<NAME>` branch.

**I. `results/classical` against the paper** (this run predates the repair, so every random-split folder in it is discarded downstream):

| Number | Paper | `results/classical` | Reading |
|---|---|---|---|
| Phylogroup only | 0.2914 | 0.2914 | identical (deterministic CPU) |
| PCA + XGBoost | 0.8033 | 0.8028 | −0.0005, GPU XGB noise |
| TabNet, random | 0.6951 | 0.7201 | +0.025; read the **unrepaired** file, will be redone |
| TabNet, holdout | 0.3255 | 0.3632 | +0.038; random-ranked genes, unrepaired file, will be redone |
| RF / LGBM / LogReg, holdout | 0.4931 / 0.5516 / 0.4408 | 0.4931 / 0.5516 / 0.4402 | identical or −0.0006 |
| XGB-gene / XGB-χ², holdout | 0.5507 / 0.6299 | 0.5630 / 0.6220 | **+0.0123 / −0.0079**: GPU XGBoost or version (finding 7), not a code bug |
| Classical stack, holdout | 0.628 | 0.6221 | −0.006, follows XGB-χ² |
| Giessen per drug (CIP/CTX/CAZ/GEN) | 0.9868 / 0.7876 / 0.7775 / 0.5480 | 0.9877 / 0.7856 / 0.7633 / 0.5394 | up to −0.014; Giessen does not read the damaged file, so this is nondeterminism |
| v9 components, random | 0.7809 … 0.8400 | — | **not produced** (χ² refuses the damaged file) |

## 4. Not checkable here, and the check to run

The data (≈10 GB) is not in the repository. Checks 4.1 and 4.10 are local, CPU-only or one-off. The rest run on Kaggle after merging the PRs, in this order: `classical` → `trilens_random` + `trilens_splits` → `standard_random` + `standard_splits` → `analysis`. Delete the old `outputs-analysis` release first (finding 5).

1. **Which file the thesis runs read.** On the thesis machine, compute the md5 of the first 200 MB of the local `data/processed/arcadia_v2_gene_filtered.pkl` and compare it with `data_check.json` (`3e6574659887f7722a1ed9adf339ee36`). Also run `python scripts/repair_gene_filtered.py --src <local file> --snp <local matrix> --out /tmp/x.pkl --report /tmp/r.json`:
   - exit 2 with PR #5 means the local copy was clean and the damage happened on upload;
   - exit 0 means the thesis numbers came from damaged data.
2. **Delivered Lens B design.** `torch.load` any `trilens_attnfix` checkpoint, or the saved state, and print the shape of `attn_Vc.weight` (or `attn_V.weight`): `(attn_dim, d_token)` means separate layers, `(attn_dim, 2*d_token)` means the concatenated design.
3. **Seed spread for `tab:attnfix`.** After PR #1, run `run_trilens_mlst` (fixed and `TRILENS_LENSB=standard`) and `trilens_v10` with `seed` ∈ {42, 123, 456}. Report mean ± sd for "Holdout, neural lenses alone" and "Random split, full stack". +0.035 is credible only if it exceeds about 2 sd.
4. **Prevalence matching to the right target.** In `analysis`, per drug, compute the random-split test prevalence, `Y[random_test, a][M[random_test, a]].mean()`. Re-run `matched_prevalence_check.py` with `prev_train` replaced by it and report the new average gap next to −0.126. CPU, seconds.
5. **Size of the TabNet leak.** `len(set(random_split["train_idx"]) & set(mlst["test_idx"]))` and the same for the phylogroup test. Optionally, run `run_tabnet_baseline_mlst.py` with `arcadia_v2_gene_filtered__mlst_holdout.pkl`.
6. **XGBoost determinism.** In one `classical` session, run `run_cascadestack_classical_mlst` twice into two folders. Diff the `solo_XGB-*` columns and print `xgboost.__version__`, `torch.__version__` and the GPU name. Non-zero diffs mean the holdout claims need run-to-run spread.
7. **ES slice composition.** On the MLST split, compare the ST and phylogroup mix of `train_idx[fit_es_local[n_fe-n_es:]]` with the rest of the fold, for each of the 5 folds (CPU, from the split pickle).
8. **`check_leakage.py` after the full run** (with PR #4): every check PASS. Read check 8 (OOF vs test gap per base learner; WARN above 0.05). Check 4: report the exact-duplicate count.
9. **Positive subsampling in matched prevalence.** Look for `note == "subsampled positives"` in `diagnostics/matched_prevalence.csv`. If present, fix the "no resistant strain was abandoned" sentence.
10. **The other two gene-filtered files.** With PR #5, run `repair_gene_filtered.py` on `arcadia_v2_gene_filtered__phylogeny_clean.pkl` and `__mlst_holdout.pkl` against the same SNP matrix: expect exit 2. Also count duplicated names, `len(pos) - len(set(pos))`, in `arcadia_v2_snp_matrix.pkl["snp_positions"]`.
11. **Final number check.** After `analysis`, open `results/diagnostics/paper_numbers.csv` and `strain_ablation.csv`. Every MISSING row must be gone, and every CHANGED row must be reflected in `report.tex`, including the Giessen table under the like-for-like reference (finding 2).

## Pull requests

| PR | Problem | Test |
|---|---|---|
| [#1](https://github.com/FaiazPronoy/TriLens-AMR/pull/1) | ExpertB: the strain-conditioned switch shifts the initialisation of every later parameter | `tests/test_expertb_init.py` |
| [#2](https://github.com/FaiazPronoy/TriLens-AMR/pull/2) | Notebooks keep pre-repair standard-model outputs (`standard/` prefix) | `tests/test_notebook_unpack.py` |
| [#3](https://github.com/FaiazPronoy/TriLens-AMR/pull/3) | Notebooks: a resumed run's old copies overwrite newer upstream outputs | `tests/test_notebook_resume_order.py` |
| [#4](https://github.com/FaiazPronoy/TriLens-AMR/pull/4) | Missing split indices are treated as matching (stacking and `check_leakage`) | `tests/test_classical_arm_loading.py`, `tests/test_check_leakage.py` |
| [#5](https://github.com/FaiazPronoy/TriLens-AMR/pull/5) | `repair_gene_filtered`: a clean-looking file is never compared with the matrix | `tests/test_repair_gene_filtered.py` |

PRs #2 and #3 both edit `_take` on adjacent lines. Whichever merges second needs a one-line conflict resolution that keeps both changes.

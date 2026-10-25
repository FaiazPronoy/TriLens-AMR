# Session Handoff — AMR Thesis (Full Session Record)

Written for another Claude/agent (or the user, later) to understand
everything done in this session, in order, without replaying the chat.
Covers the WHOLE conversation, not just the currently-active work — if you
only need current status, jump to **"Right now"** near the end. Last
updated: 2026-07-27, ~22:09, mid-run.

## What this project is

Undergraduate thesis: ML prediction of antimicrobial resistance (AMR) in
*E. coli* from genomic data (SNPs + gene presence/absence), 7 antibiotics
(AMP, CIP, AMC, SXT, GEN, CTX, CAZ), 6,962 genomes (Arcadia v2 dataset) +
Giessen external validation set (809 isolates, 4 antibiotics: CIP/CTX/CAZ/GEN,
Giessen's own column name for CAZ is `CTZ`, renamed on load). Two
architectures compared throughout: **TriLens-MoE-Stack** (3-lens neural:
motif CNN / gene presence-absence / cross-antibiotic, MoE gate fusion,
K-fold OOF + classical-base meta-stack) and **CascadeStack v9.1** (5
classical learners — XGB-gene, LGBM-gene, RF-gene, LogReg-gene,
XGB-raw-chi2 — plus an optional GeneAttentionNet 6th base, K-fold OOF,
LogReg meta-stack).

Two codebases:
- **Main repo** (`Thesis/`) — the real, historically-run pipeline.
  `src/experiments/*.py`, `src/data/*.py`, `configs/`, `results/`.
- **`Thesis/amr-genomic-data/`** — a separate, from-scratch refactor of 5
  Kaggle notebooks (`arcadia-v2-phase-a/b/c.ipynb`, `giessen-cross-validation.ipynb`,
  `autoencoder-pca-random-split.ipynb`) into a tested package. Built earlier
  in this session as its own closed deliverable (see "Part B" below).

## Absolute rules established by the user this session

1. **Never fabricate or guess a number.** Every number in every report must
   trace to a real file. Missing → say "NOT FOUND", never estimate.
2. **Never overwrite an existing results/ folder or existing file** (main
   repo). All new work goes into new, clearly-named folders/files. New
   pipeline variants of a `.pkl` get a `__suffix`
   (e.g. `arcadia_v2_gene_filtered__mlst_holdout.pkl`).
3. **Report every sub-step's result immediately** when it lands — never
   batch everything until the end.
4. **Flag methodologically significant findings (>0.02 PR-AUC deltas, split
   leaks, etc.) clearly and take NO autonomous interpretive decision** about
   what they mean for the thesis narrative. State facts; let the user decide.
5. **Verify, don't assume**, especially at three specific failure points that
   have each already caused a real bug in this project: (a) join/merge logic
   between two datasets — always join by ID, never by row position; (b)
   group/split exclusivity assertions; (c) index alignment when combining
   two saved prediction arrays.
6. Before running anything expensive: **read the code 2-3 times, verify
   signatures/shapes/imports, run a smoke test if feasible** — GPU time is
   scarce and this machine has hard-crashed mid-run twice already.
7. In the `amr-genomic-data` refactor specifically: **never silently change
   a number** (log every changed value in `docs/CHANGELOG_NUMBERS.md`,
   old vs new vs why) and **never invent a value** (unknowns become
   `# TODO(faiaz): unresolved — see docs/OPEN_QUESTIONS.md`).

---

## Part A — Result-chapter deliverables (earliest work this session)

Purely data → figures/tables, no narrative writing (that happens elsewhere).
Strict rule: no fabricated numbers, "NOT FOUND" for anything missing, never
edit existing files, output only to new `paper_assets/{figures,tables}/`.

**Tables built** (`paper_assets/tables/*.tex`, booktabs style, from real
CSV/MD data): `tab_main_results`, `tab_clinical_metrics`, `tab_layer_ablation`,
`tab_tabnet_comparison`, `tab_split_comparison`, `tab_giessen_transfer`,
`tab_gate_routing`, `tab_gene_annotation`, `tab_pipeline_progression`,
`tab_phylogroup_only`.

**Figures built** (`paper_assets/figures/*.pdf`, matplotlib, 300dpi, vector):
`fig_layer_ablation`, `fig_split_drop`, `fig_tabnet_vs_trilens`,
`fig_gate_routing`, `fig_giessen_retention`, `fig_pr_curves`,
`fig_confounding_score`.

**New experiment run for real**: `src/experiments/run_phylogroup_only_baseline.py`
(LogisticRegression + XGBoost on `phylogroup_onehot` only, i.e. can 8
phylogroup dummy variables alone predict resistance). Result:
**avg LogReg PR-AUC 0.2914, avg XGBoost PR-AUC 0.2912** — both near the
label-prevalence floor, i.e. phylogroup alone is a weak predictor (a useful
negative control).

**`paper_assets/section_gene_filtering.tex`** — plain-English (non-technical)
methodology section on Stage 1.5 gene filtering, written to match a
requested prose style, cross-checked against the actual notebook.

A double-escaping bug in the gene-annotation table generator (`esc()`
applied twice) and a bash-heredoc-corrupting-LaTeX issue (control chars
mangled) were both hit and fixed during this phase — the latter is why this
whole session uses the `Write` tool for literal file content instead of bash
heredocs.

## Part B — Number verification pass

A large list of specific numbers (main results, clinical metrics, layer
ablation, split comparison, bootstrap CIs, TabNet, Giessen, gate routing,
leakage/dedup, SHAP concordance, gene annotation, Pillar 1 locality) was
verified one by one against real files. Also resolved: **why XGBoost
PCA-1024 showed two different values (0.7992 vs 0.8033)** — both are
genuine, real numbers, from two different notebook cells with different
XGBoost hyperparameters (0.7992 = lightweight PCA-dim-search cell, 0.8033 =
the heavier `run_baseline_xgb_phylogroup.py` pipeline). Not a bug, not a
contradiction — just two different experiments that happen to share a name.

Status of 3 then-running experiments was checked
(`run_pillar_cascadestack_phylogeny`, `run_tabnet_baseline_phylogeny`,
`run_pillar_t6_multiseed`) — these become the main thread of Parts D-N below.

## Part C — Bug-fix authorization (main codebase)

User manually killed a stuck buggy process, then explicitly authorized
editing `src/data/gene_aggregated_features.py` to fix the phylogeny_split
passthrough bug (see Bug #1 in the "Bugs found" list below), with the
instruction to back up first (`.bak`), regression-test `random_split` is
byte-identical, and check all related files for the same bug pattern.

## Part D — The `amr-genomic-data` refactor (separate deliverable)

Full spec pasted verbatim ("Claude Code Prompt — AMR Dataset Pipeline
Refactor"). Two absolute rules (see rules #7 above). 11 named bugs (B1-B11)
targeted, verification manifest (E1-E8) with exact numbers to reproduce.

**Finished modules** (`src/amr_data/`): `config.py` (paths.yaml/params.yaml
loading, env auto-detect), `io.py` (pickle/npz + SHA256 sidecars), `labels.py`
(single `encode_phenotype()`, B2 fix — verified zero mismatches across
35,516 label cells), `phylogroup.py` (single canonical 57-entry
`ST_TO_PHYLOGROUP` dict, byte-identical-verified against both original
notebook copies), `splits.py` (`random_split()` B7 nan-cast fix,
`phylogroup_stratified_split()` — honest rename of the old "phylogeny_split",
documented as NOT a holdout, `lineage_holdout_split()` — NEW,
`GroupShuffleSplit`-based with mandatory disjointness assertions), `qa.py`,
`vcf_to_snp.py`, `gene_pa.py` (S2 orientation rule `present = (PLINK_value == 0)`),
`gene_filter.py` (per-split-regime gene selection, the B3 fix), `views.py`,
`metrics.py` (bootstrap CIs, clinical error rates), `baselines.py` (B8 fix —
row-chunked sparse conversion to avoid the ~9.3GB COO-array OOM), `giessen.py`
(CTZ→CAZ rename, selection-bias-fixed `honest_headline_strategy()`),
`amr_ensemblenet.py` (verbatim architecture transcription, attributed to
Siddiqui & Tarannum 2025, arXiv:2509.23552 — not executed, no TF in this env).

**`scripts/port_cluster_labels.py`** — actually run for real once
`amr_metax_dataset.pkl` (2.85GB) was supplied by the user: **100.0% match
rate**, 30 real disjoint clusters ported → `data/processed/ported_cluster_labels_v2.pkl`.
This became the grouping variable for the real `lineage_holdout_split`.

**Tests**: 85 (later 97 after Part E's viz module) pytest tests, all real-data
verified (session-scoped fixtures `pytest.skip()` if the underlying file is
missing), 1 deselected as `@pytest.mark.slow`. All passing.

**Docs**: `AUDIT_FINDINGS.md`, `CHANGELOG_NUMBERS.md`, `DATA_PROVENANCE.md`,
`OPEN_QUESTIONS.md`, `PIPELINE.md` — kept current across multiple rounds as
new files arrived.

Once `amr_metax_dataset.pkl` and `phenotype_matrix_08302024.csv` were both
confirmed present, all the open items that depended on them were resolved
for real (no more placeholders).

## Part E — Dataset visualization module (`amr-genomic-data`)

Requested dual-purpose: produce real thesis-ready figures AND simultaneously
smoke-test that the pipeline still loads/processes the real files correctly.
Used the `dataviz` skill's palette/form-selection guidance (adapted for
static PDF output, not interactive HTML — no hover/dark-mode toggle needed).

New module `src/amr_data/viz.py` + `scripts/make_figures.py`, 12 new tests
(97 total). **6 real figures generated** to `amr-genomic-data/results/figures/`:
`fig_label_overview` (resistance prevalence + label coverage per antibiotic),
`fig_population_structure` (phylogroup counts bar + top-4-and-Other donut:
B2=3542, Unknown=1697, D=723, A=317, Other=683), `fig_split_composition`
(train/val/test sizes across 4 split regimes, recomputed fresh from
`splits.py` — byte-identical cross-check against the existing
`split_diagnostics.csv`, all 28 rows), `fig_split_prevalence_shift`
(the key B1-evidence chart: diverging bars, near-zero shift for
random/phylogroup-stratified vs large shift for genuine holdouts),
`fig_gene_feature_space` (gene-size histogram before/after Stage 1.5 filter,
gene-presence-prevalence histogram, per-strain pan-genome complement
histogram), `fig_giessen_overview` (Giessen prevalence + split pie).

Two real layout bugs found and fixed via actually rendering and looking:
a y-axis-limit formula bug (used sum of parts instead of max single bar,
causing label collision with the title/legend) and a title-overlap bug in
the Giessen figure (fixed via a shorter combined single-line title instead
of a separately-positioned subtitle). Also found and did NOT silently fix: a
mismatch between a fresh `X_pa.mean(axis=0)` recompute of gene
rare/shell/core counts (17,802/14,639/0) and the pkl's own already-locked
`gene_prevalence_stats` (17,762/14,679/0) — traced to the stored stats
predating this refactor and using a slightly different row-filtering
convention; per Rule #7, the figure was changed to **cite the pkl's own
locked numbers** rather than ship a silently-different recomputation.

## Part F — Main-codebase bug-hunt continuation (5 priorities, ordered)

User's explicit priority order: (1) verify via code+logs whether TriLens's
T7 ablation fell into the same passthrough bug as CascadeStack; (2) apply
the fix with backup+regression test; (3) confirm the killed process, rename
`SEED_123` (see below); (4) rerun in order — CascadeStack-phylogeny (verify
split_key printed, GEN not near 0.70), then `run_trilens_t7_with_predictions.py`
(re-prioritized above TabNet), then `run_pillar_dedup_eval.py`, then TabNet
last; (5) verify 2 more numbers (GeneAttentionNet v1 avg, gene-agg XGBoost
avg — resolved, not detailed further here as they didn't surface issues).

**Side investigations along the way**: whether Arcadia's `strains_metadata.csv`
has a direct ST column or Phase B derived it (answered from code: direct
`MLST` column exists, needs Achtman-scheme parsing — this fed directly into
Part L's Step 0 much later); search for a previously-downloaded
`ecoli_7000_amr.zip`/extracted folder (found on disk at
`Thesis/ecoli_7000_amr/ecoli_7000_amr/`, containing
`dataset_generation/data/strains_metadata.csv` among other files — this is
the exact file Part L's join used); an ad-hoc pandas-only MLST parse
(`df['MLST'].apply(parse_mlst)`, no model/split) was requested and run at
this point in the conversation — **superseded later by the more rigorous,
cross-validated Part L Step 0 join**, whose numbers (6,487/6,962 = 93.2%
with an Achtman ST, 458 distinct STs) should be treated as authoritative.

## Bugs found this session (all fixed or deliberately worked around)

1. **`gene_aggregated_features.py::build_gene_feature_matrix_from_pkl()`**
   only passed through `random_split`, silently dropped `phylogeny_split`.
   Consuming code did `bundle.get(_split_key) or bundle["random_split"]`, so
   any phylogeny_split run silently fell back to random_split. **Fixed**
   (backed up to `.bak`, regression-tested: random_split byte-identical
   before/after, phylogeny_split now present with a genuinely different
   test_idx). This is why `results/pillar_cascadestack_phylogeny/SEED_123/`
   had to be renamed to `results/pillar_cascadestack_seed123_randomsplit/`
   with an explanatory `README.txt` — it was actually a random_split run
   mislabeled as phylogeny_split (kept, not deleted — it's a legitimate
   random_split seed-stability data point).

2. **TriLens T7's Expert A/C preprocessing** (`run_trilens_v10.py`) was fit
   on `random_split`'s train_idx even when evaluating on `phylogeny_split`
   (confirmed via direct log evidence from two real historical log files —
   the cache was built once under the outer `cfg.split_key` before any
   per-ablation loop, then silently reused for T7). New script
   `src/experiments/run_trilens_t7_with_predictions.py` fixes this (own
   isolated cache dir, fit on phylogeny_split's own train_idx). **This did
   NOT mean T7 used the wrong row-level split entirely** — separately
   confirmed via pkl inspection that phylogeny_split's row-level
   partitioning itself was correct (all 8 phylogroups present in both
   train/test, sizes matching random_split's 4872/1045/1045, 100% of
   phylogeny_split-test genomes' phylogroups present in train) — only the
   Expert A/C *preprocessing statistics* were wrong-split. Rerun result:
   **avg PR-AUC 0.7479** vs locked **0.7493** — 3/7 antibiotics moved >0.02
   in opposite directions (AMC −0.0395, CTX +0.0232, CAZ +0.0301), average
   looks stable but isn't (flagged to user per Rule #4, no interpretation
   given). Files: `results/pillar_trilens_t7_predictions/`.

3. **Classical-base predictions in `run_kfold_and_stack()`/`run_meta_stack()`
   (`run_trilens_v10.py`) always load from ONE FIXED path**
   (`cfg.v9_predictions` = `results/stage3_cascadestack_v9/predictions_saved.npz`,
   a **random_split**-trained file), regardless of `split_key`. For T7 this
   meant the classical half of the meta-stack came from different genomes
   than the neural half + labels, matched only by row position (both splits
   happen to have the same partition sizes: 4872/1045/1045). **User's own
   hint led to confirmation**: T7's avg (0.7479) is almost exactly Stage-2
   neural-only (0.7547), implying the misaligned classical arm was
   effectively disabled. **Confirmed via coefficients** (recomputed exactly,
   reproducing the real run): the meta-learner's LogReg fit gave near-zero
   weight to the 5 misaligned classical columns (range roughly −0.6 to
   +0.7) and leaned entirely on TriLens (coef ~3.5-6.0) — self-correcting,
   but still not a valid classical+neural stack; the ORIGINAL locked run's
   own historical log showed the identical pattern. Built a
   **properly-aligned version**
   (`src/experiments/diagnose_t7_classical_alignment.py`) using
   CascadeStack-phylogeny's own classical predictions (verified
   train/test-index-identical to T7's own npz, byte-for-byte, for both
   seed 123 and 456 — and separately verified the 5 classical bases are
   themselves byte-identical across those two seeds, only the excluded
   GeneAttentionNet 6th base varies by seed), giving **avg PR-AUC 0.8512**
   — dramatically higher, much closer to CascadeStack-phylogeny's own
   2-seed-avg 0.8533. Two follow-up verifications the user specifically
   requested: (a) confirmed via quoted code that the meta-learner fit is
   genuinely on OOF/train only, never test; (b) confirmed T6's own
   random_split classical-base alignment IS genuinely valid — its
   `train_indices`/`test_indices` are byte-identical (same order) to
   random_split's own, in BOTH `arcadia_v2_master_features.pkl` and
   `arcadia_v2_gene_filtered.pkl` — not just coincidentally same-shaped.

4. **Stage 1.5 gene selection leak**: the 5,000 selected genes in the
   currently-locked `arcadia_v2_gene_filtered.pkl` were ranked using ONLY
   `random_split`'s train partition (`run_xgb_gene_importance.py` line 63,
   hardcoded `master["random_split"]`), then that same gene set was reused
   for phylogeny_split evaluation — despite 73.1% of phylogeny_split's test
   genomes being in random_split's train. This inflates any
   phylogeny_split-evaluated number. **"Clean" rerun done in full** (see
   Part H below): **avg PR-AUC 0.8519** vs the leaked-gene-set's 0.8512 vs
   random_split's 0.8462 — the leak turned out to barely matter for the
   final number (+0.0007), though the actual GENE SETS differ substantially
   (only 2,726/5,000 = 54.5% overlap between the two rankings).

5. **Two more latent bugs found via a full read of everything the pipeline
   depends on** (`src/data/pergene_snp_blocks.py`, `build_expert_c_features`
   in `run_trilens_v10.py`), done specifically because this was framed as
   the LAST experiment and GPU time is scarce:
   - Cache files were trusted by **existence alone**, no identity check —
     exactly the bug class that caused #2 above. **Fixed**: added a
     `cache_identity.json` sidecar (SHA-256 of pkl_path + split_name +
     sorted train_idx) to both `build_pergene_snp_blocks` and
     `build_expert_c_features`, checked before any cache is reused;
     mismatch or missing sidecar → rebuild, never silent reuse. Fully
     backward compatible (new `split_name` param is trailing-optional with
     default `""`; all 5 existing call sites across the codebase verified
     unaffected). Regression-tested twice: synthetic (4 behaviors — fresh
     build / cache hit / different-split rebuild / missing-sidecar rebuild
     — all correct) and **real production data** (forced a full rebuild of
     Expert A blocks from a copy of the real phylogroup-clean cache; result
     was **byte-identical** to the original, already-reported cache;
     60.9s, real scale, original untouched). Also fixed a Windows-only
     `OSError` when overwriting an mmap'd file (defensive `unlink` before
     rebuild, added to both functions).
   - `build_expert_c_features`'s "fast path" via
     `build_gene_feature_matrix_from_pkl` is actually broken (tries
     `np.asarray(a_dict, np.float32)` on a function that returns a dict —
     needs `X_raw["X_gene_features"]`), always throws, always falls into
     the `except` → slower but mathematically-equivalent "inline" fallback.
     This exact failure was already visible in production logs this
     session as "Feature builder unavailable (float() argument must be a
     string or a real number, not 'dict'); building inline". **Deliberately
     NOT fixed** — every run all session has already gone through the
     inline fallback successfully; fixing an unrelated path right before
     the final run was judged not worth the risk.

6. **`GroupShuffleSplit(test_size=f)` controls the fraction of GROUPS, not
   samples** (verified empirically against sklearn 1.6.1: a 30% target gave
   30.4% of groups but 56.0% of samples on an uneven synthetic set).
   Directly relevant because Achtman ST sizes span 1 to 884 genomes — using
   `GroupShuffleSplit` naively for the MLST split (Part L) could have
   silently blown the required 800-1400 test-size bound. Worked around by
   walking a seeded (42) permutation of eligible STs, assigning whole STs
   until a genome-COUNT target is hit (deterministic, fixed before seeing
   any result, group-exclusive by construction).

## Part G — Overnight queue + two hard crashes

An unattended background queue (`overnight_queue.sh` and later per-step
wrapper scripts, each polling for the previous step's own output marker
before starting) was set up so the user could sleep while GPU jobs ran.

**Crash #1**: machine hard-hung and was force-power-cycled sometime around
10:21-10:53 the following morning, mid-way through `run_pillar_t6_multiseed.py`
seed=456's LAST fold (epoch 23/50). Diagnosed via Windows Event Log (ID
41/6008 — "unexpected shutdown", no BSOD/bugcheck event, no GPU-driver TDR
event, complete log silence for the whole gap) as a genuine hard freeze
requiring a manual power-cycle, NOT a sleep-timer issue (verified: sleep-
after is set to Never on the active "High performance" power plan) — most
likely GPU/resource contention from a concurrent screen-share session,
unconfirmed. **seed=123 was fully saved and intact**
(`results/pillar_t6_multiseed/SEED_123/`, avg PR-AUC 0.8466: AMP 0.9405,
CIP 0.9726, AMC 0.7557, SXT 0.8357, GEN 0.7037, CTX 0.9006, CAZ 0.8171) —
recovery script `src/experiments/run_pillar_t6_multiseed_seed456_only.py`
runs ONLY seed=456 (imports `run_one_seed`/`_aggregate` from the original,
unmodified script), avoiding redoing seed=123's already-correct ~2h40min.

**Crash #2** (smaller): a second, shorter interruption killed the just-
started Stage-1.5-clean-rerun chain literally at its very first step
(directory created, nothing written) — restarted cleanly from that step,
no other work lost (Stage A/B of that chain had already completed and were
intact).

## Part H — Stage 1.5 clean rerun (phylogroup-stratified arm, "leaked gene
set" bug #4 fixed)

Rebuilt gene selection using `phylogeny_split`'s own train partition instead
of `random_split`'s. 5 new scripts (none modify existing files):
`run_xgb_gene_importance_phylogeny.py` (Step A — real result: 7/7
antibiotics trained, e.g. SXT Val PR-AUC 0.8700, GEN 0.7701, CTX 0.9290, CAZ
0.8516), `build_gene_filtered_dataset_phylogeny.py` (Step B — 5,000 genes,
54.5% overlap with the locked random_split-ranked set), 
`run_cascadestack_classical_phylogeny_clean.py` (Step C — 5 classical bases,
no GeneAttentionNet, 1 seed=42; **avg 0.8530**), `run_trilens_t7_clean_gene_set.py`
(Step D — TriLens solo on the clean gene set; **avg 0.7423**, took 138.4
min), `diagnose_t7_clean_final_comparison.py` (Step E — properly-aligned
final stack; **avg 0.8519**). Locked/leaked/clean 3-way table:

| Antibiotic | Random split (T6) | Phylogeny leaked gene set | Phylogeny clean gene set |
|---|---|---|---|
| AMP | 0.9408 | 0.9338 | 0.9335 |
| CIP | 0.9729 | 0.9600 | 0.9596 |
| AMC | 0.7550 | 0.7957 | 0.7974 |
| SXT | 0.8315 | 0.8295 | 0.8292 |
| GEN | 0.7030 | 0.7108 | 0.7175 |
| CTX | 0.8986 | 0.8818 | 0.8786 |
| CAZ | 0.8212 | 0.8466 | 0.8473 |
| **Average** | **0.8462** | **0.8512** | **0.8519** |

Also ran, same background batch: **TabNet phylogeny baseline**
(`run_tabnet_baseline_phylogeny.py`, already split-safe by design — reads
`phylogeny_split` directly from the pkl, bypassing the buggy passthrough
entirely): AMP 0.8744, CIP 0.8852, AMC 0.6043, SXT 0.5176, GEN 0.4664, CTX
0.8280, CAZ 0.7035, **avg 0.6971**. And **dedup eval extended to
phylogeny_split** (`run_pillar_dedup_eval_phylogeny.py`, new script reusing
the existing `compute_dedup_eval()` unchanged): 51/1045 (4.9%) of
phylogeny_split test genomes were near-duplicates of a training genome;
after removing them, avg PR-AUC 0.7479→0.7426 (using the misaligned-
classical T7 predictions, since that was what existed at the time; CAZ
alone shifted −0.0264, the only >0.02 move).

## Part I — Prevalence-invariant metrics (ROC-AUC, MCC)

User noticed all 7 MLST-split antibiotics show negative prevalence shift
(train→test) because the big MDR clones are forced to train-only — since
PR-AUC's baseline IS prevalence, cross-split PR-AUC comparison is
confounded. Added ROC-AUC + MCC (fixed threshold=0.5, matching
`run_trilens_v10.py`'s own `compute_metrics()` convention, stated
explicitly) computed from already-saved predictions, no retraining:
`src/experiments/diagnose_prevalence_invariant_metrics.py`.

| Split | avg PR-AUC | avg ROC-AUC | avg MCC |
|---|---|---|---|
| random_split (T6) | 0.8461 (sanity-check recompute of 0.8462) | 0.9350 | 0.7084 |
| phylogroup-stratified (clean) | 0.8519 (exact match) | 0.9252 | 0.7342 |
| MLST lineage holdout | pending | pending | pending |

Per-antibiotic breakdown + test-set prevalence is in
`results/pillar_prevalence_invariant_metrics/prevalence_invariant_metrics.csv`.
The identical computation was added directly into
`diagnose_mlst_final_comparison.py` (Part L Step 5) so the MLST arm appears
automatically once that step runs.

## Part J — "Is this enough for defense / Q1/Q2?" (honest-opinion exchange)

User asked directly whether the session's work is enough for thesis defense
and Q1/Q2 journal publication. Answered honestly that this can't be fully
assessed from inside a coding session (no visibility into the actual written
manuscript, no literature-SOTA comparison, no committee-specific
expectations) but flagged: the methodological rigor (leak-hunting) is
genuinely strong and could itself be framed as a contribution, BUT most of
this session's work is *validating/fixing evaluation*, not *architecture
novelty* — if the committee's bar is novelty, this doesn't move that needle;
if the bar is rigor, it moves it a lot. The pending MLST holdout number was
flagged as the single most important remaining data point for the
"genuine generalization" claim. This was pure discussion/opinion, no code
changed.

## Part K — Deep "professor-level" code scan

Requested explicitly because "we don't want to catch this repeatedly, want
to solve it once." Read, in full, everything the queued MLST pipeline
depends on that had only been partially read before: `build_expert_a_blocks`,
`build_expert_c_features`, the 5 classical base trainers in
`run_cascadestack_v9.py`, `bootstrap_pr_auc`/`bootstrap_mcc`,
`compute_gene_features`, all cache-check logic. Findings: the two latent
cache-identity issues (fixed — see Bug #5 above), a minor pre-existing
inconsistency (`bootstrap_mcc()` is called without an explicit `threshold=`
in `run_cascadestack_v9.py`, so its CI uses 0.5 while the point-estimate MCC
uses a val-tuned threshold — not a blocker, affects only the exact CI
bound, inherited faithfully by this session's new scripts to match already-
reported numbers), and confirmation that everything else (hardcoded shape
assumptions, degenerate-gene edge cases, K-fold minimum-sample floors,
antibiotic ordering across `AB_NAMES`/pkl `label_names`/local lists) is
either structurally safe or already correctly handled.

## Part L — The MLST lineage holdout experiment (current work)

**Why**: the dataset's own `phylogeny_split` was discovered to NOT be a real
holdout — it's a *stratified re-partition* (every phylogroup proportionally
present on both sides, nothing held out). The thesis's Chapter 2 claims this
gap is addressed; it wasn't. This experiment builds a genuine group holdout
by Achtman sequence type (ST), so no test genome's ST appears in train —
what the "phylogeny-aware evaluation" claim actually requires.

**Step 0+1 — DONE.** `src/experiments/build_mlst_lineage_holdout_split.py`.
Joined `strains_metadata.csv`
(`ecoli_7000_amr/ecoli_7000_amr/dataset_generation/data/`, 6,985 rows) to
the pkl by TWO independent, duplicate-free ID columns (`Genome.Name`,
`SRA.Accession`), cross-validated they agree on all 6,962 genomes. Achtman
ST parsed via `MLST\.Escherichia_coli_1\.(\d+)` (Pasteur `_2` scheme
genomes NOT mixed in — different scheme, different lineage meaning for the
same integer). **93.2% match rate (6,487/6,962), 458 distinct STs** — above
the 90% stop-and-check threshold, so the pipeline proceeded. Split rules
(fixed before seeing any result): unknown-ST → train; any ST >5% of all
genomes → train (ST73=884, ST131=872, ST95=606, ST69=477 forced to train,
2,839 genomes total); remaining 454 STs assigned whole via a seeded (42)
permutation walk targeting ~15%/~15% test/val by genome count (not
`GroupShuffleSplit` directly — see Bug #6). Result:
**train=4,836 (240 STs), val=1,081 (115 STs), test=1,045 (103 STs)**. All
hard assertions passed (ST/index disjointness, ≥20 resistant AND ≥20
susceptible per antibiotic in test, test size in [800,1400]). Prevalence
shift (train→test) per antibiotic: AMP −8.2pp, CIP **−12.1pp** (largest),
AMC −5.9, SXT −2.5, GEN −10.4, CTX −8.8, CAZ −10.6 — under the 20pp stop
threshold the user set in advance, so the chain proceeded automatically
without asking. Split saved to `data/processed/mlst_lineage_holdout_split.pkl`
+ `results/mlst_lineage_holdout/` (diagnostics CSV, `SUMMARY.md`,
`SPLIT_OK.marker` gate file that downstream steps check for).

**Steps 2-5 — queued, chained, running unattended** (5 new scripts, none
modify existing files):
- Step 2a `run_xgb_gene_importance_mlst.py` — gene importance ranked on
  this split's own train (verbatim-locked hyperparameters from the original
  Hydra config, not reimplemented from memory).
- Step 2b `build_gene_filtered_dataset_mlst.py` — builds
  `data/processed/arcadia_v2_gene_filtered__mlst_holdout.pkl` (5000 genes,
  same locked filter formula/params as the other two arms; also computes
  and reports gene-set overlap with both other arms' selected genes).
- Step 3 `run_cascadestack_classical_mlst.py` — 5 classical bases (no
  GeneAttentionNet, 1 seed=42 — proven seed-invariant this session via
  Bug #3's investigation: the 5 classical bases were confirmed
  byte-identical across CascadeStack-phylogeny's seed 123 vs 456, only the
  excluded GeneAttentionNet varies), reuses `_run_one_antibiotic()` from
  `run_cascadestack_v9.py` unchanged via a dummy-GAN-array trick, discards
  its GAN-contaminated meta-fit, refits its own 5-base meta-learner.
  Re-verifies ST disjointness at this stage too, independently.
- Step 4 `run_trilens_mlst.py` — TriLens neural on the MLST split. Two bugs
  pre-emptively worked around (found by reading code before writing this,
  not by crashing): (a) `load_all_data()` drops unknown split keys →
  injected `mlst_lineage_holdout_split` into the data dict manually; (b)
  `run_kfold_and_stack()` would've loaded random_split's classical file
  (wrong length: 4872 vs this split's 4836 train) → pointed
  `cfg.v9_predictions` at a deliberately non-existent path (asserted it
  doesn't exist, as a guard), so it cleanly falls back to TriLens-solo.
  Also re-verifies ST disjointness independently at this stage.
- Step 5 `diagnose_mlst_final_comparison.py` — index-aligned meta-stack
  (asserts train/test indices identical between the neural and classical
  npz's before combining, and re-verifies ST disjointness a third time),
  builds the final 3-way comparison table AND the prevalence-invariant
  metrics (ROC-AUC/MCC, same convention as Part I) for this arm, all in
  one script.

**Queue mechanics**: each step is a SEPARATE backgrounded shell wrapper
(`.../scratchpad/mlst_1_importance.sh` .. `mlst_5_final.sh`) that polls for
the previous step's own output marker file before starting its Python
module — not one long script, so each step's completion fires its own
notification, satisfying Rule #3. Step 1's wrapper additionally gates on
`results/mlst_lineage_holdout/SPLIT_OK.marker` (exists — proceed) and on
`results/pillar_t6_multiseed/SUMMARY.md` (T6 seed=456 must finish first —
GPU exclusivity; could not safely reorder T6/TabNet since they were already
running inside a live, unstoppable script when this ordering request came
in, so the actual order became T6 → TabNet → MLST-steps-1-through-5, not
the originally-requested T6 → MLST → TabNet — user was told this explicitly
and accepted it).

## Part M — Cache identity-check fix

Covered under Bug #5 above in full detail (what was found, what was fixed,
how it was tested). Summarizing only the file list here:
`src/data/pergene_snp_blocks.py` (new `compute_cache_identity_hash()`,
`_load_and_check_identity()`, `_save_identity()` helpers + wired into
`build_pergene_snp_blocks`), `src/experiments/run_trilens_v10.py`
(`build_expert_a_blocks` threads `split_name` through; `build_expert_c_features`
gets the identical identity-check treatment, importing the same helpers),
`src/experiments/run_trilens_mlst.py` (updated, not-yet-run at the time, to
pass `split_name=SPLIT_KEY` for clearer logs).

## Part N — Calibration + TabNet/dedup on the MLST split (completeness pass)

Requested once MLST Steps 0-3 confirmed the experiment was real and worth
finishing off symmetrically with the other two splits.

**Calibration**: `results/posthoc_analysis/calibration.csv` already had
real, computed `brier_raw`/`ece_raw` per antibiotic (only `brier_cal`/
`ece_cal` — the POST-recalibration columns — were empty; 0 of 70 rows in
`reliability_bins.csv` have `calibrated=True`, confirming that half was
never even started). Traced the source
(`configs/experiment/posthoc_analysis.yaml`: `v9_results_dir: stage3_cascadestack_v9`)
— **this is CascadeStack v9.1 on random_split, not T6/TriLens** — labelled
precisely as such when reporting, not conflated with T6. Standard
methodology confirmed by reading the source (`src/experiments/run_posthoc_analysis.py`):
sklearn `brier_score_loss`, equal-width 10-bin ECE. Nothing computed from
scratch — just verified and reported the existing raw numbers; explicitly
did NOT touch the empty `_cal` columns (would require fitting/applying a
recalibration method, out of scope per instruction).

| AB | Brier (raw) | ECE (raw) |
|---|---|---|
| AMP | 0.0898 | 0.0318 |
| CIP | 0.0260 | 0.0550 |
| AMC | 0.1445 | 0.1022 |
| SXT | 0.1095 | 0.0788 |
| GEN | 0.0982 | 0.2106 |
| CTX | 0.0524 | 0.1132 |
| CAZ | 0.0800 | 0.1598 |

**TabNet on MLST** (`src/experiments/run_tabnet_baseline_mlst.py`, queued):
deliberately uses the SAME gene-filtered pkl as the other 2 TabNet runs
(`arcadia_v2_gene_filtered.pkl`, random_split-ranked genes — NOT the
MLST-clean-ranked one) and identical hyperparameters, verbatim from
`configs/experiment/tabnet_baseline_phylogeny.yaml` — only the split source
differs (MLST split lives in its own `mlst_lineage_holdout_split.pkl`, not
inside the gene-filtered pkl). This is intentional: the point of a
TabNet-across-3-splits table is "same features/model, only the split
changes"; using a different gene set here would break that comparability.
Re-verifies ST disjointness independently before training.

**Dedup eval on MLST** (`src/experiments/run_pillar_dedup_eval_mlst.py`,
queued): **could not** call `run_pillar_dedup_eval.compute_dedup_eval()`
directly like the phylogeny_split version did — that function does
`d["random_split"-style key]` lookups hardcoded to `arcadia_v2_gene_filtered.pkl`,
which structurally cannot resolve a split living in a separate file. Worked
around by reusing `_find_near_duplicates()` (the actual algorithm, same SNP
feature space, same thresholds — unchanged) directly, supplying MLST's own
train_idx/test_idx instead of going through the pkl-embedded-split lookup.
**Deliberately made to depend on Step 5's output specifically**, not an
earlier intermediate: this fixes a known imprecision in the EXISTING
phylogeny_split dedup number (Part H), which had to use the pre-alignment-
fix T7-solo predictions because the properly-aligned version didn't exist
as a saved artifact yet at the time. To make this possible,
`diagnose_mlst_final_comparison.py` (Step 5, not yet run at the time of this
edit — confirmed safe) was given one small addition: it now also saves
`results/pillar_mlst_final_comparison/predictions_saved.npz`
(`test_ens_probs`/`test_indices`/`train_indices`) alongside its existing
CSV outputs, so a dedup eval has something correct to point at. No other
change to that script's logic.

Both queued as `mlst_6_tabnet.sh` (waits for Step 5's
`three_way_comparison_FINAL.csv`) → `mlst_7_dedup.sh` (waits for Step 6's
`results/pillar_tabnet_mlst/SUMMARY.md`) — same poll-then-run pattern as
everything else, launched and confirmed correctly waiting.

---

## ALL QUEUED WORK COMPLETE (as of 2026-07-28, ~00:15)

Steps 1-7 of the MLST pipeline (Part L + Part N) all finished successfully.
**Final 3-way comparison** (`results/pillar_mlst_final_comparison/three_way_comparison_FINAL.csv`):

| Antibiotic | random_split (T6) | phylogroup-stratified (clean) | **MLST lineage holdout** |
|---|---|---|---|
| AMP | 0.9408 | 0.9335 | 0.9036 |
| CIP | 0.9729 | 0.9596 | 0.9156 |
| AMC | 0.7550 | 0.7974 | 0.4764 |
| SXT | 0.8315 | 0.8292 | 0.7836 |
| GEN | 0.7030 | 0.7175 | 0.2408 |
| CTX | 0.8986 | 0.8786 | 0.6063 |
| CAZ | 0.8212 | 0.8473 | 0.4834 |
| **AVERAGE** | **0.8462** | **0.8519** | **0.6300** |

Component averages: classical-only (MLST) 0.6278, TriLens-solo (MLST)
0.4369. TabNet (MLST) avg 0.2947 (`results/pillar_tabnet_mlst/`). Dedup
eval (MLST): **0/1045 test genomes near-duplicate a train genome** (0%,
vs non-zero for the other two splits — consistent with holding out whole
STs, which removes near-identical clone-mates from crossing the split by
construction) — PR-AUC before/after dedup therefore identical, no drop.
Nothing left queued. This is the terminal state of the session unless new
work is requested.

## Right now (check this first, then update it) [SUPERSEDED — see above]

As of 23:10: **T6 multiseed is fully done** (3-seed avg PR-AUC **0.8467 ±
0.0007**: AMC 0.7553, AMP 0.9409, CAZ 0.8210, CIP 0.9734, CTX 0.9005, GEN
0.7020, SXT 0.8338 — `results/pillar_t6_multiseed/SUMMARY.md`). TabNet (the
phylogroup-clean-arm one, already reported: avg 0.6971) also already ran as
part of that earlier chain. **MLST pipeline progress**: Steps 0-1 (split,
done, see Part L), **Step 2a done** (gene importance, e.g. AMP 0.9070, CIP
0.9375, AMC 0.6127, SXT 0.8012, GEN 0.3650, CTX 0.5001, CAZ 0.5448), **Step
2b done** (filtered pkl, 5000 genes, only 48.2%/48.0% gene-set overlap with
the random_split/phylogroup-clean rankings respectively), **Step 3 done**
(5 classical bases, **avg PR-AUC 0.6278** — AMP 0.9020, CIP 0.9183, AMC
0.4759, SXT 0.7795, GEN **0.2349**, CTX 0.5994, CAZ 0.4842 — a real,
substantial drop vs the other two splits' classical arms of 0.8469/0.8530,
reported factually, no interpretation given). **Step 4 (TriLens neural) is
currently running** — the long one, ~138min based on the phylogroup-clean
arm's equivalent run. Steps 5, 6 (TabNet-MLST), 7 (dedup-MLST) are queued
and correctly waiting on their markers.

**To check progress**: find the currently-running Step 4 background task by
its log content (prints `ep X/50` lines like the other TriLens runs — do
not trust a specific stale task ID, session may have restarted), or check
whether `results/pillar_trilens_mlst/predictions_saved.npz` exists yet (→
Step 4 done, Step 5 running or done). Then check
`results/pillar_mlst_final_comparison/three_way_comparison_FINAL.csv` (→
Step 5 done — the core 3-way answer), then
`results/pillar_tabnet_mlst/SUMMARY.md` (→ Step 6 done), then
`results/pillar_dedup_eval/SUMMARY_mlst.md` (→ Step 7 done, everything
queued this session is finished). Each wrapper script announces "waiting
for X" / "prerequisite ready, starting Y" / "DONE" / "FAILED" in its own
output — check those directly.

**When Step 5 lands**: report the 3-way table plainly, no interpretation,
per Rule #4. Given Step 3's classical result already dropped substantially
(0.6278 vs 0.8469/0.8530) and GEN cratered to 0.2349, do **not** be
surprised if Step 5's full-stack MLST number is meaningfully lower than
0.8462/0.8519 too — that is the expected, honest, defensible finding this
whole experiment exists to surface (see Part J), not a sign anything is
broken. Cross-check against the already-known Step 3 number as a sanity
floor (the full stack, with TriLens added, should be >= Step 3's classical-
only 0.6278 in a healthy run, though not necessarily by much).

## Where things live (quick reference)

- Real per-run scripts: `src/experiments/*.py` — files with `_mlst`,
  `_phylogeny_clean`, `_phylogeny`, `_clean_gene_set`, `_seed456_only`
  suffixes, plus `diagnose_*.py` and `build_mlst_lineage_holdout_split.py`,
  are all new this session; everything else predates this session.
- Shared-library edits this session (identity check only, otherwise
  untouched): `src/data/pergene_snp_blocks.py`,
  `src/experiments/run_trilens_v10.py` (`build_expert_a_blocks`/
  `build_expert_c_features` only). `src/data/gene_aggregated_features.py`
  (Part C fix, `.bak` backup exists alongside it).
- Background wrapper scripts (poll-then-run pattern), all in the session
  scratchpad directory: `mlst_1_importance.sh` .. `mlst_7_dedup.sh`,
  `stage_g_t6_seed456.sh`, and the earlier `stage_a`..`stage_f` scripts for
  the phylogroup-clean arm + TabNet (already completed, kept for
  reference), plus the original `overnight_queue.sh` (superseded, dead
  after crash #1).
- Locked/reference numbers: `results/trilens_v10/ablation_results.csv` (T6,
  original locked T7), `results/stage3_cascadestack_v9/` (random_split
  classical, 2-seed).
- `amr-genomic-data/` — separate package, self-contained, see its own
  `README.md`. Not part of the active GPU pipeline described above.
- `paper_assets/{tables,figures}/` — Part A's deliverables, main repo root.

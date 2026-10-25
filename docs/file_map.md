# File map: where the old files went

The project used to be spread over `Desktop\Thesis`, `D:\`, `I:\` and Kaggle. This is where
each piece is now, and what was deliberately left behind.

---

## Which old folder was the latest

| Old folder | What it was | Last change | Now |
|---|---|---|---|
| `Desktop\Thesis\Thesis` | **the working code, data and results** | 8 Aug 2026 | `code/` |
| `Desktop\Thesis\Latex` | **final thesis LaTeX** | 17 Aug 2026 | `thesis/` |
| `Desktop\Thesis\22301061_..._21301474.pdf` | final submitted PDF (= `Latex\main.pdf` through iLovePDF) | 17 Aug 2026 | `thesis/thesis.pdf` |
| `Desktop\Thesis\IEEE_Conference_Template` | IEEE versions | 12 Aug 2026 | `paper/` |
| `Desktop\Thesis\Thesis\trilens-moe-stack` | draft repository docs | 8 Aug 2026 | merged into `code/` and `code/docs/` |
| `Latex 29_07_26`, `30_07_26`, `1_08_26`, `04_08_26`, `7-8-26`, `Submitted Version Backup`, `Claude Edit`, `Original Book Version`, `Latex old`, `P3 draft pronoy` | older thesis copies | Jul to 16 Aug | not copied (superseded by `Latex`) |
| `Desktop\Thesis\Thesis 2`, `Thesis Final`, `Thesis Claude Project`, `6 23`, `Claude` | older notebooks / empty | Feb to May | not copied |
| `D:\THESIS 1_08_26` | snapshot of the code folder on 1 Aug, before the delivered run | 1 Aug | not copied (same pre-fix results) |
| `D:\Thesis`, `D:\Thesis 1`, `I:\Thesis_23_5_26`, `I:\Thesis (Research Lab) - 23-5-26` | old pipeline data, notebooks, drafts | 2025 to May 2026 | not copied, except two defence decks |
| `D:\ecoli_7000_amr` | Arcadia raw release 1 (VCFs, metadata, pan-genome FASTA) | Aug 2025 | small files in `code/data/raw/` (by `setup_data.ps1`) |
| `I:\arcadia_v2_snp_matrix.pkl`, `I:\arcadia_v2_gene_presence_absence_v2.pkl` | copies of processed data | | `code/data/processed/` has the working copies |
| Kaggle notebooks (18 files) | data building + old pipeline | | 3 in `code/notebooks/`, 10 in `archive/`, 5 were empty (0 bytes) |

---

## Code

| Old | New |
|---|---|
| `Thesis\src\`, `Thesis\configs\` | `code/src/`, `code/configs/` (backups, `rmrnet`, `diagnose_t7_classical_alignment` removed) |
| `Thesis\attention_diagnosis_attnfix.py` | `code/scripts/diagnostics/attention_diagnosis.py` |
| `Thesis\lineage_marker_test_attnfix.py` | `code/scripts/diagnostics/lineage_marker_test.py` |
| `Thesis\prefix_baserate_lensB_attnfix.py` | `code/scripts/diagnostics/prefix_baserate_lens_b.py` |
| `Thesis\prove_lensB_invariance.py` | `code/scripts/diagnostics/prove_lens_b_invariance.py` |
| `Thesis\holdout_clinical_metrics.py`, `matched_prevalence_check.py`, `threshold_tuning.py`, `verify_thesis_numbers.py`, `verify_giessen.py`, `scripts\verify_env.py` | `code/scripts/diagnostics/` (same names) |
| `Latex\make_all_figures.py`, `Latex\make_thesis_figures.py` | `code/scripts/figures/` |
| `Thesis\make_figures.py` | `code/scripts/figures/make_results_figures.py` |
| `Thesis\amr-genomic-data\` | `code/amr-genomic-data/` |
| Kaggle `arcadia-v2-phase-a/b/c` | `code/notebooks/snp_matrix_from_vcf.ipynb`, `master_features_and_splits.ipynb`, `tabular_baselines_and_gene_pa.ipynb` |
| `Thesis\make_split_figures.py`, `make_st_figure.py`, `make_thesis_figures.ps1`, `preflight.py`, the non-`attnfix` diagnostic scripts, `trilens-moe-stack\build_repo.ps1` | not copied (superseded; reasons in `code/docs/EXCLUDED.md`) |

## Results and data

| Old | New |
|---|---|
| `Thesis\data\processed\`, `Thesis\data\Giessen_dataset\` | `code/data/processed/`, `code/data/Giessen_dataset/` |
| `Thesis\results\<run>\` | `code/results/<run>/` |
| `Thesis\results\pillar_trilens_mlst`, `pillar_trilens_t7_clean_gene_set`, `pillar_mlst_final_comparison`, `pillar_t7_clean_final_comparison`, `pillar_metrics_full`, `pillar_dedup_eval`, `pillar_prevalence_invariant_metrics`, `pillar_trilens_t7_predictions` | `code/results/pre_fix/<run>/` (computed from the model before the fix) |
| `Thesis\*_attnfix.csv`, `attention_amr_hits.csv` | `code/results/diagnostics/` (renamed to match their scripts) |
| `Thesis\threshold_tuning_results.csv`, `holdout_clinical_metrics.csv`, `matched_prevalence_results.csv`, other loose CSVs | `code/results/diagnostics/pre_fix/` |
| `Thesis\experiments\` (Hydra run logs), `Model Code.zip`, `thesis_audit.zip`, `audit.csv` | not copied |

## Thesis, paper, defence

| Old | New |
|---|---|
| `Latex\main.tex`, `chapters\chapter_1..6.tex`, `core\`, `bibliography\references.bib` | `thesis/` (same structure) |
| `Latex\images\` (only the files the thesis includes) and the signature images | `thesis/images/`, `thesis/` |
| `IEEE_Conference_Template\report.tex` (journal) | `paper/journal/` (recompiled: `report.pdf`, 33 pages) |
| `IEEE_Conference_Template\paper.tex` (conference) | `paper/conference/` |
| `Latex\DEFENCE_*.md`, `TRILENS_PURO_BUJHO.md`, `ML_SHIKHO_BANGLA.md`, `SHEETA_PORO.md`, `AMAR_PART_22_29.md` | `defense/guides/` (lower-case names) |
| `I:\...\Thesis-Defence-TriLens-MoE-Stack-2.pptx`, `AMR_Thesis_Defense.pptx` | `defense/slides/defence_slides.pptx`, `defence_slides_earlier.pptx` |
| `Thesis\SESSION_HANDOFF.md`, `RERUN_ALL.md`, `RUN_GUIDE.md`, `Latex\CODE_AUDIT.md`, `SELF_INFLICTED_WEAKNESSES.md`, `REVIEW_minimal_fixes.md`, `THESIS_HANDOVER_CONTEXT.md` | `docs/history/` |
| `Latex\REWRITE_*`, `SLIDE_*`, `THESIS_AUDIT_*`, agent prompts, other groups' PDFs and slides, friends' book PDFs | not copied (working notes for finished tasks) |

The final defence deck used on the day was not found on any drive (only a lock file
`~$T2510649 updated.pptx` on the Desktop). The two decks in `defense/slides/` are the latest
ones that exist.

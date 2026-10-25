# What is not in this repository, and why

Being explicit about this so nobody wastes time looking for something that was never here.

---

## 1. Large binary files

Not in git, by design. `.npz` prediction arrays and `.pkl` datasets are listed in `DATA.md`.
Committing them would push the repository past GitHub's limits.

## 2. Raw source data

The Arcadia and Giessen raw releases are not redistributed. They belong to their publishers
and are openly available from the original sources. `DATA.md` says where.

---

## 3. Code that was left out

| Excluded | Why |
|---|---|
| `run_rmrnet.py`, `src/models/rmrnet.py`, `configs/experiment/rmrnet.yaml` | The standalone RMRNet was implemented but never run to completion. No results folder exists. TriLens's Expert C (the cross-antibiotic lens) lives inside `src/models/trilens.py` and does not import it. |
| `diagnose_t7_classical_alignment.py` | Debugging trail for a prediction-alignment bug that was fixed before any reported result. |
| `preflight.py` | One-time check run before the strain-conditioned model was trained (August 2026). It asserted that `results/trilens_attnfix/` did not yet exist, so it fails on purpose now. |
| `make_split_figures.py`, `make_st_figure.py`, `make_thesis_figures.ps1` | Superseded by `scripts/figures/make_thesis_figures.py` and `make_all_figures.py`, which draw the same figures in the final palette. |
| The non-`attnfix` versions of the attention, lineage-marker and prefix scripts | Pre-fix versions of the scripts now in `scripts/diagnostics/`. Their outputs are kept in `results/diagnostics/pre_fix/`. |
| `Thesis_Stage1/`, `paper_assets/`, Hydra run folders (`experiments/`) | Early snapshots and run logs, superseded. |
| `*.bak`, `*.stage0_backup`, `*.pre_fix_backup`, `models.rar` | Backup copies of files that are included. |

The old 80,984-feature pipeline (autoencoder, CascadeStack v3, untuned baselines) is kept
outside the repository in `../../archive/legacy_pipeline_notebooks/`. No number in the thesis
comes from it.

---

## 4. Experiments that were run but are not in the paper

| Experiment | Why it is not in the paper |
|---|---|
| Multi-seed reruns (`pillar_t6_multiseed`, `pillar_cascadestack_seed123_randomsplit`) | Stability checks. They confirm the headline is not seed luck, but the thesis reports the seed-42 run throughout. Worth reporting in the journal version. |
| `pillar_t7_classical_alignment_fix` | Output of the debugging trail above. |

---

## 5. What is *not* excluded, despite looking like a dead end

The earlier model versions are included: `run_cascadestack_v7/v8/v9.py`,
`run_deepmeta_v11.py`, `run_gene_attention_v2.py`, `run_stage2_*.py`. The design-progression
table in Chapter 5 reports their scores as evidence for why the final architecture is hybrid
rather than purely neural, so removing them would break a claim the paper makes.

Same for the two "failed" experiments: the naive 1-D CNN and the phylogroup-stratified split.
Both are reported as findings. The CNN failure is what motivates convolving only inside gene
blocks; the phylogroup split scoring *higher* than the random split (0.853 against 0.849) is
the evidence that it never tested generalisation at all.

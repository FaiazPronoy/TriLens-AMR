# Stage 1 Pillar 1 — Run Guide

Generated for Faiaz Abrar Zaman, BRAC University CSE, AMR-PhyloGene-MTL thesis.
Target hardware: RTX 4080 Super 16 GB, Ryzen, Windows 11, `D:` drive.

---

## 0. File map

Copy these 19 files into your existing Stage 0 project tree at
`D:\T2510649\Thesis\amr_thesis\` (or wherever your Stage 0 lives):

```
configs/
  model/
    cnn_1d_naive.yaml                       (NEW)
    cnn_1d_contig_aware.yaml                (NEW)
  experiment/
    pillar1_exp1.yaml                       (NEW)
    pillar1_exp2.yaml                       (NEW)
    pillar1_exp3.yaml                       (NEW)
    baseline_xgb_phylogroup.yaml            (NEW)

src/
  data/
    contig_parser.py                        (NEW)
    transforms.py                           (NEW)
    arcadia_snp_loader.py                   (NEW)
    giessen_loader.py                       (NEW)
  models/
    cnn_variants.py                         (NEW)
  experiments/
    __init__.py                             (NEW)
    _shared.py                              (NEW — internal helper)
    run_pillar1_exp1.py                     (NEW)
    run_pillar1_exp2.py                     (NEW)
    run_pillar1_exp3.py                     (NEW)
    run_baseline_xgb_phylogroup.py          (NEW)
    aggregate_pillar1_results.py            (NEW)
  run_experiment.py                         (OVERWRITE Stage 0 stub)
```

---

## 1. PowerShell — create missing directories

Run once from the project root in PowerShell:

```powershell
cd D:\T2510649\Thesis\amr_thesis      # adjust to your actual project root

# create directories (no-op if they already exist)
New-Item -ItemType Directory -Force -Path `
    "configs\model", `
    "configs\experiment", `
    "src\data", `
    "src\models", `
    "src\experiments", `
    "results\pillar1\exp1\ckpts", `
    "results\pillar1\exp2\ckpts", `
    "results\pillar1\exp3\ckpts", `
    "results\pillar1\figures", `
    "results\baselines", `
    "paper" | Out-Null

Write-Host "Directories ready."
```

---

## 2. Stage 0 config sanity check

Before running, open `configs/config.yaml` and confirm these keys exist
(my experiment configs reference them). If a key is missing, add it:

```yaml
seed: 42
use_wandb: true

paths:
  arcadia_snp_matrix_pkl: D:/T2510649/Thesis/data/processed/arcadia_v2_snp_matrix.pkl
  arcadia_master_pkl:     D:/T2510649/Thesis/data/processed/arcadia_v2_master_features.pkl
  giessen_data_csv:       D:/T2510649/Thesis/Giessen_dataset/cip_ctx_ctz_gen_multi_data.csv
  giessen_pheno_csv:      D:/T2510649/Thesis/Giessen_dataset/cip_ctx_ctz_gen_pheno.csv
  results_dir:            ${oc.env:PROJECT_ROOT,D:/T2510649/Thesis/amr_thesis}/results

data:
  arcadia:
    split_type: random      # used by Pillar 1; switch to 'phylogeny' for Pillar 4
```

Note: forward slashes work in Windows paths inside YAML strings. Hydra
resolves them correctly via `pathlib.Path`.

---

## 3. Execution sequence

Run these from the project root, in this order. The `WANDB_MODE=offline`
prefix is optional but recommended for the first dry run.

```powershell
# Activate env
conda activate D:\T2510649\miniconda3\envs\amr
cd D:\T2510649\Thesis\amr_thesis

# 3.1 — Quick baseline (15 min)
python -m src.run_experiment experiment=baseline_xgb_phylogroup

# 3.2 — Exp 1.1: Naive CNN on both datasets, 4 antibiotics each (~8 runs)
python -m src.run_experiment experiment=pillar1_exp1

# 3.3 — Exp 1.3: Position shuffling, 1 antibiotic × 2 datasets × 2 orderings (4 runs)
python -m src.run_experiment experiment=pillar1_exp3

# 3.4 — Exp 1.2: Naive vs Contig-aware on Arcadia (8 runs total)
#                Heavy. Run last so the cheap stuff is done first.
python -m src.run_experiment experiment=pillar1_exp2

# 3.5 — Aggregator (reads all CSVs above)
python -m src.experiments.aggregate_pillar1_results --results-dir results
```

### Optional disable WandB

If you want to skip WandB entirely:

```powershell
$env:WANDB_MODE = "disabled"
python -m src.run_experiment experiment=pillar1_exp1 use_wandb=false
```

### Optional single-AB smoke test (15 min)

```powershell
python -m src.run_experiment experiment=pillar1_exp1 \
       experiment.antibiotics=[CIP] \
       experiment.datasets=[giessen] \
       experiment.training.max_epochs=3
```

---

## 4. Expected runtime on RTX 4080 Super 16 GB

Estimates assume `batch_size_arcadia=16`, `batch_size_giessen=32`,
mixed-precision, early stop at patience=8. Real runs may stop earlier.

| Experiment | Configs run | Per-run time | Total |
|---|---|---|---|
| `baseline_xgb_phylogroup` | 7 abs × 2 variants = 14 fits | ~30–60 s | **~10–15 min** |
| `pillar1_exp1` (Naive CNN, Giessen) | 4 abs × ~10 epochs × ~5 s/epoch | ~5 min | **~20 min** |
| `pillar1_exp1` (Naive CNN, Arcadia) | 4 abs × ~10 epochs × ~3–4 min/epoch | ~35 min | **~2.5 h** |
| `pillar1_exp3` (shuffling, CIP only) | 4 runs (2 datasets × 2 orders) | mixed | **~1.5 h** |
| `pillar1_exp2` (contig-aware, Arcadia) | 4 abs × 2 models | Naive ~35 min, Contig-aware ~1–2 h | **~10–12 h** |
| `aggregate_pillar1_results` | post-processing only | seconds | **<1 min** |

**Grand total Pillar 1 ≈ 14–16 hours on RTX 4080 Super.**

Bottlenecks to watch:
- Arcadia Naive CNN: 535 K seq through 4 conv blocks → ~3 min/epoch at bs=16.
- Contig-aware Variant A: 1271 contigs gathered + 2-layer CNN; should be
  ~1.5–2× naive depending on `max_contig_len` (95th percentile auto-chosen).
- If Variant A OOMs at bs=16: drop to bs=8 (config override
  `experiment.training.batch_size=8`).

---

## 5. Success criteria — automated checks

The aggregator computes these verdicts automatically and writes them to
`results/pillar1/SUMMARY.md` + `results/pillar1/verdicts.json`.

| Check | Statistic | Pass threshold |
|---|---|---|
| Exp 1.1 — Gap test | (avg Giessen PR-AUC) − (avg Arcadia PR-AUC) | ≥ +0.10 |
| Exp 1.2 — Recovery | (avg Contig-aware PR-AUC) − (avg Naive PR-AUC) | ≥ +0.05 |
| Exp 1.3 — Asymmetric drop | Giessen avg drop ≥ 0.05 AND Arcadia avg drop ≤ 0.02 | both true |
| XGB phylogroup signal | avg Δ PR-AUC (with − without phylogroup) | > 0 (any) |

If all four pass → Pillar 1 hypothesis confirmed; proceed to Stage 1.5
(Bakta gene mapping) and Pillar 2 architecture.

If any fails:
- Inspect per-antibiotic rows in CSV.
- Check label balance in `results/pillar1/exp1/exp1_results.csv` (`n_pos_test`).
- Re-run with `experiment.training.max_epochs=50` if early stopping fired too soon.

---

## 6. What to share back with the other agent

After all runs finish, share these files (small, text-only):

```
results/pillar1/SUMMARY.md
results/pillar1/verdicts.json
results/pillar1/pillar1_combined.csv
results/pillar1/figures/pillar1_overview.png      (image)
paper/outline.md
```

Plus the per-experiment CSVs:
```
results/pillar1/exp1/exp1_results.csv
results/pillar1/exp2/exp2_results.csv
results/pillar1/exp3/exp3_results.csv
results/baselines/xgb_phylogroup_results.csv
```

Do not share `*.npz` prediction files or `*.ckpt` checkpoints — they are
large and not needed for cross-agent sync.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'arcadia_snp_matrix_pkl'` | `paths.*` missing from Stage 0 config | Add the `paths:` block from § 2 |
| OOM during Variant A | `max_contig_len` too high | Override: `model.max_contig_len=300` |
| Slow Arcadia epoch (>5 min) | DataLoader workers blocked on Windows | Override: `experiment.training.num_workers=2` |
| WandB login prompt | First-time setup | Either `wandb login` once, or set `WANDB_MODE=disabled` |
| `xgboost.core.XGBoostError: GPU plugin is not enabled` | XGB GPU not built | The runner auto-falls back to CPU; no action needed |
| `RuntimeError: CUDA out of memory` on Naive Arcadia | Batch too big for 16 GB | Override: `experiment.training.batch_size_arcadia=8` |
| Hydra `Could not override 'experiment.antibiotics'` | Quoting issue | Use bracketed list: `experiment.antibiotics=[CIP,CTX]` |

---

## 8. After Pillar 1 passes — next stage

Hand-off to Stage 1.5:
1. Run Bakta on the 1,271 Arcadia contigs to produce gene annotations.
2. Map SNPs → genes using the parsed contig structure (already in
   `contig_parser.py`).
3. Filter to the `(variance > 5 % ∩ XGB top-10K) ∪ CARD-known` gene set,
   cap at 5 K genes.
4. Stage 2 begins on this gene-mapped tensor.

Stage 1 deliverables are independent of Stage 1.5, so you can begin
Bakta runs in parallel while Pillar 1 experiments execute.

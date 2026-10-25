# TriLens-MoE-Stack

Code and results for **"TriLens-MoE-Stack: A Multi-View Hybrid Framework for Lineage-Aware
Multi-Drug Antibiotic Resistance Prediction in *Escherichia coli* Using Genomic Data"**.

BSc thesis, Department of Computer Science and Engineering, BRAC University, 2026.

---

## What this repository is

A model that predicts resistance to seven antibiotics (AMP, CIP, AMC, SXT, GEN, CTX, CAZ)
from an *E. coli* genome, and an evaluation protocol built to find out how much of that
prediction is real biology and how much is family resemblance between related strains.

| Test | Average PR-AUC |
|---|---|
| Random split (what most papers report) | **0.849** |
| Sequence-type lineage holdout | **0.634** |
| Lineage holdout, corrected for prevalence | **0.723** |

Under the lineage holdout the seven antibiotics split into two groups with nothing in
between. Three lose 0.017 to 0.031. Four lose 0.176 to 0.244.

> **Status (Oct 2026):** every paper number is being recomputed from the processed data by
> the Kaggle re-run (`../kaggle/`). See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md), issue 1.

---

## Layout

```
code/
├── README.md               you are here
├── requirements.txt        pinned dependencies (requirements-kaggle.txt for Kaggle)
├── environment.yml         conda environment
├── configs/                Hydra configs, one per experiment
├── src/
│   ├── data/               loaders, gene filtering, SNP blocks
│   ├── models/             TriLens, gene-attention nets, CNN variants
│   ├── evaluation/         metrics and bootstrap statistics
│   ├── utils/              seeds, logging
│   ├── experiments/        one runnable script per experiment
│   └── run_experiment.py   Hydra entry point
├── scripts/
│   ├── diagnostics/        post-hoc checks that feed thesis tables
│   └── figures/            every thesis figure
├── notebooks/              the Kaggle notebooks that built the processed data
├── amr-genomic-data/       data QA package: labels, splits, metrics, tests
├── docs/                   paper map, reproduce guide, data, figures, known issues
├── data/                   (not in git) processed datasets, see docs/DATA.md
└── results/                one folder per run; small text files are in git
```

---

## Names used in file and folder names

Script and result names keep the project's working names, because every result folder,
config and document refers to them; renaming would break the link between a number and the
file it came from. What they mean:

| In a name | Means | Paper |
|---|---|---|
| `trilens_v10`, `trilens_attnfix` | the proposed model (config name / result folder of the delivered run) | TriLens-MoE-Stack |
| `standard/` | the same model with standard (not strain-conditioned) Lens B attention | tab:attnfix, "Standard" |
| `cascadestack_v9`, `classical` | the five classical base learners and their meta-learner | the classical half of the stack |
| `random_split` | stratified random split | random split |
| `phylogeny`, `t7`, `phylogroup` | phylogroup-stratified split | phylogroup-stratified split |
| `mlst` | sequence-type lineage holdout | lineage holdout |
| `pillar1` | locality test (1D CNN, ordered vs shuffled variants) | Motif Pattern Encoder rationale |
| `pillar2` | layer ablation | tab:ablation |
| `pillar3` | interpretability (SHAP, gain importance) | tab:umark, markers |
| `pillar4` | Giessen external validation | tab:giessen_transfer |
| `T6` | the full stack on the random split (row label inside result files) | headline model |
| `dedup` | near-duplicate removal check | statistical analysis |
| `gene_filtered` | the gene-filtered SNP dataset; `__mlst_holdout` and `__phylogeny_clean` are refitted on that split's training strains | feature selection |

---

## Start here

| If you want to... | Read |
|---|---|
| check a number in the paper | [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md) |
| re-run the work | [`docs/REPRODUCE.md`](docs/REPRODUCE.md) |
| know where each figure comes from | [`docs/FIGURES.md`](docs/FIGURES.md) |
| get the data | [`docs/DATA.md`](docs/DATA.md) |
| know what is wrong or missing | [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) |
| know what was left out on purpose | [`docs/EXCLUDED.md`](docs/EXCLUDED.md) |
| check for leakage | `python scripts/diagnostics/check_leakage.py`, and [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md), "Methods the paper must state" |

---

## Quick setup

```bash
conda env create -f environment.yml
conda activate amr
```

**Always run commands from inside `code/`.** Every path in the configs is relative to it.
To run from somewhere else, set `TRILENS_ROOT` to the absolute path of this folder.

```bash
python -m src.run_experiment experiment=trilens_v10        # Hydra experiments
python -m src.experiments.run_trilens_mlst                 # standalone experiment scripts
python scripts/diagnostics/verify_thesis_numbers.py         # post-hoc checks
```

Runs log to Weights & Biases in **offline** mode by default. Set `WANDB_MODE=online` to sync.

Seed is fixed at 42 everywhere.

---

## Data

Two public collections, neither produced by us.

| Collection | Strains | Use | Source |
|---|---|---|---|
| Arcadia Science *E. coli* AMR | 6,962 | training and evaluation | Morin (2024), York and Sandler (2025) |
| Giessen | 809 | external validation (retrained, not zero-shot) | Ren et al. (2022) |

No human or patient data is involved; every genome is bacterial.

---

## A note on what this model is not

No antibiotic in this work meets both CLSI error limits, so the system is a research
prototype and not a diagnostic tool. The gene-level interpretability results are leads
rather than settled biology.

## Citation and licence

See [`CITATION.cff`](CITATION.cff). Code is MIT-licensed (see `LICENSE`). The datasets keep
the licences of their original publishers.

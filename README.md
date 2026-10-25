# TriLens-AMR

Everything for the TriLens-MoE-Stack thesis and the paper that comes out of it, in one place.
Home of this folder: **`D:\TriLens-AMR`**.
Work only from this folder. The old folders (`Desktop\Thesis`, `D:\Thesis*`, `I:\Thesis*`)
are no longer needed once `setup_data.ps1` has run; keep them as a backup until you have
checked this one.

> **Shuru korar niyom (one-time):** folder ta `D:\TriLens-AMR` e rakho, tarpor `setup_data.ps1` e right-click → *Run with PowerShell*.
> Eta boro data (about 11 GB) purono folder theke `code\data` ar `code\results` e copy kore.
> Purono kichu muchbe na. Tarpor Kaggle e kaj korte `kaggle\README.md` poro.

---

## What is where

```
TriLens-AMR/
├── README.md            you are here
├── setup_data.ps1       run once: copies the large data and result files in
│
├── code/                the research code (on GitHub with kaggle/, docs/, paper/; see .gitignore)
│   ├── src/             models, data loaders, one script per experiment
│   ├── configs/         Hydra configs
│   ├── scripts/         diagnostics/ (post-hoc checks) and figures/ (every thesis figure)
│   ├── notebooks/       the Kaggle notebooks that built the processed data
│   ├── data/            datasets (filled by setup_data.ps1, not in git)
│   ├── results/         one folder per run; pre_fix/ holds the old-model outputs
│   └── docs/            PAPER_MAP, REPRODUCE, FIGURES, DATA, KNOWN_ISSUES, EXCLUDED
│
├── kaggle/              how to run on Kaggle + ready notebooks
├── thesis/              final thesis LaTeX + thesis.pdf (the submitted version)
├── paper/               IEEE versions: journal/ (report.tex) and conference/ (paper.tex)
├── defense/             defence guides (Bangla/Banglish) and slides
├── docs/                project-level notes: next_steps.md, file_map.md, history/
└── archive/             the old 80,984-feature pipeline notebooks (not used by the thesis)
```

---

## Status (October 2026)

| | |
|---|---|
| Thesis | submitted and defended (August 2026); final PDF in `thesis/thesis.pdf` |
| Code | organised, paths fixed, configs run from any machine (`code/docs/KNOWN_ISSUES.md`, "Fixed") |
| Blocking issue | several results on disk are pre-fix or lost; the full re-run in `kaggle/` recomputes every paper number |
| Next | `docs/next_steps.md`: confirm the CTX-M finding, then the pilot-gated experiments |

---

## Where to start

| I want to... | Open |
|---|---|
| see what to do next | `docs/next_steps.md` |
| find where an old file went | `docs/file_map.md` |
| check a number in the thesis | `code/docs/PAPER_MAP.md` |
| know what is broken or pending | `code/docs/KNOWN_ISSUES.md` |
| run something on Kaggle | `kaggle/README.md` (code on GitHub `FaiazPronoy/TriLens-AMR`, data on Kaggle) |
| redraw a figure | `code/docs/FIGURES.md` |
| compile the thesis or paper | `thesis/README.md`, `paper/README.md` |

## Naming rules used here

Folders and files have short, plain names: no dates, no serial numbers, no personal names.
Pre-fix (old model) outputs always live under a `pre_fix/` folder. A new experiment gets one
script in `code/src/experiments/`, one config if it uses Hydra, one results folder, and one
Kaggle notebook in `kaggle/`.

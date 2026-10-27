# Full re-run on Kaggle, connected to GitHub

> **Short version (Banglish):** Code thake GitHub e (`FaiazPronoy/TriLens-AMR`, public).
> Kaggle e shudhu ekta dataset lage: `trilens-data`. Protita notebook GitHub theke code clone
> kore, ager notebook er output GitHub release theke namay, ar nijer output GitHub e pathay.
> Tai kono zip download/upload korte hoy na, ar Claude GitHub theke result sorasori porte pare.

Everything in the paper is recomputed from the processed data. No result file from an
earlier run is used: every notebook starts from an empty `results/` folder and only reads
what the notebooks before it wrote.

```
GitHub repo (code)  ──clone──►  Kaggle notebook  ──outputs──►  GitHub release  outputs-<name>
                                    ▲     │                      (the full zip)
            trilens-data (Kaggle) ──┘     └─────────────────►  GitHub branch   results/<name>
                                                                 (csv / md / json, readable by Claude)
```

---

## One-time setup

1. **GitHub repository** `FaiazPronoy/TriLens-AMR` (public). It holds `code/`, `kaggle/`,
   `docs/`, `paper/` and the thesis sources (no data, no results arrays, no signatures;
   see `../.gitignore`).
2. **A GitHub token for Kaggle.** github.com → Settings → Developer settings →
   Fine-grained tokens → Generate. Repository access: only `TriLens-AMR`. Permissions:
   **Contents: Read and write**. Copy it.
3. **Kaggle secret.** In any Kaggle notebook: Add-ons → Secrets → Add → label
   `GITHUB_TOKEN`, value = the token. In each of the seven notebooks switch that secret on
   once (Add-ons → Secrets → tick it).
4. **Kaggle dataset `trilens-data`** (private): the 12 files listed by `pack_for_kaggle.ps1`
   (about 9.9 GB): everything in `code/data/processed/` plus the two CSVs in
   `code/data/Giessen_dataset/`. Upload them as they are, no zip.

The token never appears in the code or the outputs; it lives only in Kaggle's secret store.

---

## The eight notebooks, in order

Import each one from `notebooks/`: Kaggle → Code → New Notebook → File → Import Notebook.
In every notebook: Add Input → `trilens-data`; Internet **On**; secret `GITHUB_TOKEN` on;
then **Save Version → Save & Run All (Commit)**. It keeps running after you close the tab.

| # | Notebook | Needs | What it computes | Paper |
|---|---|---|---|---|
| 0 | `pa_fix` | nothing | rebuilds the gene presence/absence matrix with the correct orientation (CPU, minutes; KNOWN_ISSUES 12) | every Lens B number |
| 1 | `classical` | nothing | classical arm on all three splits, PCA / phylogroup / TabNet baselines, Giessen, SHAP | tab:baselines, tab:giessen_transfer, classical half of every stack |
| 2 | `trilens_random` | pa_fix, classical | the delivered model, random split; the layer ablation | tab:main, tab:ci, tab:gate, tab:clinical, tab:ablation |
| 3 | `trilens_splits` | pa_fix, classical | the neural half on the phylogroup split and the lineage holdout | tab:mlst_per_ab, tab:mlst_components |
| 4 | `standard_random` | pa_fix, classical | the ablation model (standard Lens B), random split | tab:attnfix, Standard column |
| 5 | `standard_splits` | pa_fix, classical | the ablation model on the two harder splits | tab:attnfix, Standard column |
| 6 | `shuffle_test` | nothing | 1D CNN, ordered against shuffled variants | the locality argument for the Motif Pattern Encoder |
| 7 | `analysis` | all seven | stacking per split, prevalence correction, clinical metrics, near-duplicates, gene annotation, lineage markers, attention diagnostics, leakage checks, the ablation table, figures, `paper_numbers.csv` | everything else |

"Needs" is fetched automatically from the GitHub releases; you do not attach anything.
Notebooks 2 to 6 do not depend on each other: once `classical` is done they can run in any
order, two at a time if your Kaggle account allows two GPU sessions. `analysis` needs no GPU;
turn the accelerator off there to save quota.

Kaggle gives about 30 GPU hours a week. Expect the GPU notebooks together to take more than
one week of quota. No run has been timed on a T4 yet; the earlier runs were on an RTX 4080.

---

## Two accounts (the plan in use)

Two Kaggle accounts, each run by its own owner, split the GPU work. Each account runs one
GPU notebook at a time; the next one starts when the previous one has finished.

| Order | Account A (Faiaz) | Account B (team member) |
|---|---|---|
| 1, now | `classical` | `shuffle_test` |
| 1b, any time (CPU, alongside) | `pa_fix` | — |
| 2, after `classical` and `pa_fix` are finished | `trilens_random` | `trilens_splits` |
| 3 | `standard_random` | `standard_splits` |
| 4, after all seven | `analysis` (CPU, no GPU) | — |

`classical` is finished when the notebook's last line says "All steps finished". A later
notebook checks this itself: it stops with "not finished yet" if an earlier notebook it needs
has not reached its last cell (its zip has no `_COMPLETE.txt`). A notebook that was started
before this check existed writes no `_COMPLETE.txt`; run it once more (Save & Run All), it
skips every finished step and only adds the marker.

Account B, once:
1. Accept the GitHub invitation to `FaiazPronoy/TriLens-AMR` (repo Settings → Collaborators).
2. Make its own token: github.com/settings/tokens → Generate new token (classic), scope
   **`public_repo`** only. (A fine-grained token cannot write to a repository owned by
   another personal account.) Store it as the Kaggle secret `GITHUB_TOKEN` in account B.
3. Add the dataset shared by account A (Dataset → Settings → Sharing → add account B).

Never run the same notebook from both accounts at once: both would write the same release.

---

## Journal extension notebooks (branch `extension`)

These clone the branch `extension` (`CODE_REF`), not `main`. The new splits (country, time) are rebuilt
inside each notebook from `code/results/splits/*.json`; nothing has to be uploaded for them. Plan,
hours and pass/fail rules: `../docs/PLAN.md` and `../docs/experiments/`.

| notebook | needs | accelerator | est. time | what it computes |
|---|---|---|---|---|
| `w1_gene_sets` | nothing | CPU | about 12 h (resumes) | gene ranking + 5,000-gene set on S4a, S4b, S5 |
| `w1_classical` | `w1_gene_sets` | GPU T4 x2 | about 4.5 h | classical arm and solo learners on S4a, S4b, S5 |
| `w2_snp_knn` | nothing | CPU | about 1 h | SNP Hamming distances; lineage-lookup baselines on all six splits |
| `w4_amrfinder` | `pa_fix` | CPU, Internet on | about 0.5 h | AMRFinderPlus on the pan-genome; determinants mapped to drugs and P/A columns |

`w4_amrfinder` needs `pangenome_cds.fa` (from `small_data.zip`, `raw/pangenome/`) in a dataset
attached to the notebook, for example added to `trilens-data`. The setup cell finds it by name.

---

## Outputs

| Where | What |
|---|---|
| GitHub release `outputs-<name>` | `<name>_outputs.zip`, sent after each step (at most every 10 minutes) and at the end |
| GitHub branch `results/<name>` | the text results of that notebook (csv, md, json, yaml), replaced each time |
| Kaggle Output tab | the same zip, as a backup |

Every zip and branch records the exact code commit it was run with (`_CODE_VERSION.txt`).

The standard-model notebooks keep their files under `results/standard/`, so the two models
never mix.

When `analysis` is done, `results/analysis` holds `diagnostics/paper_numbers.csv` (every paper
number, paper against new) and `diagnostics/leakage_checks.csv`. Claude reads these straight
from the branch. The full `analysis_outputs.zip` (predictions, figures) goes into
`../code/results/` on the computer.

---

## If a session stops before the end

Just run the notebook again (Save & Run All). It downloads its own release, finds the steps
that already finished and skips them. Nothing to upload.

---

## Without GitHub

If the secret is missing, nothing is uploaded and nothing is downloaded. Then attach a dataset
`trilens-code` (made by `pack_for_kaggle.ps1`) and the earlier notebooks' outputs as inputs, as
before; the setup cell finds them.

---

## Paths inside a notebook

The first cell clones the repository to `/tmp/trilens/repo`, copies `code/` to
`/tmp/trilens/code`, links the data into `/tmp/trilens/code/data/` and sets:

```python
TRILENS_ROOT     = "/tmp/trilens/code"
TRILENS_MAIN_RUN = "trilens_attnfix"   # folder name of the main TriLens run
TRILENS_LENSB    = "fixed"             # "standard" in notebooks 4 and 5
WANDB_MODE       = "disabled"
```

`GITHUB_REPO` and `CODE_REF` at the top of the first cell choose the repository and the
branch, tag or commit to run. For the paper run, tag the commit (for example `paper-v1`) and
set `CODE_REF` to that tag, so every notebook runs the identical code.

New experiments get their own notebook built on the same first cell. Rule for every new
experiment: a cheap pilot first, with its pass/fail rule written down before it runs. See
`../docs/next_steps.md`.

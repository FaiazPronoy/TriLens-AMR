# Adding something new

The repository is built so that one new experiment touches five small places and
nothing else. Nothing in `trilens/` imports from `pipeline/`, so a new module
cannot break an existing step.

## 1. Write the module

Put it in the package folder that matches what it does:

| Folder | For |
|---|---|
| `trilens/data/` | building or repairing a data file |
| `trilens/splits/` | a new way of splitting the strains |
| `trilens/models/` | a new network or branch |
| `trilens/experiments/` | training a model, or an ablation |
| `trilens/determinants/` | anything from AMRFinderPlus or NCBI Pathogen Detection |
| `trilens/analysis/` | a measurement on saved predictions |

Two shapes are allowed, and the step runner supports both:

- a script module, with `argparse` and a `main()` under `if __name__ == "__main__":`
- a configured module, with `def run(cfg)` that takes a Hydra config

Use the shared helpers instead of writing your own: `trilens.evaluation.cluster_bootstrap`
for nAP, intervals and within-lineage concordance, `trilens.evaluation.clinical_errors`
for very major and major error rates, and `trilens.utils.seeds.set_global_seed` for
seeding. Write every output under `results/<name>/`.

## 2. Add a config, if the module takes one

Copy the closest file in `configs/experiment/` and change what differs. Each file
needs a `name`, a one-line `description` and an `output_dir`. The runner reads it with:

```python
from trilens.config import load_config

cfg = load_config("my_experiment")
```

## 3. Add the pipeline step

Make `pipeline/stepNN_short_name.py`. It is a thin wrapper, usually four lines:

```python
from trilens.cli import configured, module, run_step

run_step(
    "One line saying what this step does.",
    {
        "fit": configured("my_experiment", "trilens.experiments.my_experiment"),
        "report": module("trilens.analysis.my_report"),
    },
)
```

`run_step` gives you `--help`, the `--part` choices and `all` for free. Use
`--split` or `--model` in place of `--part` with `choose="--split"`, and
`env_options={"--attention": ["conditioned", "standard"]}` for a switch that the
module reads from the environment.

Numbering: the steps follow the order of the paper, so a new step usually belongs
next to the one it extends. If the number is taken, it is fine to insert a letter
(`step19b_...`); nothing depends on the numbers being consecutive.

## 4. Add a test

Tests in `tests/` run with no data: they build a tiny synthetic input, run the
module and check the thing that would silently go wrong. The pattern to copy is
`tests/test_determinant_rescue.py` for an experiment and
`tests/test_gene_presence_rebuild.py` for a data builder. Test the guard, not just
the happy path: that it refuses bad input, and that no test strain reaches the
training set.

```bash
pytest -q
ruff check .
ruff format .
```

## 5. Add the Kaggle stage, if it needs a GPU or hours of CPU

Add one entry to `STAGES` in `kaggle/make_notebooks.py`:

```python
"stage18_my_experiment": dict(
    title="My experiment",
    about="What it does, in two sentences.",
    gpu=GPU, needs=["stage01_gene_presence_matrix"], writes=["my_experiment"],
    steps=[step("Fit", "step31_my_experiment --part fit", "my_experiment/verdict.json")],
),
```

Then `python kaggle/make_notebooks.py`. `needs` are the stages whose outputs are
downloaded first, `writes` are the result folders the stage uploads, and the third
argument of `step` is a file that proves the step finished, so a resumed run skips it.

## If the experiment is pre-registered

Write `docs/preregistration/wNN_short_name.md` before reading the data, with the
question, the inputs, the arms, the endpoint and the pass rule, and commit it. Add
the row to `docs/preregistration/README.md`. Later changes go at the bottom of the
file as dated amendments; the original text stays as it was.

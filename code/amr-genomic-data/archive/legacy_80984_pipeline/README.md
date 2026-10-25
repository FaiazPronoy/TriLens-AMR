# Archived: superseded 80,984-feature pipeline

This directory intentionally does **not** contain a copy of
`autoencoder-pca-random-split.ipynb` (it lives in the parent thesis
directory, one level up from this repository, and was not moved or deleted
by this refactor -- it is the user's original file). This README exists so
the fact of its supersession, and exactly what was and wasn't carried
forward from it, is recorded somewhere inside this repository too.

## What that notebook is

A dataset (`amr_metax_dataset.pkl`, shape `(6962, 80984)`) and a family of
models built on top of it: **AMR-CascadeStack v3/v6/v6.1, AMR-UnifiedFusion
v1, AMR-DSCS v1.1**, and several fusion strategies. None of this is used by
the current Arcadia v2 pipeline (`arcadia-v2-phase-a/b/c.ipynb` and this
refactor of it) -- it is a different, earlier feature space entirely, not
a predecessor version of the same one.

## What WAS carried forward (per the refactor spec, Part B)

1. **The 30 disjoint cluster labels** -- a genuine lineage holdout grouping,
   sizes 4481/1078/1403, verified group-disjoint by the old notebook's own
   QA cell. `scripts/port_cluster_labels.py` in this repository is ready to
   port them onto the current sample order, but **cannot run yet** --
   `amr_metax_dataset.pkl` itself is not present anywhere this refactor had
   access to. See `docs/OPEN_QUESTIONS.md` #1 and #2.
2. **The dataset verification script** (the old notebook's cell 1) --
   generalized into `src/amr_data/qa.py::verify_dataset()` (shape/key checks
   parameterized instead of hardcoded to `(6962, 80984)`).
3. **The representation-disagreement analysis** (the old notebook's cell 9)
   -- explicitly NOT ported into working code, per the spec ("archive the
   code, do not port it -- it may become a journal-version analysis
   later"). If wanted later, it lives in the original notebook, cell index
   9, one directory up from this repository.

## What was NOT carried forward

The models themselves (AMR-CascadeStack v3/v6/v6.1, AMR-UnifiedFusion v1,
AMR-DSCS v1.1) and the 80,984-feature dataset construction. If any thesis
number is attributed to one of these models, its provenance is that old
notebook, not this refactored repository.

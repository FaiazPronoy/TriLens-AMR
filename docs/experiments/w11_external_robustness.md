# W11. Does the external result survive losing one BioProject, and is the cohort really ours-free?

**Status:** pre-registered 2026-10-29, before any leave-one-out fit. W8's headline numbers are
already read and are stated here as the quantity being tested, not re-derived.

## Why

Two reviews made the same point about W8: the clustered bootstrap says how much the result would
wobble if the BioProjects were resampled, but not whether it depends on one influential project.
The external cohort is 10,232 isolates in 62 BioProjects, 68% of them from the USA, where a single
surveillance programme can dominate. A result carried by one project is a collection-specific
result.

The second point is independence. `ext_probe` already removes any isolate whose SRA run or
BioSample matches one of ours. It does not check the assembly accession, and two accession records
can describe the same genome.

## Design

- **Leave-one-BioProject-out.** For each BioProject with at least 100 labelled isolates, recompute
  the W8 endpoints with that project's isolates dropped, from the **already-saved external scores**
  (`scores.csv.gz`). No model is refitted: the frozen models and their predictions are unchanged,
  so this is a re-scoring and cannot leak.
- **Accession audit.** Compare the external cohort's `asm_acc` and `target_acc` against our
  strains' assembly accessions, and report any overlap. If any isolate overlaps, it is dropped and
  every W8 endpoint is recomputed without it.
- **Project concentration.** Report each BioProject's isolate count, country, source and per-drug
  label counts, so a reader can see how concentrated the cohort is.

## Endpoints and rules

**Primary (AP − M), the W8 headline.** Pass if, across every leave-one-project-out replicate,
the point estimate stays ≥ 0.10 and the clustered CI low stays > 0 — the W8 rule, applied to each
replicate.

**Secondary (AP − R), the fragile one.** Reported, with no pass/fail: the range of the point
estimate and the number of replicates whose CI excludes zero, out of the replicates run. W8
already reports this endpoint as inconclusive with intermediates excluded, and W11 cannot make it
conclusive; it can only show whether the small positive margin is spread across projects or comes
from one.

**Accession audit.** Any overlap found is reported with the count, and all W8 endpoints are
restated on the de-overlapped cohort. An overlap of more than 1% of the cohort would mean the
"not ours" claim in W8 was wrong, and W8's amendment would be corrected rather than defended.

## Stated limits

- Dropping a project changes the population as well as the sample size, so a drop in one replicate
  is not by itself evidence of dependence on that project; the range across replicates is what is
  interpreted.
- An isolate sequenced twice under unrelated accessions, by different submitters, would not be
  caught by any accession comparison. Only a sequence-level comparison would catch it, and the
  external assemblies are not downloaded here, so this limit stands.
- Projects with fewer than 100 labelled isolates are not dropped individually; their joint effect
  is not tested.

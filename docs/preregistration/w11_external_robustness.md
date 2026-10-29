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
the point estimate stays ≥ 0.10 and the clustered CI low stays > 0 - the W8 rule, applied to each
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

## Result (read 2026-10-29, after the pre-registration above was committed)

**Primary (AP − M): pass.** Nine BioProjects have ≥ 100 labelled primary-drug isolates. Dropping
each in turn, the mean nAP advantage over the assembly arm stays in **0.723-0.777** and the
clustered CI lower bound never falls below **0.643**. Every replicate clears the W8 rule, so the
external result does not depend on any one project.

**Accession audit: no overlap.** Zero external isolates match any identifier we hold for our own
strains, checking `asm_acc`, `target_acc`, `Run` and `biosample_acc` against our `genome_id`,
`genome_name`, `sra`, `biosample` and `assembly_accession`. Overlap fraction 0.000, so W8's
"not ours" claim stands as written. (The limit above still applies: an isolate deposited twice
under unrelated accessions would not be caught by any accession comparison.)

**Secondary (AP − R), reported without a pass rule.** The margin over the determinant rule is
spread across projects rather than produced by one: point estimates 0.037-0.086 across the nine
replicates, with the CI above zero in **6 of 9**. The three replicates whose CI touches zero are the
drops of PRJNA809394 (0.037, CI low −0.005), PRJNA278886 (0.056, −0.004) and PRJNA592140
(0.061, −0.001). So the small external margin is consistent in sign across the cohort but its
interval is fragile, exactly as W8 reported it.

*Correction, same day:* these figures are from the Kaggle run (commit 64df73c, 2,000 bootstrap
replicates, log of `w11_w12_robustness`). An earlier version of this section quoted a local
run with 400 replicates (CI low 0.648; 7 of 9), in which PRJNA592140's interval fell just above
zero. The W12 figures are identical in both runs.

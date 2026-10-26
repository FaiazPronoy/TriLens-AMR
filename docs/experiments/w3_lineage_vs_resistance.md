# W3. Is the model reading lineage or resistance? Pre-registration

**Status:** pre-registered, not run. Committed before any W3 run. The code change it needs (each lens'
representation `z` saved for train, validation and test) is on `extension` in the same branch.

## What changed since the thesis

The thesis table `tab:attnfix` (standard vs strain-conditioned Lens B, U_ST/U_res of attended genes)
was computed on the **minority-state presence/absence matrix** (KNOWN_ISSUES 12). On that matrix Lens B
could never attend to most core and common genes. Every W3 number therefore comes from new runs on the
rebuilt matrix, with the init-parity fix (`tests/test_expertb_init.py`). No thesis value is reused.

## Question

Does strain conditioning make Lens B (and the whole neural arm) read less lineage and more
resistance? Does reading less lineage go with performing better when lineage shifts?

## Design

- **Arms:** `TRILENS_LENSB=fixed` (strain-conditioned) and `TRILENS_LENSB=standard`. Nothing else differs.
- **Splits:** S1 random, S3 ST holdout, S5 time (primary). S2 and S4a only if GPU budget is left.
- **Seeds:** 42, 123, 456 per arm and split. Seed 42 of `fixed` on S1/S3 comes from the re-run if it
  used the rebuilt matrix (its zip carries `_PA_FIXED.txt`); otherwise it is rerun.

### Measures, per run

1. **Lineage probe on the representations.**
   - Inputs: `z_A`, `z_B`, `z_C` and the gate-fused representation, standardised on training strains.
   - Model: multinomial logistic regression (sklearn, L2, C = 1.0, max_iter 2000; fixed, not tuned).
   - **ST probe** (S1, S5, where test STs also occur in training):
     - classes are the STs with ≥ 20 training strains, the rest merged into "other"; strains of
       unknown ST are left out;
     - fitted on training strains, scored on the split's test strains.
   - **Phylogroup probe** (all splits, including S3, whose test STs never occur in training): the same,
     with the 7 phylogroups ("Unknown" left out).
   - Metrics: balanced accuracy (primary) and macro-F1.
   - References, computed the same way:
     - (i) the same probe on the raw P/A profile (a ceiling for "lineage is in the input");
     - (ii) shuffled training labels (chance);
     - (iii) the representations of the untrained, freshly initialised model with the same seed
       (architecture-only).
2. **U_ST and U_res of attended genes:** `scripts/diagnostics/lineage_marker_test.py` (uncertainty
   coefficient of presence given ST, and given the resistance label), top-20 attended genes per drug,
   on the rebuilt matrix, against 500 random 20-gene sets.
3. **Performance:** PR-AUC per drug, as W1.

## Pass/fail (decided now)

Let Δ be fixed − standard and SD the pooled seed standard deviation of the two arms.

| # | claim | passes if |
|---|---|---|
| H1 | strain conditioning reads less lineage | probe balanced accuracy on `z_B` is lower for `fixed` by more than 2·SD on ≥ 2 of the 3 primary splits (ST probe on S1 and S5, phylogroup probe on S3) |
| H2 | … and more resistance | mean U_res of the attended genes (over 7 drugs) is higher for `fixed` on ≥ 2 of 3 splits, by more than 2·SD |
| H3 | the whole neural arm follows | H1 holds also for the gate-fused representation |
| H4 | less lineage goes with better transfer | on S3, the neural arm's mean PR-AUC is higher for `fixed`, paired bootstrap on the same test strains (per seed, pooled over seeds by averaging the per-seed Δ), Holm over the 7 drugs, for ≥ 4 of 7 |

The thesis claim ("strain conditioning reads less lineage") stands only if H1 and H2 pass. If they
fail, the paper reports that strain conditioning does not change what the lens reads, and drops the
claim. A partial result (for example H1 without H4) is reported as exactly that.

## Pilot

Before the full grid:

- one run of `fixed` and one of `standard`, seed 42, S1, with the probe on CIP's run only;
- check: the `z` arrays are saved with the right shapes and the probe reproduces on rerun.

The pilot numbers are reported here, whatever they are.

## Compute

`w3_seeds_fixed`, `w3_seeds_standard` (GPU), `w3_probe` (CPU): about 45 GPU h in total (see
`docs/PLAN.md`; to be updated with the re-run's timings).

## Deviations

(none)

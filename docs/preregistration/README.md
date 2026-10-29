# Pre-registration documents

Each experiment from W6 to W14 was written down before its data were read: the question, the
design, the endpoint and the rule for pass or fail. Later changes are added at the bottom of each
file as dated amendments, and the original text is never edited.

## Index

| ID | Question | Document | Pipeline step | Results folder |
|---|---|---|---|---|
| W6 | Do per-strain determinant calls rescue unseen lineages? | [w06_determinant_rescue](w06_determinant_rescue.md) | `step22`, `step23` | `results/determinant_rescue/` |
| W7 | Does assembly size predict resistance inside one sequence type? | [w07_assembly_size_within_lineage](w07_assembly_size_within_lineage.md) | `step24 --part within_lineage` | `results/diagnostics/assembly_size_*` |
| W8 | Do frozen models transfer to an external cohort? | [w08_external_cohort](w08_external_cohort.md) | `step25`, `step26 --part transfer` | `results/external_cohort/` |
| W9 | Is the holdout gain stable across seeds? | [w09_seed_stability](w09_seed_stability.md) | `step20` | `results/seed_stability/` |
| W10 | Is the rescue a coverage artefact, and does a model beat a fixed rule? | [w10_coverage_and_rule](w10_coverage_and_rule.md) | `step27` | `results/coverage_and_rule/` |
| W11 | Does the external result survive leaving out one BioProject? | [w11_external_robustness](w11_external_robustness.md) | `step26 --part robustness` | `results/external_robustness/` |
| W12 | Is the assembly signal batch or biology? | [w12_assembly_size_mechanism](w12_assembly_size_mechanism.md) | `step24 --part mechanism` | `results/assembly_size_mechanism/` |
| W13 | Do the findings hold in a second species and against MIC values? | [w13_second_species_and_mic](w13_second_species_and_mic.md) | `step28`, `step29` | `results/campylobacter/`, `results/mic_harmonisation/` |
| W14 | What are the clinical error rates, and is a linear model enough? | [w14_linear_model_and_clinical_errors](w14_linear_model_and_clinical_errors.md) | `step30` | `results/linear_model_and_clinical_errors/` |

## How to read a document

Every file has the same shape:

| Section | What it holds |
|---|---|
| Status | the date the file was committed, and what had and had not been run by then |
| Question | the one thing the experiment answers |
| Inputs | the exact data, filters and label rules |
| Arms | the models or feature sets being compared, all on the same folds |
| Endpoint | the single number that decides the outcome |
| Pass rule | the threshold, fixed in advance |
| Amendments | dated notes added after the first result was read |

Three tests failed their own pass rule: W10-A, W13-a and W13-c. They are reported as failures in
the paper, which is the point of fixing the rule beforehand.

Some documents name module paths from the older layout of this code, such as `src.mechanism.arg_pd`.
They are kept word for word. `README.md` in the repository root maps every old name to its current
module.

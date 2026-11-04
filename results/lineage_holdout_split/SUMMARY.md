# MLST lineage holdout split -- Step 0 + Step 1 diagnostics

- metadata rows: 6985, dataset genomes: 6962
- join: `Genome.Name` AND `SRA.Accession`, cross-validated, agree on all 6962
- genomes with Achtman ST: **6487 (93.2%)**; without: 475
- distinct STs: 458
- STs forced to train (>5% of data): {'73': 884, '131': 872, '95': 606, '69': 477}

## Partition sizes

| Partition | Genomes | Distinct STs |
|---|---|---|
| train | 4836 | 240 |
| val | 1081 | 115 |
| test | 1045 | 103 |

All hard assertions passed: ST sets pairwise disjoint, genome-index sets pairwise disjoint and complete, test size within bounds, >=20 resistant and >=20 susceptible per antibiotic in test.

## Prevalence shift (train -> test)

| AB | n_train | n_test | prev_train | prev_test | shift (pp) |
|---|---|---|---|---|---|
| AMP | 3441 | 701 | 51.0% | 42.8% | -8.2 |
| CIP | 4167 | 913 | 26.6% | 14.5% | -12.1 |
| AMC | 2666 | 540 | 32.7% | 26.9% | -5.9 |
| SXT | 2351 | 517 | 30.0% | 27.5% | -2.5 |
| GEN | 4170 | 914 | 14.6% | 4.2% | -10.4 |
| CTX | 3783 | 808 | 17.6% | 8.8% | -8.8 |
| CAZ | 4117 | 912 | 15.3% | 4.7% | -10.6 |

**Largest shift: -12.1 pp (CIP)**

# Component ablation: TriLens stack (full_stack)

Isolates the contribution of each architectural layer: (1) the three experts averaged without gating, (2) the learned MoE gate fusing them (TriLens standalone), (3) the 5 classical base learners + meta-learner on top (full full_stack headline). All stages evaluated on the identical test set (same `random_split`, seed=42, kfold_seed=42).

## Average PR-AUC by stage

| Stage | Description | Avg PR-AUC | Δ vs previous stage |
|---|---|---|---|
| Stage 1 | 3 experts, unweighted average (no gate) | 0.7800 | - |
| Stage 2 | + MoE gate (TriLens standalone) | 0.7955 | +0.0155 |
| Stage 3 | + 5 classical bases + LogReg meta (full_stack headline) | 0.8502 | +0.0547 |

## Per-antibiotic PR-AUC

| Antibiotic | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| AMP | 0.9198 | 0.9223 | 0.9431 |
| CIP | 0.9300 | 0.9351 | 0.9750 |
| AMC | 0.6562 | 0.6703 | 0.7539 |
| SXT | 0.7927 | 0.8242 | 0.8423 |
| GEN | 0.5596 | 0.5645 | 0.7093 |
| CTX | 0.8436 | 0.8622 | 0.9043 |
| CAZ | 0.7581 | 0.7896 | 0.8231 |

## Stage 3 clinical metrics (threshold = 0.5)

| AB | VME% | ME% | CA% | CLSI (VME<=1.5% and ME<=3%) |
|---|---|---|---|---|
| AMP | 10.66 | 9.86 | 89.74 | no |
| CIP | 7.07 | 1.54 | 97.25 | no |
| AMC | 39.43 | 10.51 | 80.53 | no |
| SXT | 21.82 | 8.59 | 87.26 | no |
| GEN | 41.49 | 2.33 | 93.63 | no |
| CTX | 19.05 | 2.56 | 95.30 | no |
| CAZ | 29.09 | 3.42 | 93.44 | no |

**CLSI criteria met (VME<=1.5% and ME<=3%): 0/7 antibiotics.**

## Verdict

- The MoE gate adds **+0.0155** avg PR-AUC over naive unweighted expert averaging (Stage 1 → Stage 2).
- Classical-base stacking + meta-learner adds **+0.0547** avg PR-AUC over TriLens standalone (Stage 2 → Stage 3).

- **Both added components (gate, stacking) contribute positively - neither is pure complexity.**
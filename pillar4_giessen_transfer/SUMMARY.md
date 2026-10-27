# Pillar 4 — Cross-dataset methodology generalization

## Scope

We evaluate whether the CascadeStack v9.1 methodology — the same 5-base
stacking architecture and identical hyperparameters that achieved 0.847
avg test PR-AUC on Arcadia v2 — also succeeds on the Giessen E. coli AMR
dataset. Because the two datasets use incompatible feature spaces
(Arcadia: pan-genome 535K SNPs over Prokka-annotated genes; Giessen:
single-reference 60K SNPs), pure zero-shot **model** transfer is not
feasible without sequence-level BLAST mapping. We therefore demonstrate
**methodology** transfer: same architecture, same hyperparameters, same
K-fold OOF protocol, same evaluation — applied independently to Giessen.

## Per-antibiotic results on Giessen

| AB | PR-AUC | 95% CI | MCC | F2 | Sens@90 | n_test |
|---|---|---|---|---|---|---|
| CIP | 0.9877 | [0.9622, 1.0000] | 0.9512 | 0.9563 | 0.9459 | 81 |
| CTX | 0.7856 | [0.6419, 0.9100] | 0.5740 | 0.7542 | 0.5278 | 81 |
| CAZ | 0.7633 | [0.5948, 0.8916] | 0.5088 | 0.6786 | 0.6786 | 81 |
| GEN | 0.5394 | [0.3395, 0.7449] | 0.2975 | 0.5051 | 0.3684 | 81 |

**Giessen avg PR-AUC: 0.7690**

## Arcadia vs Giessen — generalization gap

| AB | Arcadia v9 PR-AUC | Giessen PR-AUC | Δ | Retention |
|---|---|---|---|---|
| CIP | 0.9774 | 0.9877 | +0.0103 | 101.1% |
| CTX | 0.8931 | 0.7856 | -0.1075 | 88.0% |
| CAZ | 0.8215 | 0.7633 | -0.0582 | 92.9% |
| GEN | 0.7037 | 0.5394 | -0.1643 | 76.7% |

**Avg retention: 89.6%** (Arcadia avg 0.8489 vs Giessen avg 0.7690, Δ = -0.0799).

## Defence narrative

> *"To evaluate cross-dataset generalization of our approach, we
applied the CascadeStack v9.1 methodology — identical architecture
(5 classical base learners + L2 logistic-regression meta) and
hyperparameters — to the Giessen E. coli dataset (Witschel et al.,
2022) for the four antibiotics common to both collections (CIP, CTX,
CAZ, GEN). Despite substantial differences in feature space (Arcadia:
pan-genome 535K SNPs; Giessen: single-reference 60K SNPs), the same
methodology achieves avg test PR-AUC 0.7690 on Giessen — 89.6% of the corresponding Arcadia v9 performance
(0.8489). This demonstrates that the architectural and
methodological choices motivated by our Pillar 1 locality-mismatch
finding generalise across genomically distinct E. coli collections.
True zero-shot model transfer between these datasets is an open
problem requiring BLAST-based feature alignment, which we leave for
future work."*

## Honest limitations
- **Not zero-shot model transfer**: each dataset uses its own trained
  ensemble. Methodology transfer is the appropriate framing.
- **No GeneAttentionNet on Giessen**: GAN is parameterised on Arcadia's
  gene-level structure (5000 genes × 4 statistics). Giessen lacks an
  equivalent grouping. Pillar 4 ensemble uses the 5 classical bases only.
- **Smaller Giessen sample size (~800 vs ~6900)** means wider CIs and
  greater fold-to-fold variance. We report bootstrap 95% CIs for honesty.

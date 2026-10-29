# TriLens stack - Experiment Summary

## Reference: the classical stack
| classical stack avg test PR-AUC | 0.8469 |

---

## Ablation Results (test set)

| Ablation | Description | Avg PR-AUC | Δ vs classical stack | Notes |
|----------|-------------|-----------|-----------|-------|
| full_stack | full stack: K-fold OOF + 5 classical base | 0.8502 | +0.0033  | random_split |

## Per-AB Detail: full_stack (full stack) vs classical stack

| AB | stack PR-AUC | classical stack | Δ | CI |
|----|-----------|------|---|---|
| AMP | 0.9431 | 0.9412 | +0.0019 | [0.919, 0.963] |
| CIP | 0.9750 | 0.9774 | -0.0024 | [0.957, 0.989] |
| AMC | 0.7539 | 0.7569 | -0.0030 | [0.687, 0.813] |
| SXT | 0.8423 | 0.8346 | +0.0077 | [0.779, 0.899] |
| GEN | 0.7093 | 0.7037 | +0.0056 | [0.619, 0.790] |
| CTX | 0.9043 | 0.8931 | +0.0112 | [0.853, 0.945] |
| CAZ | 0.8231 | 0.8215 | +0.0016 | [0.754, 0.877] |

## Expert Routing (gate weights, full_stack test set)

| AB | Expert A (motif CNN) | Expert B (P/A) | Expert C (RMRNet) |
|----|---------------------|----------------|------------------|
| AMP | 0.153 | 0.649 | 0.198 |
| CIP | 0.081 | 0.553 | 0.367 |
| AMC | 0.098 | 0.641 | 0.261 |
| SXT | 0.100 | 0.807 | 0.093 |
| GEN | 0.129 | 0.676 | 0.195 |
| CTX | 0.076 | 0.735 | 0.189 |
| CAZ | 0.068 | 0.748 | 0.183 |

## Top Expert B Genes (attention_B_topgenes.csv)
These are candidate acquired-resistance markers to be cross-referenced with CARD/RGI.

*See attention_B_topgenes.csv for full per-AB tables.*

## Verdict

**PARTIAL SUCCESS:** full stack avg PR-AUC 0.8502 beats classical stack by 0.0033. Marginal but consistent with 35-45% prior.

*Split: random_split | Seed: 42 | N_folds: 5*

# TriLens stack - Experiment Summary

## Reference: the classical stack
| classical stack avg test PR-AUC | 0.8469 |

---

## Ablation Results (test set)

| Ablation | Description | Avg PR-AUC | Δ vs classical stack | Notes |
|----------|-------------|-----------|-----------|-------|
| full_stack | full stack: K-fold OOF + 5 classical base | 0.8526 | +0.0057 * | random_split |

## Per-AB Detail: full_stack (full stack) vs classical stack

| AB | stack PR-AUC | classical stack | Δ | CI |
|----|-----------|------|---|---|
| AMP | 0.9508 | 0.9412 | +0.0096 | [0.930, 0.969] |
| CIP | 0.9751 | 0.9774 | -0.0023 | [0.957, 0.989] |
| AMC | 0.7558 | 0.7569 | -0.0011 | [0.687, 0.815] |
| SXT | 0.8505 | 0.8346 | +0.0159 | [0.788, 0.904] |
| GEN | 0.7093 | 0.7037 | +0.0056 | [0.616, 0.792] |
| CTX | 0.9020 | 0.8931 | +0.0089 | [0.849, 0.944] |
| CAZ | 0.8248 | 0.8215 | +0.0033 | [0.755, 0.882] |

## Expert Routing (gate weights, full_stack test set)

| AB | Expert A (motif CNN) | Expert B (P/A) | Expert C (RMRNet) |
|----|---------------------|----------------|------------------|
| AMP | 0.129 | 0.713 | 0.159 |
| CIP | 0.064 | 0.568 | 0.368 |
| AMC | 0.130 | 0.590 | 0.279 |
| SXT | 0.096 | 0.766 | 0.138 |
| GEN | 0.125 | 0.711 | 0.164 |
| CTX | 0.066 | 0.718 | 0.215 |
| CAZ | 0.065 | 0.726 | 0.209 |

## Top Expert B Genes (attention_B_topgenes.csv)
These are candidate acquired-resistance markers to be cross-referenced with CARD/RGI.

*See attention_B_topgenes.csv for full per-AB tables.*

## Verdict

**PARTIAL SUCCESS:** full stack avg PR-AUC 0.8526 beats classical stack by 0.0057. Marginal but consistent with 35-45% prior.

*Split: random_split | Seed: 42 | N_folds: 5*

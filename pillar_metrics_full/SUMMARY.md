# T6 (headline) — Full Per-Antibiotic Metric Suite

All metrics computed on the T6 test set (`results/trilens_v10/predictions_saved.npz`, `test_ens_probs`), threshold=0.5 for MCC/F1/F2/Accuracy (PR-AUC is threshold-free). No dashes — every cell is a real number; see `notes` for any degenerate case.

## MCC methodology (Gra Sir)

MCC is computed per antibiotic as an independent binary problem (resistant vs susceptible for that drug only, at thr=0.5) — verified against `src/experiments/run_trilens_v10.py::compute_metrics()`, which already does this correctly (slices `Y`/`probs` to a single antibiotic column before calling `matthews_corrcoef`). This is not a macro-averaged multi-label MCC. Values below were independently recomputed from raw test predictions and cross-checked against `ablation_results.csv`'s T6 row (no mismatch found beyond floating-point tolerance).

## Metrics table

| Antibiotic | PR-AUC | MCC | F1 | F2 | Accuracy | Notes |
|---|---|---|---|---|---|---|
| AMP | 0.9431 | 0.7948 | 0.8971 | 0.8949 | 0.8974 | computed normally, no degenerate case |
| CIP | 0.9750 | 0.9189 | 0.9364 | 0.9321 | 0.9725 | computed normally, no degenerate case |
| AMC | 0.7539 | 0.5276 | 0.6584 | 0.6257 | 0.8053 | computed normally, no degenerate case |
| SXT | 0.8423 | 0.7019 | 0.7938 | 0.7866 | 0.8726 | computed normally, no degenerate case |
| GEN | 0.7093 | 0.6256 | 0.6548 | 0.6111 | 0.9363 | computed normally, no degenerate case |
| CTX | 0.9043 | 0.7904 | 0.8173 | 0.8126 | 0.9530 | computed normally, no degenerate case |
| CAZ | 0.8231 | 0.6886 | 0.7256 | 0.7156 | 0.9344 | computed normally, no degenerate case |

**Averages** — PR-AUC: 0.8502, MCC: 0.7211, F1: 0.7833, F2: 0.7684, Accuracy: 0.9102

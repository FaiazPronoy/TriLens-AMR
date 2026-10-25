"""Recompute every random-split number in the thesis from the saved predictions
and check it against what is actually typeset in the .tex files.

Nothing here is read from a results CSV. Every value on the "computed" side is
recomputed from results/trilens_attnfix/predictions_saved.npz plus the locked
master-features pickle, using the same metric definitions the pipeline uses
(sklearn average_precision_score / matthews_corrcoef / f1 / fbeta, and the
VME/ME/CA definitions in amr-genomic-data/src/amr_data/metrics.py).

The "thesis" side is hardcoded, transcribed by hand from:
    thesis/chapters/chapter_5.tex   tab:main, tab:clinical, tab:ci,
                                         tab:ablation, tab:baselines, tab:gate

Run:
    python scripts/diagnostics/verify_thesis_numbers.py      (from inside code/)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    f1_score,
    matthews_corrcoef,
)

ROOT = Path(__file__).resolve().parents[2]   # the code/ folder
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)
MASTER = ROOT / "data" / "processed" / "arcadia_v2_master_features.pkl"
PRED = ROOT / "results" / "trilens_attnfix" / "predictions_saved.npz"

AB = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]

# ---------------------------------------------------------------- thesis side
# tab:main  -- PR-AUC, MCC, F1, F2, Accuracy
TAB_MAIN = {
    "CIP": (0.9773, 0.9088, 0.9282, 0.9197, 0.9692),
    "AMP": (0.9440, 0.7866, 0.8932, 0.8917, 0.8933),
    "CTX": (0.9019, 0.7904, 0.8173, 0.8126, 0.9530),
    "SXT": (0.8352, 0.6954, 0.7875, 0.7730, 0.8707),
    "CAZ": (0.8299, 0.6928, 0.7290, 0.7169, 0.9356),
    "AMC": (0.7482, 0.5237, 0.6563, 0.6250, 0.8035),
    "GEN": (0.7037, 0.6095, 0.6386, 0.5915, 0.9341),
}
TAB_MAIN_AVG = (0.8486, 0.7153, 0.7786, 0.7615, 0.9085)

# tab:clinical -- VME, ME, CA at threshold 0.5
TAB_CLINICAL = {
    "CIP": (0.086, 0.015, 0.969),
    "AMP": (0.109, 0.104, 0.893),
    "CTX": (0.190, 0.026, 0.953),
    "SXT": (0.236, 0.080, 0.871),
    "CAZ": (0.291, 0.033, 0.936),
    "AMC": (0.394, 0.108, 0.804),
    "GEN": (0.436, 0.023, 0.934),
}

# tab:ci -- 95% bootstrap interval, 1000 resamples, seed 42
TAB_CI = {
    "AMP": (0.919, 0.965),
    "CIP": (0.961, 0.990),
    "AMC": (0.680, 0.809),
    "SXT": (0.773, 0.891),
    "GEN": (0.612, 0.786),
    "CTX": (0.851, 0.943),
    "CAZ": (0.762, 0.882),
}

# tab:baselines -- random-split column, the five classical learners
TAB_BASELINES = {
    "LogReg":   0.7923,
    "RF":       0.7809,
    "LGBM":     0.8338,
    "XGB_gene": 0.8375,
    "XGB_raw":  0.8430,
}
# tab:ablation -- Stage 2 is the gated neural model with no classical stacking,
# which is exactly the TriLens column of test_base_probs.
TAB_STAGE2 = {
    "AMP": 0.8970, "CIP": 0.9489, "AMC": 0.6347, "SXT": 0.7453,
    "GEN": 0.5835, "CTX": 0.8659, "CAZ": 0.7893,
}
TAB_STAGE2_AVG = 0.7807
TAB_STAGE3_AVG = 0.8486

TOL_3DP = 0.0006   # a value printed to 3 dp may differ by up to half a unit
TOL_4DP = 0.00006

n_pass = n_fail = 0
failures: list[str] = []


def check(label: str, thesis: float, computed: float, tol: float) -> None:
    global n_pass, n_fail
    ok = abs(thesis - computed) <= tol
    if ok:
        n_pass += 1
    else:
        n_fail += 1
        failures.append(f"{label}: thesis {thesis:.4f}  computed {computed:.4f}  "
                        f"diff {computed - thesis:+.4f}")
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<28} thesis {thesis:>8.4f}   "
          f"computed {computed:>8.4f}   diff {computed - thesis:+.4f}")


def bootstrap_ci(y, p, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(average_precision_score(y[idx], p[idx]))
    a = np.asarray(scores)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main() -> None:
    for f in (MASTER, PRED):
        if not f.exists():
            sys.exit(f"missing: {f}")

    with open(MASTER, "rb") as fh:
        mf = pickle.load(fh)
    Y = np.asarray(mf["Y"]).astype(int)
    mask = np.asarray(mf["label_mask"]).astype(bool)
    names = [str(x) for x in mf["label_names"]]
    if names != AB:
        sys.exit(f"label_names in the pkl are {names}, not {AB}. The column "
                 "order assumed by this script does not hold -- stop and "
                 "re-check before trusting anything below.")

    z = np.load(PRED, allow_pickle=True)
    test_idx = z["test_indices"]
    ens = z["test_ens_probs"]
    base = z["test_base_probs"]
    base_names = [str(x) for x in z["base_names"]]
    print(f"model_name in npz : {z['model_name']}")
    print(f"test strains      : {len(test_idx)}")
    print(f"base_names        : {base_names}\n")

    if base.shape[1] != len(base_names):
        sys.exit("test_base_probs column count does not match base_names.")

    # ---------------------------------------------------------- tab:main
    print("tab:main  (PR-AUC, MCC, F1, F2, Accuracy)")
    acc_cols = {k: [] for k in range(5)}
    for j, ab in enumerate(AB):
        m = mask[test_idx, j]
        y = Y[test_idx, j][m]
        p = ens[m, j]
        pred = (p >= 0.5).astype(int)
        got = (
            average_precision_score(y, p),
            matthews_corrcoef(y, pred),
            f1_score(y, pred, zero_division=0),
            fbeta_score(y, pred, beta=2.0, zero_division=0),
            (pred == y).mean(),
        )
        for k in range(5):
            acc_cols[k].append(got[k])
        for k, cname in enumerate(["PR-AUC", "MCC", "F1", "F2", "Acc"]):
            check(f"{ab} {cname}", TAB_MAIN[ab][k], got[k], TOL_4DP)
    print()
    for k, cname in enumerate(["PR-AUC", "MCC", "F1", "F2", "Acc"]):
        check(f"AVERAGE {cname}", TAB_MAIN_AVG[k], float(np.mean(acc_cols[k])), TOL_4DP)

    # ------------------------------------------------------ tab:clinical
    print("\ntab:clinical  (VME, ME, CA at threshold 0.5)")
    for j, ab in enumerate(AB):
        m = mask[test_idx, j]
        y = Y[test_idx, j][m]
        pred = (ens[m, j] >= 0.5).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        got = (fn / (tp + fn), fp / (tn + fp), (tp + tn) / len(y))
        for k, cname in enumerate(["VME", "ME", "CA"]):
            check(f"{ab} {cname}", TAB_CLINICAL[ab][k], got[k], TOL_3DP)

    # ------------------------------------------------------------ tab:ci
    print("\ntab:ci  (95% bootstrap interval, 1000 resamples, seed 42)")
    for j, ab in enumerate(AB):
        m = mask[test_idx, j]
        lo, hi = bootstrap_ci(Y[test_idx, j][m], ens[m, j])
        check(f"{ab} CI low", TAB_CI[ab][0], lo, TOL_3DP)
        check(f"{ab} CI high", TAB_CI[ab][1], hi, TOL_3DP)

    # ---------------------------------- tab:baselines + tab:ablation St.2
    print("\ntab:baselines  (five classical learners, random split)")
    per_base = {}
    for c, bname in enumerate(base_names):
        vals = []
        for j, ab in enumerate(AB):
            m = mask[test_idx, j]
            vals.append(average_precision_score(Y[test_idx, j][m], base[m, c, j]))
        per_base[bname] = vals
        if bname in TAB_BASELINES:
            check(bname, TAB_BASELINES[bname], float(np.mean(vals)), TOL_4DP)

    print("\ntab:ablation  Stage 2 (gated neural model, no classical stacking)")
    if "TriLens" in per_base:
        for j, ab in enumerate(AB):
            check(f"Stage2 {ab}", TAB_STAGE2[ab], per_base["TriLens"][j], TOL_4DP)
        check("Stage2 AVERAGE", TAB_STAGE2_AVG,
              float(np.mean(per_base["TriLens"])), TOL_4DP)
    check("Stage3 AVERAGE", TAB_STAGE3_AVG, float(np.mean(acc_cols[0])), TOL_4DP)

    # ------------------------------------------------------------ verdict
    print("\n" + "=" * 72)
    print(f"{n_pass} passed, {n_fail} failed")
    if failures:
        print("\nMISMATCHES -- these are the numbers to fix in the .tex:")
        for f in failures:
            print("  " + f)
    print("=" * 72)
    print("""
ALREADY CHECKED BY HAND against CSVs recovered from the chat uploads, so a
FAIL here would mean this script disagrees with a file that already agrees
with the thesis -- investigate the script first:
  tab:main (full_metric_suite.csv), tab:clinical (stage3_clinical_metrics.csv),
  tab:ablation all three stages (layer_ablation_results.csv),
  tab:previnv and the holdout column of tab:mlst_per_ab
  (matched_prevalence_results.csv), tab:prefix, tab:umark, tab:gate,
  tab:baselines (both columns), tab:giessen_transfer.

STILL UNVERIFIED -- no artifact anywhere, needs the run repeated:
  - tab:mlst_per_ab, phylogroup-stratified column (avg 0.8531)
  - tab:holdoutclin, the three holdout columns
  - tab:threshold, the tuned columns
  - tab:mlst_components (classical 0.628 / neural 0.472 under the holdout)
  - the strain-conditioned interpretability rows of tab:attnfix
    (U_res 0.0419, U_ST 0.3569, enrichment 9.0x, one gene recovered)
  - tab:phylogroup, tab:progression
Every one of these is internally consistent (row means, differences and
derived claims in the prose all reproduce), but that is consistency, not
verification.
""")


if __name__ == "__main__":
    main()

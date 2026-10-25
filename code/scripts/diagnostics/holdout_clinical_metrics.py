"""
Clinical error rates under the MLST lineage holdout.
====================================================

WHY THIS EXISTS
---------------
Chapter 5 (Section 5.6.3) and Chapter 6 both state that the clinical error rates
were computed on the random split only, and that recomputing them under the
lineage holdout is a next step. This script does that. It is pure post-processing
of predictions that are already saved - no retraining, runs in seconds.

It reports VME / ME / CA under the holdout at threshold 0.5, next to the random
split values already in Table 5.2, and also under the prevalence-matched holdout
so the two effects can be told apart.

HOW TO RUN
----------
    conda activate amr
    cd code
    python scripts/diagnostics/holdout_clinical_metrics.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # the code/ folder
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

# Holdout predictions from the FULL hybrid model on the MLST lineage holdout.
# First existing path wins, so the strain-conditioned run is preferred over the
# pre-fix one. Both write the same filename, and an earlier version of this
# script hardcoded the second path, which is how holdout_clinical_metrics.csv
# ended up describing the pre-fix model.
PRED_CANDIDATES = [
    ROOT / "results" / "pillar_mlst_final_comparison_attnfix" / "predictions_saved.npz",
    ROOT / "results" / "pillar_mlst_final_comparison" / "predictions_saved.npz",
]

# Random-split predictions, used to compute the comparison columns. These were
# previously hardcoded as a literal dict, which silently froze the pre-fix
# numbers into every rerun. They are now recomputed from the same saved
# predictions the rest of Chapter 5 uses, so the two columns cannot drift.
RANDOM_CANDIDATES = [
    ROOT / "results" / "trilens_attnfix" / "predictions_saved.npz",
    ROOT / "results" / "trilens_v10" / "predictions_saved.npz",
]

DATA_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__mlst_holdout.pkl"
OUT_CSV = DIAG / "holdout_clinical_metrics.csv"

THRESHOLD = 0.5
N_BOOT = 2000
SEED = 42


def _first_existing(paths, what):
    for p in paths:
        if p.exists():
            return p
    sys.exit(f"MISSING {what}: none of\n  " + "\n  ".join(str(p) for p in paths))


def clinical(y, yhat):
    """Return VME, ME, CA. VME = resistant called susceptible."""
    tp = int(((y == 1) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    vme = fn / (tp + fn) if (tp + fn) else np.nan
    me = fp / (fp + tn) if (fp + tn) else np.nan
    ca = (tp + tn) / len(y)
    return vme, me, ca, tp, tn, fp, fn


def main():
    if not DATA_PKL.exists():
        sys.exit(f"MISSING: {DATA_PKL}")
    pred_npz = _first_existing(PRED_CANDIDATES, "holdout predictions")
    rand_npz = _first_existing(RANDOM_CANDIDATES, "random-split predictions")
    print(f"Holdout predictions : {pred_npz.parent.name}")
    print(f"Random-split ref    : {rand_npz.parent.name}\n")

    npz = np.load(pred_npz, allow_pickle=True)
    with open(DATA_PKL, "rb") as fh:
        data = pickle.load(fh)

    probs = np.asarray(npz["test_ens_probs"], dtype=float)
    test_idx = np.asarray(npz["test_indices"]).ravel()
    train_idx = np.asarray(npz["train_indices"]).ravel()
    Y = np.asarray(data["Y"]).astype(int)
    M = np.asarray(data["label_mask"]).astype(bool)
    names = [str(x) for x in data["label_names"]]

    # Random-split comparison columns, recomputed rather than hardcoded.
    rz = np.load(rand_npz, allow_pickle=True)
    r_probs = np.asarray(rz["test_ens_probs"], dtype=float)
    r_test_idx = np.asarray(rz["test_indices"]).ravel()
    random_split = {}
    for a, ab in enumerate(names):
        obs = M[r_test_idx, a]
        v, m_, c, *_ = clinical(Y[r_test_idx, a][obs],
                                (r_probs[obs, a] >= THRESHOLD).astype(int))
        random_split[ab] = (round(v, 3), round(m_, 3), round(c, 3))

    rng = np.random.default_rng(SEED)
    rows = []

    for a, ab in enumerate(names):
        obs = M[test_idx, a]
        y = Y[test_idx, a][obs]
        p = probs[obs, a]
        yhat = (p >= THRESHOLD).astype(int)

        vme, me, ca, tp, tn, fp, fn = clinical(y, yhat)

        # prevalence-matched version
        tr = M[train_idx, a]
        prev_train = Y[train_idx, a][tr].mean()
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n_neg = int(round(len(pos) * (1 - prev_train) / prev_train))

        vs, ms, cs = [], [], []
        for _ in range(N_BOOT):
            if n_neg <= len(neg):
                keep = np.r_[pos, rng.choice(neg, n_neg, replace=False)]
            else:
                n_pos = int(round(len(neg) * prev_train / (1 - prev_train)))
                keep = np.r_[rng.choice(pos, n_pos, replace=False), neg]
            v, m_, c, *_ = clinical(y[keep], yhat[keep])
            vs.append(v); ms.append(m_); cs.append(c)

        r_vme, r_me, r_ca = random_split.get(ab, (np.nan,) * 3)
        rows.append(dict(
            antibiotic=ab, n_test=len(y), n_resistant=len(pos),
            source_holdout=pred_npz.parent.name,
            source_random=rand_npz.parent.name,
            random_VME=r_vme, random_ME=r_me, random_CA=r_ca,
            holdout_VME=round(vme, 3), holdout_ME=round(me, 3), holdout_CA=round(ca, 3),
            matched_VME=round(float(np.mean(vs)), 3),
            matched_ME=round(float(np.mean(ms)), 3),
            matched_CA=round(float(np.mean(cs)), 3),
            VME_increase=round(vme - r_vme, 3),
            TP=tp, TN=tn, FP=fp, FN=fn))

    df = pd.DataFrame(rows)
    order = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
    if set(order) == set(df.antibiotic):
        df["_o"] = df.antibiotic.map({a: i for i, a in enumerate(order)})
        df = df.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100)
    print(f"CLINICAL ERROR RATES UNDER THE MLST LINEAGE HOLDOUT  (threshold = {THRESHOLD})")
    print("=" * 100)
    print(df[["antibiotic", "n_test", "n_resistant",
              "random_VME", "holdout_VME", "matched_VME", "VME_increase",
              "random_ME", "holdout_ME", "random_CA", "holdout_CA"]].to_string(index=False))

    print("\n" + "-" * 100)
    print(f"Mean VME  random split : {df.random_VME.mean():.3f}")
    print(f"Mean VME  holdout      : {df.holdout_VME.mean():.3f}")
    print(f"Mean VME  holdout, prevalence-matched : {df.matched_VME.mean():.3f}")
    print(f"CLSI met (VME<0.03 AND ME<0.03) under holdout : "
          f"{int(((df.holdout_VME < 0.03) & (df.holdout_ME < 0.03)).sum())} / 7")
    print("-" * 100)
    print("""
NOTE ON READING THIS
  holdout_VME  is the number Chapter 6 currently lists as future work.
  matched_VME  removes the prevalence shift, so it is comparable to the
               random-split column on equal footing.
  A large VME_increase means the clinical risk grows on unfamiliar lineages -
  which is the claim Section 5.6.3 makes but does not currently measure.
""")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()

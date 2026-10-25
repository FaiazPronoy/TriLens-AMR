"""
Per-antibiotic decision thresholds, tuned on validation.
=======================================================

FR5 in Chapter 3 is recorded as "Not met" because every antibiotic is reported at
a fixed threshold of 0.5. That single choice is the largest controllable reason
your clinical error rates look as bad as they do: at 0.5 the model retreats to the
majority class for the rarer drugs, which is why gentamicin's very major error
rate is 0.447 while its major error rate is 0.023.

This script tunes one threshold per antibiotic on the VALIDATION set and reports
the resulting clinical metrics on the TEST set, which is the methodologically
clean way to do it. Nothing is tuned on test.

Two objectives are reported:
    F2      recall-weighted, matching Chapter 4's stated rationale
    VME<=X  the lowest threshold whose major error rate stays inside the CLSI
            limit, i.e. push very major errors down as far as the ME budget allows

It needs validation predictions. If they were not saved, the script says so and
tells you exactly what to add to the training run - that one line is worth more
than any model change this week.

    conda activate amr
    cd code
    python scripts/diagnostics/threshold_tuning.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # the code/ folder
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)
DATA_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__mlst_holdout.pkl"
OUT = DIAG / "threshold_tuning.csv"

CLSI_ME = 0.03          # major error limit
GRID = np.arange(0.02, 0.99, 0.01)

# ORDER MATTERS. trilens_attnfix is the delivered strain-conditioned model and
# is what every table in Chapter 5 reports; trilens_v10 is the same architecture
# BEFORE that fix. Both write a file of the same name, so whichever comes first
# here silently decides which model the thesis table describes. This list used
# to start at trilens_v10, which is how an earlier version of
# threshold_tuning_results.csv ended up describing the pre-fix model
# (VME_at_0p5 for AMP = 0.12 instead of 0.109). Do not reorder.
# Only the main run. A fallback list used to let the table silently describe
# another model when this file was missing.
CANDIDATES = [
    ROOT / "results" / os.environ.get("TRILENS_MAIN_RUN", "trilens_attnfix") / "predictions_saved.npz",
]


def clinical(y, yhat):
    tp = int(((y == 1) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    vme = fn / (tp + fn) if (tp + fn) else np.nan
    me = fp / (fp + tn) if (fp + tn) else np.nan
    ca = (tp + tn) / max(len(y), 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f2 = (5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) else 0.0
    return vme, me, ca, f2


def main():
    with open(DATA_PKL, "rb") as fh:
        data = pickle.load(fh)
    Y = np.asarray(data["Y"]).astype(int)
    M = np.asarray(data["label_mask"]).astype(bool)
    names = [str(x) for x in data["label_names"]]

    src = None
    for c in CANDIDATES:
        if not c.exists():
            continue
        z = np.load(c, allow_pickle=True)
        if "val_ens_probs" in z.files and "val_indices" in z.files and z["val_ens_probs"].size:
            src = (c, z)
            break

    if src is None:
        print("=" * 78)
        print("NO VALIDATION PREDICTIONS FOUND")
        print("=" * 78)
        for c in CANDIDATES:
            if c.exists():
                print(f"  {c.name:28s} keys: {sorted(np.load(c, allow_pickle=True).files)}")
        print("""
This is the blocker. Threshold tuning cannot be done honestly without validation
predictions, so FR5 stays "Not met" until they exist.

ADD THIS to the training run, right where test_ens_probs is saved:

    np.savez_compressed(
        out_dir / "predictions_saved.npz",
        test_ens_probs = test_ens_probs,
        test_indices   = test_idx,
        val_ens_probs  = val_ens_probs,     # <-- add
        val_indices    = val_idx,           # <-- add
        train_indices  = train_idx,
        model_name     = "TriLens-MoE-Stack T6",   # <-- add, saves confusion later
    )

One line each. Then rerun this script and FR5 becomes satisfiable.
""")
        sys.exit(0)

    path, z = src
    print(f"Using: {path}\n")
    # Thresholds tuned on validation transfer to test only if both come from the
    # same model. Files written before the fix saved TriLens-alone validation
    # probabilities next to stacked test probabilities.
    val_is_stack = bool(z["val_is_stack"]) if "val_is_stack" in z.files else False
    if not val_is_stack:
        print("WARNING: val_ens_probs is not the stacked model (old file). Only the fixed-0.5")
        print("         columns are valid; the tuned columns are left empty.\n")
    val_p = np.asarray(z["val_ens_probs"], dtype=float)
    val_i = np.asarray(z["val_indices"]).ravel()
    test_p = np.asarray(z["test_ens_probs"], dtype=float)
    test_i = np.asarray(z["test_indices"]).ravel()

    rows = []
    for a, ab in enumerate(names):
        vm = M[val_i, a]
        tm = M[test_i, a]
        yv, pv = Y[val_i, a][vm], val_p[vm, a]
        yt, pt = Y[test_i, a][tm], test_p[tm, a]
        if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
            continue

        # objective 1: best F2 on validation
        f2s = [clinical(yv, (pv >= t).astype(int))[3] for t in GRID]
        t_f2 = float(GRID[int(np.argmax(f2s))])

        # objective 2: lowest threshold keeping validation ME within the CLSI limit
        ok = [t for t in GRID if clinical(yv, (pv >= t).astype(int))[1] <= CLSI_ME]
        t_me = float(min(ok)) if ok else np.nan

        base = clinical(yt, (pt >= 0.5).astype(int))
        if not val_is_stack:
            t_f2, t_me = np.nan, np.nan
        tuned_f2 = clinical(yt, (pt >= t_f2).astype(int)) if not np.isnan(t_f2) else (np.nan,) * 4
        tuned_me = clinical(yt, (pt >= t_me).astype(int)) if not np.isnan(t_me) else (np.nan,) * 4

        rows.append(dict(
            antibiotic=ab,
            VME_at_0p5=round(base[0], 3), ME_at_0p5=round(base[1], 3),
            thr_F2=round(t_f2, 2) if not np.isnan(t_f2) else np.nan,
            VME_F2=round(tuned_f2[0], 3), ME_F2=round(tuned_f2[1], 3),
            thr_MEcap=round(t_me, 2) if not np.isnan(t_me) else np.nan,
            VME_MEcap=round(tuned_me[0], 3) if not np.isnan(t_me) else np.nan,
            ME_MEcap=round(tuned_me[1], 3) if not np.isnan(t_me) else np.nan,
            VME_gain=round(base[0] - tuned_f2[0], 3) if not np.isnan(t_f2) else np.nan))

    df = pd.DataFrame(rows)
    # Provenance, written into the file itself. Without this a stale CSV is
    # indistinguishable from a fresh one once it leaves this machine.
    df["source_run"] = path.parent.name
    df["val_is_stack"] = val_is_stack
    pd.set_option("display.width", 220)
    print("=" * 100)
    print("THRESHOLDS TUNED ON VALIDATION, EVALUATED ON TEST")
    print("=" * 100)
    print(df.to_string(index=False))
    print("\n" + "-" * 100)
    print(f"Mean VME at 0.5          : {df.VME_at_0p5.mean():.3f}")
    print(f"Mean VME, F2-tuned       : {df.VME_F2.mean():.3f}")
    print(f"Mean VME, ME-capped      : {df.VME_MEcap.mean():.3f}")
    print(f"Antibiotics meeting BOTH CLSI limits: "
          f"{int(((df.VME_F2 < 0.03) & (df.ME_F2 < 0.03)).sum())} / {len(df)}")
    print("-" * 100)
    df.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")
    print("""
If VME_gain is large, FR5 moves from "Not met" to "Met", Table 5.2 gains a tuned
column, and the clinical section stops being the weakest part of the thesis -
without retraining anything.
""")


if __name__ == "__main__":
    main()

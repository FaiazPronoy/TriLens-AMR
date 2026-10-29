import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)
DATA_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__mlst_holdout.pkl"
OUT = DIAG / "threshold_tuning.csv"

CLSI_ME = 0.03
CLSI_VME = 0.015
GRID = np.arange(0.02, 0.99, 0.01)


CANDIDATES = [
    ROOT / "results" / "trilens_stack_random" / "predictions_saved.npz",
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
        print(
            "\nThreshold tuning needs validation predictions. Run step11 for the random split first;"
            "\nit saves val_ens_probs and val_indices in predictions_saved.npz."
        )
        sys.exit(0)

    path, z = src
    print(f"Using: {path}\n")

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

        f2s = [clinical(yv, (pv >= t).astype(int))[3] for t in GRID]
        t_f2 = float(GRID[int(np.argmax(f2s))])

        ok = [t for t in GRID if clinical(yv, (pv >= t).astype(int))[1] <= CLSI_ME]
        t_me = float(min(ok)) if ok else np.nan

        base = clinical(yt, (pt >= 0.5).astype(int))
        if not val_is_stack:
            t_f2, t_me = np.nan, np.nan
        tuned_f2 = clinical(yt, (pt >= t_f2).astype(int)) if not np.isnan(t_f2) else (np.nan,) * 4
        tuned_me = clinical(yt, (pt >= t_me).astype(int)) if not np.isnan(t_me) else (np.nan,) * 4

        rows.append(
            dict(
                antibiotic=ab,
                VME_at_0p5=round(base[0], 3),
                ME_at_0p5=round(base[1], 3),
                thr_F2=round(t_f2, 2) if not np.isnan(t_f2) else np.nan,
                VME_F2=round(tuned_f2[0], 3),
                ME_F2=round(tuned_f2[1], 3),
                thr_MEcap=round(t_me, 2) if not np.isnan(t_me) else np.nan,
                VME_MEcap=round(tuned_me[0], 3) if not np.isnan(t_me) else np.nan,
                ME_MEcap=round(tuned_me[1], 3) if not np.isnan(t_me) else np.nan,
                VME_gain=round(base[0] - tuned_f2[0], 3) if not np.isnan(t_f2) else np.nan,
            )
        )

    df = pd.DataFrame(rows)

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
    print(
        f"Antibiotics meeting BOTH CLSI limits: "
        f"{int(((df.VME_F2 <= CLSI_VME) & (df.ME_F2 <= CLSI_ME)).sum())} / {len(df)}"
    )
    print("-" * 100)
    df.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

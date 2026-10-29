import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)


PRED_CANDIDATES = [
    ROOT / "results" / "trilens_stack_lineage" / "predictions_saved.npz",
]
PRED_NPZ = next((p for p in PRED_CANDIDATES if p.exists()), PRED_CANDIDATES[-1])
SPLIT_PKL = ROOT / "data" / "processed" / "mlst_lineage_holdout_split.pkl"
DATA_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__mlst_holdout.pkl"

N_BOOT = 2000
SEED = 42
OUT_CSV = DIAG / "matched_prevalence.csv"

AB_ORDER = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def describe(name, obj):
    print(f"\n--- {name} ---")
    if isinstance(obj, dict):
        for k, v in obj.items():
            shape = getattr(v, "shape", None)
            print(f"  {k:32s} {type(v).__name__:12s} {shape if shape is not None else ''}")
    elif hasattr(obj, "files"):
        for k in obj.files:
            print(f"  {k:32s} {obj[k].dtype}  {obj[k].shape}")
    else:
        print(f"  {type(obj)}")


def pick(container, *candidates):
    keys = container.files if hasattr(container, "files") else container.keys()
    for c in candidates:
        if c in keys:
            return container[c]
    return None


def main():
    for p in (PRED_NPZ, SPLIT_PKL, DATA_PKL):
        if not p.exists():
            sys.exit(f"MISSING: {p}\nFix the paths at the top of this script.")

    npz = np.load(PRED_NPZ, allow_pickle=True)
    split = load_pickle(SPLIT_PKL)
    data = load_pickle(DATA_PKL)

    describe("predictions_saved.npz", npz)
    describe("mlst_lineage_holdout_split.pkl", split)
    describe("arcadia_v2_gene_filtered__mlst_holdout.pkl", data)

    probs = pick(npz, "test_ens_probs", "test_probs", "ens_test_probs")
    test_idx = pick(npz, "test_indices", "test_idx")
    train_idx = pick(npz, "train_indices", "train_idx")

    if train_idx is None:
        train_idx = pick(split, "train_idx", "train_indices", "train")
    if test_idx is None:
        test_idx = pick(split, "test_idx", "test_indices", "test")

    Y = pick(data, "Y", "y", "labels")
    M = pick(data, "label_mask", "mask", "Y_mask")
    names = pick(data, "label_names", "antibiotics", "ab_names")

    missing = [
        n
        for n, v in [("probs", probs), ("test_idx", test_idx), ("train_idx", train_idx), ("Y", Y), ("mask", M)]
        if v is None
    ]
    if missing:
        sys.exit(f"\nCould not find: {missing}\nLook at the structure printed above and edit pick(...) accordingly.")

    names = [str(x) for x in names] if names is not None else AB_ORDER
    Y = np.asarray(Y).astype(int)
    M = np.asarray(M).astype(bool)
    probs = np.asarray(probs, dtype=float)
    test_idx = np.asarray(test_idx).ravel()
    train_idx = np.asarray(train_idx).ravel()

    if probs.shape[0] != test_idx.shape[0]:
        sys.exit(f"\nShape mismatch: probs has {probs.shape[0]} rows, test_indices has {test_idx.shape[0]}.")

    rng = np.random.default_rng(SEED)
    rows = []

    print("\n" + "=" * 88)
    print("MATCHED-PREVALENCE RE-EVALUATION - MLST lineage holdout")
    print("=" * 88)

    for a, ab in enumerate(names):
        obs = M[test_idx, a]
        y = Y[test_idx, a][obs]
        p = probs[obs, a]

        tr_obs = M[train_idx, a]
        prev_train = Y[train_idx, a][tr_obs].mean()
        prev_test = y.mean()

        pr_raw = average_precision_score(y, p)
        roc = roc_auc_score(y, p)

        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)

        n_neg_target = int(round(len(pos) * (1 - prev_train) / prev_train))

        if n_neg_target > len(neg):
            n_pos_target = int(round(len(neg) * prev_train / (1 - prev_train)))
            draws = [
                average_precision_score(
                    np.r_[np.ones(n_pos_target), np.zeros(len(neg))],
                    np.r_[p[rng.choice(pos, n_pos_target, replace=False)], p[neg]],
                )
                for _ in range(N_BOOT)
            ]
            note = "subsampled positives"
        else:
            draws = [
                average_precision_score(
                    np.r_[np.ones(len(pos)), np.zeros(n_neg_target)],
                    np.r_[p[pos], p[rng.choice(neg, n_neg_target, replace=False)]],
                )
                for _ in range(N_BOOT)
            ]
            note = "subsampled negatives"

        draws = np.array(draws)
        lo, hi = np.percentile(draws, [2.5, 97.5])

        rows.append(
            dict(
                antibiotic=ab,
                n_pos=len(pos),
                n_neg=len(neg),
                prev_train=round(prev_train, 4),
                prev_test=round(prev_test, 4),
                pr_auc_raw=round(pr_raw, 4),
                pr_auc_matched=round(draws.mean(), 4),
                matched_lo=round(lo, 4),
                matched_hi=round(hi, 4),
                roc_auc=round(roc, 4),
                recovered=round(draws.mean() - pr_raw, 4),
                note=note,
            )
        )

    df = pd.DataFrame(rows)
    if set(AB_ORDER) == set(df.antibiotic):
        df["_o"] = df.antibiotic.map({a: i for i, a in enumerate(AB_ORDER)})
        df = df.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    pd.set_option("display.width", 200)
    print()
    print(df.to_string(index=False))

    print("\n" + "-" * 88)
    print(f"Mean PR-AUC as reported (unmatched) : {df.pr_auc_raw.mean():.4f}")
    print(f"Mean PR-AUC prevalence-matched      : {df.pr_auc_matched.mean():.4f}")
    print(f"Mean ROC-AUC                        : {df.roc_auc.mean():.4f}")
    print("-" * 88)
    print("""
How to read this
  pr_auc_raw      average precision on the holdout as measured
  pr_auc_matched  the same predictions with test prevalence set equal to
                  training prevalence
  recovered       how much of the drop is explained by prevalence

  A low matched score means a real failure to generalise.
  A matched score that jumps up means most of the drop was prevalence.
""")

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()

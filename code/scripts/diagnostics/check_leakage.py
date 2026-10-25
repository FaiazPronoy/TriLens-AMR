"""
check_leakage.py
================
Automatic checks that no test information reaches training, and that every
saved prediction file describes the split it claims to. Run it after the
training runs (the Kaggle analysis notebook runs it first).

    python scripts/diagnostics/check_leakage.py

Writes results/diagnostics/leakage_checks.csv (check, status, detail).
status: PASS, WARN (disclose in the paper, not an error), FAIL.
Exit code 1 if any check FAILs.

Checks
  1. every split: train / val / test disjoint, together exactly the 6,962 strains
  2. the three gene-filtered datasets carry the same split as the split files
  3. lineage holdout: no sequence type shared between train and test, or val and test
  4. exact duplicate genomes between train and test (WARN: report, do not fail)
  5. every predictions_saved.npz: its train / test indices are the split's indices
  6. neural and classical arms of each stack: identical train / test indices
  7. the main run's validation probabilities come from the stack, not TriLens alone
     (needed for honest threshold tuning)
  8. over- / under-fitting on the random split: out-of-fold PR-AUC on train against
     test PR-AUC for every base learner (WARN if they differ by more than 0.05; both
     are unseen data from the same distribution, so they should agree)
"""
from __future__ import annotations

import gc
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
N = 6962

rows = []


def add(check, ok, detail, warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    rows.append(dict(check=check, status=status, detail=detail))
    print(f"[{status}] {check}: {detail}")


def idx(sp, k):
    return np.asarray(sp.get(k, sp.get(k.replace("_idx", ""), []))).ravel().astype(np.int64)


def check_split(name, sp):
    tr, va, te = idx(sp, "train_idx"), idx(sp, "val_idx"), idx(sp, "test_idx")
    every = np.concatenate([tr, va, te])
    ok = (len(set(tr) & set(te)) == 0 and len(set(tr) & set(va)) == 0 and len(set(va) & set(te)) == 0
          and len(every) == len(set(every.tolist())) == N)
    add(f"split {name}: disjoint and complete", ok,
        f"train {len(tr)}, val {len(va)}, test {len(te)}, unique {len(set(every.tolist()))}/{N}")
    return tr, va, te


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main() -> int:
    splits = {}
    master = load(PROC / "arcadia_v2_master_features.pkl")
    for key in ("random_split", "phylogeny_split"):
        splits[key] = check_split(key, master[key])
    mlst = load(PROC / "mlst_lineage_holdout_split.pkl")
    splits["mlst_lineage_holdout_split"] = check_split("mlst_lineage_holdout_split", mlst)
    del master

    # 3. sequence types
    st = np.asarray(mlst["st_per_genome"], dtype=object)
    tr, va, te = splits["mlst_lineage_holdout_split"]
    known = lambda ix: {s for s in st[ix] if s is not None and str(s) not in ("", "nan", "None")}
    shared_tt, shared_vt = known(tr) & known(te), known(va) & known(te)
    add("lineage holdout: no sequence type in both train and test", not shared_tt,
        f"{len(known(tr))} train STs, {len(known(te))} test STs, shared {len(shared_tt)}")
    add("lineage holdout: no sequence type in both val and test", not shared_vt, f"shared {len(shared_vt)}")
    n_unknown = sum(1 for s in st if s is None or str(s) in ("", "nan", "None"))
    unknown_in_test = sum(1 for s in st[te] if s is None or str(s) in ("", "nan", "None"))
    add("lineage holdout: strains without a sequence type kept out of test", unknown_in_test == 0,
        f"{n_unknown} strains without ST, {unknown_in_test} in test")

    # 2 + 4. the gene-filtered datasets
    for fname, key in (("arcadia_v2_gene_filtered.pkl", "random_split"),
                       ("arcadia_v2_gene_filtered__phylogeny_clean.pkl", "phylogeny_split"),
                       ("arcadia_v2_gene_filtered__mlst_holdout.pkl", "mlst_lineage_holdout_split")):
        p = PROC / fname
        if not p.exists():
            add(f"{fname}: present", False, "file missing")
            continue
        gf = load(p)
        tr, va, te = splits[key]
        if key in gf:
            g_tr, g_va, g_te = idx(gf[key], "train_idx"), idx(gf[key], "val_idx"), idx(gf[key], "test_idx")
            same = np.array_equal(g_tr, tr) and np.array_equal(g_va, va) and np.array_equal(g_te, te)
            add(f"{fname}: carries the same {key}", same, "identical indices and order" if same else "DIFFERENT")
        else:
            add(f"{fname}: carries {key}", False, f"keys: {sorted(gf)[:12]}")
        X = gf.get("X_snp_filtered")
        if X is not None:
            h = np.array([hashlib.blake2b(np.ascontiguousarray(X[i]).tobytes(), digest_size=16).digest()
                          for i in range(X.shape[0])], dtype=object)
            train_h = set(h[tr].tolist())
            dup = int(sum(1 for x in h[te] if x in train_h))
            add(f"{fname}: test genomes identical to a training genome (on the filtered SNPs)", dup == 0,
                f"{dup} of {len(te)}; near-duplicates are measured by run_pillar_dedup_eval",
                warn_only=True)
        del gf, X
        gc.collect()

    # 5 + 6. saved predictions
    run_split = {
        "stage3_cascadestack_v9": "random_split", "trilens_attnfix": "random_split",
        "pillar_cascadestack_classical_phylogeny_clean": "phylogeny_split",
        "pillar_trilens_t7_clean_gene_set": "phylogeny_split",
        "pillar_cascadestack_classical_mlst": "mlst_lineage_holdout_split",
        "pillar_trilens_mlst": "mlst_lineage_holdout_split",
        "pillar_mlst_final_comparison": "mlst_lineage_holdout_split",
    }
    for base in (RES, RES / "standard"):
        for run, key in run_split.items():
            p = base / run / "predictions_saved.npz"
            if not p.exists():
                continue
            tag = f"{p.parent.relative_to(RES).as_posix()}"
            z = np.load(p, allow_pickle=True)
            tr, va, te = splits[key]
            for k, ref in (("test_indices", te), ("train_indices", tr), ("val_indices", va)):
                if k in z.files and z[k].size:
                    same = np.array_equal(np.asarray(z[k]).ravel(), ref)
                    add(f"{tag}: {k} = {key}", same, "match" if same else "DIFFERENT")
        for neural, classical in (("trilens_attnfix", "stage3_cascadestack_v9"),
                                  ("pillar_trilens_t7_clean_gene_set", "pillar_cascadestack_classical_phylogeny_clean"),
                                  ("pillar_trilens_mlst", "pillar_cascadestack_classical_mlst")):
            a, b = base / neural / "predictions_saved.npz", RES / classical / "predictions_saved.npz"
            if a.exists() and b.exists():
                za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
                same = all(np.array_equal(np.asarray(za[k]).ravel(), np.asarray(zb[k]).ravel())
                           for k in ("train_indices", "test_indices") if k in za.files and k in zb.files)
                add(f"{a.parent.relative_to(RES).as_posix()} + {classical}: same strains, same order", same,
                    "match" if same else "DIFFERENT - stacking would pair different strains")

        # 7. validation probabilities of the main run
        p = base / "trilens_attnfix" / "predictions_saved.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            flag = bool(z["val_is_stack"]) if "val_is_stack" in z.files else False
            add(f"{p.parent.relative_to(RES).as_posix()}: validation probabilities come from the stack", flag,
                "yes" if flag else "no flag: file is from the code before the fix; tab:threshold must not use it")

    # 8. out-of-fold against test, per base learner
    from sklearn.metrics import average_precision_score
    _m = load(PROC / "arcadia_v2_master_features.pkl")
    Yall = np.nan_to_num(np.asarray(_m["Y"], dtype=float))
    Mall = np.asarray(_m["label_mask"]).astype(bool)
    del _m
    for base in (RES, RES / "standard"):
        # random split only: under the lineage holdout a large gap is the finding itself
        for run in ("trilens_attnfix", "stage3_cascadestack_v9"):
            p = base / run / "predictions_saved.npz"
            if not p.exists() or (base != RES and run.startswith("stage3")):
                continue
            z = np.load(p, allow_pickle=True)
            if not {"oof_base_probs", "test_base_probs", "train_indices", "test_indices"} <= set(z.files):
                continue
            tr, te = np.asarray(z["train_indices"]).ravel(), np.asarray(z["test_indices"]).ravel()
            names = [str(n) for n in z["base_names"]] if "base_names" in z.files else \
                [f"base{i}" for i in range(z["oof_base_probs"].shape[1])]
            for b, bn in enumerate(names):
                o, t = [], []
                for a in range(Yall.shape[1]):
                    mo, mt = Mall[tr, a], Mall[te, a]
                    yo, yt = Yall[tr, a][mo], Yall[te, a][mt]
                    if len(np.unique(yo)) < 2 or len(np.unique(yt)) < 2:
                        continue
                    o.append(average_precision_score(yo, z["oof_base_probs"][mo, b, a]))
                    t.append(average_precision_score(yt, z["test_base_probs"][mt, b, a]))
                if o:
                    gap = float(np.mean(o) - np.mean(t))
                    add(f"{p.parent.relative_to(RES).as_posix()} / {bn}: out-of-fold vs test PR-AUC",
                        abs(gap) <= 0.05, f"OOF {np.mean(o):.4f}, test {np.mean(t):.4f}, gap {gap:+.4f}",
                        warn_only=True)

    df = pd.DataFrame(rows)
    out = RES / "diagnostics" / "leakage_checks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_fail, n_warn = int((df.status == "FAIL").sum()), int((df.status == "WARN").sum())
    print(f"\n{(df.status == 'PASS').sum()} pass, {n_warn} warn, {n_fail} fail  ->  {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

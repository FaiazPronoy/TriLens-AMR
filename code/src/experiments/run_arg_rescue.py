"""
W6: does adding each strain's own acquired resistance genes fix lineage-held-out prediction?
Pre-registration: docs/experiments/w6_arg_rescue.md (arms, model, splits, endpoint, rule).

    python -m src.experiments.run_arg_rescue [--splits st_grouped_5fold,random_5fold,country_uk_to_norway,...]
        [--drugs paper|all16] [--labels ri|no_intermediate] [--out results/w6_rescue]

Arms: F0 catalogue presence/absence; F0A = F0 + acquired genes; A acquired genes only;
M metadata only. Every arm sees the same strains: profiled (presence/absence) AND with an
AMRFinderPlus table. Out-of-fold scores are cached per (split, arm, drug), so a stopped run
continues. Writes scores_<split>.csv.gz, metrics.csv, within_st.csv, deltas.csv and verdict.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.metadata import PROCESSED, ROOT
from src.evaluation.grouped import cluster_bootstrap, group_codes, nap, summary, within_st_concordance

PAPER = {"AMP": "ampicillin", "CIP": "ciprofloxacin", "AMC": "amoxicillin/clavulanic acid",
         "SXT": "trimethoprim/sulfamethoxazole", "GEN": "gentamicin", "CTX": "cefotaxime",
         "CAZ": "ceftazidime"}
PRIMARY, CONTROL = ["CTX", "CAZ", "GEN"], "CIP"
PASS_DELTA, CONTROL_TOL = 0.10, 0.03
XGB = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
           tree_method="hist", random_state=0, n_jobs=os.cpu_count() or 4, verbosity=0)
PA_MIN, PA_MAX = 0.01, 0.99
META_NUM = ["Contigs", "Size", "GC.Content", "Contig.N50", "CDS", "TRNA", "Collection.Year"]
META_CAT = ["Isolation.Country", "Sequencing.Platform", "Publication"]


def repo_results() -> Path:
    r = os.environ.get("TRILENS_REPO_RESULTS")
    return Path(r) if r else ROOT / "results"


# ── inputs ──────────────────────────────────────────────────────────────────────────────
def load_labels(drugs, mode):
    raw = pd.read_csv(repo_results() / "data_inventory" / "labels16.csv").set_index("strain_index")
    cols = {**PAPER, **{c: c for c in raw.columns}} if drugs == "all16" else PAPER
    mapping = {"R": 1.0, "I": 1.0 if mode == "ri" else np.nan, "S": 0.0}
    return pd.DataFrame({k: raw[v].map(mapping) for k, v in cols.items()})


def load_pa():
    with open(PROCESSED / "arcadia_v2_gene_presence_absence.pkl", "rb") as f:
        pa = pickle.load(f)
    if not str(pa.get("encoding_rule", "")).startswith("present = (allele letter"):
        raise ValueError("use the rebuilt presence/absence matrix (outputs-pa_fix)")
    X = np.asarray(pa["X_pa"], dtype=np.uint8)
    matched = np.asarray(pa["matched_mask"]).astype(bool)
    prev = X[matched].mean(axis=0)
    keep = (prev >= PA_MIN) & (prev <= PA_MAX)
    return X[:, keep], matched, np.asarray(pa["gene_ids"]).astype(str)[keep]


def load_arg(path):
    A = pd.read_csv(path, index_col=0)
    A.index = A.index.astype(int)
    return A


def load_meta():
    c = pd.read_csv(repo_results() / "data_inventory" / "covariates.csv").set_index("strain_index")
    X = c[META_NUM].astype(float)
    for col in META_CAT:
        X[col] = pd.Categorical(c[col].astype(str).where(c[col].notna(), "missing")).codes
    return X


def load_splits(names):
    from src.splits.kfold import load_kfold
    out = {}
    for n in names:
        if (repo_results() / "splits_kfold" / f"{n}.json").exists():
            k = load_kfold(n)
            out[n] = [(np.setdiff1d(np.arange(k["n_strains"]), f), f) for f in k["folds"]]
        else:   # a train/val/test split: one fold, validation strains join training
            d = json.loads((repo_results() / "splits" / f"{n}.json").read_text())
            tr = np.r_[np.asarray(d["train_idx"], int), np.asarray(d["val_idx"], int)]
            out[n] = [(tr, np.asarray(d["test_idx"], int))]
    return out


# ── models ──────────────────────────────────────────────────────────────────────────────
def fit_predict(arm, Xtr, ytr, Xte):
    if arm == "M":
        from sklearn.ensemble import HistGradientBoostingClassifier
        cat = [Xtr.columns.get_loc(c) for c in META_CAT if c in Xtr.columns]
        m = HistGradientBoostingClassifier(random_state=0, categorical_features=cat or None)
        return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    from xgboost import XGBClassifier
    return XGBClassifier(**XGB).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def features(arm, data, idx):
    if arm == "M":
        return data["meta"].loc[idx]
    parts = []
    if arm in ("F0", "F0A"):
        parts.append(data["pa"][idx])
    if arm in ("A", "F0A"):
        parts.append(data["arg"].loc[idx].to_numpy(np.uint8))
    return np.hstack(parts)


def oof(arm, drug, folds, data, keep):
    y = data["Y"][drug].to_numpy()
    s = np.full(len(y), np.nan)
    for tr, te in folds:
        tr = tr[keep[tr] & ~np.isnan(y[tr])]
        te = te[keep[te] & ~np.isnan(y[te])]
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        s[te] = fit_predict(arm, features(arm, data, tr), y[tr].astype(int), features(arm, data, te))
    return s


# ── evaluation ──────────────────────────────────────────────────────────────────────────
def evaluate(scores, Y, st, arms, drugs, n_boot):
    rows, wst, deltas = [], [], []
    groups = group_codes(st)
    for split, S in scores.items():
        for arm in arms:
            for d in drugs:
                col = f"{arm}|{d}"
                if col not in S:
                    continue
                rows.append(dict(split=split, arm=arm, drug=d, **summary(Y[d], S[col])))
                wst.append(dict(split=split, arm=arm, drug=d, **within_st_concordance(Y[d], S[col], st)))
        if {"F0", "F0A"} <= set(arms):
            a = pd.DataFrame({d: S.get(f"F0|{d}") for d in drugs})
            b = pd.DataFrame({d: S.get(f"F0A|{d}") for d in drugs})
            for name, ds in [(d, [d]) for d in drugs] + [("primary_mean", [d for d in PRIMARY if d in drugs])]:
                if not ds or a[ds].isna().all().all():
                    continue
                pt, lo, hi, ng = cluster_bootstrap(Y, a, b, groups, ds, n_boot=n_boot)
                deltas.append(dict(split=split, endpoint=name, delta=pt, ci_low=lo, ci_high=hi, n_groups=ng))
    return pd.DataFrame(rows), pd.DataFrame(wst), pd.DataFrame(deltas)


def verdict(deltas, split="st_grouped_5fold"):
    d = deltas[deltas.split == split].set_index("endpoint")
    if "primary_mean" not in d.index:
        return {"status": "not evaluated", "reason": f"no primary endpoint on {split}"}
    p = d.loc["primary_mean"]
    out = {"split": split, "delta": p.delta, "ci": [p.ci_low, p.ci_high],
           "pass_rule": f"delta >= {PASS_DELTA} and ci_low > 0",
           "primary_pass": bool(p.delta >= PASS_DELTA and p.ci_low > 0)}
    if CONTROL in d.index:
        c = d.loc[CONTROL]
        out["control_cip_delta"] = c.delta
        out["control_pass"] = bool(abs(c.delta) <= CONTROL_TOL)
    out["reading"] = ("supported: the catalogue omits determinants that predict unseen lineages"
                      if out["primary_pass"] and out.get("control_pass", False) else
                      "added block helps every drug, not specifically the missing ones"
                      if out["primary_pass"] else "not supported: the paper is the benchmark alone")
    return out


def run(data, splits, arms, drugs, out, n_boot=2000):
    out.mkdir(parents=True, exist_ok=True)
    keep = data["keep"]
    scores = {}
    for split, folds in splits.items():
        p = out / f"scores_{split}.csv.gz"
        S = pd.read_csv(p, index_col=0) if p.exists() else pd.DataFrame(index=np.arange(len(keep)))
        for arm in arms:
            for d in drugs:
                col = f"{arm}|{d}"
                if col in S:
                    continue
                t0 = time.time()
                S[col] = oof(arm, d, folds, data, keep)
                S.to_csv(p)
                print(f"{split:24s} {arm:4s} {d:5s} nAP={nap(data['Y'][d], S[col]):.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        scores[split] = S
    m, w, dl = evaluate(scores, data["Y"], data["st"], arms, drugs, n_boot)
    m.to_csv(out / "metrics.csv", index=False)
    w.to_csv(out / "within_st.csv", index=False)
    dl.to_csv(out / "deltas.csv", index=False)
    v = verdict(dl) if len(dl) else {"status": "no deltas"}
    v.update(strains_used=int(keep.sum()), strains_total=int(len(keep)))
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(json.dumps(v, indent=1, default=float))
    return m, w, dl, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="st_grouped_5fold,random_5fold")
    ap.add_argument("--arms", default="F0,F0A,A,M")
    ap.add_argument("--drugs", default="paper", choices=["paper", "all16"])
    ap.add_argument("--labels", default="ri", choices=["ri", "no_intermediate"])
    ap.add_argument("--arg-matrix", default="results/arg_matrix/acquired.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out or f"results/w6_rescue/{a.drugs}_{a.labels}")
    Y = load_labels(a.drugs, a.labels)
    pa, matched, _ = load_pa()
    arg = load_arg(a.arg_matrix)
    has_arg = np.zeros(len(Y), bool)
    has_arg[arg.index.to_numpy()] = True
    keep = matched & has_arg
    st = pd.read_csv(repo_results() / "data_inventory" / "strain_table.csv", dtype={"st": str}).st.to_numpy(object)
    print(f"strains: {keep.sum()} of {len(keep)} (profiled {matched.sum()}, with ARG table {has_arg.sum()}); "
          f"catalogue columns {pa.shape[1]}, acquired-gene columns {arg.shape[1]}", flush=True)
    data = dict(Y=Y, pa=pa, arg=arg.reindex(np.arange(len(Y)), fill_value=0), meta=load_meta(), keep=keep, st=st)
    run(data, load_splits(a.splits.split(",")), a.arms.split(","), list(Y.columns), out, a.n_boot)


if __name__ == "__main__":
    main()

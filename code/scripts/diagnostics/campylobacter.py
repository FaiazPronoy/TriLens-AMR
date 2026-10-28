"""
campylobacter.py
================
W13, the second species. Pre-registration: the Campylobacter amendment in
docs/experiments/w13_second_species.md. Uses only the species_probe candidates file.

  primary   model (XGBoost on AMRFinderPlus calls) - determinant rule, mean nAP over the four
            drugs, SNP-cluster-grouped 5-fold, cluster-clustered CI (replicates W10-B)
  W12       within-SNP-cluster concordance of assembly length, contigs, N50 with each phenotype
  also      metadata arm on the three assembly statistics, same folds

    python scripts/diagnostics/campylobacter.py --clindamycin-23s yes|no
        [--candidates results/species_probe/candidates_Campylobacter.tsv.gz]
        [--out results/w13_campylobacter] [--n-boot 2000]

--clindamycin-23s must be given: whether 23S point mutations count as clindamycin determinants,
settled from the literature before this runs (see the amendment).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.grouped import cluster_bootstrap, nap, summary, within_st_concordance  # noqa: E402
from src.experiments.run_arg_rescue import XGB  # noqa: E402
from src.experiments.run_external import calls  # noqa: E402
from src.splits.kfold import st_grouped  # noqa: E402

DRUGS = {"CIP": "ciprofloxacin", "NAL": "nalidixic acid", "TET": "tetracycline", "CLI": "clindamycin"}
MIN_CARRIERS = 5
PASS_DELTA, ASM_PASS = 0.05, 0.55
ASM = {"length": "asm_stats_length_bp", "contigs": "asm_stats_n_contig", "n50": "asm_stats_contig_n50"}


def rules(cli_23s):
    cli = r"^erm\(|^lnu\(|^cfr\(" + (r"|^23S_.*\|POINT$" if cli_23s else "")
    return {"CIP": r"^gyrA_.*\|POINT$", "NAL": r"^gyrA_.*\|POINT$", "TET": r"^tet\(", "CLI": cli}


def rule_score(acq, pt, pattern):
    syms = list(acq) + [p + "|POINT" for p in pt]
    return float(any(re.search(pattern, s) for s in syms))


def label(s, name):
    if not isinstance(s, str):
        return np.nan
    for tok in s.split(","):
        if "=" in tok:
            d, c = tok.rsplit("=", 1)
            if d.strip() == name:
                return {"R": 1.0, "S": 0.0, "I": 1.0}.get(c.strip(), np.nan)
    return np.nan


def load(path, cli_23s):
    f = pd.read_csv(path, sep="\t", dtype=str).reset_index(drop=True)
    parsed = [calls(g) for g in f.AMR_genotypes]
    acq_all = pd.Series([s for a, _ in parsed for s in a]).value_counts()
    pt_all = pd.Series([s for _, p in parsed for s in p]).value_counts()
    a_cols = sorted(acq_all[acq_all >= MIN_CARRIERS].index)
    p_cols = sorted(pt_all[pt_all >= MIN_CARRIERS].index)
    X = pd.DataFrame([{**{c: int(c in a) for c in a_cols}, **{f"{c}|POINT": int(c in p) for c in p_cols}}
                      for a, p in parsed], dtype="uint8")
    Y = pd.DataFrame({k: [label(s, v) for s in f.AST_phenotypes] for k, v in DRUGS.items()})
    pats = rules(cli_23s)
    R = pd.DataFrame({k: [rule_score(a, p, pats[k]) for a, p in parsed] for k in DRUGS})
    M = pd.DataFrame({k: pd.to_numeric(f[v], errors="coerce") for k, v in ASM.items()})
    cluster = f.PDS_acc.where(f.PDS_acc.notna(), None).to_numpy(object)
    return dict(X=X, Y=Y, R=R, M=M, cluster=cluster, frame=f, a_cols=a_cols, p_cols=p_cols)


def folds(cluster):
    test, never = st_grouped(cluster)
    n = len(cluster)
    return [(np.setdiff1d(np.arange(n), t), t) for t in test], never


def oof(X, y, fl, kind):
    s = np.full(len(y), np.nan)
    for tr, te in fl:
        tr, te = tr[~np.isnan(y[tr])], te[~np.isnan(y[te])]
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        if kind == "model":
            from xgboost import XGBClassifier
            m = XGBClassifier(**XGB).fit(X[tr], y[tr].astype(int))
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr].astype(int))
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="results/species_probe/candidates_Campylobacter.tsv.gz")
    ap.add_argument("--clindamycin-23s", required=True, choices=["yes", "no"])
    ap.add_argument("--out", default="results/w13_campylobacter")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(a.candidates, a.clindamycin_23s == "yes")
    fl, never = folds(d["cluster"])
    print(f"isolates {len(d['Y'])}, features {d['X'].shape[1]} ({len(d['a_cols'])} genes, "
          f"{len(d['p_cols'])} point mutations), clusters {pd.Series(d['cluster']).nunique()}, "
          f"never tested (no cluster) {len(never)}", flush=True)

    S = pd.DataFrame(index=range(len(d["Y"])))
    X, Mx = d["X"].to_numpy(np.uint8), d["M"].to_numpy(float)
    for k in DRUGS:
        y = d["Y"][k].to_numpy()
        S[f"model|{k}"] = oof(X, y, fl, "model")
        S[f"meta|{k}"] = oof(Mx, y, fl, "meta")
        S[f"rule|{k}"] = d["R"][k].where(pd.Series(S[f"model|{k}"]).notna())
    S.to_csv(out / "scores.csv.gz")

    rows = [dict(arm=arm, drug=k, **summary(d["Y"][k], S[f"{arm}|{k}"]))
            for arm in ("model", "rule", "meta") for k in DRUGS]
    m = pd.DataFrame(rows)
    m.to_csv(out / "metrics.csv", index=False)

    codes = pd.factorize(pd.Series(d["cluster"]).fillna("__none__"))[0]
    deltas = []
    for b, c in (("model", "rule"), ("model", "meta")):
        Sa = pd.DataFrame({k: S[f"{c}|{k}"] for k in DRUGS})
        Sb = pd.DataFrame({k: S[f"{b}|{k}"] for k in DRUGS})
        for name, ds in [(k, [k]) for k in DRUGS] + [("mean_4", list(DRUGS))]:
            pt, lo, hi, ng = cluster_bootstrap(d["Y"], Sa, Sb, codes, ds, n_boot=a.n_boot)
            deltas.append(dict(comparison=f"{b} - {c}", endpoint=name, delta=pt, ci_low=lo, ci_high=hi,
                               n_clusters=ng))
    dl = pd.DataFrame(deltas)
    dl.to_csv(out / "deltas.csv", index=False)

    conc = []
    for f_ in ASM:
        for k in DRUGS:
            r = within_st_concordance(d["Y"][k], d["M"][f_], d["cluster"])
            conc.append(dict(feature=f_, drug=k, concordance=r["macro"], n_clusters=r["n_st"], n_pairs=r["n_pairs"]))
    conc = pd.DataFrame(conc)
    conc.to_csv(out / "assembly_within_cluster.csv", index=False)

    p = dl[(dl.comparison == "model - rule") & (dl.endpoint == "mean_4")].iloc[0]
    by = m.set_index(["arm", "drug"]).nap
    three = {arm: float(np.mean([(by[(arm, "CIP")] + by[(arm, "NAL")]) / 2, by[(arm, "TET")], by[(arm, "CLI")]]))
             for arm in ("model", "rule")}
    length = float(conc[conc.feature == "length"].concordance.mean())
    v = {"clindamycin_23s_in_rule": a.clindamycin_23s, "isolates": int(len(d["Y"])),
         "model_minus_rule": p.delta, "ci": [p.ci_low, p.ci_high],
         "pass_rule": f"delta >= {PASS_DELTA} and ci_low > 0",
         "primary_pass": bool(p.delta >= PASS_DELTA and p.ci_low > 0),
         "model_minus_rule_three_classes": round(three["model"] - three["rule"], 4),
         "length_within_cluster_concordance": round(length, 4),
         "w12_pass_rule": f"length mean >= {ASM_PASS}", "w12_pass": bool(length >= ASM_PASS),
         "mean_nap": m.groupby("arm").nap.mean().round(4).to_dict()}
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(m.round(3).to_string(index=False))
    print(dl.round(3).to_string(index=False))
    print(conc.round(3).to_string(index=False))
    print(json.dumps(v, indent=1, default=float))


if __name__ == "__main__":
    main()

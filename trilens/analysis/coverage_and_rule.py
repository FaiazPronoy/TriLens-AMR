from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.evaluation.cluster_bootstrap import cluster_bootstrap, group_codes, nap, summary
from trilens.experiments.determinant_rescue import (
    PRIMARY,
    load_arg,
    load_labels,
    load_meta,
    load_pa,
    oof,
    repo_results,
)
from trilens.experiments.external_cohort import RULES, rule_score
from trilens.splits.fold_splits import st_grouped

ROOT = Path(__file__).resolve().parents[2]
A_DRUGS = ["CTX", "CAZ", "GEN", "CIP"]
A_TOL = 0.10
B_DELTA = 0.05


def strain_table():
    return pd.read_csv(repo_results() / "data_inventory" / "strain_table.csv", dtype={"st": str, "study": str})


def strata(tab, covered, Y):
    rows = []
    for name, m in [("covered", covered), ("uncovered", ~covered)]:
        r = dict(
            stratum=name,
            strains=int(m.sum()),
            n_st=int(tab.st[m].nunique()),
            n_study=int(tab.study[m].nunique()),
            top_study=tab.study[m].value_counts().idxmax() if m.any() else None,
            top_country=tab.country[m].value_counts().idxmax() if m.any() else None,
            pct_uk=round(100 * float(tab.country[m].isin(["United Kingdom", "England"]).mean()), 1),
            pct_norway=round(100 * float(tab.country[m].eq("Norway").mean()), 1),
        )
        for d in Y.columns:
            y = Y[d][m]
            r[f"n_{d}"] = int(y.notna().sum())
            r[f"prev_{d}"] = round(float(y.mean()), 3) if y.notna().any() else None
        rows.append(r)
    return pd.DataFrame(rows)


def subset_folds(st_sub):
    folds, never = st_grouped(st_sub)
    n = len(st_sub)
    return [(np.setdiff1d(np.arange(n), f), f) for f in folds], never


def part_a(pa, meta, Y, st, covered, uncovered, n_boot):
    rows, scores = [], {}
    for name, m in [("covered", covered), ("uncovered", uncovered)]:
        idx = np.flatnonzero(m)
        folds, _ = subset_folds(st[idx])
        local = [(idx[tr], idx[te]) for tr, te in folds]
        data = dict(Y=Y, pa=pa, meta=meta, arg=None, keep=m, st=st)
        S = pd.DataFrame(index=np.arange(len(Y)))
        for arm in ("F0", "M"):
            for d in A_DRUGS:
                S[f"{arm}|{d}"] = oof(arm, d, local, data, m)
                rows.append(dict(stratum=name, arm=arm, drug=d, **summary(Y[d], S[f"{arm}|{d}"])))
                print(f"A {name:9s} {arm:3s} {d:4s} nAP={nap(Y[d], S[f'{arm}|{d}']):.3f}", flush=True)
        scores[name] = S
    df = pd.DataFrame(rows)
    mean = {
        (s, a): float(df[(df.stratum == s) & (df.arm == a) & df.drug.isin(PRIMARY)].nap.mean())
        for s in ("covered", "uncovered")
        for a in ("F0", "M")
    }
    return df, scores, mean


def rule_arm(arg_dir, index, n_strains):
    A = pd.read_csv(Path(arg_dir) / "acquired.csv", index_col=0)
    P = pd.read_csv(Path(arg_dir) / "point.csv", index_col=0).reindex(A.index, fill_value=0)
    acq = [set(A.columns[r.astype(bool)]) for _, r in A.iterrows()]
    pt = [set(P.columns[r.astype(bool)]) for _, r in P.iterrows()]
    R = pd.DataFrame(np.nan, index=np.arange(n_strains), columns=list(RULES))
    for d in RULES:
        R.loc[A.index, d] = [rule_score(a, p, d) for a, p in zip(acq, pt)]
    return R.loc[:, [c for c in R.columns if c in RULES]] if len(index) else R


def part_b(arg_dir, Y, st, covered, rescue_dir, n_boot):
    R = rule_arm(arg_dir, np.flatnonzero(covered), len(Y))
    p = Path(rescue_dir) / "scores_st_grouped_5fold.csv.gz"
    if not p.exists():
        raise SystemExit(f"{p} missing: attach the determinant_rescue output (its saved out-of-fold scores)")
    S = pd.read_csv(p, index_col=0)
    drugs = [d for d in PRIMARY + ["CIP"] if f"F0A|{d}" in S]
    rows = []
    for d in drugs:
        for arm, col in [("F0", S.get(f"F0|{d}")), ("F0A", S.get(f"F0A|{d}")), ("R", R[d])]:
            if col is None:
                continue
            s = pd.Series(col).where(pd.Series(S[f"F0A|{d}"]).notna())
            rows.append(dict(arm=arm, drug=d, **summary(Y[d], s)))
    df = pd.DataFrame(rows)
    groups = group_codes(st)
    a = pd.DataFrame({d: R[d].where(pd.Series(S[f"F0A|{d}"]).notna()) for d in drugs})
    b = pd.DataFrame({d: S[f"F0A|{d}"] for d in drugs})
    deltas = []
    for name, ds in [(d, [d]) for d in drugs] + [("primary_mean", [d for d in PRIMARY if d in drugs])]:
        pt_, lo, hi, ng = cluster_bootstrap(Y, a, b, groups, ds, n_boot=n_boot)
        deltas.append(dict(comparison="F0A - R", endpoint=name, delta=pt_, ci_low=lo, ci_high=hi, n_groups=ng))
    return df, pd.DataFrame(deltas)


def part_c(Y, st, tab, rescue_dir, n_boot):
    S = pd.read_csv(Path(rescue_dir) / "scores_st_grouped_5fold.csv.gz", index_col=0)
    drugs = [d for d in PRIMARY if f"F0A|{d}" in S and f"F0|{d}" in S]
    uk = tab.country.isin(["United Kingdom", "England"]).to_numpy()
    groups = group_codes(st)
    rows = []
    for name, m in [("united_kingdom", uk), ("norway", tab.country.eq("Norway").to_numpy())]:
        Ym = Y.where(pd.Series(m, index=Y.index), other=np.nan)
        a = pd.DataFrame({d: S[f"F0|{d}"] for d in drugs})
        b = pd.DataFrame({d: S[f"F0A|{d}"] for d in drugs})
        pt_, lo, hi, ng = cluster_bootstrap(Ym, a, b, groups, drugs, n_boot=n_boot)
        n = int(Ym[drugs].notna().any(axis=1).sum())
        rows.append(
            dict(stratum=name, endpoint="primary_mean", delta=pt_, ci_low=lo, ci_high=hi, n_groups=ng, strains=n)
        )
        for d in drugs:
            rows.append(
                dict(
                    stratum=name,
                    endpoint=d,
                    delta=nap(Ym[d], b[d]) - nap(Ym[d], a[d]),
                    ci_low=None,
                    ci_high=None,
                    n_groups=None,
                    strains=int(Ym[d].notna().sum()),
                )
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arg-dir", default="results/determinant_matrix")
    ap.add_argument("--rescue-dir", default="results/determinant_rescue/paper_ri")
    ap.add_argument("--out", default="results/coverage_and_rule")
    ap.add_argument("--labels", default="ri", choices=["ri", "no_intermediate"])
    ap.add_argument("--parts", default="abc", help="which parts to run: any of a, b, c")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = set(a.parts.lower())

    Y = load_labels("paper", a.labels)
    arg = load_arg(Path(a.arg_dir) / "acquired.csv")
    has = np.zeros(len(Y), bool)
    has[arg.index.to_numpy()] = True
    tab = strain_table()
    st = tab.st.to_numpy(object)
    v = {"covered_strains": int(has.sum()), "uncovered_strains": int((~has).sum()), "parts_run": sorted(parts)}

    if "a" in parts:
        s = strata(tab, has, Y)
        s.to_csv(out / "strata.csv", index=False)
        print(s.to_string(index=False), flush=True)
        pa, matched, _ = load_pa()

        A, _, amean = part_a(pa, load_meta(), Y, st, matched & has, matched & ~has, a.n_boot)
        A.to_csv(out / "part_a.csv", index=False)
        gap = amean[("uncovered", "F0")] - amean[("covered", "F0")]
        v.update(
            {
                "a_f0_covered": round(amean[("covered", "F0")], 4),
                "a_f0_uncovered": round(amean[("uncovered", "F0")], 4),
                "a_difference": round(float(gap), 4),
                "a_pass_rule": f"abs(uncovered - covered) <= {A_TOL}",
                "a_pass": bool(abs(gap) <= A_TOL),
            }
        )
        v["a_reading"] = (
            "the catalogue-only collapse is present in both strata, so the rescue is not a coverage artefact"
            if v["a_pass"]
            else "the uncovered strains behave differently from the catalogue alone: the "
            "rescue is demonstrated on the annotated subset and may not extend beyond it"
        )

    if "b" in parts:
        B, bdel = part_b(a.arg_dir, Y, st, has, a.rescue_dir, a.n_boot)
        B.to_csv(out / "part_b.csv", index=False)
        bdel.to_csv(out / "deltas.csv", index=False)
        bp = bdel[bdel.endpoint == "primary_mean"].iloc[0]
        v.update(
            {
                "b_f0a_minus_rule": bp.delta,
                "b_ci": [bp.ci_low, bp.ci_high],
                "b_pass_rule": f"delta >= {B_DELTA} and ci_low > 0",
                "b_model_adds_over_rule": bool(bp.delta >= B_DELTA and bp.ci_low > 0),
            }
        )
        v["b_reading"] = (
            "the model adds measurably over the determinant rule on unseen lineages"
            if v["b_model_adds_over_rule"]
            else "the determinant rule recovers internally what the model recovers: the "
            "rescue is a rule effect, not a modelling effect"
        )
        print(B.round(3).to_string(index=False))
        print(bdel.round(3).to_string(index=False))

    if "c" in parts:
        C = part_c(Y, st, tab, a.rescue_dir, a.n_boot)
        C.to_csv(out / "part_c.csv", index=False)
        for name in ("united_kingdom", "norway"):
            r = C[(C.stratum == name) & (C.endpoint == "primary_mean")]
            v[f"c_{name}_delta"] = float(r.delta.iloc[0]) if len(r) else None
            v[f"c_{name}_ci"] = [float(r.ci_low.iloc[0]), float(r.ci_high.iloc[0])] if len(r) else None
        print(C.round(3).to_string(index=False))

    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(json.dumps(v, indent=1, default=float))


if __name__ == "__main__":
    main()

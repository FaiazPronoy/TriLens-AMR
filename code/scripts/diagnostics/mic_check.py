"""
mic_check.py
============
W13 MIC analysis on the W8 external cohort. Pre-registration: the MIC amendment in
docs/experiments/w13_second_species.md. Nothing is refitted: the frozen W8 scores are re-read.

  M1  W8's AP - M and AP - R with EUCAST-harmonised labels in place of the submitter's.
  M2  error rate near the breakpoint (within one doubling dilution) against far from it, per arm
      (model at 0.5, rule), pooled over drugs, BioProject-clustered bootstrap CI of the ratio.
  M3  agreement of the submitter's call with the EUCAST call, per drug.

    python scripts/diagnostics/mic_check.py
        [--mic results/data_inventory/ncbi_ast_ecoli.tsv.gz]
        [--breakpoints results/data_inventory/eucast_v16_1_enterobacterales.csv]
        [--w8 results/w8_external/primary_ri] [--candidates results/ext_probe/candidates.tsv.gz]
        [--out results/w13_mic]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.grouped import cluster_bootstrap  # noqa: E402
from src.experiments.run_arg_rescue import repo_results  # noqa: E402
from src.experiments.run_external import PD_NAME, load_external, load_training  # noqa: E402

DRUGS = ["AMP", "CIP", "AMC", "SXT", "GEN", "CAZ"]        # CTX: 950 isolates with an MIC, under 1,000
MIN_ISOLATES = 1000
SMX_VALUES = {9.5, 19.0, 38.0, 76.0, 152.0, 304.0}        # 19 x a trimethoprim dilution (Vitek)
SXT_EXCLUDED = {26.95}
RATIO, RATIO_CI = 2.0, 1.0


def interval(sign, v):
    """MIC as [lo, hi] (mg/L) from the measurement sign, as fixed in the amendment."""
    if sign == "==":
        return v, v
    if sign == "<=":
        return 0.0, v
    if sign == "<":
        return 0.0, v / 2
    if sign == ">=":
        return v, np.inf
    if sign == ">":
        return 2 * v, np.inf
    return np.nan, np.nan


def sxt_component(v):
    """Trimethoprim component; NaN for a value that fits neither convention."""
    if v in SMX_VALUES:
        return v / 19
    if v in SXT_EXCLUDED:                # fits neither convention (named in the amendment)
        return np.nan
    return v if v <= 32 else np.nan


def harmonise(lo, hi, s, r):
    """1 resistant (or I), 0 susceptible, NaN undetermined (the interval straddles a breakpoint)."""
    if np.isnan(lo):
        return np.nan
    if lo > r:
        return 1.0
    if hi <= s:
        return 0.0
    if lo > s and hi <= r:
        return 1.0                       # I, counted as resistant throughout the paper
    return np.nan


def proximity(lo, hi, s, r):
    """'near' within one doubling dilution [s/2, 2r], 'far' wholly outside it, else None."""
    if np.isnan(lo):
        return None
    a, b = s / 2, 2 * r
    if lo >= a and hi <= b:
        return "near"
    if hi < a or lo > b:
        return "far"
    return None


def load_mic(path, bps):
    d = pd.read_csv(path, sep="\t", dtype=str).rename(columns={"#BioSample": "BioSample"})
    name_to_drug = {v: k for k, v in PD_NAME.items() if k in bps.index}
    d = d[d.Antibiotic.isin(name_to_drug) & d["MIC (mg/L)"].notna()].copy()
    d["drug"] = d.Antibiotic.map(name_to_drug)
    v = d["MIC (mg/L)"].astype(float)
    v = np.where(d.drug.eq("SXT"), [sxt_component(x) for x in v], v)
    d["value"] = v
    iv = [interval(sg, x) if not np.isnan(x) else (np.nan, np.nan)
          for sg, x in zip(d["Measurement sign"], d["value"])]
    d["lo"], d["hi"] = [i[0] for i in iv], [i[1] for i in iv]
    s = d.drug.map(bps.S_le).astype(float)
    r = d.drug.map(bps.R_gt).astype(float)
    d["eucast"] = [harmonise(lo, hi, a, b) for lo, hi, a, b in zip(d.lo, d.hi, s, r)]
    d["proximity"] = [proximity(lo, hi, a, b) for lo, hi, a, b in zip(d.lo, d.hi, s, r)]
    # one row per (isolate, drug): an isolate tested twice keeps its first row
    return d.drop_duplicates(["Isolate", "drug"])


def m2(Y, S, R, prox, groups, drugs, n_boot, seed=0):
    """Near/far error-rate ratio per arm, pooled over drugs, bootstrap over BioProjects."""
    rows = []
    for d in drugs:
        y = Y[d].to_numpy()
        for arm, pred in (("model", (S[f"AP|{d}"].to_numpy() >= 0.5).astype(float)),
                          ("rule", R[d].to_numpy())):
            ok = ~np.isnan(y) & pd.Series(prox[d]).notna().to_numpy()
            rows.append(pd.DataFrame({"drug": d, "arm": arm, "near": prox[d][ok] == "near",
                                      "error": (pred[ok] != y[ok]).astype(float), "g": groups[ok]}))
    t = pd.concat(rows, ignore_index=True)
    out, per = {}, []
    rng = np.random.default_rng(seed)
    for arm, a in t.groupby("arm"):
        def ratio(x):
            n, f = x[x.near], x[~x.near]
            return (n.error.mean() / f.error.mean()) if len(n) and len(f) and f.error.mean() > 0 else np.nan
        pt = ratio(a)
        gs = a.g.unique()
        idx = {g: np.flatnonzero(a.g.to_numpy() == g) for g in gs}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(gs, len(gs), replace=True)
            boots.append(ratio(a.iloc[np.concatenate([idx[g] for g in pick])]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out[arm] = dict(ratio=pt, ci=[lo, hi], near_error=a[a.near].error.mean(), far_error=a[~a.near].error.mean(),
                        n_near=int(a.near.sum()), n_far=int((~a.near).sum()),
                        concentrated=bool(pt >= RATIO and lo > RATIO_CI))
        for d, x in a.groupby("drug"):
            per.append(dict(arm=arm, drug=d, ratio=ratio(x), near_error=x[x.near].error.mean(),
                            far_error=x[~x.near].error.mean(), n_near=int(x.near.sum()), n_far=int((~x.near).sum())))
    return out, pd.DataFrame(per)


def main():
    ap = argparse.ArgumentParser()
    inv = repo_results() / "data_inventory"
    ap.add_argument("--mic", default=str(inv / "ncbi_ast_ecoli.tsv.gz"))
    ap.add_argument("--breakpoints", default=str(inv / "eucast_v16_1_enterobacterales.csv"))
    ap.add_argument("--w8", default="results/w8_external/primary_ri")
    ap.add_argument("--candidates", default="results/ext_probe/candidates.tsv.gz")
    ap.add_argument("--arg-dir", default="results/arg_matrix")
    ap.add_argument("--out", default="results/w13_mic")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    bps = pd.read_csv(a.breakpoints).set_index("drug")
    missing = [d for d in DRUGS if d not in bps.index or bps.loc[d, ["S_le", "R_gt"]].isna().any()]
    if missing:
        raise SystemExit(f"no EUCAST breakpoint for {missing}: transcribe them from the official table first")
    mic = load_mic(a.mic, bps)

    train = load_training(a.arg_dir, "ri")
    ext = load_external(a.candidates, "primary", "ri", list(train["A"].columns), list(train["P"].columns))
    S = pd.read_csv(Path(a.w8) / "scores.csv.gz", index_col=0)
    if len(S) != ext["n"]:
        raise SystemExit(f"scores have {len(S)} rows, the W8 primary cohort {ext['n']}")
    acc = ext["frame"].target_acc.to_numpy()
    groups = pd.Series(ext["groups"], dtype=object).fillna("__none__").to_numpy(object)

    Ye, prox, m3 = pd.DataFrame(index=range(ext["n"])), {}, []
    drugs = []
    for d in DRUGS:
        x = mic[mic.drug == d].set_index("Isolate")
        if x.index.isin(acc).sum() < MIN_ISOLATES:
            print(f"{d}: fewer than {MIN_ISOLATES} W8 isolates with an MIC, not analysed")
            continue
        drugs.append(d)
        Ye[d] = pd.Series(acc).map(x.eucast).to_numpy(float)
        prox[d] = pd.Series(acc).map(x.proximity).to_numpy(object)
        sub, eu = ext["Y"][d].to_numpy(), Ye[d].to_numpy()
        both = ~np.isnan(sub) & ~np.isnan(eu)
        m3.append(dict(drug=d, with_mic=int(x.index.isin(acc).sum()),
                       undetermined=int(pd.Series(acc).map(x.eucast).isna().sum() - (~pd.Series(acc).isin(x.index)).sum()),
                       compared=int(both.sum()), agreement=float((sub[both] == eu[both]).mean()) if both.any() else np.nan,
                       submitter_R_eucast_S=int(((sub == 1) & (eu == 0)).sum()),
                       submitter_S_eucast_R=int(((sub == 0) & (eu == 1)).sum())))
    pd.DataFrame(m3).to_csv(out / "m3_label_agreement.csv", index=False)

    R = ext["R"]
    m1 = []
    codes = pd.factorize(groups)[0]
    for labels, Y in (("submitter", ext["Y"][drugs]), ("eucast", Ye[drugs])):
        for b, c in (("AP", "M"), ("AP", "R")):
            Sa = pd.DataFrame({d: (S[f"{c}|{d}"] if c != "R" else R[d]) for d in drugs})
            Sb = pd.DataFrame({d: S[f"{b}|{d}"] for d in drugs})
            pt, lo, hi, ng = cluster_bootstrap(Y, Sa, Sb, codes, drugs, n_boot=a.n_boot)
            m1.append(dict(labels=labels, comparison=f"{b} - {c}", delta=pt, ci_low=lo, ci_high=hi, n_groups=ng))
    m1 = pd.DataFrame(m1)
    m1.to_csv(out / "m1_endpoints.csv", index=False)

    m2_out, m2_per = m2(Ye[drugs], S, R, prox, groups, drugs, a.n_boot)
    m2_per.to_csv(out / "m2_per_drug.csv", index=False)
    v = {"breakpoints": Path(a.breakpoints).name, "drugs": drugs, "m1": m1.round(4).to_dict(orient="records"),
         "m2": m2_out, "m2_rule": f"ratio >= {RATIO} and ci_low > {RATIO_CI}"}
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(pd.DataFrame(m3).round(3).to_string(index=False))
    print(m1.round(3).to_string(index=False))
    print(m2_per.round(3).to_string(index=False))
    print(json.dumps(v, indent=1, default=float))


if __name__ == "__main__":
    main()

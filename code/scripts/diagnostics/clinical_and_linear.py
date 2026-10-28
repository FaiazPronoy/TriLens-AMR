"""
clinical_and_linear.py
======================
W14. Pre-registration: docs/experiments/w14_clinical_and_linear.md.

  B  L1 logistic regression (C=1.0, liblinear, no tuning)
       internal: W6 ST-grouped 5-fold, covered strains, acquired-gene block, CTX/CAZ/GEN,
                 against W6's saved XGBoost arm A and the rule
       external: trained once on all called strains (acquired + point), frozen, applied to the
                 W8 primary cohort, against W8's saved XGBoost AP and the rule
  A  categorical agreement, major and very major error on the external cohort with EUCAST labels,
     for XGBoost AP, the rule and the linear model, at score >= 0.5

    python scripts/diagnostics/clinical_and_linear.py
        [--w6 results/w6_rescue/paper_ri] [--w8 results/w8_external/primary_ri]
        [--candidates results/ext_probe/candidates.tsv.gz] [--out results/w14_clinical_linear]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.grouped import cluster_bootstrap, group_codes, summary  # noqa: E402
from src.experiments.run_arg_rescue import (  # noqa: E402
    PRIMARY, load_arg, load_labels, load_pa, load_splits, repo_results,
)
from src.experiments.run_external import PD_NAME, load_external, load_training  # noqa: E402
from src.experiments.run_external import PRIMARY as EXT_PRIMARY  # noqa: E402

LR = dict(penalty="l1", C=1.0, solver="liblinear", max_iter=2000)
MARGIN = 0.05
CA_MIN, ME_MAX, VME_UPPER_MAX = 0.899, 0.03, 0.075        # FDA 2009, section X.A
EXT_DRUGS = ["AMP", "CIP", "AMC", "SXT", "GEN", "CAZ"]     # CTX: under 1,000 MICs (W13)


def _mic_module():
    p = ROOT / "scripts" / "diagnostics" / "mic_check.py"
    spec = importlib.util.spec_from_file_location("mic_check", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fit_lr(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**LR).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def reading(delta, lo, hi):
    """The pre-registered reading of XGBoost - linear."""
    if lo > -MARGIN and hi < MARGIN:
        return "a linear model suffices"
    if delta >= MARGIN and lo > 0:
        return "non-linearity matters"
    if -delta >= MARGIN and hi < 0:
        return "the linear model is better"
    return "inconclusive"


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def errors(y, call):
    """CA, ME (over reference S), VME (over reference R), with the counts."""
    ok = ~np.isnan(y) & ~np.isnan(call)
    y, c = y[ok].astype(int), call[ok].astype(int)
    nS, nR = int((y == 0).sum()), int((y == 1).sum())
    me, vme = int(((y == 0) & (c == 1)).sum()), int(((y == 1) & (c == 0)).sum())
    lo, hi = clopper_pearson(vme, nR)
    ca = float((y == c).mean()) if len(y) else np.nan
    return dict(n=int(len(y)), n_S=nS, n_R=nR, CA=ca, ME=me / nS if nS else np.nan,
                VME=vme / nR if nR else np.nan, VME_ci_low=lo, VME_ci_high=hi,
                meets_CA=bool(ca > CA_MIN), meets_ME=bool(nS and me / nS < ME_MAX),
                meets_VME=bool(hi < VME_UPPER_MAX))


def pooled_error_diff(Y, calls_a, calls_b, groups, drugs, kind, n_boot, seed=0):
    """Pooled (over drugs) error rate of arm a minus arm b, bootstrap over BioProjects."""
    rows = []
    for d in drugs:
        y = Y[d].to_numpy()
        ref = 1 if kind == "VME" else 0
        m = (y == ref)
        rows.append(pd.DataFrame({"g": groups[m], "a": calls_a[d].to_numpy()[m] != ref,
                                  "b": calls_b[d].to_numpy()[m] != ref}))
    t = pd.concat(rows, ignore_index=True)
    pt = float(t.a.mean() - t.b.mean())
    gs = t.g.unique()
    idx = {g: np.flatnonzero(t.g.to_numpy() == g) for g in gs}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(gs, len(gs), replace=True)])
        x = t.iloc[pick]
        boots.append(x.a.mean() - x.b.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pt, float(lo), float(hi), int(len(t))


def part_b_internal(arg_dir, w6, n_boot):
    Y = load_labels("paper", "ri")
    _, matched, _ = load_pa()
    A = load_arg(Path(arg_dir) / "acquired.csv")
    has = np.zeros(len(Y), bool)
    has[A.index.to_numpy()] = True
    keep = matched & has
    X = A.reindex(np.arange(len(Y)), fill_value=0).to_numpy(np.uint8)
    folds = load_splits(["st_grouped_5fold"])["st_grouped_5fold"]
    S6 = pd.read_csv(Path(w6) / "scores_st_grouped_5fold.csv.gz", index_col=0)
    rmod = _rule_internal(arg_dir, len(Y))
    st = pd.read_csv(repo_results() / "data_inventory" / "strain_table.csv", dtype={"st": str}).st.to_numpy(object)
    lin = pd.DataFrame(index=range(len(Y)))
    for d in PRIMARY + ["CIP"]:
        y = Y[d].to_numpy()
        s = np.full(len(y), np.nan)
        for tr, te in folds:
            tr = tr[keep[tr] & ~np.isnan(y[tr])]
            te = te[keep[te] & ~np.isnan(y[te])]
            if len(te) and len(np.unique(y[tr])) == 2:
                s[te] = fit_lr(X[tr], y[tr].astype(int), X[te])
        lin[d] = s
    drugs = PRIMARY + ["CIP"]
    xgb = pd.DataFrame({d: S6[f"A|{d}"] for d in drugs})
    rule = pd.DataFrame({d: rmod[d].where(xgb[d].notna()) for d in drugs})
    lin = lin[drugs].where(xgb.notna())
    rows = [dict(arm=arm, drug=d, **summary(Y[d], S[d]))
            for arm, S in (("xgboost_A", xgb), ("linear_A", lin), ("rule", rule)) for d in drugs]
    g = group_codes(st)
    dl = []
    for name, (a, b) in {"xgboost - linear": (lin, xgb), "linear - rule": (rule, lin)}.items():
        for ep, ds in [(d, [d]) for d in drugs] + [("primary_mean", PRIMARY)]:
            pt, lo, hi, ng = cluster_bootstrap(Y, a, b, g, ds, n_boot=n_boot)
            dl.append(dict(setting="internal", comparison=name, endpoint=ep, delta=pt, ci_low=lo, ci_high=hi,
                           n_groups=ng))
    return pd.DataFrame(rows), pd.DataFrame(dl)


def _rule_internal(arg_dir, n):
    spec = importlib.util.spec_from_file_location("cr", ROOT / "scripts" / "diagnostics" / "coverage_and_rule.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    A = pd.read_csv(Path(arg_dir) / "acquired.csv", index_col=0)
    return cr.rule_arm(arg_dir, A.index.to_numpy(), n)


def part_b_external(arg_dir, w8, candidates, n_boot):
    train = load_training(arg_dir, "ri")
    ext = load_external(candidates, "primary", "ri", list(train["A"].columns), list(train["P"].columns))
    S8 = pd.read_csv(Path(w8) / "scores.csv.gz", index_col=0)
    if len(S8) != ext["n"]:
        raise SystemExit(f"W8 scores have {len(S8)} rows, the cohort {ext['n']}")
    Xtr = np.hstack([train["A"].to_numpy(np.uint8), train["P"].to_numpy(np.uint8)])
    Xte = np.hstack([ext["A"].to_numpy(np.uint8), ext["P"].to_numpy(np.uint8)])
    lin = pd.DataFrame(index=range(ext["n"]))
    for d in PD_NAME:
        y = train["Y"][d].to_numpy()
        ok = ~np.isnan(y)
        lin[d] = fit_lr(Xtr[ok], y[ok].astype(int), Xte)
    xgb = pd.DataFrame({d: S8[f"AP|{d}"] for d in PD_NAME})
    rule = ext["R"][list(PD_NAME)]
    groups = pd.Series(ext["groups"], dtype=object).fillna("__none__").to_numpy(object)
    codes = pd.factorize(groups)[0]
    rows = [dict(arm=arm, drug=d, **summary(ext["Y"][d], S[d]))
            for arm, S in (("xgboost_AP", xgb), ("linear_AP", lin), ("rule", rule)) for d in PD_NAME]
    dl = []
    for name, (a, b) in {"xgboost - linear": (lin, xgb), "linear - rule": (rule, lin)}.items():
        for ep, ds in [(d, [d]) for d in PD_NAME] + [("primary_mean", EXT_PRIMARY)]:
            pt, lo, hi, ng = cluster_bootstrap(ext["Y"], a, b, codes, ds, n_boot=n_boot)
            dl.append(dict(setting="external", comparison=name, endpoint=ep, delta=pt, ci_low=lo, ci_high=hi,
                           n_groups=ng))
    return pd.DataFrame(rows), pd.DataFrame(dl), ext, xgb, lin, rule, groups


def part_a(ext, xgb, lin, rule, groups, mic_path, bp_path, n_boot):
    mc = _mic_module()
    mic = mc.load_mic(mic_path, pd.read_csv(bp_path).set_index("drug"))
    acc = ext["frame"].target_acc.to_numpy()
    Ye = pd.DataFrame({d: pd.Series(acc).map(mic[mic.drug == d].set_index("Isolate").eucast).to_numpy(float)
                       for d in EXT_DRUGS})
    calls = {"xgboost_AP": (xgb[EXT_DRUGS] >= 0.5).astype(float),
             "linear_AP": (lin[EXT_DRUGS] >= 0.5).astype(float),
             "rule": rule[EXT_DRUGS].astype(float)}
    rows = []
    for arm, C in calls.items():
        for d in EXT_DRUGS:
            rows.append(dict(arm=arm, drug=d, **errors(Ye[d].to_numpy(), C[d].to_numpy())))
        allY = np.concatenate([Ye[d].to_numpy() for d in EXT_DRUGS])
        allC = np.concatenate([C[d].to_numpy() for d in EXT_DRUGS])
        rows.append(dict(arm=arm, drug="POOLED", **errors(allY, allC)))
    diffs = []
    for kind in ("VME", "ME"):
        for a, b in (("xgboost_AP", "rule"), ("linear_AP", "rule"), ("xgboost_AP", "linear_AP")):
            pt, lo, hi, n = pooled_error_diff(Ye, calls[a], calls[b], groups, EXT_DRUGS, kind, n_boot)
            diffs.append(dict(error=kind, comparison=f"{a} - {b}", delta=pt, ci_low=lo, ci_high=hi, isolates=n))
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def main():
    ap = argparse.ArgumentParser()
    inv = repo_results() / "data_inventory"
    ap.add_argument("--arg-dir", default="results/arg_matrix")
    ap.add_argument("--w6", default="results/w6_rescue/paper_ri")
    ap.add_argument("--w8", default="results/w8_external/primary_ri")
    ap.add_argument("--candidates", default="results/ext_probe/candidates.tsv.gz")
    ap.add_argument("--mic", default=str(inv / "ncbi_ast_ecoli.tsv.gz"))
    ap.add_argument("--breakpoints", default=str(inv / "eucast_v16_1_enterobacterales.csv"))
    ap.add_argument("--out", default="results/w14_clinical_linear")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mi, di = part_b_internal(a.arg_dir, a.w6, a.n_boot)
    me, de, ext, xgb, lin, rule, groups = part_b_external(a.arg_dir, a.w8, a.candidates, a.n_boot)
    pd.concat([mi.assign(setting="internal"), me.assign(setting="external")]).to_csv(out / "b_metrics.csv", index=False)
    dl = pd.concat([di, de], ignore_index=True)
    dl.to_csv(out / "b_deltas.csv", index=False)
    ca, cd = part_a(ext, xgb, lin, rule, groups, a.mic, a.breakpoints, a.n_boot)
    ca.to_csv(out / "a_clinical.csv", index=False)
    cd.to_csv(out / "a_error_differences.csv", index=False)

    def pick(setting):
        r = dl[(dl.setting == setting) & (dl.comparison == "xgboost - linear") & (dl.endpoint == "primary_mean")].iloc[0]
        return dict(delta=r.delta, ci=[r.ci_low, r.ci_high], reading=reading(r.delta, r.ci_low, r.ci_high))
    v = {"internal_xgboost_minus_linear": pick("internal"),
         "external_xgboost_minus_linear": pick("external"),
         "linear_minus_rule": {s: dl[(dl.setting == s) & (dl.comparison == "linear - rule")
                                     & (dl.endpoint == "primary_mean")].iloc[0][["delta", "ci_low", "ci_high"]].round(4).to_dict()
                               for s in ("internal", "external")},
         "reading_rule": f"suffices if the CI lies in (-{MARGIN}, {MARGIN}); non-linearity matters if delta >= {MARGIN} and CI > 0",
         "pooled_errors": ca[ca.drug == "POOLED"].set_index("arm")[["CA", "ME", "VME", "VME_ci_high"]].round(4).to_dict(orient="index"),
         "error_differences": cd.round(4).to_dict(orient="records"),
         "fda_benchmarks": {"CA_gt": CA_MIN, "ME_lt": ME_MAX, "VME_upper95_lt": VME_UPPER_MAX}}
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(mi.round(3).to_string(index=False))
    print(me.round(3).to_string(index=False))
    print(dl.round(3).to_string(index=False))
    print(ca.round(3).to_string(index=False))
    print(cd.round(4).to_string(index=False))
    print(json.dumps(v, indent=1, default=float))


if __name__ == "__main__":
    main()

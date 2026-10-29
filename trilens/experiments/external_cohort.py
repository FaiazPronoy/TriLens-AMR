from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.determinants.determinant_matrix import NOT_COUNTED, PRIMARY_DB, SOFTWARE, tokens
from trilens.evaluation.cluster_bootstrap import cluster_bootstrap, summary
from trilens.experiments.determinant_rescue import XGB, load_labels, repo_results

PD_NAME = {
    "AMP": "ampicillin",
    "CIP": "ciprofloxacin",
    "AMC": "amoxicillin-clavulanic acid",
    "SXT": "trimethoprim-sulfamethoxazole",
    "GEN": "gentamicin",
    "CTX": "cefotaxime",
    "CAZ": "ceftazidime",
}
PRIMARY = ["AMP", "CIP", "AMC", "SXT", "GEN", "CAZ"]
PASS_DELTA = 0.10
ASM_OURS = ["Size", "Contigs", "Contig.N50"]
ASM_PD = ["asm_stats_length_bp", "asm_stats_n_contig", "asm_stats_contig_n50"]

_AMPC = r"blaCMY|blaDHA|blaFOX|blaACC"
_CARB = r"blaKPC|blaNDM|blaVIM|blaIMP|blaOXA-48\b|blaOXA-181\b|blaOXA-232\b|blaOXA-244\b"
_ESBL = rf"blaCTX-M|blaSHV-(?!1\b|11\b)\d|blaTEM-(?!1\b)\d|{_AMPC}|{_CARB}"
RULES = {
    "AMP": r"^bla(?!EC)",
    "AMC": rf"^blaOXA-1\b|{_AMPC}|{_CARB}",
    "CTX": _ESBL,
    "CAZ": _ESBL,
    "GEN": r"aac\(3\)|ant\(2''\)|^armA|^rmt",
    "SXT": r"^dfrA|^dfrB",
    "CIP": r"^gyrA_.*\|POINT$|^qnr|aac\(6'\)-Ib-cr",
}


def calls(genotypes):
    acq, pt = set(), set()
    for t in tokens(genotypes):
        sym, _, suf = t.partition("=")
        if suf == "POINT":
            pt.add(sym)
        elif suf == "":
            acq.add(sym)
        elif suf not in NOT_COUNTED:
            raise SystemExit(f"unknown AMR_genotypes suffix {suf!r} in {t!r}: decide a rule first")
    return acq, pt


def ast(s, drug, mode):
    if not isinstance(s, str):
        return np.nan
    for tok in s.split(","):
        if "=" in tok:
            d, c = tok.rsplit("=", 1)
            if d.strip() == PD_NAME[drug]:
                c = c.strip()
                return {"R": 1.0, "S": 0.0, "I": 1.0 if mode == "ri" else np.nan}.get(c, np.nan)
    return np.nan


def rule_score(acq, pt, drug):
    syms = list(acq) + [p + "|POINT" for p in pt]
    return float(any(re.search(RULES[drug], s) for s in syms))


def load_external(path, db, mode, a_cols, p_cols):
    ext = pd.read_csv(path, sep="\t", dtype=str)
    ext = ext[ext.amrfinder_version.eq(SOFTWARE)]
    if db == "primary":
        ext = ext[ext.refgene_db_version.eq(PRIMARY_DB)]
    ext = ext.reset_index(drop=True)
    parsed = [calls(g) for g in ext.AMR_genotypes]
    A = pd.DataFrame([{c: int(c in a) for c in a_cols} for a, _ in parsed], dtype="uint8")
    P = pd.DataFrame([{c: int(c in p) for c in p_cols} for _, p in parsed], dtype="uint8")
    unseen = sorted(set().union(*[a for a, _ in parsed]) - set(a_cols)) if parsed else []
    M = pd.DataFrame({o: pd.to_numeric(ext[e], errors="coerce").to_numpy() for o, e in zip(ASM_OURS, ASM_PD)})
    Y = pd.DataFrame({d: [ast(s, d, mode) for s in ext.AST_phenotypes] for d in PD_NAME})
    R = pd.DataFrame({d: [rule_score(a, p, d) for a, p in parsed] for d in PD_NAME})
    groups = ext.get("bioproject_acc", pd.Series([np.nan] * len(ext))).to_numpy(object)
    return dict(A=A, P=P, M=M, Y=Y, R=R, groups=groups, n=len(ext), unseen_genes=unseen, frame=ext)


def load_training(arg_dir, mode):
    A = pd.read_csv(Path(arg_dir) / "acquired.csv", index_col=0)
    P = pd.read_csv(Path(arg_dir) / "point.csv", index_col=0).reindex(A.index, fill_value=0)
    cov = pd.read_csv(repo_results() / "data_inventory" / "covariates.csv").set_index("strain_index")
    Y = load_labels("paper", mode).loc[A.index]
    return dict(A=A.astype("uint8"), P=P.astype("uint8"), M=cov.loc[A.index, ASM_OURS].astype(float), Y=Y)


def features(arm, d):
    if arm == "M":
        return d["M"].to_numpy(float)
    if arm == "A":
        return d["A"].to_numpy(np.uint8)
    return np.hstack([d["A"].to_numpy(np.uint8), d["P"].to_numpy(np.uint8)])


def fit_predict(arm, Xtr, ytr, Xte):
    if arm == "M":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    from xgboost import XGBClassifier

    return XGBClassifier(**XGB).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def run(train, ext, out, n_boot=2000):
    out.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(index=range(ext["n"]))
    for d in PD_NAME:
        y = train["Y"][d].to_numpy()
        ok = ~np.isnan(y)
        for arm in ("AP", "A", "M"):
            S[f"{arm}|{d}"] = fit_predict(arm, features(arm, train)[ok], y[ok].astype(int), features(arm, ext))
        S[f"R|{d}"] = ext["R"][d].to_numpy()
    S.to_csv(out / "scores.csv.gz")

    Y = ext["Y"]
    rows = [dict(arm=arm, drug=d, **summary(Y[d], S[f"{arm}|{d}"])) for arm in ("AP", "A", "M", "R") for d in PD_NAME]
    m = pd.DataFrame(rows)
    m.to_csv(out / "metrics.csv", index=False)

    groups = np.array(
        [g if isinstance(g, str) and g else f"__none_{i}" for i, g in enumerate(ext["groups"])], dtype=object
    )
    codes = pd.factorize(groups)[0]
    deltas = []
    for b, a in [("AP", "M"), ("AP", "R"), ("A", "M")]:
        Sa = pd.DataFrame({d: S[f"{a}|{d}"] for d in PD_NAME})
        Sb = pd.DataFrame({d: S[f"{b}|{d}"] for d in PD_NAME})
        for name, ds in [(d, [d]) for d in PD_NAME] + [("primary_mean", PRIMARY)]:
            pt, lo, hi, ng = cluster_bootstrap(Y, Sa, Sb, codes, ds, n_boot=n_boot)
            deltas.append(
                dict(comparison=f"{b} - {a}", endpoint=name, delta=pt, ci_low=lo, ci_high=hi, n_bioprojects=ng)
            )
    dl = pd.DataFrame(deltas)
    dl.to_csv(out / "deltas.csv", index=False)

    p = dl[(dl.comparison == "AP - M") & (dl.endpoint == "primary_mean")].iloc[0]
    v = {
        "delta_AP_minus_M": p.delta,
        "ci": [p.ci_low, p.ci_high],
        "pass_rule": f"delta >= {PASS_DELTA} and ci_low > 0",
        "primary_pass": bool(p.delta >= PASS_DELTA and p.ci_low > 0),
        "external_isolates": int(ext["n"]),
        "bioprojects": int(p.n_bioprojects),
        "unseen_external_genes_dropped": len(ext["unseen_genes"]),
        "mean_nap": m[m.drug.isin(PRIMARY)].groupby("arm").nap.mean().round(4).to_dict(),
    }
    v["reading"] = (
        "supported: a model on our two-country strains transfers to external isolates beyond the assembly shortcut"
        if v["primary_pass"]
        else "not supported: the gene-based model does not beat the assembly shortcut externally"
    )
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(m.round(3).to_string(index=False))
    print(dl.round(3).to_string(index=False))
    print(json.dumps(v, indent=1, default=float))
    return m, dl, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="primary", choices=["primary", "all"])
    ap.add_argument("--labels", default="ri", choices=["ri", "no_intermediate"])
    ap.add_argument("--candidates", default="results/external_cohort_screen/candidates.tsv.gz")
    ap.add_argument("--arg-dir", default="results/determinant_matrix")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out or f"results/external_cohort/{a.db}_{a.labels}")
    train = load_training(a.arg_dir, a.labels)
    ext = load_external(a.candidates, a.db, a.labels, list(train["A"].columns), list(train["P"].columns))
    if "bioproject_acc" not in ext["frame"].columns:
        raise SystemExit("candidates have no bioproject_acc column: re-run external_cohort_screen (it now keeps it)")
    info = {
        "candidates_file": a.candidates,
        "sha256": hashlib.sha256(Path(a.candidates).read_bytes()).hexdigest(),
        "training_strains": int(len(train["A"])),
        "external_isolates": int(ext["n"]),
        "acquired_columns": int(train["A"].shape[1]),
        "point_columns": int(train["P"].shape[1]),
        "external_genes_not_in_training": ext["unseen_genes"][:200],
        "labelled_external_per_drug": {d: int(ext["Y"][d].notna().sum()) for d in PD_NAME},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "external_genes_not_in_training"}, indent=1))
    run(train, ext, out, a.n_boot)


if __name__ == "__main__":
    main()

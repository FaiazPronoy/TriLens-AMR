from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.evaluation.cluster_bootstrap import cluster_bootstrap
from trilens.experiments.determinant_rescue import repo_results
from trilens.experiments.external_cohort import (
    PASS_DELTA,
    PD_NAME,
    PRIMARY,
    load_external,
    load_training,
)

ROOT = Path(__file__).resolve().parents[2]
MIN_LABELLED = 100
OVERLAP_FRACTION = 0.01


def ours_accessions():
    ids = pd.read_csv(repo_results() / "data_inventory" / "genome_ids.csv", dtype=str)
    out = set()
    for col in ids.columns:
        out |= {v.strip() for v in ids[col].dropna().astype(str) if v.strip()}
    return out, list(ids.columns)


def project_table(ext):
    f = ext["frame"]
    g = pd.Series(ext["groups"], dtype=object).fillna("__none__")
    rows = []
    for proj, sub in f.groupby(g.to_numpy(), dropna=False):
        lab = {d: int(ext["Y"].loc[sub.index, d].notna().sum()) for d in PD_NAME}
        rows.append(
            dict(
                bioproject=proj,
                isolates=len(sub),
                country=(
                    sub.geo_loc_name.dropna().str.split(":").str[0].mode().iat[0]
                    if sub.geo_loc_name.notna().any()
                    else None
                ),
                epi_type=(sub.epi_type.mode().iat[0] if sub.epi_type.notna().any() else None),
                labelled_primary=int(ext["Y"].loc[sub.index, PRIMARY].notna().any(axis=1).sum()),
                **{f"n_{d}": v for d, v in lab.items()},
            )
        )
    return pd.DataFrame(rows).sort_values("isolates", ascending=False).reset_index(drop=True)


def endpoints(Y, S, groups, keep, n_boot):
    Yk = Y.where(pd.Series(keep, index=Y.index), other=np.nan)
    codes = pd.factorize(np.asarray(groups, dtype=object))[0]
    out = {}
    for b, a in [("AP", "M"), ("AP", "R")]:
        Sa = pd.DataFrame({d: S[f"{a}|{d}"] for d in PD_NAME})
        Sb = pd.DataFrame({d: S[f"{b}|{d}"] for d in PD_NAME})
        pt, lo, hi, ng = cluster_bootstrap(Yk, Sa, Sb, codes, PRIMARY, n_boot=n_boot)
        out[f"{b}_minus_{a}"] = (pt, lo, hi, ng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-dir", default="results/external_cohort/primary_ri")
    ap.add_argument("--candidates", default="results/external_cohort_screen/candidates.tsv.gz")
    ap.add_argument("--arg-dir", default="results/determinant_matrix")
    ap.add_argument("--db", default="primary", choices=["primary", "all"])
    ap.add_argument("--labels", default="ri", choices=["ri", "no_intermediate"])
    ap.add_argument("--out", default="results/external_robustness")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sp = Path(a.external_dir) / "scores.csv.gz"
    if not sp.exists():
        raise SystemExit(f"{sp} missing: attach the external_cohort output")
    S = pd.read_csv(sp, index_col=0)
    train = load_training(a.arg_dir, a.labels)
    ext = load_external(a.candidates, a.db, a.labels, list(train["A"].columns), list(train["P"].columns))
    if len(S) != ext["n"]:
        raise SystemExit(
            f"scores have {len(S)} rows, the candidates give {ext['n']}: "
            "the --db/--labels/--candidates must match the external cohort run that wrote them"
        )
    for d in PD_NAME:
        S[f"R|{d}"] = ext["R"][d].to_numpy()
    Y, groups = ext["Y"], ext["groups"]
    g = pd.Series(groups, dtype=object).fillna("__none__").to_numpy(object)

    proj = project_table(ext)
    proj.to_csv(out / "projects.csv", index=False)

    ours, id_cols = ours_accessions()
    hits = []
    for col in ("asm_acc", "target_acc", "Run", "biosample_acc"):
        if col in ext["frame"].columns:
            m = ext["frame"][col].astype(str).isin(ours)
            for i in np.flatnonzero(m.to_numpy()):
                hits.append(dict(row=int(i), column=col, value=ext["frame"][col].iloc[i]))
    ov = pd.DataFrame(hits, columns=["row", "column", "value"])
    ov.to_csv(out / "overlap.csv", index=False)
    overlap_rows = sorted(set(ov.row)) if len(ov) else []

    base_keep = np.ones(ext["n"], bool)
    base_keep[overlap_rows] = False
    rows = []
    base = endpoints(Y, S, g, base_keep, a.n_boot)
    for k, (pt, lo, hi, ng) in base.items():
        rows.append(
            dict(
                dropped="none (overlap removed)",
                comparison=k,
                delta=pt,
                ci_low=lo,
                ci_high=hi,
                n_groups=ng,
                isolates=int(base_keep.sum()),
            )
        )
    big = proj[proj.labelled_primary >= MIN_LABELLED].bioproject.tolist()
    for p in big:
        keep = base_keep & (g != p)
        e = endpoints(Y, S, g, keep, a.n_boot)
        for k, (pt, lo, hi, ng) in e.items():
            rows.append(
                dict(dropped=p, comparison=k, delta=pt, ci_low=lo, ci_high=hi, n_groups=ng, isolates=int(keep.sum()))
            )
        print(
            f"dropped {p:14s} AP-M {e['AP_minus_M'][0]:+.3f} [{e['AP_minus_M'][1]:+.3f}] "
            f"AP-R {e['AP_minus_R'][0]:+.3f} [{e['AP_minus_R'][1]:+.3f}]",
            flush=True,
        )
    lo_df = pd.DataFrame(rows)
    lo_df.to_csv(out / "leave_one_out.csv", index=False)

    m = lo_df[(lo_df.comparison == "AP_minus_M") & (lo_df.dropped != "none (overlap removed)")]
    r = lo_df[(lo_df.comparison == "AP_minus_R") & (lo_df.dropped != "none (overlap removed)")]
    v = {
        "projects_tested": len(big),
        "min_labelled_primary": MIN_LABELLED,
        "overlap_isolates": len(overlap_rows),
        "overlap_fraction": round(len(overlap_rows) / ext["n"], 5),
        "our_id_columns_checked": id_cols,
        "ap_minus_m_range": [round(float(m.delta.min()), 4), round(float(m.delta.max()), 4)],
        "ap_minus_m_min_ci_low": round(float(m.ci_low.min()), 4),
        "pass_rule": f"every replicate: delta >= {PASS_DELTA} and ci_low > 0; overlap fraction <= {OVERLAP_FRACTION}",
        "ap_minus_r_range": [round(float(r.delta.min()), 4), round(float(r.delta.max()), 4)],
        "ap_minus_r_replicates_ci_above_zero": int((r.ci_low > 0).sum()),
        "ap_minus_r_replicates": len(r),
        "candidates_sha256": hashlib.sha256(Path(a.candidates).read_bytes()).hexdigest(),
    }
    v["primary_pass"] = bool(
        len(m) and m.delta.min() >= PASS_DELTA and m.ci_low.min() > 0 and v["overlap_fraction"] <= OVERLAP_FRACTION
    )
    v["reading"] = (
        "the external gain over the assembly arm does not depend on any one BioProject"
        if v["primary_pass"]
        else "the external result changes when a single BioProject is dropped, or the cohort "
        "overlaps ours: it is collection-specific as reported"
    )
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(proj.head(12).to_string(index=False))
    print(json.dumps(v, indent=1, default=float))


if __name__ == "__main__":
    main()

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.data.strain_metadata import AB_NAMES, PROCESSED, RAW, load_strain_table

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "data_inventory"


MIN_TEST_R, MIN_TEST_S = 50, 50
MIN_TRAIN_R, MIN_TRAIN_S = 100, 100


def counts(t, key, ab):
    d = t[t[ab].notna()]
    g = d.groupby(d[key].astype(object).where(d[key].notna(), "unknown").astype(str))[ab]
    out = pd.DataFrame({"n_tested": g.size(), "n_resistant": g.sum().astype(int)})
    out["frac_resistant"] = (out.n_resistant / out.n_tested).round(3)
    out.insert(0, "antibiotic", ab)
    return out.reset_index().rename(columns={key: "value"}).sort_values("n_tested", ascending=False)


def by_key(t, key, top=None):
    rows = []
    for ab in AB_NAMES:
        c = counts(t, key, ab)
        if top:
            keep = c[c.value != "unknown"].head(top)
            rest = c[~c.value.isin(keep.value) & (c.value != "unknown")]
            unk = c[c.value == "unknown"]
            if len(rest):
                other = pd.DataFrame(
                    [
                        {
                            "antibiotic": ab,
                            "value": f"other ({len(rest)} values)",
                            "n_tested": rest.n_tested.sum(),
                            "n_resistant": rest.n_resistant.sum(),
                        }
                    ]
                )
                other["frac_resistant"] = (other.n_resistant / other.n_tested).round(3)
                keep = pd.concat([keep, other])
            c = pd.concat([keep, unk])
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


def side(t, mask, ab):
    y = t.loc[mask, ab].dropna()
    return int((y == 1).sum()), int((y == 0).sum())


def country_splits(t):
    rows = []
    for train_c, test_c in (("United Kingdom", "Norway"), ("Norway", "United Kingdom")):
        for ab in AB_NAMES:
            tr_r, tr_s = side(t, t.country == train_c, ab)
            te_r, te_s = side(t, t.country == test_c, ab)
            rows.append(
                dict(
                    split=f"train {train_c} -> test {test_c}",
                    antibiotic=ab,
                    train_R=tr_r,
                    train_S=tr_s,
                    test_R=te_r,
                    test_S=te_s,
                    train_prev=round(tr_r / max(tr_r + tr_s, 1), 3),
                    test_prev=round(te_r / max(te_r + te_s, 1), 3),
                    feasible=tr_r >= MIN_TRAIN_R and tr_s >= MIN_TRAIN_S and te_r >= MIN_TEST_R and te_s >= MIN_TEST_S,
                )
            )
    return pd.DataFrame(rows)


def time_splits(t):
    rows = []
    scopes = {
        "Norway only": t.country == "Norway",
        "United Kingdom only": t.country == "United Kingdom",
        "all strains with a year": t.year.notna(),
    }
    for scope, sm in scopes.items():
        years = t.loc[sm, "year"].dropna()
        if years.empty:
            continue
        for T in range(int(years.min()), int(years.max())):
            for ab in AB_NAMES:
                tr_r, tr_s = side(t, sm & (t.year <= T), ab)
                te_r, te_s = side(t, sm & (t.year > T), ab)
                rows.append(
                    dict(
                        scope=scope,
                        train_years=f"<= {T}",
                        test_years=f"> {T}",
                        antibiotic=ab,
                        train_R=tr_r,
                        train_S=tr_s,
                        test_R=te_r,
                        test_S=te_s,
                        train_prev=round(tr_r / max(tr_r + tr_s, 1), 3),
                        test_prev=round(te_r / max(te_r + te_s, 1), 3),
                        feasible=tr_r >= MIN_TRAIN_R
                        and tr_s >= MIN_TRAIN_S
                        and te_r >= MIN_TEST_R
                        and te_s >= MIN_TEST_S,
                    )
                )
    return pd.DataFrame(rows)


def pa_orientation():
    with open(PROCESSED / "arcadia_v2_gene_presence_absence.pkl", "rb") as f:
        pa = pickle.load(f)
    ids = np.asarray(pa["gene_ids"]).astype(str)
    prev = np.asarray(pa["X_pa"]).mean(axis=0)
    pg = pd.read_csv(RAW / "pangenome" / "gene_presence_absence.csv", low_memory=False)
    row_of = {}
    for r, cells in enumerate(pg[pg.columns[14:]].values):
        for v in cells:
            if isinstance(v, str):
                for tag in v.split("\t"):
                    row_of[tag.strip()] = r
    n_ecor = len(pg.columns) - 14
    cds = np.array([g in row_of for g in ids])
    f = np.array([pg["No. isolates"].values[row_of[g]] / n_ecor for g in ids[cds]])
    d = pd.DataFrame(
        {
            "gene": ids[cds],
            "name": [pg["Gene"].values[row_of[g]] for g in ids[cds]],
            "ecor_frequency": f,
            "observed_prevalence": prev[cds],
        }
    )
    d["ecor_bin"] = pd.cut(d.ecor_frequency, [0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1.0]).astype(str)
    summary = d.groupby("ecor_bin", sort=False).observed_prevalence.agg(["count", "mean", "median", "max"]).round(3)
    summary = summary.reindex(sorted(summary.index, key=lambda s: float(s.strip("(]").split(",")[0])))
    core = d[d.ecor_frequency >= 0.99]
    facts = dict(
        n_columns=len(ids),
        n_cds_columns=int(cds.sum()),
        n_igr_columns=int((~cds).sum()),
        max_observed_prevalence=round(float(prev.max()), 4),
        n_ecor_core_genes=len(core),
        n_ecor_core_genes_observed_below_1pct=int((core.observed_prevalence < 0.01).sum()),
        gyrA_observed=float(d.loc[d.name == "gyrA", "observed_prevalence"].iloc[0]),
        parC_observed=float(d.loc[d.name == "parC", "observed_prevalence"].iloc[0]),
    )
    return summary.reset_index(), facts


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_strain_table()
    t.drop(columns=["genome_name", "sra"]).to_csv(OUT / "strain_table.csv", index_label="strain_index")
    tabs = {
        "by_country": by_key(t, "country"),
        "by_year": by_key(t, "year"),
        "by_host": by_key(t, "host"),
        "by_study": by_key(t, "study"),
        "by_phylogroup": by_key(t, "phylogroup"),
        "by_st": by_key(t, "st", top=15),
    }
    tabs["by_year"]["value"] = tabs["by_year"]["value"].str.replace(".0", "", regex=False)
    for k, v in tabs.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    cs, ts = country_splits(t), time_splits(t)
    cs.to_csv(OUT / "split_feasibility_country.csv", index=False)
    ts.to_csv(OUT / "split_feasibility_time.csv", index=False)
    orient, facts = pa_orientation()
    orient.to_csv(OUT / "pa_orientation_by_ecor_frequency.csv", index=False)
    pd.Series(facts).to_csv(OUT / "pa_orientation_facts.csv", header=["value"])

    xt = pd.crosstab(t.year.fillna(-1).astype(int).replace(-1, "unknown"), t.country.fillna("unknown"))
    xt.to_csv(OUT / "year_by_country.csv")

    sides = {
        "train United Kingdom -> test Norway": (t.country == "United Kingdom", t.country == "Norway"),
        "train Norway -> test United Kingdom": (t.country == "Norway", t.country == "United Kingdom"),
        "Norway, train <= 2013 -> test > 2013": (
            (t.country == "Norway") & (t.year <= 2013),
            (t.country == "Norway") & (t.year > 2013),
        ),
    }
    ov = []
    for name, (tr, te) in sides.items():
        tr_st = set(t.loc[tr, "st"].dropna())
        te = t.loc[te]
        known = te.st.notna()
        ov.append(
            dict(
                split=name,
                n_test=len(te),
                test_with_known_st=int(known.sum()),
                test_st_seen_in_train=int(te.st[known].isin(tr_st).sum()),
                frac_test_st_seen_in_train=round(float(te.st[known].isin(tr_st).mean()), 3),
                n_test_sts=te.st.nunique(),
                n_test_sts_seen_in_train=int(len(set(te.st.dropna()) & tr_st)),
            )
        )
    ov = pd.DataFrame(ov)
    ov.to_csv(OUT / "split_lineage_overlap.csv", index=False)

    md = ["# Data inventory tables (generated by trilens/analysis/data_inventory.py)", ""]
    for k, v in tabs.items():
        md += [f"## {k}", ""]
        for ab in AB_NAMES:
            sub = v[v.antibiotic == ab].drop(columns="antibiotic")
            md += [f"### {ab}", "", md_table(sub), ""]
    md += [
        "## Country splits",
        "",
        md_table(cs),
        "",
        "## Time splits (feasible rows only)",
        "",
        md_table(ts[ts.feasible]),
        "",
        "## Lineage shared across the candidate splits",
        "",
        md_table(ov),
        "",
        "## Presence/absence orientation",
        "",
        md_table(orient),
        "",
        md_table(pd.Series(facts).reset_index().rename(columns={"index": "fact", 0: "value"})),
        "",
    ]
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

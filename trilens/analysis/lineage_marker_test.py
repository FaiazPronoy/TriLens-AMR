import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)
PA_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_presence_absence.pkl"
MLST_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered__mlst_holdout.pkl"
ATT_B = ROOT / "results" / "trilens_stack_random" / "attention_B_topgenes.csv"
SHAP_DIR = ROOT / "results" / "gene_importance_shap"
OUT = DIAG / "lineage_marker_test.csv"

N_RANDOM_DRAWS = 500
TOP_K = 20
SEED = 42
ABS = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]


def load_pickle(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def uncertainty_coefficient(x_bin, y_cat):
    p = x_bin.mean()
    if p <= 0 or p >= 1:
        return np.nan
    Hx = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return mutual_info_score(x_bin, y_cat) / Hx


def gene_col_of(df):
    for c in ("gene_id", "gene", "feature", "gene_name"):
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == object]
    return obj[0] if obj else df.columns[0]


def main():
    for p in (PA_PKL, MLST_PKL, ATT_B):
        if not p.exists():
            sys.exit(f"MISSING: {p}")

    pa = load_pickle(PA_PKL)
    mlst = load_pickle(MLST_PKL)

    X = np.asarray(pa["X_pa"])
    gene_ids = [str(g) for g in pa["gene_ids"]]
    gidx = {g: i for i, g in enumerate(gene_ids)}

    st = np.asarray(mlst["st_per_genome"]).astype(str).ravel()
    Y = np.asarray(mlst["Y"]).astype(int)
    M = np.asarray(mlst["label_mask"]).astype(bool)
    names = [str(x) for x in mlst["label_names"]]

    ok = ~np.isin(st, ["nan", "None", "", "unknown", "Unknown", "-1"])
    print(f"Genomes: {X.shape[0]}   genes: {X.shape[1]}")
    print(f"With a resolved sequence type: {ok.sum()} ({ok.mean() * 100:.1f}%)")
    print(f"Distinct sequence types: {len(set(st[ok]))}\n")

    Xs, sts = X[ok], st[ok]

    print("Computing U_ST for all genes (this is the slow part)...")
    prev = Xs.mean(axis=0)
    usable = np.flatnonzero((prev > 0.01) & (prev < 0.99))
    print(f"Genes with usable variation (1%-99% prevalence): {len(usable)}\n")

    U_ST = np.full(X.shape[1], np.nan)
    for n, j in enumerate(usable):
        U_ST[j] = uncertainty_coefficient(Xs[:, j], sts)
        if n % 2000 == 0 and n:
            print(f"   {n}/{len(usable)}")

    att = pd.read_csv(ATT_B)
    acol = gene_col_of(att)

    def resolve(gene_list):
        out = []
        for g in gene_list:
            if g in gidx:
                out.append(gidx[g])
                continue
            for tag in re.findall(r"\b[A-Z]{8}_\d{4,5}\b", str(g)):
                if tag in gidx:
                    out.append(gidx[tag])
                    break
        return [i for i in out if not np.isnan(U_ST[i])]

    rng = np.random.default_rng(SEED)
    rows = []

    for a, ab in enumerate(names):
        if ab not in ABS:
            continue
        obs = M[:, a] & ok
        y = Y[:, a][obs]

        def u_res(j):
            return uncertainty_coefficient(X[obs, j], y)

        att_ids = att.loc[att.ab == ab, acol].head(TOP_K).tolist()
        A = resolve(att_ids)

        S = []
        for cand in (SHAP_DIR / f"shap_gene_level_{ab}.csv", SHAP_DIR / f"shap_snp_level_{ab}.csv"):
            if cand.exists():
                d = pd.read_csv(cand)
                S = resolve(d[gene_col_of(d)].head(TOP_K).tolist())
                if S:
                    break

        pool = [j for j in usable if not np.isnan(U_ST[j])]
        null = np.array([np.nanmean(U_ST[rng.choice(pool, TOP_K, replace=False)]) for _ in range(N_RANDOM_DRAWS)])

        a_st = float(np.nanmean(U_ST[A])) if A else np.nan
        s_st = float(np.nanmean(U_ST[S])) if S else np.nan
        a_rs = float(np.nanmean([u_res(j) for j in A])) if A else np.nan
        s_rs = float(np.nanmean([u_res(j) for j in S])) if S else np.nan

        rows.append(
            dict(
                antibiotic=ab,
                n_attention=len(A),
                n_shap=len(S),
                attention_U_ST=round(a_st, 4),
                shap_U_ST=round(s_st, 4),
                random_U_ST=round(float(null.mean()), 4),
                attention_vs_random=round(a_st / null.mean(), 2) if null.mean() else np.nan,
                p_attention=round(float((null >= a_st).mean()), 4),
                attention_U_res=round(a_rs, 4),
                shap_U_res=round(s_rs, 4),
            )
        )

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)

    print("\n" + "=" * 96)
    print("HOW MUCH OF EACH GENE SET IS EXPLAINED BY LINEAGE (U_ST) VS RESISTANCE (U_res)")
    print("=" * 96)
    print(res.to_string(index=False))

    print("\n" + "-" * 96)
    print(
        f"Mean U_ST   attention {res.attention_U_ST.mean():.4f} | "
        f"SHAP {res.shap_U_ST.mean():.4f} | random {res.random_U_ST.mean():.4f}"
    )
    print(f"Mean U_res  attention {res.attention_U_res.mean():.4f} | SHAP {res.shap_U_res.mean():.4f}")
    print("-" * 96)

    res.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")
    print("""
How to read this
  attention_U_ST much higher than random_U_ST, with p_attention near 0:
      the attended genes are strongly determined by sequence type.
  attention_U_ST > shap_U_ST and attention_U_res < shap_U_res:
      the neural branch follows lineage more than the classical learners do.
  attention_U_ST close to random_U_ST:
      no lineage effect.
""")


if __name__ == "__main__":
    main()

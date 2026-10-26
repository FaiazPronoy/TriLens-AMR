"""
W2: lineage-lookup baselines on every split (docs/experiments/w2_lineage_lookup.md).

    python -m src.experiments.run_lineage_lookup --pilot     # S1, CIP only
    python -m src.experiments.run_lineage_lookup             # all splits, all antibiotics

Distance: gene presence/absence (Hamming), the CPU stand-in. The SNP-distance variant runs on
Kaggle (w2_snp_knn) with --distance snp -> results/w2_lineage_lookup_snp/. P/A output:
results/w2_lineage_lookup_pa_pilot/ (step 1, CIP on S1, in step1_cip_random/).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.baselines.lineage_lookup import (K_GRID, choose_k, hamming_matrix, knn_scores,
                                          phylogroup_lookup, st_lookup)
from src.data.metadata import AB_NAMES, PROCESSED, ROOT, load_strain_table
from src.evaluation.shift import bootstrap_ci, clinical, prevalence_matched_pr_auc, safe_roc_auc

CACHE = ROOT / "data" / "cache"
SPLIT_NAMES = {
    "S1_random": "random_split", "S2_phylogroup": "phylogeny_split", "S3_st_holdout": "mlst_lineage_holdout_split",
    "S4a_uk_to_norway": "country_uk_to_norway", "S4b_norway_to_uk": "country_norway_to_uk",
    "S5_time_norway": "time_norway_2013",
}


def load_splits():
    with open(PROCESSED / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)
    out = {"S1_random": master["random_split"], "S2_phylogroup": master["phylogeny_split"]}
    for sid, fname in (("S3_st_holdout", "mlst_lineage_holdout_split.pkl"),
                       ("S4a_uk_to_norway", "country_uk_to_norway_split.pkl"),
                       ("S4b_norway_to_uk", "country_norway_to_uk_split.pkl"),
                       ("S5_time_norway", "time_norway_2013_split.pkl")):
        with open(PROCESSED / fname, "rb") as f:
            out[sid] = pickle.load(f)
    return {k: {p: np.asarray(v[f"{p}_idx"]).astype(np.int64) for p in ("train", "val", "test")} | {
        "antibiotics": v.get("antibiotics", {"primary": AB_NAMES, "secondary": []})} for k, v in out.items()}


def pa_distance():
    """Hamming distances on the rebuilt presence/absence matrix (KNOWN_ISSUES 12), and the mask of
    strains that have a presence/absence profile at all (13 do not: all zeros)."""
    with open(PROCESSED / "arcadia_v2_gene_presence_absence.pkl", "rb") as f:
        pa = pickle.load(f)
    if not str(pa.get("encoding_rule", "")).startswith("present = (allele letter"):
        raise ValueError("arcadia_v2_gene_presence_absence.pkl is the old minority-state file; "
                         "use the rebuilt one from the outputs-pa_fix release")
    matched = np.asarray(pa["matched_mask"]).astype(bool)
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "pa_hamming_rebuilt.npy"
    if p.exists():
        return np.load(p), matched
    t0 = time.time()
    D = hamming_matrix(pa["X_pa"])
    np.save(p, D)
    print(f"P/A Hamming matrix {D.shape} in {time.time() - t0:.0f}s -> {p}")
    return D, matched


def tier(split, ab):
    a = split["antibiotics"]
    return "primary" if ab in a["primary"] else ("secondary" if ab in a.get("secondary", []) else "below minimum")


def run(t, D, splits, abs_, random_test_prev, out_scores, keep, knn_name="knn_pa"):
    rows = []
    st = np.array(t.st.values, dtype=object)
    pg = np.array(t.phylogroup.values, dtype=object)
    for sid, sp in splits.items():
        for ab in abs_:
            y_all = t[ab].values
            lab = lambda idx: idx[~np.isnan(y_all[idx]) & keep[idx]]
            tr, va, te = lab(sp["train"]), lab(sp["val"]), lab(sp["test"])
            ytr, yva, yte = (y_all[i].astype(int) for i in (tr, va, te))
            if min(ytr.sum(), (1 - ytr).sum(), yte.sum(), (1 - yte).sum(), yva.sum(), (1 - yva).sum()) == 0:
                rows.append(dict(split=sid, antibiotic=ab, tier=tier(sp, ab), status="skipped: a side has one class",
                                 n_train=len(tr), n_val=len(va), n_test=len(te)))
                continue
            base = {}
            s_va, _ = st_lookup(st[tr], pg[tr], ytr, st[va], pg[va])
            s_te, level = st_lookup(st[tr], pg[tr], ytr, st[te], pg[te])
            base["st_lookup"] = (s_va, s_te, {"test_share_st_seen": float((level == "st").mean())})
            base["phylogroup"] = (phylogroup_lookup(pg[tr], ytr, pg[va]), phylogroup_lookup(pg[tr], ytr, pg[te]), {})
            kv = knn_scores(D[np.ix_(va, tr)], ytr)
            k, _ = choose_k(kv, yva)
            kt = knn_scores(D[np.ix_(te, tr)], ytr, ks=(k,))[k]
            base[knn_name] = (kv[k], kt, {"k": k})
            val_ap = {b: average_precision_score(yva, v[0]) for b, v in base.items()}
            best = max(val_ap, key=val_ap.get)
            for b, (sv, sc, extra) in base.items():
                ap, lo, hi = bootstrap_ci(yte, sc)
                vme, me = clinical(yte, sc, 0.5)
                m_tr = prevalence_matched_pr_auc(yte, sc, ytr.mean())
                m_rd = prevalence_matched_pr_auc(yte, sc, random_test_prev[ab])
                rows.append(dict(
                    split=sid, antibiotic=ab, tier=tier(sp, ab), baseline=b, status="ok",
                    n_train=len(tr), n_val=len(va), n_test=len(te), n_test_resistant=int(yte.sum()),
                    train_prev=round(float(ytr.mean()), 4), test_prev=round(float(yte.mean()), 4),
                    val_pr_auc=round(val_ap[b], 4), best_on_val=(b == best),
                    test_pr_auc=round(ap, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                    test_roc_auc=round(safe_roc_auc(yte, sc), 4),
                    pr_auc_matched_train_prev=round(m_tr[0], 4),
                    pr_auc_matched_random_test_prev=round(m_rd[0], 4),
                    vme_0p5=round(vme, 4), me_0p5=round(me, 4), **extra))
                out_scores[f"{sid}|{ab}|{b}"] = sc
            out_scores[f"{sid}|{ab}|test_idx"] = te
            out_scores[f"{sid}|{ab}|y_test"] = yte
            print(f"{sid:18s} {ab}  " + "  ".join(f"{b}={r['test_pr_auc']:.3f}" for b, r in
                                                   ((r['baseline'], r) for r in rows[-3:])) + f"  (k={k}, best on val: {best})")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="S1 random split, CIP only")
    ap.add_argument("--distance", default="pa", choices=["pa", "snp"])
    a = ap.parse_args()
    t = load_strain_table()
    splits = load_splits()
    random_test_prev = {ab: float(np.nanmean(t[ab].values[splits["S1_random"]["test"]])) for ab in AB_NAMES}
    if a.pilot:
        splits, abs_ = {"S1_random": splits["S1_random"]}, ["CIP"]
    else:
        abs_ = AB_NAMES
    if a.distance == "pa":
        D, keep = pa_distance()
        print(f"{int((~keep).sum())} strains without a presence/absence profile are left out of every partition")
        out = ROOT / "results" / "w2_lineage_lookup_pa_pilot"
        note = "P/A-only pilot (rebuilt presence/absence matrix; the SNP variant runs on Kaggle)"
        about = ("Distance = Hamming on the 32,441 gene presence/absence columns of the rebuilt matrix "
                 "(KNOWN_ISSUES 12), with the 13 strains that have no presence/absence profile left out of "
                 "every partition; a stand-in for the SNP distance that runs on Kaggle.")
    else:
        snp = CACHE / "snp_hamming.npy"
        if not snp.exists():
            raise FileNotFoundError(f"{snp} missing: run python -m src.baselines.snp_distance first")
        D, keep = np.load(snp), np.ones(len(t), bool)       # SNP analyses keep all strains
        out = ROOT / "results" / "w2_lineage_lookup_snp"
        note = "SNP-distance lookups (the paper's W2 table)"
        about = ("Distance = Hamming on the 535,439 SNP columns, every non-reference code counted as "
                 "'differs from reference' (`src/baselines/snp_distance.py`); all strains kept.")
    if a.pilot:
        out = out / "step1_cip_random"
    out.mkdir(parents=True, exist_ok=True)
    scores = {}
    df = run(t, D, splits, abs_, random_test_prev, scores, keep, knn_name=f"knn_{a.distance}")
    df.insert(0, "note", note)
    df["n_unmatched_excluded"] = int((~keep).sum())
    df.to_csv(out / "lookup_results.csv", index=False)
    np.savez_compressed(out / "test_scores.npz", **{k.replace("|", "__"): v for k, v in scores.items()})
    (out / "README.md").write_text(
        f"# W2 lineage-lookup baselines ({a.distance.upper()} distance)\n\n"
        f"Pre-registration: `docs/experiments/w2_lineage_lookup.md`. {about} Produced by "
        f"`python -m src.experiments.run_lineage_lookup --distance {a.distance}{' --pilot' if a.pilot else ''}`.\n\n"
        f"k grid: {list(K_GRID)}; smoothing alpha = 1; bootstrap 2,000 resamples.\n", encoding="utf-8")
    (out / "random_test_prevalence.json").write_text(json.dumps(random_test_prev, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

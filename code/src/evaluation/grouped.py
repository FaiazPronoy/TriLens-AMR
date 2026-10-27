"""
Prevalence-aware scores, lineage-clustered bootstrap and within-lineage concordance.
Rules: docs/experiments/w6_arg_rescue.md.

    nap(y, s)                     (AP - prevalence) / (1 - prevalence); 0 = no better than chance
    summary(y, s)                 n, positives, prevalence, AP, nAP, AUROC
    cluster_bootstrap(...)        paired CI of a difference in mean nAP, resampling whole groups (STs)
    within_st_concordance(...)    among resistant-susceptible pairs from the same ST, share ranked right

Groups: strains with no ST get a group of their own (each is its own lineage for resampling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _valid(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~np.isnan(y) & ~np.isnan(s)
    return y[m], s[m], m


def nap(y, s, w=None):
    y, s, m = _valid(y, s)
    w = None if w is None else np.asarray(w, float)[m]
    if w is not None:
        keep = w > 0
        y, s, w = y[keep], s[keep], w[keep]
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    prev = np.average(y, weights=w)
    return (average_precision_score(y, s, sample_weight=w) - prev) / (1 - prev)


def summary(y, s):
    y, s, _ = _valid(y, s)
    both = len(y) and y.min() != y.max()
    prev = float(y.mean()) if len(y) else np.nan
    ap = float(average_precision_score(y, s)) if both else np.nan
    return dict(n=int(len(y)), positives=int(y.sum()), prevalence=prev, ap=ap,
                nap=(ap - prev) / (1 - prev) if both else np.nan,
                auroc=float(roc_auc_score(y, s)) if both else np.nan)


def group_codes(st):
    """Integer group per strain; a missing ST becomes a singleton group."""
    st = pd.Series(np.asarray(st, dtype=object))
    missing = st.isna() | st.astype(str).isin(["", "None", "nan"])
    labels = st.astype(str).where(~missing, "__single__" + pd.Series(np.arange(len(st))).astype(str))
    return pd.factorize(labels)[0]


def cluster_bootstrap(Y, S_a, S_b, groups, drugs, n_boot=2000, seed=0):
    """
    Y, S_a, S_b: DataFrames (strains x drugs) of labels and the two arms' scores (NaN = not scored).
    Difference = mean over `drugs` of nAP(b) - nAP(a). Each resample draws groups with replacement;
    a strain's weight is how often its group was drawn, the same weight for both arms (paired).
    Returns (point estimate, lower 2.5%, upper 97.5%, n_groups).
    """
    g = np.asarray(groups)
    scored = np.zeros(len(g), bool)
    for d in drugs:
        scored |= Y[d].notna().to_numpy() & S_a[d].notna().to_numpy()
    uniq = np.unique(g[scored])
    pos = {u: i for i, u in enumerate(uniq)}
    gi = np.array([pos.get(x, -1) for x in g])

    def diff(w):
        return np.nanmean([nap(Y[d], S_b[d], w) - nap(Y[d], S_a[d], w) for d in drugs])

    point = diff(None)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        w = np.where(gi >= 0, counts[np.clip(gi, 0, None)], 0)
        out[b] = diff(w)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(point), float(lo), float(hi), int(len(uniq))


def within_st_concordance(y, s, st, min_pairs=1):
    """
    For each ST holding both phenotypes: share of (resistant, susceptible) pairs with the resistant
    strain scored higher (ties count half) = that ST's AUROC. Returns macro mean over STs with
    >= min_pairs pairs, the pair-weighted mean, and the number of STs and pairs used.
    A score that is constant within each ST gets exactly 0.5.
    """
    df = pd.DataFrame({"y": np.asarray(y, float), "s": np.asarray(s, float),
                       "st": pd.Series(np.asarray(st, dtype=object)).astype(str)})
    df = df[df.y.notna() & df.s.notna() & ~df.st.isin(["", "None", "nan"])]
    rows = []
    for k, d in df.groupby("st"):
        npos, nneg = int(d.y.sum()), int((1 - d.y).sum())
        if npos and nneg and npos * nneg >= min_pairs:
            rows.append((k, roc_auc_score(d.y, d.s), npos * nneg))
    if not rows:
        return dict(macro=np.nan, pair_weighted=np.nan, n_st=0, n_pairs=0)
    a = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows], float)
    return dict(macro=float(a.mean()), pair_weighted=float(np.average(a, weights=p)),
                n_st=len(rows), n_pairs=int(p.sum()))

"""
Lineage-lookup baselines (W2): predict resistance from a strain's relatives only.
Definitions are pre-registered in docs/experiments/w2_lineage_lookup.md.

Every function receives the training strains' labels explicitly and never sees a label of
the strains it scores.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

K_GRID = (1, 3, 5, 10, 20, 50)
ALPHA = 1.0


def _known(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) not in ("", "nan", "None")


def group_fraction(train_groups, train_y, groups, prior, alpha=ALPHA):
    """Smoothed resistant fraction of each strain's group in training: (r + a*p) / (n + a).
    Returns the score and a mask of strains whose group occurred in training."""
    r, n = {}, {}
    for g, y in zip(train_groups, train_y):
        if _known(g):
            r[g] = r.get(g, 0.0) + y
            n[g] = n.get(g, 0) + 1
    score = np.full(len(groups), np.nan)
    seen = np.zeros(len(groups), bool)
    for i, g in enumerate(groups):
        if _known(g) and g in n:
            score[i] = (r[g] + alpha * prior) / (n[g] + alpha)
            seen[i] = True
    return score, seen


def st_lookup(train_st, train_pg, train_y, st, pg, alpha=ALPHA):
    """(a) ST lookup with phylogroup, then prior, fallback. Returns score and the level used."""
    prior = float(np.mean(train_y))
    s_st, seen_st = group_fraction(train_st, train_y, st, prior, alpha)
    s_pg, seen_pg = group_fraction(train_pg, train_y, pg, prior, alpha)
    score = np.where(seen_st, s_st, np.where(seen_pg, s_pg, prior))
    level = np.where(seen_st, "st", np.where(seen_pg, "phylogroup", "prior"))
    return score, level


def phylogroup_lookup(train_pg, train_y, pg, alpha=ALPHA):
    """(c) smoothed resistant fraction of the phylogroup, prior fallback."""
    prior = float(np.mean(train_y))
    s, seen = group_fraction(train_pg, train_y, pg, prior, alpha)
    return np.where(seen, s, prior)


def hamming_matrix(X, chunk=1024):
    """All-pairs Hamming distance of a binary matrix (strains x features), float32 matmul.
    Exact while the number of features is below 2**24."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] >= 2 ** 24:
        raise ValueError("too many features for exact float32 counts")
    s = X.sum(axis=1)
    D = np.empty((X.shape[0], X.shape[0]), dtype=np.float32)
    for a in range(0, X.shape[0], chunk):
        D[a:a + chunk] = s[a:a + chunk, None] + s[None, :] - 2.0 * (X[a:a + chunk] @ X.T)
    return np.rint(D).astype(np.int32)


def knn_scores(D_eval_train, train_y, ks=K_GRID):
    """(b) resistant fraction among the k nearest training strains, for every k in ks.
    D_eval_train: distances (n_eval x n_train). Ties are broken by the lower training
    position (stable sort), which is the lower strain index when train columns are sorted."""
    order = np.argsort(D_eval_train, axis=1, kind="stable")
    ys = np.asarray(train_y, float)[order]
    csum = np.cumsum(ys, axis=1)
    return {k: csum[:, min(k, ys.shape[1]) - 1] / min(k, ys.shape[1]) for k in ks}


def choose_k(val_scores, val_y):
    """Smallest k with the highest validation PR-AUC."""
    best_k, best = None, -np.inf
    for k in sorted(val_scores):
        ap = average_precision_score(val_y, val_scores[k])
        if ap > best + 1e-12:
            best_k, best = k, ap
    return best_k, best

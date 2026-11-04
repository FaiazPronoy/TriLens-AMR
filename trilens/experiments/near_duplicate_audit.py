from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "near_duplicate_audit"
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered.pkl"
DUP_THRESHOLD = 0.005
SAMPLE_SCREEN_THRESHOLD = 0.02
N_SAMPLE_COLS = 2000
SEED = 42


def _load_data():
    with open(GENE_FILTERED_PKL, "rb") as f:
        d = pickle.load(f)
    return d


def _find_near_duplicates(X_snp: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
    rng = np.random.default_rng(SEED)
    cols = rng.choice(X_snp.shape[1], size=N_SAMPLE_COLS, replace=False)
    cols.sort()

    Xtr_s = X_snp[np.ix_(train_idx, cols)].astype(np.float32)
    Xte_s = X_snp[np.ix_(test_idx, cols)].astype(np.float32)
    D = cdist(Xte_s, Xtr_s, metric="hamming")
    nn_train_local = D.argmin(axis=1)
    sampled_min = D.min(axis=1)

    exact_dist = np.full(len(test_idx), np.nan, dtype=np.float64)
    candidates = np.where(sampled_min < SAMPLE_SCREEN_THRESHOLD)[0]
    n_snps = X_snp.shape[1]
    for te_local in candidates:
        te_global = test_idx[te_local]
        tr_global = train_idx[nn_train_local[te_local]]
        diff = (X_snp[te_global] != X_snp[tr_global]).sum()
        exact_dist[te_local] = diff / n_snps

    non_candidates = np.where(sampled_min >= SAMPLE_SCREEN_THRESHOLD)[0]
    exact_dist[non_candidates] = sampled_min[non_candidates]

    dup_mask = exact_dist < DUP_THRESHOLD
    return dup_mask, exact_dist


def compute_dedup_eval(
    split_name: str, predictions_npz_path: Path, prob_key: str = "test_ens_probs", idx_key: str = "test_indices"
) -> dict:
    d = _load_data()
    Y, mask, label_names = d["Y"], d["label_mask"], list(d["label_names"])
    X_snp = d["X_snp_filtered"]
    split = d[split_name]
    train_idx = np.asarray(split.get("train_idx", split.get("train", [])))

    npz = np.load(predictions_npz_path, allow_pickle=True)
    test_idx = npz[idx_key]
    test_probs = npz[prob_key]

    dup_mask, exact_dist = _find_near_duplicates(X_snp, train_idx, test_idx)
    n_dup = int(dup_mask.sum())
    print(
        f"[{split_name}] {n_dup}/{len(test_idx)} test samples are near-duplicates "
        f"(<{DUP_THRESHOLD * 100:.1f}% SNP divergence from a training genome) -- dropping them"
    )

    keep_mask = ~dup_mask
    dedup_test_idx = test_idx[keep_mask]

    Yt, Mt = Y[test_idx], mask[test_idx]
    rows = []
    for i, ab in enumerate(label_names):
        valid_full = Mt[:, i].astype(bool)
        valid_dedup = valid_full & keep_mask
        pr_before = float(average_precision_score(Yt[valid_full, i], test_probs[valid_full, i]))
        if valid_dedup.sum() >= 2 and len(np.unique(Yt[valid_dedup, i])) > 1:
            pr_after = float(average_precision_score(Yt[valid_dedup, i], test_probs[valid_dedup, i]))
        else:
            pr_after = float("nan")
        rows.append(
            {
                "split": split_name,
                "antibiotic": ab,
                "pr_auc_before": pr_before,
                "pr_auc_after": pr_after,
                "delta": pr_after - pr_before if not np.isnan(pr_after) else float("nan"),
                "n_test_before": int(valid_full.sum()),
                "n_test_after": int(valid_dedup.sum()),
            }
        )

    return {
        "rows": rows,
        "n_dup": n_dup,
        "n_test_total": len(test_idx),
        "dedup_test_idx": dedup_test_idx,
        "exact_dist": exact_dist,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = compute_dedup_eval(
        split_name="random_split",
        predictions_npz_path=next(
            (d for d in (RESULTS_DIR / "trilens_stack_random", RESULTS_DIR / "trilens_stack_random") if d.exists()),
            RESULTS_DIR / "trilens_stack_random",
        )
        / "predictions_saved.npz",
    )
    df = pd.DataFrame(result["rows"])
    df.to_csv(OUTPUT_DIR / "near_duplicate_results.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'near_duplicate_results.csv'}")
    np.save(OUTPUT_DIR / "random_split_dedup_test_idx.npy", result["dedup_test_idx"])

    avg_before = df["pr_auc_before"].mean()
    avg_after = df["pr_auc_after"].mean()

    lines = ["# Deduplicated-test-set evaluation (full_stack, random_split)\n"]
    lines.append(
        f"{result['n_dup']}/{result['n_test_total']} test samples "
        f"(random_split) were flagged as near-duplicates of a training "
        f"genome (<{DUP_THRESHOLD * 100:.1f}% SNP divergence) and removed. "
        f"Training set unchanged (leakage direction is test-contaminated-by-"
        f"train, not the reverse).\n"
    )
    lines.append(
        f"**Avg PR-AUC before dedup: {avg_before:.4f}**  |  **after dedup: {avg_after:.4f}**  "
        f"|  **delta: {avg_after - avg_before:+.4f}**\n"
    )
    lines.append("## Per-antibiotic\n")
    lines.append("| Antibiotic | PR-AUC before | PR-AUC after | Delta | N test before | N test after |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['pr_auc_before']:.4f} | {r['pr_auc_after']:.4f} | "
            f"{r['delta']:+.4f} | {r['n_test_before']} | {r['n_test_after']} |"
        )
    lines.append("")
    lines.append("## phylogeny_split - NOT done yet (scope limitation found during implementation)\n")
    lines.append(
        "`results/trilens_stack_random/predictions_saved.npz` only covers "
        "`random_split` (full_stack). full_stack_phylogroup's (phylogeny_split) raw per-sample "
        "predictions were never saved to disk by `trilens_stack.py` - only "
        "its aggregate PR-AUC/MCC/CI are saved in `ablation_results.csv`. "
        "The phylogroup-stratified split is therefore not audited here."
    )
    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY.md'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

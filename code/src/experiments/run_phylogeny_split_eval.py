"""
src/experiments/run_phylogeny_split_eval.py
============================================
EXP-1: Phylogeny-aware split re-evaluation (phylogenetic leakage check).

Motivation
----------
Genomic ML reviewers require phylogeny-aware splits to rule out phylogenetic
signal leakage.  Our headline 0.8469 avg PR-AUC is on random_split, which may
have phylogenetically related strains in both train and test.  The pkl contains
`phylogeny_split` (train/val/test indices built by phylogenetic grouping).
Re-running on phylogeny_split gives the leakage-corrected estimate.

What this script does
---------------------
1. Gene-aggregated XGBoost baseline (Stage 1.5) on both random_split and
   phylogeny_split independently.
2. Full CascadeStack v9.1 (6 bases + LogReg meta) on phylogeny_split.
   This works by calling run_cascadestack_v9.run() with a modified config
   that sets split_key="phylogeny_split".
3. Comparison table: per-AB PR-AUC on random_split vs phylogeny_split,
   with bootstrap 95% CIs and Δ (leakage estimate).
4. SUMMARY.md stating which number should be primary in the paper.

PREREQUISITE — add split_key support to run_cascadestack_v9.py
--------------------------------------------------------------
In src/experiments/run_cascadestack_v9.py, replace every occurrence of:

    data["random_split"]

with:

    data.get(cfg.get("split_key", "random_split")) or data["random_split"]

And in configs/experiment/cascadestack_v9.yaml add:

    split_key: random_split   # default; override to phylogeny_split for EXP-1

This change is backward-compatible: default behavior is unchanged.

Dispatch line (add to src/run_experiment.py _DISPATCH):
    "phylogeny_split_eval": "src.experiments.run_phylogeny_split_eval",
"""
from __future__ import annotations

import gc
import importlib
import inspect
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score

log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

AB_NAMES = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]

# Locked CascadeStack v9.1 results on random_split (test set, seed=42)
# These are the existing headline numbers — do NOT overwrite.
V9_RANDOM_PRAUC = dict(zip(AB_NAMES,
    [0.9412, 0.9774, 0.7569, 0.8346, 0.7037, 0.8931, 0.8215]))
V9_RANDOM_AVG = 0.8469

# Stage 1.5 gene-agg XGB on random_split (locked reference)
XGB_GENE_RANDOM_PRAUC = dict(zip(AB_NAMES,
    [0.9432, 0.9770, 0.7490, 0.8183, 0.7128, 0.8846, 0.8158]))


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(pkl_path: str) -> Dict:
    """Load full dataset from pkl."""
    log.info(f"Loading data from {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    if "phylogeny_split" not in raw:
        raise KeyError(
            "phylogeny_split not found in pkl.  "
            "The pkl must contain phylogeny_split with keys "
            "{'train': [...], 'val': [...], 'test': [...]}.  "
            "If only random_split is available, phylogenetic leakage "
            "cannot be assessed — note this limitation in the paper."
        )
    return {
        "X_snp":          raw.get("X_snp_filtered"),
        "gene_to_range":  raw.get("gene_to_range", {}),
        "selected_genes": raw.get("selected_gene_ids", []),
        "Y":              raw["Y"].astype(np.float32),
        "label_mask":     raw["label_mask"].astype(bool),
        "random_split":   raw["random_split"],
        "phylogeny_split": raw["phylogeny_split"],
    }


# ─── Gene Feature Building ───────────────────────────────────────────────────

def _build_or_load_features(
    data:      Dict,
    cache_dir: Path,
) -> np.ndarray:
    """
    Load gene feature matrix (N, n_genes, 4) from cache or build inline.
    Uses same logic / cache as GAN experiments to avoid duplication.
    """
    n_genes = len(data["selected_genes"])
    cache_path = cache_dir / f"gene_features_{n_genes}x4.npy"

    if cache_path.exists():
        log.info(f"Gene features loaded from cache: {cache_path}")
        return np.load(str(cache_path))

    log.info(f"Building gene features ({n_genes} genes) inline …")
    X_snp  = data["X_snp"]
    gt_rng = data["gene_to_range"]
    genes  = data["selected_genes"]
    N      = X_snp.shape[0]
    out    = np.zeros((N, n_genes, 4), dtype=np.float32)
    X      = X_snp.astype(np.float32)

    for g_idx, gid in enumerate(genes):
        start, end = gt_rng[gid]
        if end <= start:
            continue
        cols             = X[:, start:end]
        out[:, g_idx, 0] = cols.mean(axis=1)
        out[:, g_idx, 1] = cols.var(axis=1)
        out[:, g_idx, 2] = (cols > 0).sum(axis=1)
        out[:, g_idx, 3] = cols.max(axis=1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), out)
    log.info(f"Gene features saved to cache: {cache_path}")
    return out


# ─── Bootstrap CI helper ─────────────────────────────────────────────────────

def _bootstrap_ci(
    y_true:  np.ndarray,
    y_score: np.ndarray,
    n_boot:  int = 1000,
    seed:    int = 42,
) -> Tuple[float, float, float]:
    """Returns (point_estimate, ci_lo, ci_hi)."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan, np.nan
    point = float(average_precision_score(y_true, y_score))
    rng   = np.random.default_rng(seed)
    n     = len(y_true)
    boot  = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        try:
            boot.append(average_precision_score(yt, ys))
        except Exception:
            pass
    if len(boot) < 50:
        return point, np.nan, np.nan
    arr = np.array(boot)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ─── Gene-Aggregated XGBoost Baseline ────────────────────────────────────────

def run_gene_agg_xgb(
    data:       Dict,
    X_feat:     np.ndarray,    # (N, n_genes, 4)
    split_key:  str,
    n_bootstrap: int = 1000,
    seed:       int = 42,
    xgb_params: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Run per-AB XGBoost on gene-aggregated features using the specified split.
    Returns {ab_name: {'prauc': float, 'ci_lo': float, 'ci_hi': float}}.
    """
    split = data[split_key]
    train_idx = np.array(split.get("train") or split["train_idx"])
    val_idx   = np.array(split.get("val") or split["val_idx"])
    test_idx  = np.array(split.get("test") or split["test_idx"])

    Y          = data["Y"]
    label_mask = data["label_mask"]

    log.info(
        f"XGB gene-agg on {split_key}: "
        f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
    )

    # Flatten and standardize on train
    N, G, S = X_feat.shape
    X_flat  = X_feat.reshape(N, G * S).astype(np.float32)
    mean    = X_flat[train_idx].mean(axis=0)
    std     = X_flat[train_idx].std(axis=0).clip(min=1e-8)
    X_flat  = (X_flat - mean) / std

    default_params = dict(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.3,   # aggressive for ~20K features
        min_child_weight=1,
        eval_metric="aucpr",
        early_stopping_rounds=25,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    if xgb_params:
        default_params.update(xgb_params)

    results = {}
    for ab_idx, ab_name in enumerate(AB_NAMES):
        valid_tr = label_mask[train_idx, ab_idx]
        valid_va = label_mask[val_idx,   ab_idx]
        valid_te = label_mask[test_idx,  ab_idx]

        if valid_tr.sum() < 20 or valid_te.sum() < 2:
            results[ab_name] = {"prauc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
            continue

        X_tr = X_flat[train_idx][valid_tr]
        y_tr = Y[train_idx][valid_tr, ab_idx]
        X_va = X_flat[val_idx][valid_va]
        y_va = Y[val_idx][valid_va, ab_idx]
        X_te = X_flat[test_idx]                 # all test samples

        n_pos = float(y_tr.sum())
        n_neg = float(len(y_tr) - n_pos)
        params = dict(**default_params, scale_pos_weight=n_neg / max(n_pos, 1.0))

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        prob_te = model.predict_proba(X_te)[:, 1]
        y_true  = Y[test_idx][valid_te, ab_idx]
        y_score = prob_te[valid_te]

        pt, lo, hi = _bootstrap_ci(y_true, y_score, n_bootstrap, seed)
        results[ab_name] = {
            "prauc": round(float(pt), 5) if not np.isnan(pt) else np.nan,
            "ci_lo": round(float(lo), 5) if not np.isnan(lo) else np.nan,
            "ci_hi": round(float(hi), 5) if not np.isnan(hi) else np.nan,
        }
        log.info(
            f"  XGB/{split_key}/{ab_name}: PR-AUC={pt:.4f} "
            f"[{lo:.4f},{hi:.4f}]"
        )

        del model
        gc.collect()

    avg = float(np.nanmean([r["prauc"] for r in results.values()]))
    log.info(
        f"  XGB/{split_key}: avg PR-AUC = {avg:.4f}  "
        + "  ".join(f"{ab}={r['prauc']:.3f}" for ab, r in results.items())
    )
    return results


# ─── CascadeStack v9.1 on Phylogeny Split ────────────────────────────────────

_SPLIT_KEY_INSTRUCTIONS = """
═══════════════════════════════════════════════════════════════════════
 HOW TO ADD split_key SUPPORT TO run_cascadestack_v9.py  (one-time)
═══════════════════════════════════════════════════════════════════════
1. In src/experiments/run_cascadestack_v9.py, find every line that reads:
       data["random_split"]
   and replace it with:
       data.get(cfg.get("split_key", "random_split")) or data["random_split"]

2. In configs/experiment/cascadestack_v9.yaml, add ONE line at the bottom:
       split_key: random_split

   (Default is random_split — existing behavior is 100% unchanged.)

3. Re-run phylo eval:
       python src/run_experiment.py +experiment=phylogeny_split_eval

The change is backward-compatible and adds ~2 lines to v9.1.
═══════════════════════════════════════════════════════════════════════
"""


def _v9_supports_split_key() -> bool:
    """Return True if the v9.1 runner has split_key support."""
    try:
        mod = importlib.import_module("src.experiments.run_cascadestack_v9")
        src = inspect.getsource(mod)
        return "split_key" in src
    except Exception:
        return False


def run_cascadestack_on_split(
    cfg:         DictConfig,
    split_key:   str,
    output_base: Path,
    n_bootstrap: int = 1000,
    seed:        int = 42,
) -> Optional[Dict[str, Dict]]:
    """
    Run CascadeStack v9.1 on the given split key by calling v9's run() with
    a modified config.  Returns per-AB results dict or None on failure.
    """
    if not _v9_supports_split_key():
        log.error(_SPLIT_KEY_INSTRUCTIONS)
        return None

    try:
        v9_module = importlib.import_module("src.experiments.run_cascadestack_v9")
    except ImportError as exc:
        log.error(f"Cannot import run_cascadestack_v9: {exc}")
        return None

    run_output_dir = output_base / f"v9_{split_key}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Build modified config: same as current cfg but override split and output dir
    base_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    base_cfg_dict.update({
        "split_key": split_key,
        "output_dir": str(run_output_dir),
    })
    v9_cfg = OmegaConf.create(base_cfg_dict)

    log.info(f"Running CascadeStack v9.1 on split='{split_key}' → {run_output_dir}")
    t0 = time.time()

    try:
        v9_module.run(v9_cfg)
    except Exception as exc:
        log.exception(f"v9.1 run on {split_key} failed: {exc}")
        return None

    elapsed = (time.time() - t0) / 60
    log.info(f"  v9.1/{split_key} complete in {elapsed:.1f} min")

    # Try to load results from the output directory
    # The v9.1 runner saves a results CSV — try common names
    for fname in ["test_results.csv", "results.csv", "cascade_results.csv",
                  "v9_results.csv"]:
        res_csv = run_output_dir / fname
        if res_csv.exists():
            df = pd.read_csv(res_csv)
            # Parse per-AB PR-AUC from the CSV
            # Assuming format: ab_name column + prauc column (check your v9.1 output)
            if "ab" in df.columns and "prauc" in df.columns:
                return {
                    row["ab"]: {
                        "prauc": float(row["prauc"]),
                        "ci_lo": float(row.get("ci_lo", np.nan)),
                        "ci_hi": float(row.get("ci_hi", np.nan)),
                    }
                    for _, row in df.iterrows()
                    if row["ab"] in AB_NAMES
                }

    log.warning(
        f"Could not locate v9.1 results CSV in {run_output_dir}.  "
        f"Check the output directory for CSV files and update the loader."
    )
    return None


# ─── Comparison Table ─────────────────────────────────────────────────────────

def build_comparison_table(
    xgb_random:    Dict[str, Dict],
    xgb_phylo:     Dict[str, Dict],
    v9_random:     Dict[str, Dict],    # locked from session context
    v9_phylo:      Optional[Dict[str, Dict]],
) -> pd.DataFrame:
    """
    Build the per-AB comparison table:
    | AB | XGB random | XGB phylo | v9.1 random | v9.1 phylo | Δ (leakage) |
    """
    rows = []
    for ab in AB_NAMES:
        xgb_r_pr  = xgb_random.get(ab, {}).get("prauc", np.nan)
        xgb_p_pr  = xgb_phylo.get(ab,  {}).get("prauc", np.nan)
        v9_r_pr   = v9_random.get(ab,   {}).get("prauc", np.nan)
        v9_p_pr   = v9_phylo.get(ab,    {}).get("prauc", np.nan) \
                    if v9_phylo else np.nan
        delta_xgb = xgb_r_pr - xgb_p_pr \
                    if not (np.isnan(xgb_r_pr) or np.isnan(xgb_p_pr)) else np.nan
        delta_v9  = v9_r_pr - v9_p_pr \
                    if not (np.isnan(v9_r_pr) or np.isnan(v9_p_pr)) else np.nan

        rows.append({
            "ab":                 ab,
            "xgb_gene_random":    round(xgb_r_pr, 5),
            "xgb_gene_phylo":     round(xgb_p_pr, 5) if not np.isnan(xgb_p_pr) else np.nan,
            "v9_random":          round(v9_r_pr, 5)  if not np.isnan(v9_r_pr)  else np.nan,
            "v9_phylo":           round(v9_p_pr, 5)  if not np.isnan(v9_p_pr)  else np.nan,
            "delta_xgb_leakage":  round(delta_xgb, 5) if not np.isnan(delta_xgb) else np.nan,
            "delta_v9_leakage":   round(delta_v9, 5)  if not np.isnan(delta_v9)  else np.nan,
        })

    df = pd.DataFrame(rows)

    # Averages row
    avg_row = {"ab": "AVG"}
    for col in df.columns[1:]:
        avg_row[col] = round(float(df[col].dropna().mean()), 5) \
                       if df[col].dropna().shape[0] > 0 else np.nan
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df


def write_summary(
    df:         pd.DataFrame,
    v9_phylo:   Optional[Dict],
    output_dir: Path,
) -> None:
    """Write SUMMARY.md with honest leakage assessment."""
    avg_row   = df[df["ab"] == "AVG"].iloc[0]
    v9_r_avg  = V9_RANDOM_AVG
    v9_p_avg  = float(avg_row.get("v9_phylo", np.nan))
    xgb_r_avg = float(avg_row.get("xgb_gene_random", np.nan))
    xgb_p_avg = float(avg_row.get("xgb_gene_phylo",  np.nan))

    leakage_v9  = v9_r_avg  - v9_p_avg  if not np.isnan(v9_p_avg)  else np.nan
    leakage_xgb = xgb_r_avg - xgb_p_avg if not np.isnan(xgb_p_avg) else np.nan

    v9_phylo_run = v9_phylo is not None

    lines = [
        "# EXP-1: Phylogeny-Aware Split Re-evaluation",
        "",
        "## Purpose",
        "",
        "Checks whether our headline PR-AUC on random_split is inflated by "
        "phylogenetic leakage — related strains in both train and test being "
        "predictable from each other rather than from genuine AMR signals.",
        "",
        "---",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "---",
        "",
        "## Leakage Assessment",
        "",
    ]

    if not np.isnan(leakage_xgb):
        lines += [
            f"**Gene-agg XGB leakage estimate:** "
            f"{leakage_xgb:+.4f} "
            f"(random={xgb_r_avg:.4f} → phylo={xgb_p_avg:.4f})",
        ]

    if not np.isnan(leakage_v9):
        lines += [
            f"**CascadeStack v9.1 leakage estimate:** "
            f"{leakage_v9:+.4f} "
            f"(random={v9_r_avg:.4f} → phylo={v9_p_avg:.4f})",
        ]
    elif not v9_phylo_run:
        lines += [
            f"**CascadeStack v9.1 phylo split:** NOT RUN YET.  "
            f"Add split_key support to v9.1 runner first (see instructions above).",
        ]

    lines += [""]

    # Interpretation
    if not np.isnan(leakage_v9):
        if leakage_v9 > 0.05:
            assessment = (
                "**High leakage detected** (Δ > 0.05).  "
                "The random_split headline number is inflated by phylogenetic "
                "signal.  **The paper MUST report the phylo_split result as "
                f"the primary headline: v9.1 = {v9_p_avg:.4f} avg PR-AUC.**  "
                "The random_split result may be reported as an optimistic upper "
                "bound or omitted entirely.  Consider retitling the contribution "
                "around the phylo-corrected number."
            )
        elif leakage_v9 > 0.02:
            assessment = (
                f"**Moderate leakage** (Δ = {leakage_v9:.4f}).  "
                "Some phylogenetic inflation is present but limited.  "
                "**Recommendation: report phylo_split as primary result "
                f"({v9_p_avg:.4f}) and random_split as secondary / optimistic "
                f"upper bound ({v9_r_avg:.4f}).** "
                "The conclusion of the paper is unchanged; adjust claims accordingly."
            )
        elif leakage_v9 > 0.0:
            assessment = (
                f"**Low leakage** (Δ = {leakage_v9:.4f}).  "
                "Minimal phylogenetic inflation — results are robust.  "
                "**Recommendation: report both splits.  Use phylo_split as primary "
                f"({v9_p_avg:.4f}) with random_split as secondary ({v9_r_avg:.4f}).** "
                "This strengthens the paper: we show the result holds under the "
                "stricter evaluation protocol."
            )
        else:
            assessment = (
                f"**Negative / no leakage** (Δ = {leakage_v9:.4f}).  "
                "The phylo split performs as well or better than random split — "
                "the model is genuinely generalising to phylogenetically distinct "
                "strains.  **Report both splits; headline with phylo_split.**  "
                "This is the strongest possible reviewer answer."
            )
    else:
        assessment = (
            "CascadeStack v9.1 phylo split results not yet available.  "
            "Add split_key support per the instructions above and re-run.  "
            "XGB leakage (if computed) provides a lower-bound estimate."
        )

    lines += [
        assessment,
        "",
        "---",
        "",
        "## Recommendation for Paper",
        "",
        "| Number to Report | Value | Context |",
        "|---|---|---|",
    ]

    if not np.isnan(v9_p_avg):
        lines.append(
            f"| **Primary (phylo split)** | **{v9_p_avg:.4f}** | "
            "Leakage-corrected; should be main headline |"
        )
    lines += [
        f"| Secondary (random split) | {v9_r_avg:.4f} | "
        "Optimistic upper bound; valid for comparison with literature |",
        f"| XGB gene-agg (random)  | {xgb_r_avg:.4f} | Stage 1.5 baseline |",
    ]
    if not np.isnan(xgb_p_avg):
        lines.append(
            f"| XGB gene-agg (phylo)   | {xgb_p_avg:.4f} | Phylo-corrected baseline |"
        )

    lines += [
        "",
        "**Per CLSI and genomics-methods reviewer standards:**  "
        "the phylo-aware split result is the scientifically appropriate "
        "primary metric for a clinical genomics paper.  "
        "Reporting only random_split will invite mandatory revision requests.",
        "",
        "---",
        "*Generated by run_phylogeny_split_eval.py*",
    ]

    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved SUMMARY.md")


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    if OmegaConf.select(cfg, "output_dir") is None:
        try: cfg = cfg.experiment
        except: pass
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    """Hydra entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Phylogeny split eval  →  {output_dir.resolve()}")

    n_boot = int(cfg.get("n_bootstrap", 1000))
    seed   = int(cfg.get("seed",         42))

    # ── Load data ────────────────────────────────────────────────────────────
    data = load_data(cfg.data_pkl)

    # ── Build gene feature matrix ─────────────────────────────────────────────
    cache_dir = output_dir / "cache"
    X_feat = _build_or_load_features(data, cache_dir)

    # ── Gene-agg XGB on random_split (re-run for CI, compare with locked) ────
    log.info("─── XGB gene-agg on random_split ────────────────────────────────")
    xgb_random = run_gene_agg_xgb(
        data, X_feat, "random_split", n_boot, seed
    )

    # ── Gene-agg XGB on phylogeny_split ───────────────────────────────────────
    log.info("─── XGB gene-agg on phylogeny_split ─────────────────────────────")
    xgb_phylo = run_gene_agg_xgb(
        data, X_feat, "phylogeny_split", n_boot, seed
    )

    # ── CascadeStack v9.1 on phylogeny_split (if split_key support added) ────
    v9_phylo = None
    if cfg.get("run_v9_phylo", True):
        log.info("─── CascadeStack v9.1 on phylogeny_split ────────────────────")
        if not _v9_supports_split_key():
            log.warning(
                "Skipping v9.1 phylo run — split_key not yet supported.\n"
                + _SPLIT_KEY_INSTRUCTIONS
            )
        else:
            v9_phylo = run_cascadestack_on_split(
                cfg, "phylogeny_split", output_dir, n_boot, seed
            )
    else:
        log.info("run_v9_phylo=False in config — skipping v9.1 phylo run")

    # Locked random_split v9.1 results as reference dict
    v9_random = {ab: {"prauc": pr, "ci_lo": np.nan, "ci_hi": np.nan}
                 for ab, pr in V9_RANDOM_PRAUC.items()}

    # ── Build comparison table ────────────────────────────────────────────────
    df_cmp = build_comparison_table(xgb_random, xgb_phylo, v9_random, v9_phylo)
    df_cmp.to_csv(output_dir / "split_comparison.csv", index=False)
    log.info("Saved split_comparison.csv")

    # Save per-split XGB results
    xgb_rows = []
    for split_key, res in [("random_split", xgb_random), ("phylogeny_split", xgb_phylo)]:
        for ab, r in res.items():
            xgb_rows.append({"split": split_key, "ab": ab, **r})
    pd.DataFrame(xgb_rows).to_csv(output_dir / "xgb_gene_results.csv", index=False)
    log.info("Saved xgb_gene_results.csv")

    # ── SUMMARY.md ────────────────────────────────────────────────────────────
    write_summary(df_cmp, v9_phylo, output_dir)

    # Console summary
    log.info("\n── LEAKAGE SUMMARY ─────────────────────────────────────────────")
    xgb_r_avg = float(np.nanmean([r["prauc"] for r in xgb_random.values()]))
    xgb_p_avg = float(np.nanmean([r["prauc"] for r in xgb_phylo.values()]))
    log.info(
        f"  XGB gene-agg:       random={xgb_r_avg:.4f}  phylo={xgb_p_avg:.4f}  "
        f"Δ={xgb_r_avg - xgb_p_avg:+.4f}"
    )
    if v9_phylo:
        v9_p_avg = float(np.nanmean([r["prauc"] for r in v9_phylo.values()]))
        log.info(
            f"  CascadeStack v9.1:  random={V9_RANDOM_AVG:.4f}  "
            f"phylo={v9_p_avg:.4f}  "
            f"Δ={V9_RANDOM_AVG - v9_p_avg:+.4f}"
        )
    else:
        log.info(
            f"  CascadeStack v9.1:  random={V9_RANDOM_AVG:.4f}  "
            f"phylo=NOT RUN  (add split_key support per SUMMARY.md)"
        )
    log.info("────────────────────────────────────────────────────────────────")

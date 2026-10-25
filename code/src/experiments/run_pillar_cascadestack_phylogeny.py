"""
src/experiments/run_pillar_cascadestack_phylogeny.py
=====================================================
CascadeStack v9.1 on `phylogeny_split` (Iftu/committee gap — the thesis's own
phylogeny_split_eval/SUMMARY.md explicitly says "CascadeStack v9.1 phylo
split: NOT RUN YET"). Without this number, TriLens T6's phylogeny_split
result (T7, avg PR-AUC 0.7493) cannot be compared against its classical-only
counterpart, so the "does the neural architecture reduce lineage-shortcut
reliance" question is unanswerable.

This script imports and calls `src.experiments.run_cascadestack_v9.run()`
directly — nothing in that file is modified. It only builds a config with
`split_key: phylogeny_split` and a new `output_dir`, mirroring exactly how
the existing SEED 123 / SEED 456 runs were produced (confirmed by diffing
their two saved `config_used.yaml` files: the ONLY difference between those
two runs is the `seed:` value — everything else identical). Run with the
same two seeds (123, 456) for direct comparability with the existing
random_split results.

METHODOLOGICAL CAVEAT (verified from the code, stated honestly): `cfg.seed`
only reaches `set_global_seed()` (random/numpy/torch global state). Every
classical base learner's own randomness is HARDCODED in
run_cascadestack_v9.py regardless of cfg.seed:
  - StratifiedKFold/StratifiedShuffleSplit fold splits: random_state=42
  - RandomForestClassifier, LogisticRegression (base + meta): random_state=42
  - LGBMClassifier: random_state=42
  - XGBoost (_train_xgb_gene, _train_xgb_raw_chi2): no seed param passed at
    all (own internal RNG, not controlled by cfg.seed)
So the "two seeds" comparison for CascadeStack, both here and in the
existing random_split SEED 123/456 results, reflects variability almost
entirely from the neural GeneAttentionNet component's training stochasticity
(weight init, dropout, batch shuffling) plus XGBoost's own uncontrolled
internal randomness — NOT from re-shuffled classical-learner folds. This is
worth stating explicitly in the thesis rather than implying "two independent
full re-runs of every component."

NOT RUN by this session (GPU-heavy, ~20-40 min per seed, same order as the
original v9.1 headline runs) — write-only, per instructions.

Usage (run each seed separately or let main() loop both):
    conda activate amr
    python -m src.experiments.run_pillar_cascadestack_phylogeny
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.experiments.run_cascadestack_v9 import run as run_cascadestack_v9

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "pillar_cascadestack_phylogeny"
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
CASCADESTACK_CFG_PATH = ROOT / "configs" / "experiment" / "stage3_cascadestack_v9.yaml"
SEEDS = [123, 456]


def _build_cfg(seed: int):
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    exp_cfg = OmegaConf.load(CASCADESTACK_CFG_PATH)
    exp_cfg.split_key = "phylogeny_split"
    exp_cfg.output_dir = str(OUTPUT_DIR / f"SEED_{seed}")

    cfg = OmegaConf.merge(base_cfg, {"experiment": exp_cfg})
    cfg.seed = seed
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))


def run_one_seed(seed: int) -> None:
    print(f"\n{'='*80}\n  CascadeStack v9.1 on phylogeny_split — seed={seed}\n{'='*80}")
    cfg = _build_cfg(seed)
    run_cascadestack_v9(cfg)


def _aggregate() -> None:
    """Average the two seeds' v9_results.csv and compare against T6's T7
    (phylogeny_split) number already in results/trilens_v10/ablation_results.csv."""
    per_seed = []
    for seed in SEEDS:
        p = OUTPUT_DIR / f"SEED_{seed}" / "v9_results.csv"
        if not p.exists():
            print(f"  Missing {p} — run both seeds before aggregating.")
            return
        df = pd.read_csv(p)
        df["seed"] = seed
        per_seed.append(df)

    all_df = pd.concat(per_seed, ignore_index=True)
    per_ab_avg = all_df.groupby("antibiotic")["ensemble_pr_auc"].mean()
    overall_avg = float(per_ab_avg.mean())

    t7_path = RESULTS_DIR / "trilens_v10" / "ablation_results.csv"
    t7_per_ab = {}
    t7_avg = None
    if t7_path.exists():
        t7_df = pd.read_csv(t7_path)
        t7_rows = t7_df[(t7_df["ablation"] == "T7") & (t7_df["split_key"] == "phylogeny_split")]
        t7_per_ab = dict(zip(t7_rows["ab"], t7_rows["prauc"]))
        if len(t7_rows):
            t7_avg = float(t7_rows["prauc"].mean())

    lines = ["# CascadeStack v9.1 vs TriLens T6/T7 — phylogeny_split comparison\n"]
    lines.append(
        f"CascadeStack v9.1 (2-seed avg, {SEEDS}) on `phylogeny_split`: "
        f"avg ensemble PR-AUC = **{overall_avg:.4f}**.\n"
    )
    if t7_avg is not None:
        delta = t7_avg - overall_avg
        lines.append(f"TriLens T7 (phylogeny_split, seed=42 only): avg PR-AUC = **{t7_avg:.4f}**.\n")
        lines.append(f"**Delta (T7 - CascadeStack-phylo): {delta:+.4f}**\n")
        if delta > 0:
            lines.append(
                "TriLens generalizes better under phylogeny-aware evaluation — "
                "this is the strongest available justification for the neural "
                "architecture (it relies less on lineage-correlated shortcuts "
                "than the classical-only stack)."
            )
        else:
            lines.append(
                "TriLens does NOT beat CascadeStack under phylogeny-aware "
                "evaluation — report this honestly: neither architecture shows "
                "a generalization advantage on this split, which is itself an "
                "important finding for the thesis (no evidence the neural "
                "branch reduces lineage-shortcut reliance)."
            )
    else:
        lines.append("T7 (phylogeny_split) row not found in ablation_results.csv — cannot compare yet.")

    lines.append("\n## Per-antibiotic comparison\n")
    lines.append("| Antibiotic | CascadeStack-phylo (2-seed avg) | TriLens T7 | Delta (T7 - CascadeStack) |")
    lines.append("|---|---|---|---|")
    for ab in per_ab_avg.index:
        cs = per_ab_avg[ab]
        t7 = t7_per_ab.get(ab)
        if t7 is not None:
            lines.append(f"| {ab} | {cs:.4f} | {t7:.4f} | {t7 - cs:+.4f} |")
        else:
            lines.append(f"| {ab} | {cs:.4f} | n/a | n/a |")

    (OUTPUT_DIR / "COMPARISON_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'COMPARISON_SUMMARY.md'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        run_one_seed(seed)
    _aggregate()


if __name__ == "__main__":
    main()

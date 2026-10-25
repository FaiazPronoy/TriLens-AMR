"""
src/experiments/run_pillar_t6_multiseed.py
============================================
TriLens T6 with 2 additional seeds (123, 456) -- statistical rigor gap: T6's
headline number (0.8462) has only ever been run at seed=42, while
CascadeStack v9.1 has 2 seeds (123, 456). This runs the FULL T6 pipeline
(3-expert MoE neural stack + 5-fold OOF + 5 classical bases + per-AB LogReg
meta) at each new seed, via `src.experiments.run_trilens_v10.run()` (imported
directly -- that file is not modified), leaving the existing headline
seed=42 run (results/trilens_v10/) completely untouched (separate output
directories per seed here).

METHODOLOGICAL NOTES (stated explicitly, not glossed over):
1. `kfold_seed` is intentionally left at 42 for every run (per
   configs/experiment/trilens_v10.yaml's own comment: "MUST stay 42 --
   identical to v9.1 fold splits"). Only `training.seed` is overridden per
   run. This means the K-fold partition of train_idx into fit/early-stop/OOF
   groups is IDENTICAL across all 3 T6 seeds; what varies is neural network
   weight initialization, dropout, and batch-order stochasticity --
   consistent with how CascadeStack v9.1's own 2 seeds vary (see
   run_pillar_cascadestack_phylogeny.py's docstring for the same point about
   CascadeStack).
2. The 5 classical base learners' predictions are loaded from the ORIGINAL,
   fixed `results/stage3_cascadestack_v9/predictions_saved.npz` for every
   seed (cfg.v9_predictions path is unchanged) -- so the classical half of
   the T6 meta-stack input is identical across all 3 seeds; only the neural
   TriLens (Lens A/B/C + MoE gate) component varies by seed. This is an
   honest limitation: this is not "3 fully independent end-to-end reruns,"
   it isolates seed-variance specifically in the neural branch.
3. Only ablation T6 is rerun per seed (not T7/phylogeny_split) to keep
   runtime bounded, since the task is about random_split statistical rigor.

Runtime: ~2 hours per seed (matches the original T6 K-fold OOF run's ~38.6
min core training time plus this project's typical per-antibiotic overhead
at this scale) -> ~4 hours total for 2 seeds. NOT run by this session --
write-only, per instructions (heaviest task, lowest priority).

Usage:
    conda activate amr
    python -m src.experiments.run_pillar_t6_multiseed
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats as scipy_stats

from src.experiments.run_trilens_v10 import run as run_trilens_v10

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "pillar_t6_multiseed"
CONFIG_YAML_PATH = ROOT / "configs" / "config.yaml"
TRILENS_CFG_PATH = ROOT / "configs" / "experiment" / "trilens_v10.yaml"
SEEDS = [123, 456]


def _build_cfg(seed: int):
    base_cfg = OmegaConf.load(CONFIG_YAML_PATH)
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    trilens_cfg = OmegaConf.load(TRILENS_CFG_PATH)
    cfg = OmegaConf.merge(base_cfg, trilens_cfg)
    cfg.training.seed = seed
    cfg.ablations = ["T6"]
    cfg.output_dir = str(OUTPUT_DIR / f"SEED_{seed}")
    # kfold_seed intentionally left unchanged (stays 42) -- see docstring.
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))


def run_one_seed(seed: int) -> None:
    print(f"\n{'='*80}\n  TriLens T6 -- seed={seed}\n{'='*80}")
    cfg = _build_cfg(seed)
    run_trilens_v10(cfg)


def _aggregate() -> None:
    """Combine seed=42 (existing headline), seed=123, seed=456 into
    mean +/- std per antibiotic and overall, plus a paired bootstrap
    significance test (seed-level, n=3) against CascadeStack v9.1's own
    2-seed average."""
    seed_results = {}

    headline_path = RESULTS_DIR / "trilens_v10" / "ablation_results.csv"
    if headline_path.exists():
        hdf = pd.read_csv(headline_path)
        t6_42 = hdf[(hdf["ablation"] == "T6") & (hdf["split_key"] == "random_split")]
        seed_results[42] = dict(zip(t6_42["ab"], t6_42["prauc"]))

    for seed in SEEDS:
        p = OUTPUT_DIR / f"SEED_{seed}" / "ablation_results.csv"
        if not p.exists():
            print(f"  Missing {p} -- run all seeds before aggregating.")
            return
        df = pd.read_csv(p)
        t6 = df[(df["ablation"] == "T6") & (df["split_key"] == "random_split")]
        seed_results[seed] = dict(zip(t6["ab"], t6["prauc"]))

    all_seeds = sorted(seed_results.keys())
    label_names = sorted(seed_results[all_seeds[0]].keys())

    rows = []
    for ab in label_names:
        vals = [seed_results[s][ab] for s in all_seeds]
        rows.append({
            "antibiotic": ab,
            **{f"seed_{s}": seed_results[s][ab] for s in all_seeds},
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_DIR / "t6_multiseed_results.csv", index=False)

    overall_mean = df_out["mean"].mean()
    overall_std = float(np.std([np.mean([seed_results[s][ab] for ab in label_names]) for s in all_seeds], ddof=1)) if len(all_seeds) > 1 else 0.0

    lines = ["# TriLens T6 -- multi-seed (42, 123, 456) results\n"]
    lines.append(f"**Overall avg PR-AUC across {len(all_seeds)} seeds: {overall_mean:.4f} +/- {overall_std:.4f}**\n")
    lines.append("## Per-antibiotic (mean +/- std across seeds)\n")
    lines.append("| Antibiotic | " + " | ".join(f"seed={s}" for s in all_seeds) + " | Mean | Std |")
    lines.append("|---|" + "---|" * len(all_seeds) + "---|---|")
    for _, r in df_out.iterrows():
        seed_vals = " | ".join(f"{r[f'seed_{s}']:.4f}" for s in all_seeds)
        lines.append(f"| {r['antibiotic']} | {seed_vals} | {r['mean']:.4f} | {r['std']:.4f} |")
    lines.append("")

    # Paired bootstrap significance vs CascadeStack v9.1 (2-seed avg), at the
    # per-antibiotic mean-across-seeds level -- n is small (2 or 3 seeds), so
    # this is a coarse seed-level test, reported honestly as such.
    v9_seed_dirs = [RESULTS_DIR / "stage3_cascadestack_v9" / f"SEED {s}" for s in (123, 456)]
    v9_per_ab = {}
    for d in v9_seed_dirs:
        p = d / "v9_results.csv"
        if p.exists():
            vdf = pd.read_csv(p)
            for _, r in vdf.iterrows():
                v9_per_ab.setdefault(r["antibiotic"], []).append(r["ensemble_pr_auc"])
    if v9_per_ab and all(ab in v9_per_ab for ab in label_names):
        diffs = []
        for ab in label_names:
            t6_vals = [seed_results[s][ab] for s in all_seeds]
            v9_vals = v9_per_ab[ab]
            diffs.append(np.mean(t6_vals) - np.mean(v9_vals))
        diffs = np.array(diffs)
        if len(diffs) > 1:
            t_stat, p_val = scipy_stats.ttest_1samp(diffs, 0.0)
            lines.append("## T6 (3-seed) vs CascadeStack v9.1 (2-seed) -- per-antibiotic paired t-test on the mean deltas\n")
            lines.append(f"Mean delta (T6 - v9.1) across antibiotics: {diffs.mean():+.4f}, "
                         f"paired t-test t={t_stat:.3f}, p={p_val:.4f} "
                         f"(n={len(diffs)} antibiotics, NOT n=seeds -- a small-sample, "
                         f"antibiotic-level test on seed-averaged values, stated honestly "
                         f"as a coarse check, not a rigorous seed-level bootstrap).\n")
    else:
        lines.append("(CascadeStack v9.1 per-antibiotic seed data not found for comparison.)\n")

    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY.md'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        run_one_seed(seed)
    _aggregate()


if __name__ == "__main__":
    main()

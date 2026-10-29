from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GIESSEN_MIN_DROP = 0.05
ARCADIA_MAX_DROP = 0.02


def shuffle_drops(df: pd.DataFrame) -> dict:
    ok = df[df["status"] == "ok"]
    drops = {}
    for ds in ("arcadia", "giessen"):
        for ab in ok["antibiotic"].unique():
            sel = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab)]
            orig = sel[sel["variant"] == "orig"]
            shuf = sel[sel["variant"] == "shuf"]
            if len(orig) and len(shuf):
                drops[f"{ds}_{ab}"] = float(orig["test_pr_auc"].iloc[0] - shuf["test_pr_auc"].iloc[0])
    return drops


def verdict(drops: dict) -> dict:
    arcadia = [v for k, v in drops.items() if k.startswith("arcadia")]
    giessen = [v for k, v in drops.items() if k.startswith("giessen")]
    if not arcadia or not giessen:
        return {"verdict": "INCONCLUSIVE (a dataset is missing)", "per_dataset_drops": drops}
    a, g = float(np.mean(arcadia)), float(np.mean(giessen))
    asymmetric = g >= GIESSEN_MIN_DROP and a <= ARCADIA_MAX_DROP
    return {
        "verdict": "ASYMMETRIC" if asymmetric else "INCONCLUSIVE",
        "arcadia_avg_drop": a,
        "giessen_avg_drop": g,
        "per_dataset_drops": drops,
        "criterion": f"Giessen drop >= {GIESSEN_MIN_DROP} and Arcadia drop <= {ARCADIA_MAX_DROP}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Average precision lost when the variant order is shuffled.")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    a = ap.parse_args()
    out = a.results_dir / "position_shuffle_test"
    runs = out / "runs" / "shuffle_results.csv"
    if not runs.exists():
        raise SystemExit(f"missing {runs}: run the shuffle test first")
    result = verdict(shuffle_drops(pd.read_csv(runs)))
    (out / "verdict.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Position shuffle test",
        "",
        "Average precision on ciprofloxacin with the variant columns in their original order and after a "
        "random permutation. A real positional signal shows as a drop.",
        "",
        "| Run | Drop |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:+.4f} |" for k, v in result["per_dataset_drops"].items()]
    lines += ["", f"Verdict: {result['verdict']} ({result.get('criterion', '')})", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

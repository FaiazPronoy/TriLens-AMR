from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "seed_stability"
CLASSICAL = "classical_stack_lineage"


def seed_dir(seed):
    return OUT / f"seed_{seed}"


def copy_arm(
    src, dst, files=("predictions_saved.npz", "classical_stack_results.csv", "trilens_neural_lineage_results.csv")
):
    dst.mkdir(parents=True, exist_ok=True)
    for name in files:
        if (src / name).exists() and not (dst / name).exists():
            shutil.copy2(src / name, dst / name)


def train(seed):
    import trilens.experiments.trilens_lineage as m

    d = seed_dir(seed) / "trilens_neural_lineage"
    if (d / "predictions_saved.npz").exists():
        print(f"[seed {seed}] neural stack already there: {d}")
        return
    m.SEED, m.OUTPUT_DIR = seed, d
    print(f"[seed {seed}] training the neural stack -> {d}", flush=True)
    m.main()


def stack(seed):
    import trilens.experiments.stack_lineage as f

    d = seed_dir(seed)
    if seed == 42 and not (d / "trilens_neural_lineage").exists():
        src = RES / "trilens_neural_lineage"
        if not (src / "predictions_saved.npz").exists():
            raise SystemExit(f"seed 42: {src} missing - attach the trilens_splits output")
        copy_arm(src, d / "trilens_neural_lineage")
    if not (RES / CLASSICAL / "predictions_saved.npz").exists():
        raise SystemExit(f"{RES / CLASSICAL} missing - attach the classical output")
    copy_arm(RES / CLASSICAL, d / CLASSICAL)
    f.TRILENS_NPZ = d / "trilens_neural_lineage" / "predictions_saved.npz"
    f.TRILENS_CSV = d / "trilens_neural_lineage" / "trilens_neural_lineage_results.csv"
    f.CLASSICAL_NPZ = d / CLASSICAL / "predictions_saved.npz"
    f.CLASSICAL_CSV = d / CLASSICAL / "classical_stack_results.csv"
    f.OUTPUT_DIR = d / "trilens_stack_lineage"
    f.main()


def validity(seed, n_boot):
    cmd = [
        sys.executable,
        "-m",
        "trilens.analysis.holdout_margin",
        "--results",
        str(seed_dir(seed)),
        "--n-boot",
        str(n_boot),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def summary(seeds):
    rows, per = [], []
    for s in seeds:
        d = seed_dir(s) / "diagnostics"
        dl, hv = d / "holdout_margin_deltas.csv", d / "holdout_margin.csv"
        if not dl.exists():
            print(f"[seed {s}] no holdout_margin_deltas.csv yet - skipped")
            continue
        r = pd.read_csv(dl)
        r = r[r.endpoint == "mean_7"].iloc[0]
        v = pd.read_csv(hv)
        mean = v[v.drug == "MEAN"].set_index("model")
        rows.append(
            dict(
                seed=s,
                delta_nap=r.delta_nap,
                ci_low=r.ci_low,
                ci_high=r.ci_high,
                n_st=r.n_st,
                **{f"{k}_nap": mean.loc[k, "nap"] for k in ("stack", "classical", "neural") if k in mean.index},
                **{f"{k}_within_st": mean.loc[k, "within_st_macro"] for k in ("stack", "neural") if k in mean.index},
            )
        )
        per.append(v.assign(seed=s))
    if not rows:
        raise SystemExit("no seed has been read yet")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "summary.csv", index=False)
    pd.concat(per).to_csv(OUT / "per_drug_all_seeds.csv", index=False)
    n_pos = int((df.ci_low > 0).sum())
    v = {
        "seeds": df.seed.tolist(),
        "delta_per_seed": df.delta_nap.round(4).tolist(),
        "ci_low_per_seed": df.ci_low.round(4).tolist(),
        "mean_delta": round(float(df.delta_nap.mean()), 4),
        "sd_delta": round(float(df.delta_nap.std(ddof=1)), 4) if len(df) > 1 else None,
        "seeds_with_ci_above_zero": n_pos,
        "pass_rule": "all 3 seeds read, mean delta > 0 and CI above zero in at least 2 of 3 seeds",
        "complete": len(df) == 3,
    }
    v["primary_pass"] = bool(v["complete"] and v["mean_delta"] > 0 and n_pos >= 2)
    v["reading"] = (
        "stable: the stack's lineage-holdout gain over the classical arm survives the seed"
        if v["primary_pass"]
        else "not stable: the gain is reported as seed-dependent, not as a finding"
        if v["complete"]
        else "incomplete: fewer than 3 seeds read"
    )
    (OUT / "verdict.json").write_text(json.dumps(v, indent=1))
    print(df.round(4).to_string(index=False))
    print(json.dumps(v, indent=1))
    return v


def main():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="123,456")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--train-only", action="store_true", help="train the given seeds, nothing else")
    a = ap.parse_args()
    new = [int(s) for s in a.seeds.split(",") if s]
    seeds = [42] + [s for s in new if s != 42]
    if not a.summary_only:
        for s in new:
            train(s)
        if a.train_only:
            return
        for s in seeds:
            stack(s)
            validity(s, a.n_boot)
    summary(seeds)


if __name__ == "__main__":
    main()

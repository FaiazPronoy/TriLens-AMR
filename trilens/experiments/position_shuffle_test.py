from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from trilens.data.arcadia_snp_loader import (
    ArcadiaSNPSingleABDataset,
    load_arcadia_raw_for_antibiotic,
)
from trilens.data.giessen_loader import (
    GiessenSingleABDataset,
    arcadia_to_giessen_ab,
    load_giessen_for_antibiotic,
)
from trilens.data.snp_transforms import make_column_permutation
from trilens.evaluation.training_helpers import (
    error_row,
    make_loader,
    maybe_wandb_logger,
    pos_weight_from_labels,
    result_to_row,
    train_and_evaluate,
)
from trilens.models.plain_cnn import Naive1DCNN
from trilens.utils.seeds import set_global_seed


def _build_naive(n_features: int, ab: str, train_y: np.ndarray, cfg: DictConfig) -> Naive1DCNN:
    return Naive1DCNN(
        n_features=n_features,
        vocab_size=cfg.model.vocab_size,
        padding_idx=cfg.model.padding_idx,
        embed_dim=cfg.model.embed_dim,
        dropout=cfg.model.dropout,
        antibiotic_names=[ab],
        target_antibiotic=ab,
        pos_weights=[pos_weight_from_labels(train_y)],
        learning_rate=cfg.experiment.training.learning_rate,
        weight_decay=cfg.experiment.training.weight_decay,
        scheduler_t_max=cfg.experiment.training.scheduler_t_max,
    )


def _run_arcadia(ab: str, perm: np.ndarray | None, order_tag: str, cfg: DictConfig, output_dir: Path) -> dict:
    bundle = load_arcadia_raw_for_antibiotic(
        snp_pkl_path=cfg.paths.arcadia_snp_matrix_pkl,
        master_pkl_path=cfg.paths.arcadia_master_pkl,
        antibiotic=ab,
        split_type=cfg.data.arcadia.split_type,
    )
    train_ds = ArcadiaSNPSingleABDataset(bundle.train, shuffle_perm=perm)
    val_ds = ArcadiaSNPSingleABDataset(bundle.val, shuffle_perm=perm)
    test_ds = ArcadiaSNPSingleABDataset(bundle.test, shuffle_perm=perm)

    bs = cfg.experiment.training.batch_size_arcadia
    nw = cfg.experiment.training.num_workers
    train_loader = make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = make_loader(test_ds, bs, nw, shuffle=False)

    model = _build_naive(bundle.n_snps, ab, bundle.train.y, cfg)
    run_name = f"arcadia_{ab}_naive_{order_tag}"
    wandb_logger = maybe_wandb_logger(
        cfg,
        run_name,
        extra_tags=["arcadia", ab, order_tag],
        save_dir=output_dir,
    )
    result = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        training_cfg=cfg.experiment.training,
        evaluation_cfg=cfg.experiment.evaluation,
        run_name=run_name,
        ckpt_dir=output_dir / "ckpts",
        wandb_logger=wandb_logger,
    )
    return {"result": result, "n_snps": bundle.n_snps}


def _run_giessen(ab: str, perm: np.ndarray | None, order_tag: str, cfg: DictConfig, output_dir: Path) -> dict:
    giessen_ab = arcadia_to_giessen_ab(ab)
    if giessen_ab is None:
        raise ValueError(f"No Giessen mapping for {ab}")

    bundle = load_giessen_for_antibiotic(
        data_csv_path=cfg.paths.giessen_data_csv,
        pheno_csv_path=cfg.paths.giessen_pheno_csv,
        antibiotic=giessen_ab,
        seed=cfg.seed,
    )
    train_ds = GiessenSingleABDataset(bundle.train, shuffle_perm=perm)
    val_ds = GiessenSingleABDataset(bundle.val, shuffle_perm=perm)
    test_ds = GiessenSingleABDataset(bundle.test, shuffle_perm=perm)

    bs = cfg.experiment.training.batch_size_giessen
    nw = cfg.experiment.training.num_workers
    train_loader = make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = make_loader(test_ds, bs, nw, shuffle=False)

    model = _build_naive(bundle.n_snps, ab, bundle.train.y, cfg)
    run_name = f"giessen_{ab}_naive_{order_tag}"
    wandb_logger = maybe_wandb_logger(
        cfg,
        run_name,
        extra_tags=["giessen", ab, order_tag],
        save_dir=output_dir,
    )
    result = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        training_cfg=cfg.experiment.training,
        evaluation_cfg=cfg.experiment.evaluation,
        run_name=run_name,
        ckpt_dir=output_dir / "ckpts",
        wandb_logger=wandb_logger,
    )
    return {"result": result, "n_snps": bundle.n_snps}


def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ckpts").mkdir(exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config_used.yaml")

    rows: list[dict] = []
    nested: dict[str, dict] = {}
    only = {k.strip() for k in os.environ.get("SHUFFLE_ONLY", "").split(",") if k.strip()}

    def _plain(o):
        return o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))

    for ab in cfg.experiment.antibiotics:
        for dataset in cfg.experiment.datasets:
            for order_tag in ["orig", "shuf"]:
                key = f"{dataset}_{ab}_{order_tag}"
                cache = output_dir / f"{key}_run.json"
                if only and key not in only:
                    continue
                if cache.exists():
                    saved = json.loads(cache.read_text())
                    rows.append(saved["row"])
                    nested[key] = saved["nested"]
                    print(f"  [cached] {key}: PR-AUC {saved['nested']['metrics']['pr_auc']:.4f}")
                    continue
                print("=" * 80)
                print(f"  EXP 1.3 - {dataset.upper()} × {ab} × {order_tag.upper()}")
                print("=" * 80)
                t0 = time.time()

                if order_tag == "shuf":
                    if dataset == "arcadia":
                        n = 535439
                    else:
                        n = 60936
                    perm = make_column_permutation(n, seed=cfg.experiment.shuffle_seed)
                else:
                    perm = None

                try:
                    if dataset == "arcadia":
                        bundle_result = _run_arcadia(
                            ab,
                            perm,
                            order_tag,
                            cfg,
                            output_dir,
                        )
                    elif dataset == "giessen":
                        bundle_result = _run_giessen(
                            ab,
                            perm,
                            order_tag,
                            cfg,
                            output_dir,
                        )
                    else:
                        raise ValueError(f"Unknown dataset {dataset}")
                except Exception as ex:
                    print(f"  ERROR: {ex!r}")
                    rows.append(
                        error_row(
                            dataset,
                            ab,
                            f"cnn_1d_naive_{order_tag}",
                            None,
                            ex,
                        )
                    )
                    continue

                result = bundle_result["result"]
                elapsed = time.time() - t0
                row = result_to_row(
                    dataset,
                    ab,
                    "plain_cnn",
                    order_tag,
                    result,
                    elapsed,
                )
                rows.append(row)

                pred_path = output_dir / f"{key}_predictions.npz"
                np.savez_compressed(pred_path, y_true=result.y_true, y_prob=result.y_prob)
                nested[key] = {
                    "predictions_npz": str(pred_path),
                    "metrics": result.metrics,
                    "pr_auc_ci": list(result.pr_auc_ci),
                    "mcc_ci": list(result.mcc_ci),
                    "best_ckpt": result.best_ckpt,
                }

                cache.write_text(json.dumps({"row": row, "nested": nested[key]}, default=_plain, indent=1))
                m = result.metrics
                print(f"  PR-AUC: {m['pr_auc']:.4f} [{result.pr_auc_ci[0]:.4f}, {result.pr_auc_ci[1]:.4f}]")

    df = pd.DataFrame(rows)
    if only:
        bad = [r for r in rows if r.get("status", "ok") != "ok"]
        if bad or not rows:
            raise SystemExit(f"SHUFFLE_ONLY={sorted(only)}: run failed or key unknown ({bad[:1]})")
        print(f"  finished {sorted(only)}; the summary runs when all four are cached")
        return
    failed = df[df["status"] != "ok"] if "status" in df else df
    if len(failed):
        df.to_csv(output_dir / "shuffle_results_FAILED.csv", index=False)
        raise SystemExit(
            f"{len(failed)} of {len(df)} shuffle-test runs failed; "
            f"see shuffle_results_FAILED.csv (first error: {failed.iloc[0].to_dict()})"
        )
    df.to_csv(output_dir / "shuffle_results.csv", index=False)
    with open(output_dir / "shuffle_results.json", "w") as f:
        json.dump(nested, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("  EXP 1.3 SUMMARY - Asymmetric shuffling drop test")
    print("=" * 80)
    ok = df[df["status"] == "ok"]
    drops: dict[str, float] = {}
    for ds in cfg.experiment.datasets:
        for ab in cfg.experiment.antibiotics:
            orig = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab) & (ok["variant"] == "orig")]
            shuf = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab) & (ok["variant"] == "shuf")]
            if len(orig) and len(shuf):
                orig_pr = orig["test_pr_auc"].iloc[0]
                shuf_pr = shuf["test_pr_auc"].iloc[0]
                drop = orig_pr - shuf_pr
                drops[f"{ds}_{ab}"] = drop
                print(f"  {ds:<10} {ab:<6}  orig={orig_pr:.4f}  shuf={shuf_pr:.4f}  drop={drop:+.4f}")

    if drops:
        arc_drops = [v for k, v in drops.items() if k.startswith("arcadia")]
        gie_drops = [v for k, v in drops.items() if k.startswith("giessen")]
        if arc_drops and gie_drops:
            arc_avg = float(np.mean(arc_drops))
            gie_avg = float(np.mean(gie_drops))
            asym_ok = (gie_avg >= 0.05) and (arc_avg <= 0.02)
            verdict = "ASYMMETRIC" if asym_ok else "INCONCLUSIVE"
            print(f"\n  avg shuffle drop Arcadia = {arc_avg:+.4f}")
            print(f"  avg shuffle drop Giessen = {gie_avg:+.4f}")
            print(f"  Asymmetry verdict: {verdict}")
            print("  Success criteria: Giessen ≥ 0.05, Arcadia ≤ 0.02")

    print(f"\n  Outputs → {output_dir}")

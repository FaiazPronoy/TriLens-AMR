import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("torch")
import trilens.experiments.trilens_stack as stack

N, A = 60, 7
TRAINED = []


@pytest.fixture
def patched(monkeypatch):
    TRAINED.clear()
    split = {"train_idx": np.arange(40), "val_idx": np.arange(40, 50), "test_idx": np.arange(50, 60)}
    monkeypatch.setattr(
        stack, "_get_split_indices", lambda data, key: (split["train_idx"], split["val_idx"], split["test_idx"])
    )
    monkeypatch.setattr(stack, "compute_pos_weights", lambda *a, **k: None)
    monkeypatch.setattr(stack, "_build_dataset", lambda idx, *a: np.asarray(idx))
    monkeypatch.setattr(stack, "_create_model", lambda *a, **k: object())

    def train(model, fit_ds, es_ds, pw, cfg, device):
        TRAINED.append(len(fit_ds))
        return model, [{"epoch": 1, "loss": 0.5}]

    monkeypatch.setattr(stack, "_train_until_convergence", train)

    def predict(model, ds, device, bs, collect_gate=False, collect_attn=False):
        p = (np.asarray(ds)[:, None] / 100.0 + np.zeros((1, A))).astype(np.float32)
        if collect_attn:
            return p, np.ones((len(ds), A, 3), np.float32), np.ones((len(ds), 4)), np.ones((len(ds), 5)), None
        return p, None, None, None, None

    monkeypatch.setattr(stack, "_predict", predict)
    monkeypatch.setattr(stack, "compute_metrics", lambda *a, **k: {"_avg_prauc": 0.5})
    if hasattr(stack, "_representations"):
        monkeypatch.setattr(
            stack, "_representations", lambda model, ds, device, bs: {"z_A": np.zeros((len(ds), 2), np.float32)}
        )
    cfg = OmegaConf.create({"n_folds": 5, "kfold_seed": 42, "training": {"batch_size": 4}})
    data = {"Y": np.zeros((N, A)), "label_mask": np.ones((N, A), bool), "X_pa": None}
    return cfg, data


def run(cfg, data, fold_dir):
    return stack.run_kfold_oof(cfg, data, None, None, None, "random_split", 42, "cpu", fold_dir=fold_dir)


def test_folds_are_reused(patched, tmp_path, monkeypatch):
    cfg, data = patched
    monkeypatch.setenv("TRILENS_ATTENTION", "conditioned")
    first = run(cfg, data, tmp_path)
    assert len(TRAINED) == 5 and len(list(tmp_path.glob("random_split_fold*.npz"))) == 5
    (tmp_path / "random_split_fold3.npz").unlink()
    TRAINED.clear()
    second = run(cfg, data, tmp_path)
    assert len(TRAINED) == 1
    for k in ("oof_preds", "val_preds", "test_preds", "gate_test", "a_A", "a_B"):
        assert np.array_equal(first[k], second[k]), k
    TRAINED.clear()
    (tmp_path / "random_split_fold4.representations.pkl").unlink(missing_ok=True)
    run(cfg, data, tmp_path)
    assert len(TRAINED) == 1
    assert second["a_C"] is None and len(second["histories"]) == 5


def test_other_lens_b_variant_is_not_reused(patched, tmp_path, monkeypatch):
    cfg, data = patched
    monkeypatch.setenv("TRILENS_ATTENTION", "conditioned")
    run(cfg, data, tmp_path)
    TRAINED.clear()
    monkeypatch.setenv("TRILENS_ATTENTION", "standard")
    run(cfg, data, tmp_path)
    assert len(TRAINED) == 5


def test_without_fold_dir_nothing_is_saved(patched, tmp_path):
    cfg, data = patched
    run(cfg, data, None)
    assert len(TRAINED) == 5 and not list(tmp_path.iterdir())


def test_other_presence_matrix_is_not_reused(patched, tmp_path, monkeypatch):
    cfg, data = patched
    monkeypatch.setenv("TRILENS_ATTENTION", "conditioned")
    data["X_pa"] = np.zeros((N, 8), np.uint8)
    run(cfg, data, tmp_path)
    TRAINED.clear()
    data["X_pa"] = np.ones((N, 8), np.uint8)
    run(cfg, data, tmp_path)
    assert len(TRAINED) == 5

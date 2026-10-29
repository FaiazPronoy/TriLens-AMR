from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from trilens.experiments import trilens_stack as stack
from trilens.models.trilens_network import TriLens

CFG = Path(__file__).resolve().parents[1] / "configs" / "experiment" / "trilens_stack.yaml"
G, L, G_PA, N = 12, 8, 30, 20


def _model():
    cfg = OmegaConf.load(CFG)
    cfg.expert_c.n_genes, cfg.expert_a.L_max, cfg.expert_b.n_pa_genes = G, L, G_PA
    torch.manual_seed(0)
    return TriLens(cfg, torch.ones(G, L, dtype=torch.bool)), cfg


def _dataset(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return stack.TriLensDataset(
        X_blocks=rng.integers(0, 3, (n, G, L)).astype(np.int8),
        X_pa=rng.integers(0, 2, (n, G_PA)).astype(np.uint8),
        X_gene_agg=rng.normal(size=(n, G, 4)).astype(np.float32),
        Y=rng.integers(0, 2, (n, 7)).astype(np.float32),
        label_mask=np.ones((n, 7), bool),
    )


def test_forward_returns_representations():
    model, cfg = _model()
    model.eval()
    ds = _dataset(4)
    batch = [torch.stack([torch.as_tensor(ds[i][k]) for i in range(4)]) for k in range(3)]
    out = model(*batch)
    d = int(cfg.d_model)
    for k in ("z_A", "z_B", "z_C"):
        assert out[k].shape == (4, d)
    assert out["z_fused"].shape == (4, 3 * d)
    assert torch.allclose(out["z_fused"], torch.cat([out["z_A"], out["z_B"], out["z_C"]], dim=-1))


def test_representations_and_save(tmp_path):
    model, cfg = _model()
    parts = {
        p: _dataset(n, seed)
        for p, n, seed in (("fit", 8, 1), ("es", 3, 2), ("oof", 4, 3), ("val", 5, 4), ("test", 6, 5))
    }
    rng_before = torch.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        rep = {p: stack._representations(model, ds, torch.device("cpu"), 4) for p, ds in parts.items()}
    assert torch.equal(rng_before, torch.get_rng_state())
    again = {p: stack._representations(model, ds, torch.device("cpu"), 3) for p, ds in parts.items()}
    assert np.allclose(rep["test"]["z_B"], again["test"]["z_B"], atol=1e-6)

    idx = {p: np.arange(len(ds)) + 100 * i for i, (p, ds) in enumerate(parts.items())}
    result = {
        "train_idx": np.r_[idx["fit"], idx["es"], idx["oof"]],
        "val_idx": idx["val"],
        "test_idx": idx["test"],
        "representations": {
            "fold": 4,
            "fold_seed": 442,
            "fit_idx": idx["fit"],
            "es_idx": idx["es"],
            "oof_idx": idx["oof"],
            "init": rep,
            "trained": again,
        },
    }
    out = tmp_path / "representations.npz"
    assert stack.save_representations(result, out)
    z = np.load(out)
    assert z["trained_z_B_train"].shape == (15, int(cfg.d_model))
    assert np.array_equal(z["train_rep_indices"], np.r_[idx["fit"], idx["es"], idx["oof"]])
    assert z["init_z_fused_test"].shape == (6, 3 * int(cfg.d_model))
    assert not stack.save_representations({"representations": None}, tmp_path / "none.npz")

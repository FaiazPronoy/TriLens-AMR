import torch

from src.models.trilens import ExpertB


def _build(strain_conditioned):
    torch.manual_seed(42)
    m = ExpertB(G_pa=500, d_token=16, d_out=64, attn_dim=32, strain_conditioned=strain_conditioned)
    return m, torch.get_rng_state()


def test_switch_changes_only_the_context_projections():
    fixed, rng_fixed = _build(True)
    standard, rng_standard = _build(False)
    sf, ss = fixed.state_dict(), standard.state_dict()
    assert set(sf) - set(ss) == {"attn_Vc.weight", "attn_Uc.weight"}
    for k, v in ss.items():
        assert torch.equal(sf[k], v), k
    assert torch.equal(rng_fixed, rng_standard)


def test_standard_forward_ignores_context():
    torch.manual_seed(0)
    m = ExpertB(G_pa=50, d_token=8, d_out=16, attn_dim=8, strain_conditioned=False)
    x = (torch.rand(4, 50) > 0.5).float()
    z, a, logits = m(x)
    assert a.shape == (4, 50) and logits.shape[0] == 4
    assert torch.allclose(a.sum(dim=1), torch.ones(4), atol=1e-5)

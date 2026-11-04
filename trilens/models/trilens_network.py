from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TriLens", "AB_NAMES", "DRUG_GROUPS"]


AB_NAMES: List[str] = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]


DRUG_GROUPS: Dict[str, List[str]] = {
    "penicillin": ["AMP", "AMC"],
    "cephalosporin": ["CTX", "CAZ"],
    "fluoroquinolone": ["CIP"],
    "aminoglycoside": ["GEN"],
    "folate": ["SXT"],
}


_AB_TO_GROUP: List[int] = [0, 2, 0, 4, 3, 1, 1]


class GatedAttentionPool(nn.Module):
    def __init__(self, d_in: int, attn_dim: int = 32) -> None:
        super().__init__()
        self.V = nn.Linear(d_in, attn_dim, bias=False)
        self.U = nn.Linear(d_in, attn_dim, bias=False)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        e = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))
        a = torch.softmax(e, dim=-1)
        z = (a.unsqueeze(-1) * h).sum(dim=1)
        return z, a


def _flat_heads(d_in: int, n_abs: int) -> nn.ModuleList:
    return nn.ModuleList([nn.Linear(d_in, 1) for _ in range(n_abs)])


def _run_heads(heads: nn.ModuleList, z: torch.Tensor) -> torch.Tensor:
    return torch.cat([h(z) for h in heads], dim=-1)


class ExpertA(nn.Module):
    def __init__(
        self,
        G: int,
        L: int,
        d_out: int,
        conv_channels: List[int],
        kernel_size: int,
        dropout: float,
        attn_dim: int,
        gene_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.G = G
        self.L = L

        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.c_out = conv_channels[-1]

        self.gene_pool = GatedAttentionPool(self.c_out, attn_dim)

        self.proj = nn.Sequential(nn.Linear(self.c_out, d_out), nn.GELU())
        self.ab_heads = _flat_heads(d_out, len(AB_NAMES))

        self.register_buffer("gene_mask", gene_mask.bool())

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, G, L = X.shape
        C = self.c_out

        x = X.view(B * G, 1, L)
        h = self.conv(x)

        mask_bg = self.gene_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * G, L).unsqueeze(1).expand(-1, C, -1)
        h = h.masked_fill(~mask_bg, float("-inf"))

        h = self.pool(h).squeeze(-1)

        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        h = h.view(B, G, C)

        z_pool, a = self.gene_pool(h)

        z = self.proj(z_pool)
        logits = _run_heads(self.ab_heads, z)

        return z, a, logits


class ExpertB(nn.Module):
    def __init__(
        self,
        G_pa: int,
        d_token: int,
        d_out: int,
        attn_dim: int,
        strain_conditioned: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.G_pa = G_pa
        self.d_token = d_token
        if strain_conditioned is None:
            strain_conditioned = os.environ.get("TRILENS_ATTENTION", "conditioned").lower() != "standard"
        self.strain_conditioned = bool(strain_conditioned)

        self.E_pa = nn.Parameter(torch.empty(G_pa, d_token))
        nn.init.xavier_uniform_(self.E_pa)

        self.attn_V = nn.Linear(d_token, attn_dim, bias=False)
        self.attn_U = nn.Linear(d_token, attn_dim, bias=False)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)

        self.proj = nn.Sequential(nn.Linear(d_token, d_out), nn.GELU())
        self.ab_heads = _flat_heads(d_out, len(AB_NAMES))

        if self.strain_conditioned:
            with torch.random.fork_rng(devices=[]):
                self.attn_Vc = nn.Linear(d_token, attn_dim, bias=False)
                self.attn_Uc = nn.Linear(d_token, attn_dim, bias=False)

    def forward(self, X_pa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weighted = self.E_pa.unsqueeze(0) * X_pa.unsqueeze(-1)

        v = self.attn_V(weighted)
        u = self.attn_U(weighted)
        if self.strain_conditioned:
            n_present = X_pa.sum(dim=1, keepdim=True).clamp(min=1.0)
            ctx = weighted.sum(dim=1) / n_present

            v = v + self.attn_Vc(ctx).unsqueeze(1)
            u = u + self.attn_Uc(ctx).unsqueeze(1)
        e = self.attn_w(torch.tanh(v) * torch.sigmoid(u)).squeeze(-1)

        present_mask = X_pa > 0.5
        e_masked = e.masked_fill(~present_mask, float("-inf"))

        no_present = ~present_mask.any(dim=1, keepdim=True)
        e_safe = torch.where(
            no_present.expand_as(e_masked),
            torch.zeros_like(e_masked),
            e_masked,
        )

        a_B = torch.softmax(e_safe.float(), dim=-1).to(weighted.dtype)

        a_B = a_B.masked_fill(no_present.expand_as(a_B), 0.0)

        z_pre = (a_B.unsqueeze(-1) * weighted).sum(dim=1)

        z = self.proj(z_pre)
        logits = _run_heads(self.ab_heads, z)

        return z, a_B, logits


class ExpertC(nn.Module):
    def __init__(
        self,
        G: int,
        n_stats: int,
        d_model: int,
        attn_dim: int,
        n_context_layers: int,
        dropout: float,
        use_coresistance: bool,
    ) -> None:
        super().__init__()
        self.G = G
        self.n_context_layers = n_context_layers
        self.use_coresistance = use_coresistance

        self.encoder = nn.Sequential(
            nn.Linear(n_stats, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        if n_context_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            try:
                self.transformer: Optional[nn.Module] = nn.TransformerEncoder(
                    enc_layer, num_layers=n_context_layers, enable_nested_tensor=False
                )
            except TypeError:
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_context_layers)
            self.pos_embed = nn.Embedding(G, d_model)
        else:
            self.transformer = None
            self.pos_embed = None

        self.gene_pool = GatedAttentionPool(d_model, attn_dim)

        self.trunk = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout))
        self.group_projs = nn.ModuleList([nn.Sequential(nn.Linear(32, 16), nn.GELU()) for _ in range(5)])
        self.ab_classifiers = _flat_heads(16, len(AB_NAMES))
        self.register_buffer("ab_to_group", torch.tensor(_AB_TO_GROUP, dtype=torch.long))

        if use_coresistance:
            self.coresistance_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(d_model + len(AB_NAMES), 32),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(32, len(AB_NAMES)),
            )
            nn.init.zeros_(self.coresistance_mlp[-1].weight)
            nn.init.zeros_(self.coresistance_mlp[-1].bias)
            self.alpha = nn.Parameter(torch.tensor([0.1]))

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h = self.encoder(X)

        if self.n_context_layers > 0:
            pos = torch.arange(self.G, device=X.device)
            h = h + self.pos_embed(pos)
            h = self.transformer(h)

        z_pool, a = self.gene_pool(h)

        trunk_out = self.trunk(z_pool)
        group_feats = [proj(trunk_out) for proj in self.group_projs]

        logits_init = torch.stack(
            [self.ab_classifiers[i](group_feats[self.ab_to_group[i]]).squeeze(-1) for i in range(len(AB_NAMES))],
            dim=1,
        )

        if self.use_coresistance:
            ctx = torch.cat([z_pool, logits_init.detach()], dim=-1)
            r = self.coresistance_mlp(ctx)
            logits = logits_init + self.alpha * r
        else:
            logits = logits_init

        return z_pool, a, logits


class MoEGate(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        n_experts: int,
        n_abs: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout))
        self.gate_heads = nn.ModuleList([nn.Linear(d_hidden, n_experts) for _ in range(n_abs)])

    def forward(self, z_all: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z_all)
        gate_logits = torch.stack([head(h) for head in self.gate_heads], dim=1)
        return F.softmax(gate_logits, dim=-1)


class TriLens(nn.Module):
    def __init__(
        self,
        cfg,
        gene_mask: torch.Tensor,
        use_a: bool = True,
        use_b: bool = True,
        use_c: bool = True,
        fusion: str = "moe",
    ) -> None:
        super().__init__()

        if fusion not in ("single", "uniform", "moe"):
            raise ValueError(f"fusion must be 'single'/'uniform'/'moe', got {fusion!r}")

        self.use_a = use_a
        self.use_b = use_b
        self.use_c = use_c
        self.fusion = fusion
        self.n_active = int(use_a) + int(use_b) + int(use_c)

        d = int(cfg.d_model)
        attn_p = int(cfg.attn_dim_pool)
        ea = cfg.expert_a
        eb = cfg.expert_b
        ec = cfg.expert_c
        moe = cfg.moe

        if use_a:
            self.expert_a = ExpertA(
                G=int(ec.n_genes),
                L=int(ea.L_max),
                d_out=d,
                conv_channels=list(ea.conv_channels),
                kernel_size=int(ea.kernel_size),
                dropout=float(ea.dropout),
                attn_dim=attn_p,
                gene_mask=gene_mask,
            )

        if use_b:
            self.expert_b = ExpertB(
                G_pa=int(eb.n_pa_genes),
                d_token=int(eb.d_token),
                d_out=d,
                attn_dim=int(eb.attn_dim),
                strain_conditioned=eb.get("strain_conditioned", None),
            )

        if use_c:
            self.expert_c = ExpertC(
                G=int(ec.n_genes),
                n_stats=int(ec.n_stats),
                d_model=d,
                attn_dim=attn_p,
                n_context_layers=int(ec.n_context_layers),
                dropout=0.3,
                use_coresistance=bool(ec.use_coresistance),
            )

        if fusion == "moe" and self.n_active > 1:
            self.moe_gate: Optional[nn.Module] = MoEGate(
                d_in=d * self.n_active,
                d_hidden=int(moe.hidden_dim),
                n_experts=self.n_active,
                n_abs=len(AB_NAMES),
                dropout=float(moe.dropout),
            )
        else:
            self.moe_gate = None

    def forward(
        self,
        X_blocks: torch.Tensor,
        X_pa: torch.Tensor,
        X_gene_agg: torch.Tensor,
    ) -> Dict:

        expert_z: List[torch.Tensor] = []
        expert_l: List[torch.Tensor] = []

        z_A = a_A = l_A = None
        if self.use_a:
            z_A, a_A, l_A = self.expert_a(X_blocks)
            expert_z.append(z_A)
            expert_l.append(l_A)

        z_B = a_B = l_B = None
        if self.use_b:
            z_B, a_B, l_B = self.expert_b(X_pa)
            expert_z.append(z_B)
            expert_l.append(l_B)

        z_C = a_C = l_C = None
        if self.use_c:
            z_C, a_C, l_C = self.expert_c(X_gene_agg)
            expert_z.append(z_C)
            expert_l.append(l_C)

        gate = None
        z_all = None
        if self.n_active == 1 or self.fusion == "single":
            l_final = expert_l[0]

        elif self.fusion == "uniform":
            l_final = torch.stack(expert_l, dim=-1).mean(dim=-1)

        else:
            z_all = torch.cat(expert_z, dim=-1)
            gate = self.moe_gate(z_all)
            l_stk = torch.stack(expert_l, dim=-1)
            l_final = (gate * l_stk).sum(dim=-1)

        return {
            "z_A": z_A,
            "z_B": z_B,
            "z_C": z_C,
            "z_fused": z_all if self.fusion == "moe" and self.n_active > 1 else None,
            "l_final": l_final,
            "l_A": l_A,
            "l_B": l_B,
            "l_C": l_C,
            "gate": gate,
            "a_A": a_A,
            "a_B": a_B,
            "a_C": a_C,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

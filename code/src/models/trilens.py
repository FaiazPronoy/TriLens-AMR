"""
src/models/trilens.py
======================
TriLens-MoE: three-lens AMR prediction model with MoE gating.

Three orthogonal information lenses:
  Expert A  — within-gene SNP-motif 1D-CNN (trees destroy SNP order; CNN recovers it)
  Expert B  — pan-genome presence/absence sparse-attention (HGT/plasmid resistance genes)
  Expert C  — RMRNet cross-AB structured head (drug-class hierarchy + shared mechanism)

MoE gate   — per-sample per-AB adaptive routing (B, 7, 3) gate weights

All experts output d_model=64 embeddings and (B,7) per-AB logits.
The model supports partial expert activation (T1/T2/T3 ablations) and
fusion modes: 'single', 'uniform', 'moe'.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TriLens", "AB_NAMES", "DRUG_GROUPS"]

# ─── Constants ────────────────────────────────────────────────────────────────

AB_NAMES: List[str] = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]

# Drug class → AB list (grounded in SHAP co-resistance clustering AMP-AMC, CTX-CAZ)
DRUG_GROUPS: Dict[str, List[str]] = {
    "penicillin":      ["AMP", "AMC"],
    "cephalosporin":   ["CTX", "CAZ"],
    "fluoroquinolone": ["CIP"],
    "aminoglycoside":  ["GEN"],
    "folate":          ["SXT"],
}

# AB-index → group-index (groups: 0=pen, 1=ceph, 2=flq, 3=ami, 4=fol)
# AB order: AMP=0, CIP=1, AMC=2, SXT=3, GEN=4, CTX=5, CAZ=6
_AB_TO_GROUP: List[int] = [0, 2, 0, 4, 3, 1, 1]


# ─── Shared utility ───────────────────────────────────────────────────────────

class GatedAttentionPool(nn.Module):
    """
    Reusable gated attention-MIL pooling (Ilse et al. 2018).

    e_k = w^T ( tanh(V h_k) ⊙ σ(U h_k) )    (⊙ = element-wise)
    a   = softmax(e)                           (B, N)
    z   = Σ_k a_k h_k                          (B, D)

    V, U ∈ R^(attn_dim × D), w ∈ R^attn_dim, attn_dim DEFAULT 32.
    Optionally accepts a boolean mask to ignore certain positions.
    """

    def __init__(self, d_in: int, attn_dim: int = 32) -> None:
        super().__init__()
        self.V = nn.Linear(d_in, attn_dim, bias=False)
        self.U = nn.Linear(d_in, attn_dim, bias=False)
        self.w = nn.Linear(attn_dim, 1,    bias=False)

    def forward(
        self,
        h:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, N, D)
        e = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)  # (B, N)
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))
        a = torch.softmax(e, dim=-1)           # (B, N)
        z = (a.unsqueeze(-1) * h).sum(dim=1)  # (B, D)
        return z, a


def _flat_heads(d_in: int, n_abs: int) -> nn.ModuleList:
    """Helper: n_abs independent Linear(d_in→1) heads → concat → (B, n_abs)."""
    return nn.ModuleList([nn.Linear(d_in, 1) for _ in range(n_abs)])


def _run_heads(heads: nn.ModuleList, z: torch.Tensor) -> torch.Tensor:
    return torch.cat([h(z) for h in heads], dim=-1)  # (B, n_abs)


# ─── Expert A — Within-gene SNP-Motif CNN ─────────────────────────────────────

class ExpertA(nn.Module):
    """
    Per-gene shared 1D-CNN over raw SNP columns + gated attention pooling.

    Architecture (spec §2.1):
      (B, G, L) → reshape (B·G, 1, L)
        → Conv1d(1→16, k=3, p=1) → ReLU → Dropout
        → Conv1d(16→32, k=3, p=1) → ReLU → Dropout
        → mask padded positions to -inf
        → AdaptiveMaxPool1d(1) → squeeze → (B·G, 32)
        → reshape (B, G, 32)
        → GatedAttentionPool over G genes → (B, 32)
        → Linear(32→d_out) → GELU → z_A ∈ (B, d_out)
        → 7 × Linear(d_out→1) → ℓ_A ∈ (B, 7)

    gene_mask (G, L) bool registered buffer: True = real data, False = padding.
    Same for ALL samples.
    """

    def __init__(
        self,
        G:             int,
        L:             int,
        d_out:         int,
        conv_channels: List[int],
        kernel_size:   int,
        dropout:       float,
        attn_dim:      int,
        gene_mask:     torch.Tensor,  # (G, L) bool
    ) -> None:
        super().__init__()
        self.G = G
        self.L = L

        # ── Shared 1D-CNN ────────────────────────────────────────────────────
        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv   = nn.Sequential(*layers)
        self.pool   = nn.AdaptiveMaxPool1d(1)
        self.c_out  = conv_channels[-1]

        # ── Gene-level gated attention pool ──────────────────────────────────
        self.gene_pool = GatedAttentionPool(self.c_out, attn_dim)

        # ── Projection + heads ────────────────────────────────────────────────
        self.proj     = nn.Sequential(nn.Linear(self.c_out, d_out), nn.GELU())
        self.ab_heads = _flat_heads(d_out, len(AB_NAMES))

        # Register gene mask (same for all samples, only depends on gene structure)
        self.register_buffer("gene_mask", gene_mask.bool())  # (G, L)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        X : (B, G, L) float32 — per-gene SNP block matrix
        Returns z (B, d_out), a (B, G), logits (B, n_abs)
        """
        B, G, L = X.shape
        C = self.c_out

        # Reshape for shared CNN: (B, G, L) → (B·G, 1, L)
        x = X.view(B * G, 1, L)
        h = self.conv(x)              # (B·G, C, L)

        # Apply gene mask: positions where gene_mask=False are padding → set to -inf
        # gene_mask: (G, L) → (B, G, L) → (B·G, L) → (B·G, 1, L) → (B·G, C, L)
        mask_bg = (
            self.gene_mask
            .unsqueeze(0).expand(B, -1, -1)   # (B, G, L)
            .reshape(B * G, L)                 # (B·G, L)
            .unsqueeze(1).expand(-1, C, -1)    # (B·G, C, L)
        )
        h = h.masked_fill(~mask_bg, float("-inf"))

        # Max pool over L → (B·G, C, 1) → (B·G, C)
        h = self.pool(h).squeeze(-1)
        # Guard: if a gene somehow had all-inf (shouldn't happen: min len_g=5)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape to per-gene embeddings
        h = h.view(B, G, C)           # (B, G, C)

        # Gated attention pool over G genes
        z_pool, a = self.gene_pool(h)  # (B, C), (B, G)

        # Project to d_out and classify
        z      = self.proj(z_pool)                   # (B, d_out)
        logits = _run_heads(self.ab_heads, z)        # (B, 7)

        return z, a, logits


# ─── Expert B — Pan-genome P/A Sparse-Attention Encoder ──────────────────────

class ExpertB(nn.Module):
    """
    Sparse-attention encoder over 32441-gene presence/absence matrix.

    Architecture (spec §2.2):
      E_pa ∈ R^(G_pa × d_token) learnable embedding matrix
      weighted = E_pa * X_pa  (B, G_pa, d_token)  — only present genes contribute
      Gated attention (Ilse) with sparsity mask (absent genes → -inf)
      Softmax in FLOAT32 for stability over 32K elements
      z_pre_B = Σ_k a_k weighted_k  (B, d_token)
      z_B = GELU(Linear(d_token→d_out))
      ℓ_B = 7 × Linear(d_out→1)

    The 13 unmatched samples (X_pa = 0 for all genes) return z_pre_B = 0
    via the no_present guard that zeroes their attention.

    strain_conditioned
        True (default, the delivered model): every gene score also sees the
        strain's mean gene embedding. False: standard gated attention, the
        "Standard" column of the strain-conditioning ablation. None reads the
        environment variable TRILENS_LENSB ("standard" -> False), so every
        training script can run the ablation without code changes.
        With False the two context layers are not created at all, so the
        parameter initialisation order is the same as the original standard
        model.
    """

    def __init__(
        self,
        G_pa:     int,
        d_token:  int,
        d_out:    int,
        attn_dim: int,
        strain_conditioned: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.G_pa    = G_pa
        self.d_token = d_token
        if strain_conditioned is None:
            strain_conditioned = os.environ.get("TRILENS_LENSB", "fixed").lower() != "standard"
        self.strain_conditioned = bool(strain_conditioned)

        # Learnable per-gene embedding (Xavier init)
        self.E_pa = nn.Parameter(torch.empty(G_pa, d_token))
        nn.init.xavier_uniform_(self.E_pa)

        # Gated attention: V, U ∈ R^(attn_dim × d_token), w ∈ R^(attn_dim×1)
        self.attn_V = nn.Linear(d_token, attn_dim, bias=False)
        self.attn_U = nn.Linear(d_token, attn_dim, bias=False)
        # Strain-context projections. Without these the score for a present
        # gene reduces to a function of E_pa[g] alone, which is identical for
        # every strain, so the attention degenerates to one global gene
        # ranking filtered by presence and cannot express per-sample
        # importance. See Section 5.6.1 of the thesis.
        if self.strain_conditioned:
            self.attn_Vc = nn.Linear(d_token, attn_dim, bias=False)
            self.attn_Uc = nn.Linear(d_token, attn_dim, bias=False)
        self.attn_w  = nn.Linear(attn_dim, 1, bias=False)

        # Projection + heads
        self.proj     = nn.Sequential(nn.Linear(d_token, d_out), nn.GELU())
        self.ab_heads = _flat_heads(d_out, len(AB_NAMES))

    def forward(self, X_pa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        X_pa : (B, G_pa) float32, values ∈ {0.0, 1.0}
        Returns z (B, d_out), a_B (B, G_pa), logits (B, n_abs)
        """
        # Weighted gene embeddings: absent genes (0) contribute zero weight
        weighted = self.E_pa.unsqueeze(0) * X_pa.unsqueeze(-1)  # (B, G_pa, d_token)

        v = self.attn_V(weighted)                                   # (B, G_pa, attn_dim)
        u = self.attn_U(weighted)
        if self.strain_conditioned:
            # Strain profile: mean embedding over the genes this strain carries
            n_present = X_pa.sum(dim=1, keepdim=True).clamp(min=1.0)   # (B, 1)
            ctx       = weighted.sum(dim=1) / n_present                # (B, d_token)
            # Score each gene in the context of the strain carrying it. Adding two
            # projections is identical to one projection over the concatenation,
            # but broadcasts instead of materialising (B, 32441, 2*d_token).
            v = v + self.attn_Vc(ctx).unsqueeze(1)
            u = u + self.attn_Uc(ctx).unsqueeze(1)
        e = self.attn_w(torch.tanh(v) * torch.sigmoid(u)).squeeze(-1)  # (B, G_pa)

        # Mask absent genes (X_pa = 0 → not present)
        present_mask = (X_pa > 0.5)                           # (B, G_pa) bool
        e_masked     = e.masked_fill(~present_mask, float("-inf"))

        # Guard: samples where ALL genes are absent (13 unmatched) → use zero scores
        no_present = ~present_mask.any(dim=1, keepdim=True)    # (B, 1)
        e_safe     = torch.where(
            no_present.expand_as(e_masked),
            torch.zeros_like(e_masked),
            e_masked,
        )

        # Softmax in FLOAT32 for stability (softmax over 32K elements with -inf is
        # numerically catastrophic in fp16 — this is the critical safety cast)
        a_B = torch.softmax(e_safe.float(), dim=-1).to(weighted.dtype)  # (B, G_pa)
        # Zero attention for all-absent samples
        a_B = a_B.masked_fill(no_present.expand_as(a_B), 0.0)

        # Weighted sum over present genes
        z_pre = (a_B.unsqueeze(-1) * weighted).sum(dim=1)  # (B, d_token)

        # Project and classify
        z      = self.proj(z_pre)              # (B, d_out)
        logits = _run_heads(self.ab_heads, z)  # (B, 7)

        return z, a_B, logits


# ─── Expert C — RMRNet Structured Cross-AB Head ───────────────────────────────

class ExpertC(nn.Module):
    """
    Structured cross-antibiotic head with drug-class hierarchy (spec §2.3).

    Architecture:
      (B, G, 4) gene-agg stats
        → per-gene MLP encoder  Linear(4→d)→GELU→Dropout→Linear(d→d)
        → [optional] 1 pre-norm transformer block (gene-context)
        → GatedAttentionPool over G → z_C_pool ∈ (B, d)
        → drug-class head:
            trunk   Linear(d→32) → GELU → Dropout(0.3)
            5 group Linear(32→16) → GELU
            7 AB    Linear(16→1) from own group projection
        → [optional] co-resistance: ℓ_C = ℓ_C_init + α·MLP([z_C; ℓ_C_init])
    """

    def __init__(
        self,
        G:                int,
        n_stats:          int,
        d_model:          int,
        attn_dim:         int,
        n_context_layers: int,
        dropout:          float,
        use_coresistance: bool,
    ) -> None:
        super().__init__()
        self.G                = G
        self.n_context_layers = n_context_layers
        self.use_coresistance = use_coresistance

        # Per-gene MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_stats, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Optional transformer gene-context layer (pre-LN)
        if n_context_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4,
                dim_feedforward=d_model * 2,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
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
            self.pos_embed   = None

        # Gated attention pool over genes
        self.gene_pool = GatedAttentionPool(d_model, attn_dim)

        # Drug-class structured head
        self.trunk       = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout))
        self.group_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(32, 16), nn.GELU()) for _ in range(5)
        ])
        self.ab_classifiers = _flat_heads(16, len(AB_NAMES))
        self.register_buffer("ab_to_group", torch.tensor(_AB_TO_GROUP, dtype=torch.long))

        # Light co-resistance refinement (near-zero init to start inert)
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

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        X : (B, G, n_stats) float32 — standardized per-stat gene aggregated features
        Returns z (B, d_model), a (B, G), logits (B, n_abs)
        """
        B = X.shape[0]

        # Per-gene encoding
        h = self.encoder(X)                                            # (B, G, d_model)

        # Optional transformer
        if self.n_context_layers > 0:
            pos = torch.arange(self.G, device=X.device)
            h   = h + self.pos_embed(pos)
            h   = self.transformer(h)

        # Gated attention pool
        z_pool, a = self.gene_pool(h)                                  # (B, d_model), (B, G)

        # Drug-class structured head
        trunk_out   = self.trunk(z_pool)                               # (B, 32)
        group_feats = [proj(trunk_out) for proj in self.group_projs]  # 5 × (B, 16)

        logits_init = torch.stack(
            [
                self.ab_classifiers[i](
                    group_feats[self.ab_to_group[i]]
                ).squeeze(-1)
                for i in range(len(AB_NAMES))
            ],
            dim=1,
        )                                                               # (B, 7)

        # Co-resistance refinement
        if self.use_coresistance:
            ctx    = torch.cat([z_pool, logits_init.detach()], dim=-1) # (B, d+7)
            r      = self.coresistance_mlp(ctx)                        # (B, 7)
            logits = logits_init + self.alpha * r
        else:
            logits = logits_init

        return z_pool, a, logits


# ─── MoE Gate ────────────────────────────────────────────────────────────────

class MoEGate(nn.Module):
    """
    Per-sample per-AB adaptive expert routing (spec §2.4).

    Input:  concat([z_A, z_B, z_C]) ∈ (B, d_in)
    Output: gate ∈ (B, n_abs, n_experts)  — softmax weights summing to 1 per AB

    Entropy regularisation (computed externally):
      H_gate = -(gate * log(gate + ε)).sum(-1).mean()
      L_entropy = -beta * H_gate  → added to total loss to prevent expert collapse.
    """

    def __init__(
        self,
        d_in:      int,
        d_hidden:  int,
        n_experts: int,
        n_abs:     int,
        dropout:   float,
    ) -> None:
        super().__init__()
        self.trunk      = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.gate_heads = nn.ModuleList([
            nn.Linear(d_hidden, n_experts) for _ in range(n_abs)
        ])

    def forward(self, z_all: torch.Tensor) -> torch.Tensor:
        """z_all: (B, d_in) → gate: (B, n_abs, n_experts)"""
        h = self.trunk(z_all)
        gate_logits = torch.stack(
            [head(h) for head in self.gate_heads], dim=1
        )                                                   # (B, n_abs, n_experts)
        return F.softmax(gate_logits, dim=-1)               # (B, n_abs, n_experts)


# ─── TriLens Main Model ────────────────────────────────────────────────────────

class TriLens(nn.Module):
    """
    TriLens-MoE: three-lens resistance prediction model.

    Parameters
    ----------
    cfg           : Hydra DictConfig with expert/moe/d_model sub-configs
    gene_mask     : (G, L) bool tensor from Expert A block builder
    use_a/b/c     : activate individual experts (for ablations T1-T3)
    fusion        : 'single' (T1-T3), 'uniform' (T4), 'moe' (T5-T7)

    Forward returns a dict with keys:
      l_final (B,7), l_A/l_B/l_C (B,7) or None, gate (B,7,n_exp) or None,
      a_A/a_B/a_C (B,N) attention weights
    """

    def __init__(
        self,
        cfg,
        gene_mask:  torch.Tensor,
        use_a:      bool = True,
        use_b:      bool = True,
        use_c:      bool = True,
        fusion:     str  = "moe",
    ) -> None:
        super().__init__()

        if fusion not in ("single", "uniform", "moe"):
            raise ValueError(f"fusion must be 'single'/'uniform'/'moe', got {fusion!r}")

        self.use_a    = use_a
        self.use_b    = use_b
        self.use_c    = use_c
        self.fusion   = fusion
        self.n_active = int(use_a) + int(use_b) + int(use_c)

        d      = int(cfg.d_model)
        attn_p = int(cfg.attn_dim_pool)
        ea     = cfg.expert_a
        eb     = cfg.expert_b
        ec     = cfg.expert_c
        moe    = cfg.moe

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

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        X_blocks:    torch.Tensor,   # (B, G, L) float32 — Expert A input
        X_pa:        torch.Tensor,   # (B, G_pa) float32 — Expert B input
        X_gene_agg:  torch.Tensor,   # (B, G, 4) float32 — Expert C input (standardized)
    ) -> Dict:
        """
        Returns dict with:
          l_final : (B, 7)                  final logits
          l_A/B/C : (B, 7) or None          per-expert logits (for aux loss + ablation)
          gate    : (B, 7, n_active) or None per-AB routing weights
          a_A/B/C : attention arrays or None
        """
        expert_z:   List[torch.Tensor] = []
        expert_l:   List[torch.Tensor] = []

        # ── Run active experts ────────────────────────────────────────────────
        z_A = a_A = l_A = None
        if self.use_a:
            z_A, a_A, l_A = self.expert_a(X_blocks)
            expert_z.append(z_A); expert_l.append(l_A)

        z_B = a_B = l_B = None
        if self.use_b:
            z_B, a_B, l_B = self.expert_b(X_pa)
            expert_z.append(z_B); expert_l.append(l_B)

        z_C = a_C = l_C = None
        if self.use_c:
            z_C, a_C, l_C = self.expert_c(X_gene_agg)
            expert_z.append(z_C); expert_l.append(l_C)

        # ── Fusion ────────────────────────────────────────────────────────────
        gate = None
        if self.n_active == 1 or self.fusion == "single":
            l_final = expert_l[0]

        elif self.fusion == "uniform":
            l_final = torch.stack(expert_l, dim=-1).mean(dim=-1)   # (B, 7)

        else:  # "moe"
            z_all  = torch.cat(expert_z, dim=-1)                   # (B, d*n_active)
            gate   = self.moe_gate(z_all)                          # (B, 7, n_active)
            l_stk  = torch.stack(expert_l, dim=-1)                 # (B, 7, n_active)
            l_final = (gate * l_stk).sum(dim=-1)                   # (B, 7)

        return {
            "l_final": l_final,
            "l_A":     l_A,
            "l_B":     l_B,
            "l_C":     l_C,
            "gate":    gate,   # (B, 7, n_active) or None
            "a_A":     a_A,
            "a_B":     a_B,
            "a_C":     a_C,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Code audit — final

`models/` and `src/` both read. Training loops, loss functions, schedulers,
attention export, and every architecture.

**Result: one real bug. Two thesis documentation errors. Everything else clean.**

---

# 🔴 The one real bug — Lens B's attention cannot depend on the sample

**File:** `src/models/trilens.py`, `ExpertB.forward`, lines 237–246
**Affects:** interpretability only. Your PR-AUC numbers are not wrong because of it.

```python
weighted = self.E_pa.unsqueeze(0) * X_pa.unsqueeze(-1)   # (B, G_pa, d_token)
e        = self.attn_w(torch.tanh(self.attn_V(weighted)) *
                       torch.sigmoid(self.attn_U(weighted))).squeeze(-1)
```

`X_pa` is binary, so for a **present** gene `weighted[b,k,:] == E_pa[k,:]` — no
dependence on `b`. Every strain gets the same pre-softmax score for the same gene.
The softmax only renormalises over whichever genes that strain carries.

**Lens B has one global gene ranking, filtered by presence.** It cannot say
"gene 4127 matters for *this* strain". This is not Ilse et al. gated attention,
where the instance features vary per bag.

It explains, in one stroke: near-identical top genes across all seven
antibiotics, near-uniform entropy, zero AMR hits at top-20 and top-100, and
attended genes carrying neither lineage nor resistance information.

**Fix A — condition on the strain:**
```python
present = (X_pa > 0.5).float()
ctx     = weighted.sum(1) / present.sum(1, keepdim=True).clamp(min=1)   # (B, d_token)
q       = torch.cat([weighted, ctx.unsqueeze(1).expand_as(weighted)], -1)
e       = self.attn_w(torch.tanh(self.attn_V(q)) *
                      torch.sigmoid(self.attn_U(q))).squeeze(-1)
```
and widen `attn_V` / `attn_U` to `nn.Linear(2 * d_token, attn_dim, bias=False)`.

**Fix B — swap softmax for entmax15.** Your own
`results/gene_attention_v2/attention_sparsity.csv` shows entmax produces genuinely
sparse attention on this data (E2: 21 genes above 1%, entropy 5.21 against a
maximum of 8.52) where softmax and gated pooling do not.

**Do one, not both**, so the change is attributable.

---

# ✅ RETRACTED — the scheduler bug I reported earlier does not exist

I previously flagged `CosineAnnealingLR(T_max=30)` in `base_module.py` against
`max_epochs=50`, which would make the learning rate climb back up after epoch 30.
**I was wrong.** I read the default and did not check the call sites. Every one of
them passes a consistent value:

| Call site | Setting |
|---|---|
| `run_trilens_v10.py:552` | `T_max = cfg.training.max_epochs` ✓ |
| `run_cascadestack_v9.py:217,233` | `scheduler_t_max = max_epochs` ✓ |
| `run_deepmeta_v11.py:214,227` | `scheduler_t_max = max_epochs` ✓ |
| `run_cascadestack_v7.py:296,312` | consistent ✓ |
| `run_rmrnet.py:899` | `T_max = cfg.max_epochs` ✓ |
| `configs/experiment/pillar1_exp1.yaml` | `max_epochs: 30`, `scheduler_t_max: 30` ✓ |

The 30 default in `base_module.py` is never reached with a mismatched epoch count.
**No learning-rate bug anywhere. Ignore what I said before.**

---

# 🟠 Two thesis errors (the code is right, the paper is wrong)

**1. Adam vs AdamW.** Chapter 3 Table 3.8 and Chapter 4 both say **Adam**.
`base_module.py:252` and `run_trilens_v10.py:547` both use `torch.optim.AdamW`.
With `weight_decay=1e-4` these are not interchangeable. **Change the thesis.**

**2. The learning-rate scheduler is undocumented.** Every training loop uses
`CosineAnnealingLR(T_max=max_epochs, eta_min=lr*0.01)`. Table 3.8 lists optimiser,
LR, weight decay, batch size, epochs, patience, gradient clipping, mixed
precision, deterministic mode, cross-validation, seed and bootstrap rounds — but
no scheduler. **Add a row.**

---

# ✅ Verified correct

Checked specifically, found no fault:

| Component | Where | Verdict |
|---|---|---|
| Entropy regulariser | `run_trilens_v10.py:445-446` | `H = -(g·log g).sum(-1).mean()`, `L_ent = -β·H`. Minimising the loss maximises entropy. Matches Chapter 4 exactly. |
| Auxiliary loss | `run_trilens_v10.py:433-439` | Per-lens masked BCE summed, scaled by `aux_loss_weight`. Matches §4.4.5. |
| Gate-collapse guard | `run_trilens_v10.py:572-581` | Fires when `gate_avg.min() < threshold`, raises β. The `epoch >= 10` guard stops it triggering on noise. Matches §4.2.7. |
| Masked BCE | `base_module.py:117-128` | Losses zeroed on masked entries, denominator is `mask.sum()`. Untested drugs contribute nothing. |
| `compute_pos_weights` | `base_module.py:279` | `n_neg / n_pos` per antibiotic ✓ |
| Attention export | `run_pillar3_interpretability.py:79-88` | Manually mirrors the forward pass; I compared it line by line against `GeneAttentionNet` and it matches, including the pool's internal LayerNorm. |
| Shared-pool attention | `run_gene_attention_v2.py:677` | `# shared pool → same attn for all ABs` — documented and intentional. This is why E0/E1/E2 rows are identical across antibiotics in `attention_sparsity.csv`. Not a bug. |
| `ExpertA` | `trilens.py` | `h` from the CNN over each sample's SNP block → sample-adaptive ✓. `-inf` padding mask plus `nan_to_num` guard is sound. |
| `ExpertC` | `trilens.py` | `h = encoder(X)` varies per sample ✓. `logits_init.detach()` is deliberate stabilisation. Zero-init makes the co-resistance branch start inert as documented. |
| `MoEGate` | `trilens.py` | Softmax over experts per antibiotic, shapes correct ✓ |
| `gene_attention_net.py`, `_v2.py` | | Correct, and `_v2` already implements entmax/sparsemax ✓ |
| `rmrnet.py`, `cnn_variants.py`, `deep_meta_v11.py` | | Correct ✓ |

---

# What a re-run will and will not buy you

Be clear-eyed about this before spending the week.

**Will not improve:** PR-AUC. The only bug affects what the attention *selects*,
not what Lens B contributes as a feature extractor. And the ceiling is set by
feature representation, not by the model — the heritability comparison in §5.5.3
says markers explain 94–99% of the variance while your held-out scores are far
below that. No attention fix closes that gap.

**Will improve:** the interpretability chapter. Right now it reports a diagnosed
failure. After the fix it could report an actual gene ranking, which turns
Objective 5 from a negative into a result.

**The biggest available win is not a model change at all.** It is per-antibiotic
threshold tuning. FR5 is currently "not met" and your clinical numbers are the
worst they could be, purely because the threshold was fixed at 0.5 and validation
predictions were never saved. Save them this time and you can tune on validation,
report on test, and very likely move the very major error rates substantially.

---

# Suggested week

**Day 1 — fix and instrument**
- ExpertB: Fix A or Fix B, one only.
- Change the thesis to say AdamW, add a scheduler row to Table 3.8.
- Add the saving listed below. This matters more than the model fix.
- Smoke test: 2 epochs on a subset.

**Day 2–3 — retrain** (overnight)
- T6 on the random split: ~3.5–4.5 h on a 3060
- T6 on the MLST lineage holdout: same again
- Reuse the classical arm's cached out-of-fold predictions — unaffected by this
  change, so reusing them isolates the effect to the neural side.

**Day 4 — downstream reruns**
- Stage 1/2/3 ablation, gate weights, attention export, interpretability.
- Threshold tuning on the new validation predictions.
- Multiseed only if time allows; it is the least important.

**Day 5 — re-verify**
- Rerun all five analysis scripts. They read saved predictions, so they pick up
  the new numbers automatically:
  `matched_prevalence_check.py`, `holdout_clinical_metrics.py`,
  `attention_diagnosis.py`, `prefix_baserate_lensB.py`, `lineage_marker_test.py`

**Day 6–7 — thesis**
- Every TriLens number in Chapter 5 changes. Chapters 1, 3 and 6 quote several.

---

# ⚠️ Save these this time

The expensive part of the last six rounds was questions that could not be
answered without retraining.

| Save | Why it matters |
|---|---|
| **`val_ens_probs` + `val_indices`** | Without them, per-antibiotic threshold tuning is impossible. FR5 is "not met" for this reason alone. **This is the single highest-value line of code you will write this week.** |
| **Per-sample attention `(n_test, G)` per lens** | Not just the top-20 average. Every interpretability question this week wanted the full matrix. |
| **Per-sample gate weights** | Same reason. |
| **The epoch each run stopped at** | Needed to check any future scheduler or early-stopping question. |
| **A `model_name` string inside every results file** | Table 5.2 reporting CascadeStack under a TriLens caption cost a full round to find. One string would have prevented it. |

---

# Bottom line

Your pipeline is sound. One architectural bug in one lens, and it affects
interpretability rather than accuracy. The entropy regulariser, auxiliary loss,
collapse guard, masking, positive weighting and attention extraction are all
implemented correctly and match what the thesis describes.

If the goal is better numbers, the threshold work will do more than the model fix.
If the goal is a stronger interpretability chapter, fix ExpertB.

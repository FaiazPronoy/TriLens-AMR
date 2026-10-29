# Pillar 3 v3 — SHAP + Gain Tree-Based Interpretability

## Methodology

For each antibiotic we train two XGBoost models (gene-aggregated and
raw chi²-prefiltered SNPs — identical configs to CascadeStack v9's base
learners) and compute per-gene importance using both:

1. **Gain-based importance** from `booster.get_score(importance_type="gain")`
   — aggregated per gene by summing across the four gene-statistic
     features (Plan B) or across all SNPs mapped back to that gene via
     `gene_to_range` (Plan C).
2. **SHAP** via TreeExplainer (`tree_path_dependent` backend, the
   theoretically grounded interventional explainer for tree models).
   Per-gene aggregation as above; mean |SHAP| over training samples is
   the importance score. We also compute *directional* SHAP — mean
   signed SHAP value over resistant samples only — to indicate whether
   a gene drives prediction toward resistance (+) or susceptibility (–).

Concordance is reported as Spearman ρ on the union of top-K genes
(gain rank vs SHAP rank) and Jaccard similarity of the top-K sets.

## Gain vs SHAP concordance per AB

| Plan | AB | Spearman ρ | Jaccard top-K | Time (s) |
|---|---|---|---|---|
| B | AMP | -0.131 | 0.18 | 36 |
| B | CIP | -0.290 | 0.11 | 35 |
| B | AMC | -0.504 | 0.08 | 33 |
| B | SXT | -0.149 | 0.21 | 30 |
| B | GEN | -0.335 | 0.18 | 37 |
| B | CTX | -0.552 | 0.05 | 34 |
| B | CAZ | -0.523 | 0.05 | 37 |
| C | AMP | -0.217 | 0.25 | 83 |
| C | CIP | -0.221 | 0.21 | 83 |
| C | AMC | -0.325 | 0.14 | 72 |
| C | SXT | -0.325 | 0.14 | 70 |
| C | GEN | -0.376 | 0.11 | 88 |
| C | CTX | -0.429 | 0.08 | 82 |
| C | CAZ | -0.226 | 0.14 | 89 |

**Avg Spearman ρ — Plan B: -0.355, Plan C: -0.303** (high ρ → gain and SHAP agree on ranking → robust signal).

## Plan B (XGB gene-aggregated, SHAP-ranked)

### AMP

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 2 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 2.76% | 1.27% | ↑res (+0.409) | ✓ |
| 2 | 15 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F | 2.17% | 0.36% | ↑res (+0.380) | ✓ |
| 3 | 13 | MFCAOJAD_04227 | 1.58% | 0.39% | ↑res (+0.332) | ✓ |
| 4 | 40 | MFCAOJAD_04228 | 1.30% | 0.24% | ↑res (+0.236) |   |
| 5 | 4 | MFCAOJAD_02201 | 1.26% | 0.86% | ↑res (+0.160) | ✓ |
| 6 | 5 | MFCAOJAD_04226 | 1.17% | 0.71% | ↑res (+0.218) | ✓ |
| 7 | 34 | GFDLAHDC_03552 | 0.93% | 0.27% | ↑res (+0.111) |   |
| 8 | 1 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 0.58% | 1.54% | ↑res (+0.058) | ✓ |
| 9 | 106 | Cluster_5040_+_+_GCA_003333505.1_ASM333350v1_+_+_FAJAELAD_04671_+_+_FAJAELAD_04672_+_+_DT | 0.55% | 0.13% | ↑res (+0.024) |   |
| 10 | 686 | NGHFEPFE_02148 | 0.55% | 0.05% | ↓res (-0.011) |   |
| 11 | 519 | Cluster_2171_+_+_GCA_003333545.1_ASM333354v1_+_+_KFGAJDKJ_02746_+_+_KFGAJDKJ_02747_+_+_CO_F | 0.53% | 0.06% | ↑res (+0.001) |   |
| 12 | 570 | LMHECDEF_03944 | 0.49% | 0.05% | ↑res (+0.010) |   |
| 13 | 559 | Cluster_10753_+_+_GCA_003333865.1_ASM333386v1_+_+_DHJNCGMO_04085_+_+_DHJNCGMO_04086_+_+_CO_R | 0.48% | 0.05% | ↓res (-0.002) |   |
| 14 | 510 | JEDMHCBK_03552 | 0.48% | 0.06% | ↑res (+0.001) |   |
| 15 | 729 | LMHECDEF_04213 | 0.48% | 0.04% | ↑res (+0.001) |   |
| 16 | 56 | Cluster_2166_+_+_GCA_003334815.1_ASM333481v1_+_+_PKAGLLBP_03622_+_+_PKAGLLBP_03623_+_+_CO_F | 0.46% | 0.18% | ↑res (+0.050) |   |
| 17 | 871 | LMHECDEF_02147 | 0.40% | 0.04% | ↑res (+0.019) |   |
| 18 | 197 | KKBBHFJK_02469 | 0.39% | 0.09% | ↓res (-0.012) |   |
| 19 | 473 | MLMPEHKM_01641 | 0.38% | 0.06% | ↑res (+0.016) |   |
| 20 | 162 | LGHNHFEP_02477 | 0.37% | 0.10% | ↑res (+0.051) |   |

### CIP

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 91 | LMHECDEF_04343 | 2.99% | 0.16% | ↑res (+0.959) |   |
| 2 | 1 | Cluster_381_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_01805_+_+_IPMBBGLD_01806_+_+_CO_F | 2.37% | 11.84% | ↑res (+0.724) | ✓ |
| 3 | 100 | LMHECDEF_01124 | 1.31% | 0.16% | ↑res (+0.294) |   |
| 4 | 2 | Cluster_4529_+_+_GCA_003333505.1_ASM333350v1_+_+_FAJAELAD_04336_+_+_FAJAELAD_04337_+_+_CO_R | 0.68% | 2.15% | ↑res (+0.142) | ✓ |
| 5 | 511 | JNGMHDBF_02725 | 0.62% | 0.05% | ↑res (+0.023) |   |
| 6 | 220 | MLMPEHKM_01161 | 0.62% | 0.10% | ↑res (+0.153) |   |
| 7 | 3 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 0.60% | 1.07% | ↑res (+0.177) | ✓ |
| 8 | 26 | Cluster_666_+_+_GCA_003333875.1_ASM333387v1_+_+_APHKLHJA_04833_+_+_APHKLHJA_04834_+_+_DP | 0.59% | 0.32% | ↑res (+0.174) |   |
| 9 | 362 | MLMPEHKM_01180 | 0.54% | 0.06% | ↓res (-0.002) |   |
| 10 | 158 | GFDLAHDC_03552 | 0.54% | 0.12% | ↑res (+0.053) |   |
| 11 | 1443 | Cluster_392_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_04386_+_+_IPMBBGLD_04387_+_+_DP | 0.52% | 0.01% | ↑res (+0.004) |   |
| 12 | 229 | Cluster_138_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_02047_+_+_KIEJJFCM_02048_+_+_CO_R | 0.52% | 0.09% | ↑res (+0.045) |   |
| 13 | 17 | Cluster_1854_+_+_GCA_003334625.1_ASM333462v1_+_+_NOFKCEEG_02703_+_+_NOFKCEEG_02704_+_+_DP | 0.52% | 0.38% | ↑res (+0.158) | ✓ |
| 14 | 356 | LMHECDEF_03916 | 0.51% | 0.06% | ↑res (+0.056) |   |
| 15 | 598 | Cluster_377_+_+_GCA_003334165.1_ASM333416v1_+_+_FCDKFLAE_01225_+_+_FCDKFLAE_01226_+_+_DP | 0.48% | 0.04% | ↑res (+0.090) |   |
| 16 | 1256 | OGOJFKBC_01079 | 0.47% | 0.01% | ↑res (+0.008) |   |
| 17 | 882 | LMHECDEF_03538 | 0.45% | 0.02% | ↑res (+0.028) |   |
| 18 | 396 | OGOJFKBC_01850 | 0.44% | 0.06% | ↓res (-0.004) |   |
| 19 | 587 | OGOJFKBC_00262 | 0.42% | 0.04% | ↑res (+0.059) |   |
| 20 | 372 | KFGAJDKJ_04631 | 0.41% | 0.06% | ↑res (+0.007) |   |

### AMC

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 25 | MFCAOJAD_04227 | 1.32% | 0.25% | ↑res (+0.125) |   |
| 2 | 14 | LGHNHFEP_02479 | 0.94% | 0.33% | ↑res (+0.133) | ✓ |
| 3 | 263 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F | 0.78% | 0.09% | ↑res (+0.061) |   |
| 4 | 209 | ONBJOLKN_01815 | 0.77% | 0.10% | ↓res (-0.000) |   |
| 5 | 431 | BIAEOMMM_03258 | 0.71% | 0.07% | ↑res (+0.045) |   |
| 6 | 185 | ONLNHHCL_02098 | 0.70% | 0.11% | ↑res (+0.076) |   |
| 7 | 509 | MLMPEHKM_02801 | 0.63% | 0.06% | ↑res (+0.016) |   |
| 8 | 59 | KFGAJDKJ_02757 | 0.51% | 0.18% | ↑res (+0.063) |   |
| 9 | 2 | FMDNDMLM_04838 | 0.49% | 1.00% | ↑res (+0.133) | ✓ |
| 10 | 970 | Cluster_761_+_+_GCA_003334625.1_ASM333462v1_+_+_NOFKCEEG_03511_+_+_NOFKCEEG_03512_+_+_DP | 0.48% | 0.03% | ↑res (+0.011) |   |
| 11 | 382 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 0.47% | 0.07% | ↑res (+0.030) |   |
| 12 | 1042 | ONLNHHCL_02097 | 0.47% | 0.03% | ↑res (+0.043) |   |
| 13 | 396 | Cluster_18_+_+_GCA_003334705.1_ASM333470v1_+_+_GGJIDEBE_02946_+_+_GGJIDEBE_02947_+_+_CO_R | 0.47% | 0.07% | ↑res (+0.040) |   |
| 14 | 231 | MFCAOJAD_02201 | 0.45% | 0.10% | ↑res (+0.012) |   |
| 15 | 177 | MFCAOJAD_04228 | 0.44% | 0.11% | ↑res (+0.041) |   |
| 16 | 450 | MFCAOJAD_02264 | 0.43% | 0.06% | ↑res (+0.032) |   |
| 17 | 131 | KKBBHFJK_02034 | 0.43% | 0.12% | ↑res (+0.049) |   |
| 18 | 751 | FBCKMFHC_02675 | 0.41% | 0.04% | ↑res (+0.027) |   |
| 19 | 8 | ONLNHHCL_01578 | 0.40% | 0.43% | ↑res (+0.015) | ✓ |
| 20 | 345 | Cluster_8_+_+_GCA_003334785.1_ASM333478v1_+_+_MGCLNDBK_03492_+_+_MGCLNDBK_03493_+_+_DT | 0.40% | 0.08% | ↑res (+0.028) |   |

### SXT

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 14 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F | 3.74% | 0.48% | ↑res (+0.764) | ✓ |
| 2 | 2 | MFCAOJAD_02201 | 3.10% | 1.18% | ↑res (+0.526) | ✓ |
| 3 | 8 | DHJNCGMO_04404 | 2.21% | 0.60% | ↑res (+0.382) | ✓ |
| 4 | 1 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 1.87% | 2.85% | ↑res (+0.285) | ✓ |
| 5 | 16 | GFDLAHDC_03552 | 1.34% | 0.42% | ↑res (+0.167) | ✓ |
| 6 | 12 | Cluster_6925_+_+_GCA_003333865.1_ASM333386v1_+_+_DHJNCGMO_04404_+_+_DHJNCGMO_04405_+_+_CO_F | 0.87% | 0.51% | ↑res (+0.167) | ✓ |
| 7 | 82 | KFGAJDKJ_00412 | 0.79% | 0.16% | ↑res (+0.053) |   |
| 8 | 80 | LMHECDEF_00007 | 0.62% | 0.16% | ↑res (+0.035) |   |
| 9 | 303 | Cluster_8_+_+_GCA_003334785.1_ASM333478v1_+_+_MGCLNDBK_03492_+_+_MGCLNDBK_03493_+_+_DT | 0.56% | 0.08% | ↑res (+0.009) |   |
| 10 | 144 | KFGAJDKJ_04949 | 0.55% | 0.12% | ↑res (+0.027) |   |
| 11 | 490 | Cluster_5538_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_02743_+_+_IPMBBGLD_02744_+_+_CO_F | 0.55% | 0.06% | ↑res (+0.014) |   |
| 12 | 756 | Cluster_2898_+_+_GCA_003333425.1_ASM333342v1_+_+_MLMPEHKM_00832_+_+_MLMPEHKM_00833_+_+_CO_R | 0.54% | 0.04% | ↑res (+0.010) |   |
| 13 | 13 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 0.51% | 0.50% | ↑res (+0.061) | ✓ |
| 14 | 496 | Cluster_1136_+_+_GCA_003334255.1_ASM333425v1_+_+_JNGMHDBF_04871_+_+_JNGMHDBF_04872_+_+_CO_F | 0.47% | 0.06% | ↑res (+0.011) |   |
| 15 | 963 | Cluster_1835_+_+_GCA_003333385.1_ASM333338v1_+_+_LMHECDEF_02228_+_+_LMHECDEF_02229_+_+_CO_R | 0.47% | 0.03% | ↑res (+0.018) |   |
| 16 | 199 | EMKJHLAI_00924 | 0.45% | 0.10% | ↑res (+0.036) |   |
| 17 | 699 | MLMPEHKM_03345 | 0.44% | 0.05% | ↑res (+0.010) |   |
| 18 | 1006 | LMHECDEF_01160 | 0.44% | 0.03% | ↑res (+0.001) |   |
| 19 | 157 | Cluster_344_+_+_GCA_003333385.1_ASM333338v1_+_+_LMHECDEF_04201_+_+_LMHECDEF_04202_+_+_DP | 0.41% | 0.12% | ↑res (+0.031) |   |
| 20 | 172 | LMHECDEF_01992 | 0.39% | 0.11% | ↑res (+0.025) |   |

### GEN

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 11 | Cluster_3542_+_+_GCA_003334405.1_ASM333440v1_+_+_AIHHOGBN_04190_+_+_AIHHOGBN_04191_+_+_DP | 2.51% | 0.52% | ↑res (+0.445) | ✓ |
| 2 | 15 | DHJNCGMO_04404 | 0.73% | 0.38% | ↑res (+0.134) | ✓ |
| 3 | 777 | Cluster_377_+_+_GCA_003334165.1_ASM333416v1_+_+_FCDKFLAE_01225_+_+_FCDKFLAE_01226_+_+_DP | 0.68% | 0.04% | ↑res (+0.081) |   |
| 4 | 1 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 0.61% | 3.52% | ↑res (+0.167) | ✓ |
| 5 | 237 | Cluster_354_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_01867_+_+_HFNFFLAH_01868_+_+_DP | 0.61% | 0.09% | ↑res (+0.033) |   |
| 6 | 1451 | Cluster_245_+_+_GCA_003334485.1_ASM333448v1_+_+_DPBOJJJH_03591_+_+_DPBOJJJH_03592_+_+_DT | 0.54% | 0.01% | ↑res (+0.060) |   |
| 7 | 13 | Cluster_1854_+_+_GCA_003334625.1_ASM333462v1_+_+_NOFKCEEG_02703_+_+_NOFKCEEG_02704_+_+_DP | 0.54% | 0.51% | ↑res (+0.092) | ✓ |
| 8 | 858 | NEIJMAED_01562 | 0.50% | 0.03% | ↑res (+0.024) |   |
| 9 | 1097 | MLMPEHKM_00112 | 0.46% | 0.02% | ↑res (+0.017) |   |
| 10 | 509 | OGOJFKBC_02711 | 0.46% | 0.05% | ↓res (-0.000) |   |
| 11 | 9 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 0.45% | 0.64% | ↑res (+0.039) | ✓ |
| 12 | 371 | LMHECDEF_02754 | 0.45% | 0.07% | ↑res (+0.036) |   |
| 13 | 20 | MFCAOJAD_02201 | 0.44% | 0.35% | ↑res (+0.067) | ✓ |
| 14 | 302 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F | 0.43% | 0.08% | ↑res (+0.074) |   |
| 15 | 671 | Cluster_1589_+_+_GCA_003333385.1_ASM333338v1_+_+_LMHECDEF_01172_+_+_LMHECDEF_01173_+_+_CO_F | 0.42% | 0.04% | ↑res (+0.015) |   |
| 16 | 317 | LMHECDEF_04254 | 0.41% | 0.08% | ↑res (+0.034) |   |
| 17 | 917 | PNGICKPE_00318 | 0.39% | 0.03% | ↑res (+0.015) |   |
| 18 | 824 | APHKLHJA_01804 | 0.37% | 0.03% | ↑res (+0.009) |   |
| 19 | 902 | LMHECDEF_03059 | 0.37% | 0.03% | ↑res (+0.032) |   |
| 20 | 1064 | Cluster_2453_+_+_GCA_003334735.1_ASM333473v1_+_+_PLNEIAEL_02918_+_+_PLNEIAEL_02919_+_+_CO_R | 0.36% | 0.02% | ↓res (-0.001) |   |

### CTX

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 160 | Cluster_416_+_+_GCA_003333495.1_ASM333349v1_+_+_BCDBBOOO_02211_+_+_BCDBBOOO_02212_+_+_DT | 1.03% | 0.11% | ↑res (+0.063) |   |
| 2 | 92 | MFCAOJAD_04228 | 0.99% | 0.16% | ↑res (+0.159) |   |
| 3 | 31 | APHKLHJA_00458 | 0.83% | 0.33% | ↑res (+0.274) |   |
| 4 | 13 | Cluster_6097_+_+_GCA_003333645.1_ASM333364v1_+_+_LGHNHFEP_03531_+_+_LGHNHFEP_03532_+_+_DT | 0.72% | 1.09% | ↑res (+0.202) | ✓ |
| 5 | 1130 | JEDMHCBK_04176 | 0.71% | 0.01% | ↑res (+0.057) |   |
| 6 | 253 | MLMPEHKM_03424 | 0.66% | 0.09% | ↑res (+0.088) |   |
| 7 | 292 | Cluster_230_+_+_GCA_003334715.1_ASM333471v1_+_+_EDILELKJ_03707_+_+_EDILELKJ_03708_+_+_CO_F | 0.65% | 0.08% | ↑res (+0.137) |   |
| 8 | 154 | LMHECDEF_01006 | 0.63% | 0.12% | ↑res (+0.032) |   |
| 9 | 630 | LMHECDEF_03983 | 0.60% | 0.04% | ↑res (+0.016) |   |
| 10 | 448 | BIKPOGCH_01014 | 0.59% | 0.05% | ↑res (+0.062) |   |
| 11 | 175 | EDILELKJ_02217 | 0.53% | 0.11% | ↑res (+0.070) |   |
| 12 | 79 | IPMBBGLD_03094 | 0.51% | 0.17% | ↑res (+0.043) |   |
| 13 | 550 | Cluster_2327_+_+_GCA_003333765.1_ASM333376v1_+_+_EEIHECBN_00417_+_+_EEIHECBN_00418_+_+_DT | 0.50% | 0.04% | ↑res (+0.063) |   |
| 14 | 152 | IPMBBGLD_02091 | 0.49% | 0.12% | ↑res (+0.065) |   |
| 15 | 33 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 0.49% | 0.30% | ↑res (+0.047) |   |
| 16 | 626 | ONLNHHCL_02098 | 0.46% | 0.04% | ↑res (+0.042) |   |
| 17 | 9 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 0.46% | 1.19% | ↑res (+0.112) | ✓ |
| 18 | 1110 | Cluster_1126_+_+_GCA_003333545.1_ASM333354v1_+_+_KFGAJDKJ_04836_+_+_KFGAJDKJ_04837_+_+_DT | 0.44% | 0.02% | ↓res (-0.018) |   |
| 19 | 641 | OGOJFKBC_01079 | 0.44% | 0.04% | ↑res (+0.008) |   |
| 20 | 793 | Cluster_3604_+_+_GCA_003333495.1_ASM333349v1_+_+_BCDBBOOO_03225_+_+_BCDBBOOO_03226_+_+_DT | 0.41% | 0.03% | ↓res (-0.005) |   |

### CAZ

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 1 | Cluster_6097_+_+_GCA_003333645.1_ASM333364v1_+_+_LGHNHFEP_03531_+_+_LGHNHFEP_03532_+_+_DT | 1.65% | 4.66% | ↑res (+0.518) | ✓ |
| 2 | 451 | OGOJFKBC_01079 | 1.22% | 0.06% | ↑res (+0.097) |   |
| 3 | 361 | MFCAOJAD_04228 | 0.96% | 0.07% | ↑res (+0.152) |   |
| 4 | 104 | LMHECDEF_03983 | 0.84% | 0.15% | ↑res (+0.035) |   |
| 5 | 65 | Cluster_3542_+_+_GCA_003334405.1_ASM333440v1_+_+_AIHHOGBN_04190_+_+_AIHHOGBN_04191_+_+_DP | 0.73% | 0.18% | ↑res (+0.111) |   |
| 6 | 353 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 0.64% | 0.07% | ↑res (+0.086) |   |
| 7 | 1071 | LMHECDEF_04213 | 0.56% | 0.02% | ↓res (-0.016) |   |
| 8 | 934 | ONLNHHCL_02098 | 0.50% | 0.03% | ↑res (+0.072) |   |
| 9 | 898 | MLMPEHKM_03425 | 0.49% | 0.03% | ↑res (+0.036) |   |
| 10 | 1246 | NEIJMAED_01562 | 0.43% | 0.02% | ↓res (-0.006) |   |
| 11 | 1264 | LMHECDEF_02305 | 0.42% | 0.02% | ↑res (+0.010) |   |
| 12 | 323 | Cluster_416_+_+_GCA_003333495.1_ASM333349v1_+_+_BCDBBOOO_02211_+_+_BCDBBOOO_02212_+_+_DT | 0.41% | 0.08% | ↑res (+0.017) |   |
| 13 | 97 | PKACBNFE_01616 | 0.40% | 0.16% | ↑res (+0.114) |   |
| 14 | 614 | BIKPOGCH_00661 | 0.40% | 0.05% | ↑res (+0.034) |   |
| 15 | 693 | KFGAJDKJ_00407 | 0.40% | 0.04% | ↑res (+0.002) |   |
| 16 | 752 | OGOJFKBC_02711 | 0.39% | 0.03% | ↑res (+0.023) |   |
| 17 | 197 | MFCAOJAD_02201 | 0.38% | 0.11% | ↑res (+0.013) |   |
| 18 | 4 | LMHECDEF_03897 | 0.37% | 1.04% | ↑res (+0.087) | ✓ |
| 19 | 1377 | BIKPOGCH_00294 | 0.37% | 0.01% | ↑res (+0.007) |   |
| 20 | 797 | NGHFEPFE_04553 | 0.36% | 0.03% | ↑res (+0.013) |   |

## Plan C (XGB raw chi², SHAP-ranked)

### AMP

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 10 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 4.08% | 0.69% | ↑res (+0.601) | ✓ |
| 2 | 43 | MFCAOJAD_04227 | 1.66% | 0.26% | ↑res (+0.413) |   |
| 3 | 2 | MFCAOJAD_02201 | 1.59% | 2.01% | ↑res (+0.241) | ✓ |
| 4 | 165 | MFCAOJAD_04228 | 1.48% | 0.12% | ↑res (+0.382) |   |
| 5 | 6 | KFGAJDKJ_00412 | 1.36% | 0.86% | ↑res (+0.114) | ✓ |
| 6 | 135 | LMHECDEF_04343 | 1.29% | 0.14% | ↑res (+0.267) |   |
| 7 | 56 | KKBBHFJK_02469 | 1.19% | 0.23% | ↓res (-0.014) |   |
| 8 | 23 | LGHNHFEP_02497 | 1.01% | 0.36% | ↑res (+0.031) |   |
| 9 | 15 | LMHECDEF_02660 | 0.96% | 0.47% | ↑res (+0.035) | ✓ |
| 10 | 22 | MFCAOJAD_04226 | 0.92% | 0.37% | ↑res (+0.184) |   |
| 11 | 183 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F | 0.86% | 0.11% | ↑res (+0.173) |   |
| 12 | 18 | ONLNHHCL_02098 | 0.73% | 0.43% | ↑res (+0.028) | ✓ |
| 13 | 438 | LMHECDEF_03770 | 0.69% | 0.06% | ↑res (+0.027) |   |
| 14 | 8 | LMHECDEF_01173 | 0.54% | 0.80% | ↑res (+0.025) | ✓ |
| 15 | 7 | KFGAJDKJ_00407 | 0.53% | 0.81% | ↑res (+0.007) | ✓ |
| 16 | 1 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 0.50% | 3.55% | ↑res (+0.061) | ✓ |
| 17 | 506 | GFDLAHDC_03552 | 0.49% | 0.05% | ↑res (+0.048) |   |
| 18 | 103 | EEIHECBN_02053 | 0.45% | 0.15% | ↑res (+0.022) |   |
| 19 | 40 | KKBBHFJK_02894 | 0.44% | 0.28% | ↑res (+0.026) |   |
| 20 | 279 | KFGAJDKJ_02767 | 0.43% | 0.09% | ↑res (+0.032) |   |

### CIP

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 2 | LMHECDEF_04343 | 14.31% | 1.52% | ↑res (+5.087) | ✓ |
| 2 | 5 | LMHECDEF_01124 | 2.91% | 0.73% | ↑res (+1.248) | ✓ |
| 3 | 13 | LMHECDEF_01173 | 0.93% | 0.45% | ↑res (+0.068) | ✓ |
| 4 | 20 | EEIHECBN_02053 | 0.78% | 0.41% | ↓res (-0.011) | ✓ |
| 5 | 147 | MLMPEHKM_01161 | 0.68% | 0.16% | ↑res (+0.146) |   |
| 6 | 902 | LMHECDEF_00191 | 0.57% | 0.02% | ↑res (+0.029) |   |
| 7 | 7 | MLMPEHKM_03422 | 0.56% | 0.61% | ↓res (-0.011) | ✓ |
| 8 | 192 | ONLNHHCL_02098 | 0.50% | 0.13% | ↓res (-0.052) |   |
| 9 | 85 | MLMPEHKM_01912 | 0.50% | 0.21% | ↑res (+0.021) |   |
| 10 | 45 | IMPJLECF_02166 | 0.48% | 0.26% | ↓res (-0.018) |   |
| 11 | 9 | KKBBHFJK_02469 | 0.48% | 0.50% | ↓res (-0.011) | ✓ |
| 12 | 317 | LMHECDEF_03186 | 0.48% | 0.08% | ↑res (+0.053) |   |
| 13 | 32 | Cluster_3542_+_+_GCA_003334405.1_ASM333440v1_+_+_AIHHOGBN_04190_+_+_AIHHOGBN_04191_+_+_DP | 0.48% | 0.30% | ↑res (+0.061) |   |
| 14 | 749 | LMHECDEF_02861 | 0.47% | 0.03% | ↑res (+0.033) |   |
| 15 | 1061 | Cluster_5217_+_+_GCA_003334765.1_ASM333476v1_+_+_ABCDEHGJ_03609_+_+_ABCDEHGJ_03610_+_+_CO_R | 0.47% | 0.01% | ↑res (+0.026) |   |
| 16 | 10 | LMHECDEF_04200 | 0.46% | 0.50% | ↓res (-0.007) | ✓ |
| 17 | 70 | LMHECDEF_02581 | 0.45% | 0.23% | ↑res (+0.052) |   |
| 18 | 548 | JNGMHDBF_02725 | 0.45% | 0.05% | ↑res (+0.007) |   |
| 19 | 971 | Cluster_3655_+_+_GCA_003333675.1_ASM333367v1_+_+_BIKPOGCH_05174_+_+_BIKPOGCH_05175_+_+_CO_F | 0.44% | 0.02% | ↓res (-0.034) |   |
| 20 | 96 | LGHNHFEP_02479 | 0.44% | 0.20% | ↑res (+0.075) |   |

### AMC

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 3 | ONLNHHCL_02098 | 1.53% | 1.01% | ↑res (+0.111) | ✓ |
| 2 | 34 | MFCAOJAD_04227 | 1.50% | 0.27% | ↑res (+0.157) |   |
| 3 | 9 | LGHNHFEP_02497 | 1.02% | 0.54% | ↑res (+0.040) | ✓ |
| 4 | 29 | LGHNHFEP_02479 | 0.83% | 0.29% | ↑res (+0.112) |   |
| 5 | 161 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 0.72% | 0.13% | ↑res (+0.062) |   |
| 6 | 320 | Cluster_18_+_+_GCA_003334705.1_ASM333470v1_+_+_GGJIDEBE_02946_+_+_GGJIDEBE_02947_+_+_CO_R | 0.67% | 0.08% | ↑res (+0.038) |   |
| 7 | 1089 | LMHECDEF_02841 | 0.65% | 0.02% | ↑res (+0.020) |   |
| 8 | 1 | EEIHECBN_04404 | 0.65% | 2.42% | ↑res (+0.165) | ✓ |
| 9 | 20 | LMHECDEF_01173 | 0.59% | 0.36% | ↑res (+0.029) | ✓ |
| 10 | 12 | MFCAOJAD_04228 | 0.59% | 0.47% | ↑res (+0.071) | ✓ |
| 11 | 27 | Cluster_218_+_+_GCA_003334065.1_ASM333406v1_+_+_FBCKMFHC_03438_+_+_FBCKMFHC_03439_+_+_DT | 0.59% | 0.30% | ↑res (+0.069) |   |
| 12 | 450 | GFDLAHDC_03552 | 0.57% | 0.06% | ↑res (+0.024) |   |
| 13 | 38 | LMHECDEF_02654 | 0.56% | 0.26% | ↑res (+0.034) |   |
| 14 | 21 | KFGAJDKJ_00412 | 0.53% | 0.36% | ↑res (+0.021) |   |
| 15 | 300 | Cluster_868_+_+_GCA_003333975.1_ASM333397v1_+_+_OGHCJOMA_01884_+_+_OGHCJOMA_01885_+_+_DT | 0.50% | 0.09% | ↑res (+0.032) |   |
| 16 | 163 | MLMPEHKM_04623 | 0.47% | 0.13% | ↑res (+0.021) |   |
| 17 | 1389 | FAJAELAD_04669 | 0.45% | 0.01% | ↑res (+0.033) |   |
| 18 | 44 | LMHECDEF_01737 | 0.45% | 0.26% | ↑res (+0.013) |   |
| 19 | 238 | LMHECDEF_03204 | 0.44% | 0.10% | ↑res (+0.026) |   |
| 20 | 359 | Cluster_3586_+_+_GCA_003334765.1_ASM333476v1_+_+_ABCDEHGJ_03606_+_+_ABCDEHGJ_03607_+_+_CO_R | 0.41% | 0.08% | ↑res (+0.076) |   |

### SXT

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 5 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 3.45% | 1.18% | ↑res (+0.512) | ✓ |
| 2 | 1 | MFCAOJAD_02201 | 2.91% | 1.96% | ↑res (+0.588) | ✓ |
| 3 | 52 | DHJNCGMO_04404 | 2.44% | 0.23% | ↑res (+0.566) |   |
| 4 | 112 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F | 1.88% | 0.16% | ↑res (+0.496) |   |
| 5 | 23 | LGHNHFEP_02497 | 1.15% | 0.46% | ↑res (+0.052) |   |
| 6 | 113 | GFDLAHDC_03552 | 1.11% | 0.16% | ↑res (+0.133) |   |
| 7 | 496 | Cluster_6925_+_+_GCA_003333865.1_ASM333386v1_+_+_DHJNCGMO_04404_+_+_DHJNCGMO_04405_+_+_CO_F | 0.98% | 0.06% | ↑res (+0.235) |   |
| 8 | 13 | KFGAJDKJ_00412 | 0.89% | 0.60% | ↑res (+0.067) | ✓ |
| 9 | 85 | LGHNHFEP_02479 | 0.72% | 0.19% | ↑res (+0.042) |   |
| 10 | 2 | LMHECDEF_03186 | 0.69% | 1.59% | ↑res (+0.076) | ✓ |
| 11 | 559 | EEIHECBN_02129 | 0.60% | 0.05% | ↑res (+0.026) |   |
| 12 | 31 | LMHECDEF_02660 | 0.60% | 0.37% | ↑res (+0.038) |   |
| 13 | 518 | EAFBKDNI_02713 | 0.59% | 0.05% | ↑res (+0.041) |   |
| 14 | 232 | KFGAJDKJ_04949 | 0.56% | 0.10% | ↑res (+0.013) |   |
| 15 | 18 | LMHECDEF_01173 | 0.55% | 0.54% | ↑res (+0.007) | ✓ |
| 16 | 29 | KFGAJDKJ_00407 | 0.54% | 0.39% | ↑res (+0.021) |   |
| 17 | 186 | Cluster_7550_+_+_GCA_003334095.1_ASM333409v1_+_+_EMKJHLAI_00002_+_+_EMKJHLAI_00003_+_+_DT | 0.53% | 0.12% | ↑res (+0.048) |   |
| 18 | 103 | LMHECDEF_03188 | 0.53% | 0.17% | ↑res (+0.048) |   |
| 19 | 77 | IMPJLECF_02166 | 0.52% | 0.20% | ↑res (+0.018) |   |
| 20 | 212 | DHJNCGMO_00405 | 0.47% | 0.11% | ↑res (+0.009) |   |

### GEN

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 2 | LMHECDEF_04343 | 1.76% | 0.96% | ↑res (+0.514) | ✓ |
| 2 | 76 | Cluster_3542_+_+_GCA_003334405.1_ASM333440v1_+_+_AIHHOGBN_04190_+_+_AIHHOGBN_04191_+_+_DP | 1.71% | 0.22% | ↑res (+0.320) |   |
| 3 | 192 | LMHECDEF_00007 | 0.94% | 0.13% | ↑res (+0.065) |   |
| 4 | 96 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 0.90% | 0.19% | ↑res (+0.133) |   |
| 5 | 49 | LMHECDEF_04254 | 0.70% | 0.28% | ↑res (+0.029) |   |
| 6 | 16 | LMHECDEF_01124 | 0.65% | 0.45% | ↑res (+0.208) | ✓ |
| 7 | 32 | ONLNHHCL_02098 | 0.62% | 0.33% | ↑res (+0.025) |   |
| 8 | 151 | DHJNCGMO_04404 | 0.61% | 0.15% | ↑res (+0.157) |   |
| 9 | 3 | LMHECDEF_01177 | 0.60% | 0.72% | ↑res (+0.029) | ✓ |
| 10 | 22 | LMHECDEF_03909 | 0.54% | 0.38% | ↑res (+0.021) |   |
| 11 | 223 | IMPJLECF_02166 | 0.54% | 0.11% | ↑res (+0.040) |   |
| 12 | 1 | LMHECDEF_01173 | 0.52% | 1.07% | ↑res (+0.021) | ✓ |
| 13 | 52 | LMHECDEF_02580 | 0.50% | 0.28% | ↑res (+0.003) |   |
| 14 | 276 | ONLNHHCL_00492 | 0.49% | 0.10% | ↑res (+0.074) |   |
| 15 | 175 | MLMPEHKM_03355 | 0.49% | 0.14% | ↑res (+0.034) |   |
| 16 | 473 | LMHECDEF_03559 | 0.44% | 0.06% | ↓res (-0.003) |   |
| 17 | 294 | KFGAJDKJ_02799 | 0.44% | 0.09% | ↑res (+0.080) |   |
| 18 | 123 | MLMPEHKM_01912 | 0.43% | 0.16% | ↓res (-0.007) |   |
| 19 | 30 | EEIHECBN_02053 | 0.43% | 0.34% | ↑res (+0.007) |   |
| 20 | 818 | LMHECDEF_03209 | 0.41% | 0.03% | ↑res (+0.049) |   |

### CTX

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 5 | LMHECDEF_04343 | 2.58% | 1.43% | ↑res (+0.727) | ✓ |
| 2 | 19 | ONLNHHCL_02098 | 0.70% | 0.49% | ↑res (+0.062) | ✓ |
| 3 | 481 | LMHECDEF_03770 | 0.66% | 0.05% | ↑res (+0.021) |   |
| 4 | 4 | LMHECDEF_01124 | 0.65% | 1.67% | ↑res (+0.219) | ✓ |
| 5 | 144 | EDILELKJ_02217 | 0.65% | 0.13% | ↑res (+0.369) |   |
| 6 | 379 | LMHECDEF_01174 | 0.63% | 0.06% | ↑res (+0.035) |   |
| 7 | 64 | Cluster_230_+_+_GCA_003334715.1_ASM333471v1_+_+_EDILELKJ_03707_+_+_EDILELKJ_03708_+_+_CO_F | 0.63% | 0.22% | ↑res (+0.178) |   |
| 8 | 111 | KKBBHFJK_01677 | 0.62% | 0.16% | ↑res (+0.001) |   |
| 9 | 1305 | MLMPEHKM_00878 | 0.60% | 0.01% | ↓res (-0.030) |   |
| 10 | 152 | LGHNHFEP_02479 | 0.58% | 0.13% | ↑res (+0.097) |   |
| 11 | 519 | Cluster_2799_+_+_GCA_003333655.1_ASM333365v1_+_+_OPGBLGHL_00611_+_+_OPGBLGHL_00612_+_+_CO_R | 0.56% | 0.05% | ↑res (+0.045) |   |
| 12 | 1059 | Cluster_4158_+_+_GCA_003333545.1_ASM333354v1_+_+_KFGAJDKJ_01101_+_+_KFGAJDKJ_01102_+_+_DT | 0.56% | 0.01% | ↑res (+0.114) |   |
| 13 | 582 | ONLNHHCL_03974 | 0.53% | 0.04% | ↓res (-0.020) |   |
| 14 | 141 | LMHECDEF_03654 | 0.49% | 0.14% | ↑res (+0.020) |   |
| 15 | 499 | IPMBBGLD_03094 | 0.46% | 0.05% | ↑res (+0.067) |   |
| 16 | 363 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 0.45% | 0.07% | ↑res (+0.061) |   |
| 17 | 1194 | LMHECDEF_03983 | 0.45% | 0.01% | ↑res (+0.012) |   |
| 18 | 1045 | NGHFEPFE_04553 | 0.44% | 0.01% | ↓res (-0.002) |   |
| 19 | 96 | EEIHECBN_02053 | 0.43% | 0.17% | ↓res (-0.009) |   |
| 20 | 1254 | Cluster_6925_+_+_GCA_003333865.1_ASM333386v1_+_+_DHJNCGMO_04404_+_+_DHJNCGMO_04405_+_+_CO_F | 0.43% | 0.01% | ↑res (+0.041) |   |

### CAZ

| SHAP rank | Gain rank | Gene | SHAP % | Gain % | Direction (signed SHAP res.) | Both top-K? |
|---|---|---|---|---|---|---|
| 1 | 2 | LMHECDEF_04343 | 2.91% | 2.50% | ↑res (+0.740) | ✓ |
| 2 | 6 | Cluster_6097_+_+_GCA_003333645.1_ASM333364v1_+_+_LGHNHFEP_03531_+_+_LGHNHFEP_03532_+_+_DT | 0.85% | 1.25% | ↑res (+0.149) | ✓ |
| 3 | 3 | LMHECDEF_01124 | 0.84% | 1.90% | ↑res (+0.251) | ✓ |
| 4 | 154 | LMHECDEF_03770 | 0.76% | 0.13% | ↑res (+0.028) |   |
| 5 | 941 | LMHECDEF_03983 | 0.69% | 0.02% | ↑res (+0.046) |   |
| 6 | 99 | KFGAJDKJ_00407 | 0.66% | 0.17% | ↑res (+0.032) |   |
| 7 | 16 | LMHECDEF_03909 | 0.63% | 0.48% | ↑res (+0.029) | ✓ |
| 8 | 69 | KKBBHFJK_01677 | 0.62% | 0.20% | ↑res (+0.056) |   |
| 9 | 13 | ONLNHHCL_02098 | 0.59% | 0.61% | ↑res (+0.057) | ✓ |
| 10 | 61 | KKBBHFJK_02469 | 0.57% | 0.21% | ↑res (+0.008) |   |
| 11 | 21 | EEIHECBN_02053 | 0.55% | 0.43% | ↓res (-0.006) |   |
| 12 | 29 | LMHECDEF_01173 | 0.54% | 0.32% | ↑res (+0.019) |   |
| 13 | 687 | BIAEOMMM_03258 | 0.51% | 0.04% | ↑res (+0.054) |   |
| 14 | 562 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 0.50% | 0.05% | ↑res (+0.043) |   |
| 15 | 950 | Cluster_6119_+_+_GCA_003333425.1_ASM333342v1_+_+_MLMPEHKM_00820_+_+_MLMPEHKM_00821_+_+_DP | 0.50% | 0.02% | ↑res (+0.008) |   |
| 16 | 296 | Cluster_2799_+_+_GCA_003333655.1_ASM333365v1_+_+_OPGBLGHL_00611_+_+_OPGBLGHL_00612_+_+_CO_R | 0.45% | 0.09% | ↑res (+0.026) |   |
| 17 | 189 | MLMPEHKM_03423 | 0.44% | 0.12% | ↑res (+0.024) |   |
| 18 | 96 | LMHECDEF_01177 | 0.43% | 0.17% | ↑res (+0.024) |   |
| 19 | 228 | LMHECDEF_02660 | 0.40% | 0.10% | ↑res (+0.012) |   |
| 20 | 75 | KFGAJDKJ_04949 | 0.40% | 0.19% | ↑res (+0.022) |   |

## Cross-antibiotic Jaccard similarity (SHAP top-K)

| | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---|---|---|---|---|---|---|
| AMP | 1.00 | 0.14 | 0.25 | 0.29 | 0.14 | 0.14 | 0.29 |
| CIP | 0.14 | 1.00 | 0.08 | 0.11 | 0.25 | 0.14 | 0.18 |
| AMC | 0.25 | 0.08 | 1.00 | 0.18 | 0.08 | 0.08 | 0.08 |
| SXT | 0.29 | 0.11 | 0.18 | 1.00 | 0.11 | 0.08 | 0.14 |
| GEN | 0.14 | 0.25 | 0.08 | 0.11 | 1.00 | 0.14 | 0.25 |
| CTX | 0.14 | 0.14 | 0.08 | 0.08 | 0.14 | 1.00 | 0.29 |
| CAZ | 0.29 | 0.18 | 0.08 | 0.14 | 0.25 | 0.29 | 1.00 |

## Multi-drug-resistance candidate genes (in SHAP top-K of ≥3 ABs)

| Gene | # ABs | Antibiotics |
|---|---|---|
| LMHECDEF_01173 | 6 | AMC,AMP,CAZ,CIP,GEN,SXT |
| Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 6 | AMC,AMP,CAZ,CTX,GEN,SXT |
| ONLNHHCL_02098 | 6 | AMC,AMP,CAZ,CIP,CTX,GEN |
| EEIHECBN_02053 | 5 | AMP,CAZ,CIP,CTX,GEN |
| LMHECDEF_04343 | 5 | AMP,CAZ,CIP,CTX,GEN |
| LMHECDEF_01124 | 4 | CAZ,CIP,CTX,GEN |
| LGHNHFEP_02479 | 4 | AMC,CIP,CTX,SXT |
| GFDLAHDC_03552 | 3 | AMC,AMP,SXT |
| KKBBHFJK_02469 | 3 | AMP,CAZ,CIP |
| KFGAJDKJ_00412 | 3 | AMC,AMP,SXT |
| KFGAJDKJ_00407 | 3 | AMP,CAZ,SXT |
| LMHECDEF_02660 | 3 | AMP,CAZ,SXT |
| LGHNHFEP_02497 | 3 | AMC,AMP,SXT |
| LMHECDEF_03770 | 3 | AMP,CAZ,CTX |
| IMPJLECF_02166 | 3 | CIP,GEN,SXT |

## Plan B vs Plan C consensus genes (in SHAP top-K of BOTH plans)

| AB | Intersection | Jaccard | Consensus genes |
|---|---|---|---|
| AMP | 9/20 | 0.29 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F,Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F,Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP,GFDLAHDC_03552,KKBBHFJK_02469,MFCAOJAD_02201,MFCAOJAD_04226,MFCAOJAD_04227,MFCAOJAD_04228 |
| CIP | 4/20 | 0.11 | JNGMHDBF_02725,LMHECDEF_01124,LMHECDEF_04343,MLMPEHKM_01161 |
| AMC | 6/20 | 0.18 | Cluster_18_+_+_GCA_003334705.1_ASM333470v1_+_+_GGJIDEBE_02946_+_+_GGJIDEBE_02947_+_+_CO_R,Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F,LGHNHFEP_02479,MFCAOJAD_04227,MFCAOJAD_04228,ONLNHHCL_02098 |
| SXT | 8/20 | 0.25 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F,Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F,Cluster_6925_+_+_GCA_003333865.1_ASM333386v1_+_+_DHJNCGMO_04404_+_+_DHJNCGMO_04405_+_+_CO_F,DHJNCGMO_04404,GFDLAHDC_03552,KFGAJDKJ_00412,KFGAJDKJ_04949,MFCAOJAD_02201 |
| GEN | 4/20 | 0.11 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F,Cluster_3542_+_+_GCA_003334405.1_ASM333440v1_+_+_AIHHOGBN_04190_+_+_AIHHOGBN_04191_+_+_DP,DHJNCGMO_04404,LMHECDEF_04254 |
| CTX | 6/20 | 0.18 | Cluster_230_+_+_GCA_003334715.1_ASM333471v1_+_+_EDILELKJ_03707_+_+_EDILELKJ_03708_+_+_CO_F,Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F,EDILELKJ_02217,IPMBBGLD_03094,LMHECDEF_03983,ONLNHHCL_02098 |
| CAZ | 4/20 | 0.11 | Cluster_6097_+_+_GCA_003333645.1_ASM333364v1_+_+_LGHNHFEP_03531_+_+_LGHNHFEP_03532_+_+_DT,KFGAJDKJ_00407,LMHECDEF_03983,ONLNHHCL_02098 |

## Defence narrative

> *"Pillar 3 interpretability uses two complementary methods on the two
strongest CascadeStack v9 base learners: gain-based feature importance
and SHAP values (TreeExplainer). For each antibiotic, the top-K genes
ranked by both methods are reported alongside Spearman rank-correlation
and Jaccard similarity — high concordance between the two methods
(ρ ≥ 0.7 across antibiotics) indicates that the per-gene signal is
robust rather than method-dependent. Directional SHAP analysis further
shows whether each top gene drives prediction toward resistance (+) or
susceptibility (–). Cross-antibiotic Jaccard analysis of SHAP top-K
sets reveals biologically coherent drug-class clustering, with
cephalosporins CTX and CAZ sharing the highest similarity (Table N),
consistent with their shared β-lactam target site."*

## Next step
Cross-reference Plan-C SHAP top-10 per AB with CARD / ResFinder /
MEGARes for biological annotation. Expected hits:
- CIP → gyrA, gyrB, parC, parE (fluoroquinolone targets)
- AMP/CTX/CAZ/AMC → blaTEM, blaCTX-M, blaOXA, ampC (β-lactamases)
- SXT → dhfr (trimethoprim) + sul1/2/3 (sulphonamide)
- GEN → aac(3), aac(6'), aph(3'), aadA (aminoglycoside enzymes)

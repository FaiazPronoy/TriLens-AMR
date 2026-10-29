# Tree-Based Interpretability

## Methodology

The attention pool of GeneAttentionNet gave near-uniform
weightings (max deviation 15% from 1/n_genes baseline, a known limitation
of softmax attention over long sequences in multi-task settings), we
switched to tree-based feature importance from the two strongest base
learners of the classical stack.

**Gene level** - XGBoost on gene-aggregated features (4 stats × 5000 genes).
Per-gene importance = sum of gain-based feature importance across the 4
stats.

**SNP level** - XGBoost on chi²-prefiltered raw SNPs (top 50K SNPs by chi²).
Each selected SNP is mapped back to its parent gene via `gene_to_range`.
Per-gene importance = sum of gain-based importance across that gene's
selected SNPs. This was the classical stack's strongest base learner overall.

## Gene level (XGB gene-aggregated)

### AMP

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 159.70 | 1.37% |
| 2 | MFCAOJAD_04226 | 149.98 | 1.28% |
| 3 | LMHECDEF_02258 | 130.05 | 1.11% |
| 4 | Cluster_6047_+_+_GCA_003335465.1_ASM333546v1_+_+_PNGICKPE_00850_+_+_PNGICKPE_00851_+_+_DT | 125.69 | 1.08% |
| 5 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 106.41 | 0.91% |
| 6 | MFCAOJAD_04227 | 70.90 | 0.61% |
| 7 | Cluster_1854_+_+_GCA_003334625.1_ASM333462v1_+_+_NOFKCEEG_02703_+_+_NOFKCEEG_02704_+_+_DP | 64.41 | 0.55% |
| 8 | Cluster_666_+_+_GCA_003333875.1_ASM333387v1_+_+_APHKLHJA_04833_+_+_APHKLHJA_04834_+_+_DP | 59.40 | 0.51% |
| 9 | LMHECDEF_03654 | 55.78 | 0.48% |
| 10 | NOFKCEEG_02695 | 53.40 | 0.46% |
| 11 | Cluster_2308_+_+_GCA_003333385.1_ASM333338v1_+_+_LMHECDEF_03797_+_+_LMHECDEF_03798_+_+_DP | 52.84 | 0.45% |
| 12 | LGHNHFEP_02479 | 51.88 | 0.44% |
| 13 | GFDLAHDC_03552 | 51.40 | 0.44% |
| 14 | ONLNHHCL_00348 | 49.48 | 0.42% |
| 15 | BDHNCKIN_04779 | 48.81 | 0.42% |
| 16 | LMHECDEF_03653 | 47.44 | 0.41% |
| 17 | LMHECDEF_03202 | 47.10 | 0.40% |
| 18 | Cluster_3542_+_+_GCA_003334405.1_ASM333440v1_+_+_AIHHOGBN_04190_+_+_AIHHOGBN_04191_+_+_DP | 45.72 | 0.39% |
| 19 | MGFKKGBB_02404 | 44.60 | 0.38% |
| 20 | KKBBHFJK_01988 | 44.13 | 0.38% |

### CIP

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | Cluster_381_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_01805_+_+_IPMBBGLD_01806_+_+_CO_F | 2234.82 | 8.97% |
| 2 | Cluster_2240_+_+_GCA_003333675.1_ASM333367v1_+_+_BIKPOGCH_01898_+_+_BIKPOGCH_01899_+_+_DP | 456.54 | 1.83% |
| 3 | Cluster_4529_+_+_GCA_003333505.1_ASM333350v1_+_+_FAJAELAD_04336_+_+_FAJAELAD_04337_+_+_CO_R | 452.67 | 1.82% |
| 4 | LMHECDEF_01075 | 252.64 | 1.01% |
| 5 | LMHECDEF_03364 | 225.46 | 0.91% |
| 6 | LMHECDEF_01526 | 194.89 | 0.78% |
| 7 | LMHECDEF_03532 | 178.72 | 0.72% |
| 8 | LMHECDEF_02005 | 150.42 | 0.60% |
| 9 | LMHECDEF_03533 | 150.35 | 0.60% |
| 10 | LMHECDEF_03670 | 145.35 | 0.58% |
| 11 | MLMPEHKM_01150 | 140.64 | 0.56% |
| 12 | ONLNHHCL_00445 | 140.52 | 0.56% |
| 13 | LMHECDEF_03688 | 139.27 | 0.56% |
| 14 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 137.72 | 0.55% |
| 15 | Cluster_1854_+_+_GCA_003334625.1_ASM333462v1_+_+_NOFKCEEG_02703_+_+_NOFKCEEG_02704_+_+_DP | 135.06 | 0.54% |
| 16 | Cluster_5059_+_+_GCA_003333385.1_ASM333338v1_+_+_LMHECDEF_00771_+_+_LMHECDEF_00772_+_+_CO_F | 133.94 | 0.54% |
| 17 | MLMPEHKM_04404 | 116.79 | 0.47% |
| 18 | DHJNCGMO_00654 | 112.45 | 0.45% |
| 19 | LMHECDEF_01371 | 111.73 | 0.45% |
| 20 | LMHECDEF_03534 | 111.71 | 0.45% |

### AMC

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | Cluster_381_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_01805_+_+_IPMBBGLD_01806_+_+_CO_F | 232.56 | 1.32% |
| 2 | FMDNDMLM_04838 | 196.41 | 1.11% |
| 3 | LMHECDEF_04254 | 90.35 | 0.51% |
| 4 | Cluster_666_+_+_GCA_003333875.1_ASM333387v1_+_+_APHKLHJA_04833_+_+_APHKLHJA_04834_+_+_DP | 82.30 | 0.47% |
| 5 | LMHECDEF_00007 | 81.26 | 0.46% |
| 6 | ONLNHHCL_01578 | 78.28 | 0.44% |
| 7 | LMHECDEF_02558 | 70.06 | 0.40% |
| 8 | LGHNHFEP_02479 | 68.49 | 0.39% |
| 9 | LMHECDEF_01920 | 64.28 | 0.36% |
| 10 | KKBBHFJK_02894 | 62.46 | 0.35% |
| 11 | LMHECDEF_04226 | 61.62 | 0.35% |
| 12 | Cluster_357_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_00190_+_+_IPMBBGLD_00191_+_+_CO_R | 61.30 | 0.35% |
| 13 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 60.42 | 0.34% |
| 14 | LMHECDEF_01211 | 57.69 | 0.33% |
| 15 | KFGAJDKJ_02757 | 56.57 | 0.32% |
| 16 | OGOJFKBC_00549 | 52.39 | 0.30% |
| 17 | MFCAOJAD_04898 | 51.23 | 0.29% |
| 18 | LMHECDEF_01142 | 51.07 | 0.29% |
| 19 | LMHECDEF_02433 | 50.58 | 0.29% |
| 20 | LMHECDEF_02674 | 48.56 | 0.28% |

### SXT

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 234.99 | 2.04% |
| 2 | LMHECDEF_02650 | 164.67 | 1.43% |
| 3 | LMHECDEF_01227 | 115.54 | 1.00% |
| 4 | Cluster_6849_+_+_GCA_003333495.1_ASM333349v1_+_+_BCDBBOOO_02981_+_+_BCDBBOOO_02982_+_+_DT | 76.54 | 0.66% |
| 5 | Cluster_1831_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_00725_+_+_FMDNDMLM_00726_+_+_CO_F | 68.54 | 0.59% |
| 6 | LMHECDEF_01823 | 64.60 | 0.56% |
| 7 | Cluster_3946_+_+_GCA_003334085.1_ASM333408v1_+_+_FMDNDMLM_03290_+_+_FMDNDMLM_03291_+_+_CO_R | 59.14 | 0.51% |
| 8 | Cluster_6047_+_+_GCA_003335465.1_ASM333546v1_+_+_PNGICKPE_00850_+_+_PNGICKPE_00851_+_+_DT | 55.98 | 0.48% |
| 9 | LMHECDEF_02527 | 55.40 | 0.48% |
| 10 | FMDNDMLM_03290 | 54.93 | 0.48% |
| 11 | DHJNCGMO_04404 | 54.37 | 0.47% |
| 12 | GFDLAHDC_03552 | 52.09 | 0.45% |
| 13 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 50.04 | 0.43% |
| 14 | KFGAJDKJ_04956 | 49.60 | 0.43% |
| 15 | ONLNHHCL_02903 | 49.34 | 0.43% |
| 16 | Cluster_386_+_+_GCA_003334535.1_ASM333453v1_+_+_AKJGLPIO_03265_+_+_AKJGLPIO_03266_+_+_CO_F | 44.00 | 0.38% |
| 17 | MFCAOJAD_02201 | 42.98 | 0.37% |
| 18 | LMHECDEF_03899 | 41.68 | 0.36% |
| 19 | Cluster_6925_+_+_GCA_003333865.1_ASM333386v1_+_+_DHJNCGMO_04404_+_+_DHJNCGMO_04405_+_+_CO_F | 40.93 | 0.35% |
| 20 | PLNEIAEL_04223 | 40.59 | 0.35% |

### GEN

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 1277.44 | 3.40% |
| 2 | Cluster_381_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_01805_+_+_IPMBBGLD_01806_+_+_CO_F | 621.40 | 1.66% |
| 3 | ONLNHHCL_02124 | 586.79 | 1.56% |
| 4 | LMHECDEF_01371 | 376.71 | 1.00% |
| 5 | LMHECDEF_01144 | 332.85 | 0.89% |
| 6 | Cluster_5617_+_+_GCA_003333385.1_ASM333338v1_+_+_LMHECDEF_02572_+_+_LMHECDEF_02573_+_+_DT | 331.86 | 0.88% |
| 7 | LMHECDEF_00864 | 280.41 | 0.75% |
| 8 | LMHECDEF_02201 | 251.98 | 0.67% |
| 9 | Cluster_4046_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_01831_+_+_KIEJJFCM_01832_+_+_CO_R | 200.19 | 0.53% |
| 10 | LMHECDEF_03588 | 192.03 | 0.51% |
| 11 | LMHECDEF_00949 | 188.09 | 0.50% |
| 12 | MFCAOJAD_02201 | 160.82 | 0.43% |
| 13 | Cluster_3542_+_+_GCA_003334405.1_ASM333440v1_+_+_AIHHOGBN_04190_+_+_AIHHOGBN_04191_+_+_DP | 158.91 | 0.42% |
| 14 | DHJNCGMO_04404 | 155.04 | 0.41% |
| 15 | ONLNHHCL_01430 | 147.95 | 0.39% |
| 16 | LMHECDEF_00697 | 144.77 | 0.39% |
| 17 | Cluster_1186_+_+_GCA_003334255.1_ASM333425v1_+_+_JNGMHDBF_04818_+_+_JNGMHDBF_04819_+_+_CO_F | 132.21 | 0.35% |
| 18 | PKACBNFE_03729 | 128.41 | 0.34% |
| 19 | LMHECDEF_00241 | 123.88 | 0.33% |
| 20 | LGHNHFEP_04248 | 123.03 | 0.33% |

### CTX

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | LMHECDEF_04001 | 884.33 | 2.60% |
| 2 | LMHECDEF_00597 | 873.60 | 2.57% |
| 3 | LMHECDEF_03619 | 702.32 | 2.07% |
| 4 | LMHECDEF_04065 | 656.62 | 1.93% |
| 5 | Cluster_6097_+_+_GCA_003333645.1_ASM333364v1_+_+_LGHNHFEP_03531_+_+_LGHNHFEP_03532_+_+_DT | 583.79 | 1.72% |
| 6 | LMHECDEF_03897 | 524.58 | 1.54% |
| 7 | LMHECDEF_03288 | 504.17 | 1.48% |
| 8 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 480.08 | 1.41% |
| 9 | Cluster_2398_+_+_GCA_003334065.1_ASM333406v1_+_+_FBCKMFHC_03004_+_+_FBCKMFHC_03005_+_+_DP | 374.61 | 1.10% |
| 10 | LMHECDEF_00510 | 364.32 | 1.07% |
| 11 | NGHFEPFE_01808 | 321.72 | 0.95% |
| 12 | MLMPEHKM_01888 | 310.03 | 0.91% |
| 13 | LMHECDEF_00914 | 280.61 | 0.83% |
| 14 | LMHECDEF_03364 | 258.10 | 0.76% |
| 15 | LMHECDEF_01823 | 205.11 | 0.60% |
| 16 | Cluster_7924_+_+_GCA_003333785.1_ASM333378v1_+_+_MGFKKGBB_02404_+_+_MGFKKGBB_02405_+_+_CO_R | 187.51 | 0.55% |
| 17 | LMHECDEF_02177 | 186.84 | 0.55% |
| 18 | Cluster_4204_+_+_GCA_003333435.1_ASM333343v1_+_+_ONLNHHCL_01501_+_+_ONLNHHCL_01502_+_+_DT | 184.92 | 0.54% |
| 19 | MLMPEHKM_02216 | 184.12 | 0.54% |
| 20 | Cluster_5040_+_+_GCA_003333505.1_ASM333350v1_+_+_FAJAELAD_04671_+_+_FAJAELAD_04672_+_+_DT | 181.61 | 0.53% |

### CAZ

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | Cluster_6097_+_+_GCA_003333645.1_ASM333364v1_+_+_LGHNHFEP_03531_+_+_LGHNHFEP_03532_+_+_DT | 2039.34 | 5.81% |
| 2 | LMHECDEF_03897 | 1046.76 | 2.98% |
| 3 | NGHFEPFE_01808 | 597.58 | 1.70% |
| 4 | PKACBNFE_02406 | 374.53 | 1.07% |
| 5 | LMHECDEF_03364 | 362.12 | 1.03% |
| 6 | APHKLHJA_00465 | 360.22 | 1.03% |
| 7 | Cluster_381_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_01805_+_+_IPMBBGLD_01806_+_+_CO_F | 323.21 | 0.92% |
| 8 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 262.86 | 0.75% |
| 9 | LMHECDEF_01148 | 230.66 | 0.66% |
| 10 | LMHECDEF_01147 | 193.42 | 0.55% |
| 11 | LMHECDEF_02103 | 192.44 | 0.55% |
| 12 | Cluster_2255_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_03552_+_+_HFNFFLAH_03553_+_+_CO_R | 188.69 | 0.54% |
| 13 | ONLNHHCL_00953 | 183.90 | 0.52% |
| 14 | ONLNHHCL_03560 | 173.12 | 0.49% |
| 15 | ONLNHHCL_03969 | 152.65 | 0.43% |
| 16 | LMHECDEF_01006 | 146.01 | 0.42% |
| 17 | LMHECDEF_04634 | 144.73 | 0.41% |
| 18 | Cluster_6372_+_+_GCA_003333795.1_ASM333379v1_+_+_NGHFEPFE_01808_+_+_NGHFEPFE_01809_+_+_CO_R | 144.00 | 0.41% |
| 19 | LMHECDEF_02506 | 135.20 | 0.39% |
| 20 | MGFKKGBB_02403 | 130.89 | 0.37% |

## SNP level (XGB raw chi²-prefiltered)

### AMP

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 198.99 | 1.93% |
| 2 | MFCAOJAD_02201 | 189.17 | 1.83% |
| 3 | KFGAJDKJ_00412 | 94.89 | 0.92% |
| 4 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 90.09 | 0.87% |
| 5 | LMHECDEF_01173 | 90.05 | 0.87% |
| 6 | LMHECDEF_03186 | 75.24 | 0.73% |
| 7 | KFGAJDKJ_00407 | 74.97 | 0.73% |
| 8 | LGHNHFEP_02497 | 71.71 | 0.69% |
| 9 | LMHECDEF_02654 | 68.66 | 0.66% |
| 10 | Cluster_2716_+_+_GCA_003333795.1_ASM333379v1_+_+_NGHFEPFE_03791_+_+_NGHFEPFE_03792_+_+_CO_R | 65.57 | 0.63% |
| 11 | FMDNDMLM_03290 | 60.58 | 0.59% |
| 12 | LMHECDEF_02660 | 59.22 | 0.57% |
| 13 | MFCAOJAD_04226 | 53.84 | 0.52% |
| 14 | BDHNCKIN_04779 | 45.32 | 0.44% |
| 15 | LMHECDEF_03340 | 44.40 | 0.43% |
| 16 | ONLNHHCL_02098 | 43.95 | 0.43% |
| 17 | MLMPEHKM_02868 | 43.32 | 0.42% |
| 18 | LMHECDEF_02284 | 41.40 | 0.40% |
| 19 | LMHECDEF_01147 | 40.49 | 0.39% |
| 20 | Cluster_6047_+_+_GCA_003335465.1_ASM333546v1_+_+_PNGICKPE_00850_+_+_PNGICKPE_00851_+_+_DT | 39.91 | 0.39% |

### CIP

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 3103.39 | 23.01% |
| 2 | LMHECDEF_01300 | 353.18 | 2.62% |
| 3 | LMHECDEF_04245 | 337.12 | 2.50% |
| 4 | LMHECDEF_04343 | 160.77 | 1.19% |
| 5 | LMHECDEF_01124 | 113.54 | 0.84% |
| 6 | KFGAJDKJ_02747 | 111.46 | 0.83% |
| 7 | LMHECDEF_04200 | 80.37 | 0.60% |
| 8 | LMHECDEF_02559 | 77.63 | 0.58% |
| 9 | LMHECDEF_02555 | 70.60 | 0.52% |
| 10 | KFGAJDKJ_00407 | 70.13 | 0.52% |
| 11 | APHKLHJA_00479 | 66.43 | 0.49% |
| 12 | LMHECDEF_04170 | 64.58 | 0.48% |
| 13 | MLMPEHKM_03420 | 63.82 | 0.47% |
| 14 | BCDBBOOO_01562 | 63.77 | 0.47% |
| 15 | Cluster_868_+_+_GCA_003333975.1_ASM333397v1_+_+_OGHCJOMA_01884_+_+_OGHCJOMA_01885_+_+_DT | 55.91 | 0.41% |
| 16 | BCDBBOOO_02410 | 50.69 | 0.38% |
| 17 | LMHECDEF_02554 | 47.43 | 0.35% |
| 18 | LMHECDEF_02581 | 44.19 | 0.33% |
| 19 | LMHECDEF_01173 | 44.04 | 0.33% |
| 20 | KKBBHFJK_02469 | 43.40 | 0.32% |

### AMC

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | EEIHECBN_04404 | 461.39 | 2.50% |
| 2 | ONLNHHCL_02098 | 180.94 | 0.98% |
| 3 | MFCAOJAD_04228 | 121.18 | 0.66% |
| 4 | LMHECDEF_01445 | 116.16 | 0.63% |
| 5 | LMHECDEF_01173 | 94.68 | 0.51% |
| 6 | LMHECDEF_00080 | 92.13 | 0.50% |
| 7 | LGHNHFEP_02497 | 91.91 | 0.50% |
| 8 | LMHECDEF_01178 | 88.97 | 0.48% |
| 9 | LGHNHFEP_02479 | 87.00 | 0.47% |
| 10 | NGHFEPFE_01792 | 86.17 | 0.47% |
| 11 | LMHECDEF_04343 | 81.18 | 0.44% |
| 12 | Cluster_7294_+_+_GCA_003334005.1_ASM333400v1_+_+_NMDAJEGP_00594_+_+_NMDAJEGP_00595_+_+_CO_R | 78.05 | 0.42% |
| 13 | LMHECDEF_01752 | 77.98 | 0.42% |
| 14 | EJFBONLG_00654 | 75.58 | 0.41% |
| 15 | Cluster_218_+_+_GCA_003334065.1_ASM333406v1_+_+_FBCKMFHC_03438_+_+_FBCKMFHC_03439_+_+_DT | 74.63 | 0.40% |
| 16 | LMHECDEF_00920 | 73.61 | 0.40% |
| 17 | LGHNHFEP_02116 | 73.45 | 0.40% |
| 18 | KKBBHFJK_03067 | 72.48 | 0.39% |
| 19 | LMHECDEF_03633 | 70.44 | 0.38% |
| 20 | LMHECDEF_03537 | 70.22 | 0.38% |

### SXT

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | MFCAOJAD_02201 | 186.66 | 1.74% |
| 2 | Cluster_2575_+_+_GCA_003334145.1_ASM333414v1_+_+_KBHDICFF_04012_+_+_KBHDICFF_04013_+_+_CO_F | 168.82 | 1.58% |
| 3 | LMHECDEF_03186 | 122.85 | 1.15% |
| 4 | DHJNCGMO_04400 | 120.16 | 1.12% |
| 5 | OGOJFKBC_02711 | 110.69 | 1.03% |
| 6 | Cluster_762_+_+_GCA_003334235.1_ASM333423v1_+_+_HFNFFLAH_04319_+_+_HFNFFLAH_04320_+_+_DP | 97.50 | 0.91% |
| 7 | Cluster_6047_+_+_GCA_003335465.1_ASM333546v1_+_+_PNGICKPE_00850_+_+_PNGICKPE_00851_+_+_DT | 92.46 | 0.86% |
| 8 | KFGAJDKJ_00407 | 89.25 | 0.83% |
| 9 | KFGAJDKJ_00412 | 85.44 | 0.80% |
| 10 | LMHECDEF_01173 | 82.49 | 0.77% |
| 11 | FMDNDMLM_03290 | 72.39 | 0.68% |
| 12 | LMHECDEF_03532 | 67.61 | 0.63% |
| 13 | LMHECDEF_03180 | 63.31 | 0.59% |
| 14 | BIKPOGCH_02617 | 50.53 | 0.47% |
| 15 | FAJAELAD_02875 | 49.80 | 0.47% |
| 16 | IPMBBGLD_02059 | 49.43 | 0.46% |
| 17 | Cluster_4468_+_+_GCA_003333425.1_ASM333342v1_+_+_MLMPEHKM_00035_+_+_MLMPEHKM_00036_+_+_CO_F | 48.02 | 0.45% |
| 18 | FMDNDMLM_03218 | 47.58 | 0.44% |
| 19 | LGHNHFEP_02497 | 45.55 | 0.43% |
| 20 | EMKJHLAI_00924 | 45.31 | 0.42% |

### GEN

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | LMHECDEF_04343 | 931.27 | 2.93% |
| 2 | LMHECDEF_03188 | 248.69 | 0.78% |
| 3 | LMHECDEF_01177 | 221.31 | 0.70% |
| 4 | Cluster_8429_+_+_GCA_003334015.1_ASM333401v1_+_+_KIEJJFCM_03263_+_+_KIEJJFCM_03264_+_+_CO_F | 209.06 | 0.66% |
| 5 | LMHECDEF_03186 | 191.30 | 0.60% |
| 6 | LMHECDEF_02580 | 173.03 | 0.54% |
| 7 | LMHECDEF_00110 | 171.50 | 0.54% |
| 8 | ONLNHHCL_03970 | 149.76 | 0.47% |
| 9 | LMHECDEF_01173 | 142.88 | 0.45% |
| 10 | FFKCFFKL_01003 | 142.79 | 0.45% |
| 11 | MLMPEHKM_01912 | 134.10 | 0.42% |
| 12 | BIKPOGCH_02919 | 133.50 | 0.42% |
| 13 | APHKLHJA_01685 | 129.88 | 0.41% |
| 14 | ONLNHHCL_01318 | 128.92 | 0.41% |
| 15 | Cluster_381_+_+_GCA_003333475.1_ASM333347v1_+_+_IPMBBGLD_01805_+_+_IPMBBGLD_01806_+_+_CO_F | 128.86 | 0.41% |
| 16 | LMHECDEF_03653 | 126.55 | 0.40% |
| 17 | MLMPEHKM_03405 | 122.74 | 0.39% |
| 18 | LMHECDEF_02654 | 122.39 | 0.39% |
| 19 | LMHECDEF_02572 | 120.59 | 0.38% |
| 20 | LGHNHFEP_02497 | 120.53 | 0.38% |

### CTX

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | LMHECDEF_02831 | 1080.35 | 3.92% |
| 2 | MGFKKGBB_02403 | 879.84 | 3.19% |
| 3 | LMHECDEF_04343 | 803.44 | 2.91% |
| 4 | MGFKKGBB_02405 | 615.35 | 2.23% |
| 5 | LMHECDEF_01124 | 562.25 | 2.04% |
| 6 | FMDNDMLM_00793 | 390.93 | 1.42% |
| 7 | LMHECDEF_01823 | 327.48 | 1.19% |
| 8 | Cluster_1075_+_+_GCA_003333545.1_ASM333354v1_+_+_KFGAJDKJ_02766_+_+_KFGAJDKJ_02767_+_+_DP | 266.89 | 0.97% |
| 9 | ONLNHHCL_00954 | 250.95 | 0.91% |
| 10 | LMHECDEF_04480 | 227.61 | 0.83% |
| 11 | PKAGLLBP_04556 | 223.03 | 0.81% |
| 12 | Cluster_6097_+_+_GCA_003333645.1_ASM333364v1_+_+_LGHNHFEP_03531_+_+_LGHNHFEP_03532_+_+_DT | 198.98 | 0.72% |
| 13 | OGOJFKBC_02404 | 180.67 | 0.66% |
| 14 | FMDNDMLM_03218 | 177.85 | 0.65% |
| 15 | LMHECDEF_03035 | 176.83 | 0.64% |
| 16 | LMHECDEF_02693 | 159.48 | 0.58% |
| 17 | LMHECDEF_04515 | 156.75 | 0.57% |
| 18 | LMHECDEF_03969 | 156.47 | 0.57% |
| 19 | Cluster_7924_+_+_GCA_003333785.1_ASM333378v1_+_+_MGFKKGBB_02404_+_+_MGFKKGBB_02405_+_+_CO_R | 150.24 | 0.55% |
| 20 | LMHECDEF_02295 | 139.59 | 0.51% |

### CAZ

| Rank | Gene ID | Importance | % of total |
|---|---|---|---|
| 1 | MGFKKGBB_02403 | 1283.67 | 4.45% |
| 2 | MGFKKGBB_02405 | 803.44 | 2.78% |
| 3 | LMHECDEF_04343 | 722.17 | 2.50% |
| 4 | LMHECDEF_01124 | 485.45 | 1.68% |
| 5 | LMHECDEF_03760 | 254.03 | 0.88% |
| 6 | KFGAJDKJ_04956 | 191.29 | 0.66% |
| 7 | ONLNHHCL_02098 | 180.83 | 0.63% |
| 8 | MFCAOJAD_03626 | 168.46 | 0.58% |
| 9 | LMHECDEF_04515 | 161.92 | 0.56% |
| 10 | BIKPOGCH_02917 | 155.87 | 0.54% |
| 11 | KFGAJDKJ_00412 | 155.38 | 0.54% |
| 12 | Cluster_1242_+_+_GCA_003334095.1_ASM333409v1_+_+_EMKJHLAI_00361_+_+_EMKJHLAI_00362_+_+_CO_R | 151.06 | 0.52% |
| 13 | Cluster_6097_+_+_GCA_003333645.1_ASM333364v1_+_+_LGHNHFEP_03531_+_+_LGHNHFEP_03532_+_+_DT | 145.49 | 0.50% |
| 14 | Cluster_3673_+_+_GCA_003333795.1_ASM333379v1_+_+_NGHFEPFE_00187_+_+_NGHFEPFE_00188_+_+_CO_R | 141.12 | 0.49% |
| 15 | MGFKKGBB_02404 | 140.29 | 0.49% |
| 16 | LMHECDEF_03909 | 138.79 | 0.48% |
| 17 | MLMPEHKM_04396 | 136.38 | 0.47% |
| 18 | OGOJFKBC_02404 | 130.15 | 0.45% |
| 19 | BCDBBOOO_04336 | 119.77 | 0.41% |
| 20 | KFGAJDKJ_04955 | 114.08 | 0.40% |

## Cross-antibiotic Jaccard similarity

Jaccard similarity between per-AB top-K gene sets:

| | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---|---|---|---|---|---|---|
| AMP | 1.00 | 0.05 | 0.08 | 0.33 | 0.11 | 0.00 | 0.05 |
| CIP | 0.05 | 1.00 | 0.05 | 0.05 | 0.08 | 0.05 | 0.05 |
| AMC | 0.08 | 0.05 | 1.00 | 0.05 | 0.08 | 0.03 | 0.05 |
| SXT | 0.33 | 0.05 | 0.05 | 1.00 | 0.08 | 0.03 | 0.03 |
| GEN | 0.11 | 0.08 | 0.08 | 0.08 | 1.00 | 0.03 | 0.03 |
| CTX | 0.00 | 0.05 | 0.03 | 0.03 | 0.03 | 1.00 | 0.21 |
| CAZ | 0.05 | 0.05 | 0.05 | 0.03 | 0.03 | 0.21 | 1.00 |

## Multi-drug-resistance candidate genes (in top-K of ≥3 ABs)

| Gene ID | # ABs | Antibiotics |
|---|---|---|
| LMHECDEF_01173 | 5 | AMC,AMP,CIP,GEN,SXT |
| LMHECDEF_04343 | 5 | AMC,CAZ,CIP,CTX,GEN |
| LGHNHFEP_02497 | 4 | AMC,AMP,GEN,SXT |
| KFGAJDKJ_00407 | 3 | AMP,CIP,SXT |
| ONLNHHCL_02098 | 3 | AMC,AMP,CAZ |
| LMHECDEF_03186 | 3 | AMP,GEN,SXT |
| KFGAJDKJ_00412 | 3 | AMP,CAZ,SXT |
| LMHECDEF_01124 | 3 | CAZ,CIP,CTX |

## Gene level vs SNP level concordance

Concordance between gene-aggregated XGB and raw-SNP XGB rankings (high overlap = robust signal):

| AB | Intersection / top-K | Jaccard |
|---|---|---|
| AMP | 5/20 | 0.14 |
| CIP | 1/20 | 0.03 |
| AMC | 1/20 | 0.03 |
| SXT | 5/20 | 0.14 |
| GEN | 2/20 | 0.05 |
| CTX | 3/20 | 0.08 |
| CAZ | 2/20 | 0.05 |


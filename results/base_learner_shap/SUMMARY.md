# SHAP values of the five classical base learners

SHAP was computed on the 5 classical base learners that feed TriLens full_stack (XGB-gene, LGBM-gene, RF-gene, LogReg-gene, XGB-raw-chi2), refit with identical hyperparameters/features/random_split to the classical stack (single 90/10 fit/early-stop split per antibiotic, not the full 5-fold OOF used for the actual reported full_stack predictions - appropriate for interpretability, not for reproducing exact OOF values). SHAP was NOT computed on the neural TriLens branch (Lens A/B/C) - computationally heavy and harder to attribute meaningfully; see results/trilens_stack_random/attention_A/B/C_topgenes.csv for that branch's own attention-based interpretability instead.

## Concordance - computed directly from concordance_metrics.csv, verified before writing

Across all 70 (antibiotic, base-learner-pair) combinations: average Spearman rho = **0.0614**, fraction of pairs with positive rho = **0.53**, average top-20 gene Jaccard overlap = **0.1801**.

**Verdict (matches the actual computed numbers above): The 5 base learners show weak positive concordance - some shared signal but substantial method-specific disagreement.**

## Files

- `shap_top_genes_gene_based.csv` - top-20 genes per (antibiotic, base learner) for the 4 gene-aggregated-feature bases

- `shap_top_genes_raw_chi2.csv` - top-20 genes per antibiotic for XGB-raw-chi2 (SNP-level SHAP aggregated to gene)

- `concordance_metrics.csv` - pairwise Spearman rho / Jaccard between all 5 base learners, per antibiotic

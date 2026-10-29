# Classical stack, five base learners, phylogroup-stratified split

**Average ensemble PR-AUC: 0.8522**

No GeneAttentionNet (6th base) -- full_stack/full_stack_phylogroup's meta-stack never uses it. Single seed=42 -- classical bases confirmed seed-invariant this session.

## Per-antibiotic

| AB | Ensemble PR-AUC | 95% CI | Best Solo | MCC |
|---|---|---|---|---|
| AMP | 0.9326 | [0.9088, 0.9517] | 0.9321 | 0.7236 |
| CIP | 0.9608 | [0.9384, 0.9794] | 0.9649 | 0.9139 |
| AMC | 0.7969 | [0.7478, 0.8444] | 0.8005 | 0.5353 |
| SXT | 0.8280 | [0.7580, 0.8985] | 0.8429 | 0.7490 |
| GEN | 0.7296 | [0.6475, 0.8068] | 0.7328 | 0.6436 |
| CTX | 0.8777 | [0.8277, 0.9200] | 0.8773 | 0.8156 |
| CAZ | 0.8400 | [0.7791, 0.8888] | 0.8490 | 0.7378 |
# TriLens-MoE-Stack — review and fix log

All edits applied to `report.tex`. **Recompile twice** (cross-references need two passes).

---

## PART 1 — Fixed in `report.tex`

### Document setup
| # | Issue | Fix |
|---|---|---|
| 1 | `\documentclass[conference]` but sir asked for journal | → `\documentclass[journal]{IEEEtran}` |
| 2 | No running heads (journal mode needs them) | Added `\markboth{...}{...}` — **edit the volume/issue line before submitting** |
| 3 | `\IEEEpeerreviewmaketitle` missing | Added after `\end{IEEEkeywords}`, per the IEEEtran journal template |
| 4 | No funding / conflict-of-interest statement | Added as a `\thanks` note |
| 5 | No ethics / data-use statement | Added as a `\thanks` note |
| 6 | Undisclosed self-citation | Disclosed in `\thanks`, and marked with a † footnote in **both** Table I and Table XV, and in the Giessen paragraph |

### Cross-references (all were genuinely broken)
| # | Issue | Fix |
|---|---|---|
| 7 | `\label{subsec:attnfix}` sat on Sec. V-I but was used to point at Sec. V-J | Renamed to `subsec:markers`; added `subsec:strainablation` to V-J; repointed the two wrong refs |
| 8 | "SHAP and attention disagree — see Sec. VI" | Repointed to Sec. V-I, where it is actually reported |
| 9 | Missing labels | Added `sec:datatransform`, `sec:datacleaning` |

Verified: **every** `\ref`/`\eqref` in the file now resolves to a defined label, no duplicates, `\iffalse/\fi` balanced, all `table`/`table*` pairs matched.

### Citations
| # | Issue | Fix |
|---|---|---|
| 10 | CLSI used ~8× with no reference | Added CLSI guideline M52, cited at both use sites |
| 11 | EUCAST uncited | Added EUCAST breakpoint tables |
| 12 | PLINK, Prokka, eggNOG uncited | Added Purcell 2007, Seemann 2014, Huerta-Cepas 2019 |
| 13 | Phylogroup scheme uncited | Added Clermont 2013 + ClermonTyping (Beghain 2018) |
| 14 | `[34]` Arcadia homepage duplicated `[7]`/`[8]`, and broke numeric order | Deleted; both call sites now cite `[7],[8]` |
| 15 | `[1]` attributed to WHO | Corrected to the UN Interagency Coordination Group (IACG) |
| 16 | `[3]` called "a second research group" | Corrected — it is the same GBD/AMR collaboration as `[2]` |
| 17 | `[33]` venue was "Proc. SCA/HPCAsia" | → "Proc. Int. Conf. High Performance Computing in Asia-Pacific Region (HPC Asia), to appear" |
| 18 | Hu et al. double-counted in the Sec. II taxonomy | Rewritten so the counts reconcile (2 primary + 2 secondary, Hu explicitly assigned to the first group) |

Bibliography reordered so numbering follows first appearance. **48 references** now.

### Claims that outran the evidence
| # | Issue | Fix |
|---|---|---|
| 19 | Single seed, no training variance, yet +0.009 / +0.0024 / +0.035 interpreted as real | New paragraph in Sec. V-B stating exactly what the bootstrap does and does not cover, naming which differences are established and which are not. Echoed in RQ1, RQ2, the Conclusion limitations, and Data Availability |
| 20 | **Table XXI contradicted its own text** — "all interpretability measures go in the expected direction" while $U_{ST}$ moved the wrong way | Rewritten: "Three of the four… and one does not," with a new paragraph explaining the trade honestly |
| 21 | Giessen "90% retention" over-claimed in RQ1/Conclusion as evidence about "our approach" | Scoped to the classical half only, with the prevalence-floor caveat; noted that 90.0% is a mean-of-ratios (ratio-of-means is 90.8%) |
| 22 | Missing baselines never acknowledged | Added to Conclusion: lineage-only baseline not run under the holdout, and no rule-based AMR-gene caller in the comparison set |

### Method contradictions
| # | Issue | Fix |
|---|---|---|
| 23 | Sec. III-A/III-F said filters used "the collection"; Sec. III-G said "training partition only" | Unified to training-partition-only, refitted per split. Added an honest paragraph that the **variant-level** MAF/indel filters *were* collection-wide, why, and why that is a weaker form of leakage |
| 24 | ST calling described in one vague sentence | Expanded: taken from the metadata (Achtman 7-locus), joined on sequencing accession not row order, plus the clonal-complex limitation (ST131 SLVs) |
| 25 | Phylogroup lookup had no provenance | Kept the honest disclosure, added what the correct route is (ClermonTyping) with citation |
| 26 | "Each phylogroup present in both train and test" — phylogroup E has n=4 | Qualified: E carries no evidential weight at that size |
| 27 | Imputation confound silently dropped | New paragraph in Sec. VI-A: the biological story and the reference-allele imputation predict the *same* outcome on the *same* drugs; two observations that keep biology in play; the masked-imputation rerun that would settle it |
| 28 | CLSI comparison not apples-to-apples | Two-qualification paragraph: device-verification context, and the intermediate→resistant collapse, including which way the bias runs (against the paper, so the conclusion holds) |
| 29 | 13 strains get all-zero gene vectors | Stated as a defect, with the correct handling and why it is below the resolution of the intervals |
| 30 | 0.5% duplicate threshold asserted, not justified | Justified in absolute terms (~2,700 positions, an order of magnitude below same-ST separation) |
| 31 | Spearman −0.04 to −0.53 between gain and SHAP — implausible as stated | Qualified as a top-20 restricted-range statistic; Jaccard identified as the reliable one; conclusion re-based on overlap rather than sign |

### Formatting
| # | Issue | Fix |
|---|---|---|
| 32 | Abstract ~330 words, undefined acronyms | Trimmed to ~250; PR-AUC, SNP, CLSI now expanded |
| 33 | `12.1\%age points` | → "12.1 percentage points" |
| 34 | β meant both the F_β parameter and the entropy weight | Entropy weight renamed `λ_ent`; equation now shows the averaging explicitly; note added |
| 35 | F_β defined twice (Eqs. 26 and 29) | Second one removed, replaced with a back-reference |
| 36 | Table VII used "CTZ" with an apologetic caption | → CAZ; caption cleaned |
| 37 | Table XI "Average" had an empty CI cell | Filled as "not defined", with the reason in the note |
| 38 | Table XXI "Change" column mixed deltas, dashes and reference values | Split into "Change" + "Reference"; promoted to `table*` so it fits |
| 39 | Fig. 1 / Fig. 2 used "Left/Centre and right" prose | Converted to (a)/(b)/(c) |
| 40 | Fig. 10 red/green not colourblind-safe | Caption now states every cell is annotated, so it reads without colour |
| 41 | Fig. 11 "range across antibiotics" ambiguous | Specified as full min–max |

### Wording and accuracy
| # | Issue | Fix |
|---|---|---|
| 42 | "almost equal to the average frequency of resistance" (0.2914 vs 0.250) | Both numbers now stated; "buys about 0.04 over chance" |
| 43 | "49% of the original variants" | Corrected — 49% of the *retained* 535,439; 10.7% of the original 2,451,184 |
| 44 | median prevalence conflated with mean gene count | Separated, with a sentence saying why one is not the other times 32,441 |
| 45 | `cusA` called a multidrug efflux pump | Corrected to the copper/silver system, counted as efflux but not as direct drug-mechanism evidence |
| 46 | "18 to 72 hours" vs "may take days" | Reconciled |
| 47 | "mixed-precision training" vs "single precision" attention | Stated as one explicit exception, cross-referenced |
| 48 | HIV/malaria comparison uncited | Citation added |
| 49 | Contractions (`It's`, `wasn't`, `aren't`, `that's`) | All removed |
| 50 | "XGBoost, plus another gradient boosting model, equals nothing" | Rewritten |
| 51 | "each lens becomes the sole author of the very brief write-up of a lengthy document" | Rewritten |
| 52 | "Two things besides could bring a real difference about that zero" | Rewritten |
| 53 | "we did the annotation in the top 100 genes five times" | Rewritten |
| 54 | Comma splice, "It comes as a surprise to note", "the lens the model inclines on" | Rewritten |

---

## PART 2 — Still needs you

**Experiments I cannot run (I will not invent numbers).** The text now honestly states each as a limitation, so the paper is submittable as-is — but these are what a reviewer will ask for:

1. **Multi-seed runs.** 5 seeds, mean ± sd. This is the single highest-value thing you can do. Until then no architectural claim in the paper is defensible.
2. **Lineage-only baseline under the lineage holdout.** Table XIV says "not run" in exactly the cell that tests the paper's thesis. Cheap: logistic regression on the 8-dim phylogroup one-hot.
3. **A rule-based AMR-gene baseline** (ResFinder / AMRFinderPlus / CARD-RGI). Hu et al. `[22]` already benchmarked these.
4. **Masked-vs-reference imputation rerun** for the four hard drugs — separates your biology conclusion from your pipeline choice.

**Two reference details to verify against your library** (I would not guess):
- CLSI M52 edition/year — I used the 2015 guideline; confirm which edition you consulted.
- EUCAST breakpoint version — pin the version Ren et al. actually used for Giessen.
- `[29]` Gao et al.: *Front. Microbiol.* **vol. 14** with year **2024** looks like a volume/year mismatch (vol. 14 is 2023).

**Length.** Still ~31 pages. I did not delete content, because that is your call. Note you had already commented out "Summary of Design Decisions" under `\iffalse` for a 30-page limit. Fastest cuts, in order:
1. Figs. 8, 9, 10, 12 are the same numbers as Tables XIII, XVII, XVIII, XXII — the captions say so. Cutting the four figures saves ~3 pages and loses nothing.
2. Sec. IV-B gives each of XGBoost / LightGBM / LogReg / RF a textbook paragraph plus an equation. Compressing to one paragraph with citations saves ~2.5 pages.

**Title.** Left untouched as you asked. Be aware the reviewer tension is real: the title sells the architecture, the finding is the protocol. Sec. I-E already says so in the paper's own words.

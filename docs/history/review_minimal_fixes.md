# Minimal fix list for the original text

Your writing is the base. Every replacement below is the **smallest possible
repair** — same voice, same sentence shape, only the broken part changed.
Nothing has been applied yet.

Source of truth: `chapters/Backup/chapter_?.tex` (verified as your pre-edit text).

---

## GROUP A — Factual errors and internal contradictions

These are not style. An examiner comparing two pages will catch each one.

### A1. Abstract — contradicts Chapter 5 and 6

**Now:**
> ...compared with 0.803 for the gradient-boosting baseline, a margin contained
> within the bootstrap interval, while performance dropped to 0.634...

The 0.849 vs 0.803 gap is 0.046, which Ch.5 and Ch.6 both call a clear lead.
The margin that actually sits inside the interval is the 0.009 over the best
single classical learner.

**Replace with:**
> ...compared with 0.803 for the gradient-boosting baseline the project began
> with, though its margin over the strongest single classical learner was only
> 0.009 and sits inside the bootstrap interval, while performance dropped to
> 0.634 under sequence-type holdout.

---

### A2. Abstract — miscounts the studies

**Now:**
> only two used population structure for their main split, and three more
> included a structure-aware check.

Hu et al. is one of the two primary. Only Moradigaravand and Orcales are extra.

**Replace with:**
> only two used population structure for their main split, and two others
> included a structure-aware check.

---

### A3. Chapter 2, line 117 — same double-count

**Now:**
> Three do report a structure-aware analysis alongside their main result:
> Moradigaravand et al.\ hold out the ST131 lineage in a supplementary figure
> \cite{moradigaravand2018}, Orcales et al.\ provide MLST-blocked
> cross-validation as an option \cite{Orcales2024}, and Hu et al.\ compare
> random against phylogeny-aware folds throughout their benchmark
> \cite{Hu2024}.

**Replace with:**
> Two of the remaining twelve report a structure-aware analysis alongside their
> main result: Moradigaravand et al.\ hold out the ST131 lineage in a
> supplementary figure \cite{moradigaravand2018}, and Orcales et al.\ provide
> MLST-blocked cross-validation as an option \cite{Orcales2024}.

---

### A4. Chapter 5, line 1326 — number does not match the data

**Now:**
> ...therefore, abundance can not account for a twenty-fold preference.

The median enrichment you report two sentences earlier is 16.4x.

**Replace with:**
> ...therefore, abundance can not account for a sixteen-fold preference.

---

### A5. Chapter 3, line 288 — overclaim

**Now:**
> NFR10 was met on average and for all seven drugs.

Amoxicillin-clavulanate beat its baseline by 0.0005, which you state yourself
earlier in the same chapter.

**Replace with:**
> NFR10 was met on average, and every drug scored at or above its own baseline,
> though for amoxicillin-clavulanate the margin was only 0.0005.

---

## GROUP B — Sentences that do not parse

A reader stops at each of these. All repairs keep your wording around them.

### B1. Chapter 1, objective bullet 3 — says the opposite of what you mean

**Now:** `Each type of sequence is not found in more than one partition.`

**Replace with:** `No sequence type is found in more than one partition.`

---

### B2. Chapter 1, Rationale (line 9)

**Now:** `This will allow its predictions to be trusted more truthfully.`

**Replace with:** `Only then can its predictions be trusted for the right reasons.`

---

### B3. Chapter 1, Challenges (line 52)

**Now:** `We had a well-tuned classical model too as a real competitor and not a formality.`

**Replace with:** `We also kept a well-tuned classical model in the comparison, as a real competitor and not a formality.`

---

### B4. Chapter 3, line 389

**Now:**
> The system is also referred to as unvalidated for clinical use, and only used
> for decision support in addition to laboratory testing, as mentioned in
> Section~\ref{sec:societal}.

**Replace with:**
> The system is therefore described throughout as unvalidated for clinical use,
> and intended only as decision support alongside laboratory testing, as
> mentioned in Section~\ref{sec:societal}.

---

### B5. Chapter 3, line 598 (risk table R1 cell)

**Now:**
> The correction was no longer a defect that was quietly patched up, but instead
> a principal finding.

**Replace with:**
> Rather than quietly patching the defect, we report the correction as one of the
> principal findings.

---

### B6. Chapter 4, line 211

**Now:** `The obvious objection is that XGBoost, plus another gradient boosting model, equals nothing.`

**Replace with:** `The obvious objection is that adding a second gradient boosting model to XGBoost gains nothing.`

---

### B7. Chapter 4, line 178 — imperative breaks the register

**Now:**
> If the number of susceptible strains is much higher than the number of
> resistant strains for some drugs, scale the pos weight according to the ratio
> of the susceptible to the resistant samples.

**Replace with:**
> Where susceptible strains greatly outnumber resistant ones for a drug, the
> \texttt{scale\_pos\_weight} parameter is set to the ratio between them.

---

### B8. Chapter 4, architecture section

**Now:**
> A mutation-based stance will lose a resistance gene that came into the plasmid,
> since that gene is not a mutation, it is only there.

**Replace with:**
> A mutation-based view will miss a resistance gene that arrived on a plasmid,
> because such a gene is not a mutation at all. It is simply present, or it is not.

---

### B9. Chapter 5, line 123

**Now:** `...for models trained of exactly the same genomes...`

**Replace with:** `...for models trained on exactly the same genomes...`

---

### B10. Chapter 5, line 595

**Now:** `None of the ensemble or the half-neural is of value on unfamiliar strains, and all other analyses in this chapter confirm this.`

**Replace with:** `Neither the ensemble nor the hybrid design buys anything on unfamiliar strains, and every other analysis in this chapter agrees.`

---

### B11. Chapter 5, line 687

**Now:**
> ...and when training its validation score around that drug switched between
> 0.58 and 0.80 from epoch to epoch and never converged.

**Replace with:**
> ...and during training its validation score on that drug swung between 0.58 and
> 0.80 from epoch to epoch and never settled.

---

### B12. Chapter 5, line 1184

**Now:** `The 1D CNN was naive with an average of 0.4826 on the Arcadia pan-genome and 0.6889 on the Giessen data...`

**Replace with:** `The naive 1D CNN averaged 0.4826 on the Arcadia pan-genome and 0.6889 on the Giessen data...`

---

### B13. Chapter 5, line 1607 — contradicts itself

**Now:** `An incidental finding like that must give rise to a check of the pipeline, not incidental.`

**Replace with:** `A finding of that kind should trigger a check of the pipeline rather than be treated as incidental.`

---

## GROUP C — Figures to carry over

These seven exist now but were not in your original text. Their `\begin{figure}`
blocks need re-inserting. The image files in `images/` are safe either way.

| Figure | Chapter | Suggested position |
|---|---|---|
| `fig:requirements_status` | 3 | Start of the compliance subsection |
| `fig:phylogroup_composition` | 4 | After Table `tab:phylogroups` |
| `fig:layer_ablation` | 5 | After Table `tab:ablation` |
| `fig:baseline_ladder` | 5 | After Table `tab:baselines` |
| `fig:progression` | 5 | After Table `tab:progression` |
| `fig:giessen_retention` | 5 | After Table `tab:giessen_transfer` |
| `fig:mechanism_gap` | 5 | End of the two-group subsection |

Two of these need the wording fixes we already agreed:

- **`fig:evalsplit_landscape`** (Ch.2, already in your original): its caption and
  in-figure label say *"twelve of the fourteen never hold a lineage out"*, which
  contradicts Table 2.1. Change to *"do not hold a lineage out as their primary
  design"*.
- **`fig:mechanism_gap`**: labels are already fixed to *small / large corrected
  gap*. If you keep it, Section 5.6.3 needs a short paragraph on AMP and SXT,
  since neither is a clean "chromosomal / single-gene" case.

---

## Note on the abstract

`chapters/Backup/` has no abstract file. `core/abstract.tex` currently holds my
version. Your original opened:

> Antibiotic resistance prediction using bacterial genomes is more rapid than
> laboratory analysis, though high scores do not indicate the performance of a
> model on unfamiliar bacteria.

Tell me if you want that restored, and A1, A2 and the entropy sentence applied
on top of it.

---

## What I need from you

Reply with the item codes you approve, e.g. *"A1-A5, B1-B13, all of C"*, or
strike out any you disagree with. I will not touch a file until then.

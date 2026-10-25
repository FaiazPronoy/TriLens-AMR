# Findings you cannot un-discover

Everything below weakens a claim the thesis would otherwise have been able to
make. In every case the weakness exists **because an experiment was run**. Had it
not been run, the thesis could have stayed silent and looked stronger.

Sorted by how much it costs. For each: what was lost, and what can honestly be
said instead.

---

# TIER 1 — Kills a stated claim outright

## 1. The architecture ties its own baseline
`0.8462` (TriLens) vs `0.8469` (CascadeStack v9.1), difference `0.0007`,
paired t-test `p = 0.8691`.

**What was lost.** Objective 1 in Chapter 1 is *"Develop TriLens-MoE-Stack."*
That objective is met in the sense that the thing was built, but the thing does
not win. Every architectural decision in Chapter 4 — three lenses, the MoE gate,
the entropy regulariser, the per-antibiotic meta-learner — buys **−0.0007**.

**Why you can't drop it.** CascadeStack is not an external competitor. It is
five of your own six base learners. You cannot report the stack without
reporting what the stack is made of.

**Had you not run it:** you could have compared only against XGBoost-on-PCA
(0.803) and TabNet (0.695) and claimed a clear win.

**What to say instead.** The comparison is what makes the ablation meaningful.
Without CascadeStack you would not know that the classical arm carries the
model, which is the finding Chapter 5 is built on. Say: *"the architecture is
the vehicle, not the contribution."*

---

## 2. The neural half contributes ~nothing where it matters most
Lineage holdout: classical alone `0.6278`, full hybrid `0.6300`, neural lenses
alone `0.4369`.

**What was lost.** On the split the thesis argues is the only honest one, the
entire neural apparatus adds **+0.0022**. Three lenses, a gate, an entropy
penalty, a float32 softmax workaround, and 32,441-gene attention — for two
thousandths.

**Why you can't drop it.** You ran the component breakdown to answer RQ2. RQ2
exists in Chapter 1. Removing the holdout breakdown while keeping the random-split
one would be selective reporting.

**Had you not run it:** the random-split ablation alone (+0.0224 for the gate)
would have made the neural design look worthwhile.

**What to say instead.** This is the sharpest question you will face:
*"why is the neural half in the model at all?"* Your only honest answer is the
gate weights — they are interpretable and they do vary sensibly by drug
(Lens C at 0.279 for CIP, its point-mutation drug). That is a real, if modest,
return. **Do not claim more than that.**

---

## 3. Attention recovered zero resistance genes out of 420
`pillar_gene_annotation`: Lens A 0/140, Lens B 0/140, Lens C 0/140. All eight
canonical AMR genes came from SHAP on the classical learners.

**What was lost.** Objective 4 says *"Interpret the model's predictions using
its own attention and gating weights."* Half of that objective is unmet, and
the half that failed is the half that was novel. Post-hoc SHAP on gradient
boosting is not a contribution.

**Why you can't drop it.** You built the annotation pipeline specifically to
check this, and it is what recovered gyrA and parC — your best biological
result. The same table that gives you the win gives you the loss.

**Had you not run it:** you could have shown attention heatmaps and said the
model "focuses on specific genes," which is what
`pillar3_interpretability/SUMMARY.md` still claims. Nobody would have checked.

**What to say instead.** *"Attention pooling over a pan-genome does not yield
biologically meaningful gene rankings at this scale"* is a publishable negative
result. State it as a claim, not as an apology.

---

## 4. Your two attribution methods disagree with each other
Top-20 Jaccard 0.11–0.17. Spearman **negative for all seven antibiotics**,
mean −0.36.

**What was lost.** This is worse than the attention failure. Two methods applied
to the *same trained model* are anti-correlated. It means **no gene-level claim
in this thesis can be attributed to the model** — only to the agreement between
the model and pre-existing biology.

**Why you can't drop it.** You computed it to answer RQ3.

**Had you not run it:** every gene in your marker list would have looked equally
credible.

**What to say instead.** gyrA and parC are credible *because independent
literature says so*, not because your attribution vouches for them. The thesis
now says this. Keep it that way — the alternative is a claim that collapses
under one question.

---

## 5. Clinical error rates are catastrophic, and you measured them
GEN VME `0.447`, AMC `0.406`. CLSI met for **0 of 7**.

**What was lost.** NFR11 is permanently "not met." Any framing of the system as
clinically useful is gone. Nearly half of resistant gentamicin cases would be
called susceptible.

**Why you can't drop it.** You wrote NFR11 into Chapter 3 *before* building, on
purpose, precisely so it could not be quietly dropped. That was the right call
and it is now binding.

**Had you not run it:** "clinical validation is left as future work" would have
been a completely normal thing to write, and no reviewer would have objected.

**What to say instead.** The gap between PR-AUC 0.846 and VME 0.447 for the same
model is itself the finding: *machine-learning metrics and clinical metrics
disagree about whether this model works.* Very few papers in this field report
both. Lead with that.

---

# TIER 2 — Weakens a claim, but reframable

## 6. Two fifths of your headline drop was a prevalence artefact
`0.846 → 0.630` becomes `0.846 → 0.719` once prevalence is matched.

**What was lost.** Your most quotable number. "Performance falls by more than a
fifth" becomes "by about an eighth."

**Why you can't drop it.** You now know. Reporting 0.630 alone, having computed
0.719, would be misconduct — and the file was sitting in your results folder
either way.

**What to say instead.** The corrected number produces a *cleaner* result: three
drugs lose 0.015–0.030, four lose 0.175–0.245, nothing in between. And the
correction itself is a methodological contribution nobody else in this literature
reports. This one genuinely converts from weakness to strength.

## 7. Gentamicin's holdout rests on 38 resistant strains
Prevalence-matched interval `[0.396, 0.510]` — a span of 0.11.

**What was lost.** Your most dramatic per-drug number is also your least stable.

**Why you can't drop it.** The split rules that produced it (force >5% clones
into training) were fixed before results were seen, which is exactly what makes
them credible.

**What to say instead.** Report the interval, note the direction is clear even if
the value is not, and list "holdout with a minimum resistant count per drug" as
future work. Chapter 6 now does this.

## 8. Missing genotypes were set to reference — and you know where that bites
Coverage is worst on plasmids; plasmid-borne genes are exactly where GEN and AMC
resistance lives.

**What was lost.** Your two worst antibiotics have a known *data-quality*
explanation. That competes with your biological explanation. A reviewer can say:
*"GEN fails because your features are broken, not because the model learned
lineage."*

**Why you can't drop it.** You checked and documented it in §4.3.

**What to say instead.** Both explanations point the same way and are not
mutually exclusive — fragmented plasmid assembly is *why* the signal is
lineage-carried rather than gene-carried. The Arcadia team reached the same
conclusion independently. That convergence is your defence.

## 9. Lens B is the view you trust most and understand least
Gate weight 0.675–0.880 across all seven drugs. eggNOG annotation coverage
**61.4%** — the lowest of the three lenses (A: 97.9%, C: 100%).

**What was lost.** Interpretability is weakest precisely where the model leans
hardest.

**Why you can't drop it.** Both numbers are in your own results.

**⚠️ This is currently NOT in the thesis.** A sharp examiner will find it. I
recommend adding one sentence — better to raise it yourself.

## 10. External validation is 81 test strains
CIP's bootstrap CI upper bound is literally `1.0000`.

**What was lost.** "Validated on an independent collection" is not defensible at
that n.

**What to say instead.** The thesis already calls it *methodology* transfer, not
model transfer, and says so plainly. Hold that line and never use the word
"validation" for it.

---

# TIER 3 — Design choices now locked in

These cannot be changed without rerunning, and each costs something.

| Choice | What it costs you |
|---|---|
| **Threshold fixed at 0.5** | FR5 "not met". Your clinical numbers are the worst they could be. You know a tuned threshold would improve them and you cannot show it. |
| **Intermediate encoded as resistant** | CA becomes identical to accuracy, so one of your three clinical metrics carries zero extra information. |
| **Gene block L = 64** | 28% of selected genes truncated. Lens A never sees the full mutation pattern of the largest genes. |
| **Phylogroup: B2 = 50.9%, Unknown = 24.4%** | Any phylogroup analysis is really "B2 vs everything else". Your stratified split was never going to work — the label could not support it. |
| **5,000-gene cap refitted per split** | Holdout and random-split gene sets overlap by <50%. You frame this as "signal is redundant," which is fair, but it also means feature selection is unstable. |
| **Multiseed varies only the neural half** | Classical predictions loaded from cache. "Seed stability" is a partial claim you cannot complete without retraining. |
| **Giessen feature space incompatible** | True zero-shot transfer was impossible from the moment this dataset was chosen. |
| **Undocumented phylogroup lookup table** | A variable used in one of your three splits has no traceable provenance. Disclosed in §4.3. |

---

# The honest summary for your supervisor

**Every Tier-1 item exists because a check was run that most papers in this field
do not run.** Concretely:

- Most AMR papers do not report a strong classical baseline inside their own
  stack. You do → you found the tie.
- Most do not decompose the ensemble on a hard split. You did → you found the
  neural half contributes 0.002.
- Most show attention heatmaps without annotating them. You annotated → you
  found zero hits.
- Most report one attribution method. You reported two → you found they
  anti-correlate.
- Most report ML metrics only. You reported CLSI → you found 0/7.
- Most report a lineage-holdout drop without a prevalence correction. You
  corrected → you found 41% of yours was an artefact.

**The thesis is weaker than it could have been, and more trustworthy than almost
anything it cites.** That trade is the actual contribution, and it is the answer
to give when someone asks why the numbers are not better.

**The one thing that is genuinely, unfixably weak** is not on the list above:
everything rests on a single species and a single collection. That is not a
disclosure problem, it is a scope problem, and it is the only item here that a
reviewer can legitimately reject on.

---

# One recommended addition

Item 9 (Lens B, 61.4% annotation coverage) is the only Tier-2 weakness not
currently stated anywhere in the thesis. Say it before someone finds it. Suggested
wording for §5.6.1:

> One limitation of this annotation applies unevenly across the three lenses.
> The eggNOG file covered 97.9\% of Lens A's markers and all of Lens C's, but only
> 61.4\% of Lens B's. Lens B is the view the gate relies on most, so the
> representation the model trusts most is also the one we can interpret least
> well. This does not affect the eight resistance genes reported above, all of
> which came from the classical learners, but it does mean the absence of AMR
> hits among Lens B's markers is a weaker negative result than the absence among
> Lens A's and Lens C's.

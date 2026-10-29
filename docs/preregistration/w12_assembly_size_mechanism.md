# W12. Is the assembly shortcut batch or biology, and which statistics carry it?

**Status:** pre-registered 2026-10-29, before this analysis has been run. W7's results are already
read and are the quantity being tested.

## Why

W7 showed that assembly statistics predict resistance *within* sequence type (within-ST
concordance 0.552-0.676 across seven drugs, and assembly size correlates with acquired-gene count
at within-ST rho 0.325). Two reviews pointed out that this leaves the mechanism open, and that the
paper's wording depends on which answer is right:

- **Biology:** a strain carrying more plasmid and accessory DNA is both larger and more likely to
  carry acquired genes. Then assembly *size* is a crude but real resistance feature, and the
  honest word is "proxy", not "artefact".
- **Batch:** strains from one submitting study were sequenced and assembled alike, and that study's
  strains happen to be more resistant. Then N50 and contig count are the carriers, the signal is a
  study artefact, and "shortcut" is the right word.

W7's own text already argued against a pure batch explanation, because the epidemiological fields
including publication score at chance across countries and time. But it never conditioned the
within-ST analysis on the study, which is the direct test. The `study` field is complete for all
6,962 strains (16 values), so this costs nothing.

## Design

All of it is re-scoring of data already on disk; no model is fitted and no split is used.

1. **Within (ST × study) concordance.** Repeat W7's within-ST concordance with cells defined by
   sequence type *and* submitting study together, so a pair is only compared when both strains
   share a lineage and a study. Cells with no resistant-susceptible pair contribute nothing.
   Minimum for a drug to be reported: 10 eligible cells and 100 eligible pairs, fixed here.
2. **Two feature groups, fixed here before looking.** *Content* = assembly size, CDS count,
   GC content. *Technical* = contig count, contig N50. The within-ST and within-(ST × study)
   concordance of each statistic is reported inside its group.
3. **Within (ST × study) correlation** of each statistic against the acquired-gene count, the
   W7 rho recomputed in the narrower cells.

## Endpoints and rules

**Primary.** The mean over CTX, CAZ, GEN, SXT and AMC - the acquired-determinant drugs as fixed in
W7's `MECHANISM` table - of assembly size's within-(ST × study) concordance.

- **The signal is not a study artefact** if it stays ≥ 0.55, i.e. the level W7 reported within ST
  alone. The paper then says assembly size is a within-lineage proxy for accessory content.
- **It is largely a study artefact** if it falls to 0.50-0.55. The paper then says the within-ST
  assembly signal is substantially explained by submitting study, and W7's claim is narrowed to
  "assembly statistics carry signal that does not survive conditioning on the study, and does not
  transfer externally".

**Secondary, no pass/fail.** Whether the content group or the technical group carries more, in both
cell definitions. A technical-dominant pattern supports batch; a content-dominant pattern supports
accessory content. Reported as it comes out.

## Stated limits

- Conditioning on the study also removes real variation: strains from one study can share a
  lineage, a country and a sampling frame, so a fall in concordance is not proof of a technical
  artefact.
- `study` is the submitting publication, not the sequencing run or assembler. It is the finest
  batch label available in this dataset; the actual assembler and its version are not recorded for
  our strains, so the test is a proxy for batch, not batch itself.
- Concordance is a rank statistic within a cell and says nothing about effect size.
- W12 cannot establish that the assembly block inflates the *full model's* internal score. That
  would need an ablation of the full stack with the assembly features removed, which is not run
  here, so the paper does not use the word "inflates" of the full model.

## Result (read 2026-10-29, after the pre-registration above was committed)

**Primary: pass.** Assembly size's mean within-cell concordance over AMC, SXT, GEN, CTX, CAZ is
**0.613 within ST and 0.592 within (ST × study)** - a fall of only 0.022, well above the 0.55 level.
The correlation with acquired-gene count is likewise unmoved: within-ST rho 0.325, within
(ST × study) rho 0.321, on 3,951 strains in 75 cells. Conditioning on the submitting study does
not remove the signal, so the within-lineage assembly signal is not a study artefact.

**Secondary: the content group carries it, the technical group does not.** Within (ST × study),
mean concordance over the acquired-determinant drugs is 0.554 for content (Size, CDS, GC.Content)
and 0.531 for technical (Contigs, Contig.N50). Per statistic:

| | AMC | AMP | CAZ | CIP | CTX | GEN | SXT |
|---|---|---|---|---|---|---|---|
| Size | 0.623 | 0.643 | 0.576 | 0.595 | 0.616 | 0.556 | 0.588 |
| CDS | 0.600 | 0.660 | 0.597 | 0.580 | 0.636 | 0.535 | 0.597 |
| Contigs | 0.586 | 0.648 | 0.594 | 0.599 | 0.631 | 0.611 | 0.606 |
| GC.Content | 0.474 | 0.515 | 0.435 | 0.513 | 0.426 | 0.488 | 0.565 |
| Contig.N50 | 0.470 | 0.474 | 0.473 | 0.416 | 0.383 | 0.490 | 0.469 |

Contig N50 sits **below** 0.5 for every drug (0.383-0.490) and correlates *negatively* with
acquired-gene count (rho −0.185): a more contiguous assembly carries fewer acquired genes, which is
what a plasmid-content explanation predicts and the opposite of a "better sequencing, more
resistance" batch effect. GC content is near chance throughout.

**Wording this fixes.** The paper does not call this a batch artefact, and does not say assembly
quality "inflates" the full model (not tested; see the limits above). It says: assembly size and
CDS count act as a within-lineage proxy for accessory gene content, which is real biology but is a
proxy a model can lean on instead of the determinants, and it does not transfer to the external
cohort (W8: metadata-arm mean nAP 0.045).

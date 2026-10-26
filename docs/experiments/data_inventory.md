# W0. Data inventory

**Status:** done (CPU, no Kaggle). Script: `code/scripts/inventory/data_inventory.py`; loader:
`code/src/data/metadata.py`. Every number below is in `code/results/data_inventory/` (full tables in
`tables.md`, one CSV per axis). Data: the 6,962 strains of `arcadia_v2_master_features.pkl`, joined to
`data/raw/strains_metadata.csv` by genome name; the SRA accession join is checked to agree on every
strain. All 6,962 have a metadata row.

Normalisation (in `metadata.py`): "England" is merged into "United Kingdom"; host is "Human",
"Non-human" or unknown; study is the PubMed ID in the metadata, or `unpublished:<country>`; ST is
the Achtman ST of `mlst_lineage_holdout_split.pkl` (475 strains unknown).

Cells below are **tested strains / percent resistant** (intermediate counts as resistant, as in the
paper). The label totals equal `tab:labels` of the paper exactly (AMP 4,868 / 2,439 R …
CAZ 5,973 / 780 R).

## 1. Who the strains are

Five facts that shape every split:

1. **Two countries hold 91% of the strains:** United Kingdom 3,387 (three studies), Norway 2,964 (one
   collection, no PubMed ID in the metadata). The other 12 countries together have 611.
2. **Country is study, and the antibiotic panels differ by study.** Norway has no AMC
   (1 tested strain); the UK has no SXT (12 tested). One UK study (PMID 32205351, 921 strains) has
   only AMP and AMC; another (34485958, 1,015 strains) has no AMP and no year.
3. **Only Norway has a year for every strain** (2002–2017). In the UK, 1,439 strains (PMID 28720578)
   have a year (2001–2012); 1,936 do not. Overall 1,991 strains have no year.
4. **Host is not usable:** 3,878 human, 24 non-human (all AMP-resistant, from small targeted
   studies), 3,060 unknown.
5. **Several small collections were sampled for resistance:** Thailand (139 strains, 100% CTX-R and AMP-R),
   USA (222, 98% AMP-R), PMID 27208899 (27, 100% CTX-R). The 475 strains without an ST are enriched
   for these (48% CTX-R against 15% overall). The existing lineage holdout puts all unknown-ST strains in
   training, which is one reason its test prevalence is lower than its training prevalence.

Two more things to check against the source papers before writing them up: the Norway 2012 cohort has
**39% GEN-R** (78/201; every other Norway year is 0–10%); PMID 34485958 (England, 1,015 strains)
has 33% CTX-R and 43% CIP-R, far above the other UK study (8% / 19%), so it may be an
ESBL-enriched collection. We have not opened these publications yet, so nothing is claimed about
them here.

## 2. Tables per antibiotic

Tested strains / percent resistant. Full tables (every ST, every study, every year) are in `code/results/data_inventory/`.

### By country

| Country | strains | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| United Kingdom | 3387 | 1308 / 57% | 2466 / 29% | 3375 / 29% | 12 / 92% | 2464 / 10% | 2392 / 19% | 2454 / 14% |
| Norway | 2964 | 2964 / 39% | 2963 / 11% | 1 / 100% | 2964 / 26% | 2964 / 8% | 2765 / 5% | 2963 / 5% |
| USA | 227 | 224 / 98% | 226 / 77% | 218 / 64% | 227 / 62% | 225 / 38% | 54 / 83% | 173 / 71% |
| Thailand | 141 | 141 / 100% | 141 / 70% | 141 / 39% | 0 | 141 / 77% | 141 / 100% | 141 / 71% |
| other (10 countries) | 240 | 228 / 73% | 234 / 23% | 41 / 2% | 213 / 63% | 240 / 14% | 55 / 60% | 239 / 23% |
| unknown | 3 | 3 / 100% | 3 / 67% | 2 / 50% | 2 / 50% | 3 / 0% | 2 / 0% | 3 / 33% |

### By collection year, Norway

| Year | strains | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2002 | 78 | 78 / 32% | 78 / 4% | 0 | 78 / 24% | 78 / 4% | 0 | 78 / 3% |
| 2003 | 114 | 114 / 26% | 114 / 2% | 0 | 114 / 18% | 114 / 0% | 0 | 114 / 1% |
| 2004 | 120 | 120 / 31% | 120 / 2% | 0 | 120 / 18% | 120 / 1% | 120 / 0% | 120 / 1% |
| 2005 | 120 | 120 / 33% | 120 / 7% | 0 | 120 / 16% | 120 / 2% | 120 / 1% | 120 / 1% |
| 2006 | 117 | 117 / 32% | 117 / 6% | 0 | 117 / 26% | 117 / 2% | 117 / 2% | 117 / 1% |
| 2007 | 133 | 133 / 37% | 133 / 6% | 0 | 133 / 23% | 133 / 5% | 133 / 3% | 133 / 5% |
| 2008 | 149 | 149 / 34% | 149 / 10% | 0 | 149 / 21% | 149 / 4% | 149 / 2% | 149 / 2% |
| 2009 | 180 | 180 / 34% | 180 / 9% | 0 | 180 / 24% | 180 / 4% | 180 / 3% | 180 / 2% |
| 2010 | 181 | 181 / 37% | 181 / 6% | 0 | 181 / 29% | 181 / 6% | 180 / 2% | 181 / 2% |
| 2011 | 206 | 206 / 40% | 206 / 12% | 0 | 206 / 26% | 206 / 3% | 206 / 3% | 206 / 4% |
| 2012 | 201 | 201 / 39% | 201 / 10% | 0 | 201 / 34% | 201 / 39% | 200 / 14% | 201 / 13% |
| 2013 | 219 | 219 / 40% | 219 / 15% | 0 | 219 / 28% | 219 / 6% | 217 / 6% | 219 / 5% |
| 2014 | 255 | 255 / 42% | 255 / 18% | 0 | 255 / 26% | 255 / 10% | 254 / 8% | 255 / 8% |
| 2015 | 283 | 283 / 46% | 282 / 12% | 1 / 100% | 283 / 27% | 283 / 10% | 282 / 7% | 282 / 6% |
| 2016 | 280 | 280 / 44% | 280 / 16% | 0 | 280 / 30% | 280 / 6% | 279 / 7% | 280 / 7% |
| 2017 | 328 | 328 / 49% | 328 / 19% | 0 | 328 / 29% | 328 / 7% | 328 / 6% | 328 / 7% |

### By collection year, United Kingdom

| Year | strains | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2001 | 74 | 0 | 74 / 11% | 74 / 20% | 0 | 74 / 11% | 0 | 74 / 0% |
| 2002 | 106 | 0 | 106 / 8% | 106 / 27% | 0 | 106 / 6% | 106 / 4% | 106 / 5% |
| 2003 | 103 | 0 | 103 / 12% | 103 / 28% | 0 | 103 / 2% | 103 / 3% | 103 / 3% |
| 2004 | 106 | 0 | 106 / 22% | 106 / 21% | 0 | 106 / 5% | 106 / 7% | 106 / 4% |
| 2005 | 103 | 0 | 103 / 17% | 103 / 29% | 0 | 103 / 8% | 103 / 10% | 103 / 11% |
| 2006 | 107 | 4 / 50% | 107 / 30% | 107 / 43% | 0 | 107 / 12% | 107 / 14% | 107 / 13% |
| 2007 | 152 | 59 / 68% | 152 / 32% | 140 / 30% | 12 / 92% | 150 / 15% | 152 / 9% | 140 / 8% |
| 2008 | 147 | 51 / 51% | 147 / 20% | 147 / 33% | 0 | 147 / 12% | 147 / 14% | 147 / 11% |
| 2009 | 161 | 64 / 59% | 161 / 17% | 161 / 37% | 0 | 161 / 4% | 161 / 7% | 161 / 9% |
| 2010 | 145 | 60 / 58% | 145 / 18% | 145 / 28% | 0 | 145 / 8% | 145 / 4% | 145 / 3% |
| 2011 | 163 | 65 / 65% | 163 / 20% | 163 / 28% | 0 | 163 / 10% | 163 / 11% | 163 / 10% |
| 2012 | 84 | 84 / 63% | 84 / 19% | 84 / 25% | 0 | 84 / 5% | 84 / 10% | 84 / 10% |
| unknown | 1936 | 921 / 55% | 1015 / 43% | 1936 / 29% | 0 | 1015 / 13% | 1015 / 33% | 1015 / 24% |

### By host

| Host | strains | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Human | 3878 | 1795 / 65% | 2955 / 33% | 3683 / 31% | 413 / 64% | 2955 / 14% | 2524 / 23% | 2901 / 20% |
| Non-human | 24 | 24 / 100% | 24 / 100% | 12 / 0% | 12 / 92% | 22 / 59% | 24 / 54% | 12 / 100% |
| unknown | 3060 | 3049 / 41% | 3054 / 12% | 83 / 24% | 2993 / 26% | 3060 / 9% | 2861 / 8% | 3060 / 6% |

### By sequence type (12 largest)

| ST | strains | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ST73 | 884 | 681 / 37% | 734 / 2% | 414 / 33% | 471 / 15% | 734 / 4% | 687 / 1% | 734 / 2% |
| ST131 | 872 | 548 / 81% | 757 / 70% | 612 / 52% | 373 / 55% | 758 / 30% | 657 / 42% | 722 / 39% |
| ST95 | 606 | 478 / 34% | 522 / 3% | 256 / 23% | 350 / 15% | 522 / 4% | 487 / 1% | 522 / 1% |
| ST69 | 477 | 368 / 73% | 390 / 14% | 227 / 25% | 254 / 56% | 390 / 9% | 368 / 4% | 390 / 3% |
| ST12 | 235 | 181 / 45% | 195 / 2% | 106 / 39% | 130 / 22% | 195 / 4% | 179 / 1% | 195 / 3% |
| ST127 | 191 | 156 / 22% | 160 / 1% | 79 / 13% | 113 / 9% | 160 / 1% | 149 / 0% | 160 / 0% |
| ST10 | 176 | 115 / 55% | 151 / 20% | 101 / 20% | 82 / 49% | 151 / 9% | 127 / 2% | 145 / 4% |
| ST59 | 103 | 61 / 13% | 94 / 10% | 55 / 24% | 48 / 15% | 94 / 2% | 88 / 0% | 94 / 0% |
| ST141 | 103 | 78 / 15% | 98 / 1% | 31 / 0% | 72 / 6% | 98 / 4% | 93 / 1% | 98 / 1% |
| ST58 | 92 | 49 / 69% | 84 / 2% | 55 / 11% | 39 / 69% | 84 / 2% | 80 / 21% | 84 / 23% |
| ST38 | 89 | 60 / 92% | 82 / 21% | 52 / 62% | 46 / 63% | 82 / 10% | 73 / 16% | 82 / 18% |
| ST88 | 88 | 57 / 65% | 73 / 5% | 50 / 34% | 38 / 37% | 73 / 1% | 65 / 6% | 72 / 8% |
| other (446 STs) | 2571 | 1655 / 44% | 2264 / 23% | 1419 / 28% | 1234 / 31% | 2266 / 10% | 1964 / 15% | 2250 / 11% |
| unknown ST | 475 | 381 / 68% | 429 / 40% | 321 / 25% | 168 / 26% | 430 / 31% | 392 / 48% | 425 / 36% |

### By phylogroup

| Phylogroup | strains | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 | 3542 | 2521 / 46% | 3032 / 22% | 1905 / 35% | 1757 / 25% | 3033 / 11% | 2770 / 13% | 2997 / 11% |
| D | 723 | 538 / 72% | 607 / 19% | 364 / 32% | 382 / 54% | 607 / 12% | 559 / 6% | 606 / 7% |
| A | 317 | 179 / 60% | 286 / 29% | 222 / 30% | 122 / 57% | 286 / 15% | 241 / 9% | 269 / 12% |
| F | 307 | 129 / 40% | 290 / 35% | 231 / 37% | 82 / 24% | 290 / 13% | 269 / 12% | 290 / 12% |
| B1 | 258 | 137 / 57% | 241 / 13% | 163 / 13% | 102 / 58% | 241 / 5% | 203 / 10% | 240 / 11% |
| C | 114 | 96 / 70% | 80 / 24% | 60 / 50% | 63 / 37% | 81 / 9% | 70 / 14% | 80 / 19% |
| E | 4 | 3 / 67% | 4 / 0% | 2 / 50% | 2 / 50% | 4 / 0% | 3 / 0% | 4 / 0% |
| Unknown | 1697 | 1265 / 46% | 1493 / 25% | 831 / 23% | 908 / 27% | 1495 / 13% | 1294 / 27% | 1487 / 19% |

### By study (PubMed ID; 'unpublished' = no publication in the metadata)

| Study | strains | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unpublished:Norway | 2963 | 2963 / 39% | 2963 / 11% | 0 | 2963 / 26% | 2963 / 8% | 2764 / 5% | 2963 / 5% |
| 28720578 | 1439 | 375 / 60% | 1439 / 19% | 1439 / 30% | 0 | 1439 / 8% | 1365 / 8% | 1439 / 7% |
| 34485958 | 1015 | 0 | 1015 / 43% | 1015 / 23% | 0 | 1015 / 13% | 1015 / 33% | 1015 / 24% |
| 32205351 | 921 | 921 / 55% | 0 | 921 / 35% | 0 | 0 | 0 | 0 |
| unpublished:USA | 222 | 219 / 98% | 221 / 78% | 218 / 64% | 222 / 62% | 220 / 39% | 54 / 83% | 168 / 70% |
| 30127495 | 164 | 164 / 65% | 164 / 5% | 0 | 163 / 65% | 164 / 3% | 0 | 164 / 2% |
| 28877757, 30550564 | 139 | 139 / 100% | 139 / 70% | 139 / 40% | 0 | 139 / 77% | 139 / 100% | 139 / 71% |
| 27208899 | 27 | 27 / 100% | 27 / 100% | 27 / 4% | 0 | 27 / 67% | 27 / 100% | 27 / 100% |
| 27965630 | 26 | 26 / 96% | 26 / 62% | 0 | 26 / 69% | 26 / 38% | 0 | 26 / 96% |
| 29184361 | 24 | 13 / 69% | 18 / 6% | 14 / 0% | 24 / 38% | 24 / 4% | 24 / 8% | 24 / 8% |
| 32122581 | 12 | 12 / 100% | 12 / 100% | 0 | 12 / 92% | 10 / 50% | 12 / 8% | 0 |
| 27322919 | 4 | 4 / 100% | 4 / 50% | 0 | 4 / 75% | 4 / 0% | 4 / 100% | 4 / 50% |
| 28118859 | 2 | 2 / 100% | 2 / 50% | 2 / 0% | 0 | 2 / 50% | 2 / 100% | 2 / 50% |
| unpublished:unknown | 2 | 2 / 100% | 2 / 100% | 2 / 50% | 2 / 50% | 2 / 0% | 2 / 0% | 2 / 50% |
| 29112974 | 1 | 1 / 100% | 0 | 1 / 100% | 1 / 100% | 1 / 0% | 1 / 0% | 0 |
| unpublished:Brazil | 1 | 0 | 1 / 100% | 0 | 1 / 100% | 1 / 0% | 0 | 0 |


## 3. Which country and time splits are possible

### Minimum counts (fixed before any model was run on these splits)

A split side is usable for an antibiotic only if

| side | resistant | susceptible |
|---|---:|---:|
| test | ≥ 50 | ≥ 50 |
| train (before validation is carved out) | ≥ 100 | ≥ 100 |

Reason: with 50 positives the bootstrap interval of PR-AUC is still about ±0.07, so a smaller test
side cannot separate models; 100 training positives is roughly the smallest count at which the
per-antibiotic classical learners trained stably in the thesis runs (GEN on the lineage holdout had
38 test positives and an interval of [0.398, 0.511]).

A **secondary tier** (reported, marked, left out of the Holm family of headline tests) is
allowed when the test side passes and the training side has ≥ 50 resistant strains.

These thresholds were chosen by me after seeing the counts in this file, but before any model
output on these splits existed. **Open question for the owner** (`docs/PROGRESS.md`).

### Country: United Kingdom ↔ Norway (both directions)

`results/data_inventory/split_feasibility_country.csv`

| antibiotic | UK R / S | Norway R / S | UK prevalence | Norway prevalence | usable |
|---|---:|---:|---:|---:|---|
| AMP | 744 / 564 | 1,165 / 1,799 | 0.569 | 0.393 | yes |
| CIP | 713 / 1,753 | 337 / 2,626 | 0.289 | 0.114 | yes |
| AMC | 982 / 2,393 | 1 / 0 | 0.291 | — | **no** (not tested in Norway) |
| SXT | 11 / 1 | 773 / 2,191 | — | 0.261 | **no** (not tested in the UK) |
| GEN | 252 / 2,212 | 231 / 2,733 | 0.102 | 0.078 | yes |
| CTX | 453 / 1,939 | 151 / 2,614 | 0.189 | 0.055 | yes |
| CAZ | 353 / 2,101 | 148 / 2,815 | 0.144 | 0.050 | yes |

**Judgement:** possible for AMP, CIP, GEN, CTX, CAZ, in both directions. The other 611 strains (USA,
Thailand, …) are left out of both sides: they are small, partly resistance-selected, and would make
"country" mean "everything but Norway".

Three caveats belong in the paper:

1. Prevalence differs by up to 3.4× (CTX 0.189 vs 0.055), so prevalence-matched PR-AUC and ROC-AUC are
   required.
2. **The two countries share lineages:** 93.8% of Norway strains with a known ST have an ST that also
   occurs in the UK (`split_lineage_overlap.csv`). This axis tests geography, study and
   phenotyping method, not new lineages. That is the point of a separate axis, but it must be said.
3. Country is confounded with study and probably with the susceptibility-testing method; the
   metadata does not record the method.

### Time

`results/data_inventory/split_feasibility_time.csv` (every cut-off year, three scopes)

- **Pooled over countries: rejected.** Its recent side is almost only Norway, USA and Thailand
  (1,365 / 217 / 141 of the strains after 2012), and its early side carries all UK strains with a year,
  so "time" would mostly mean "country". All AMC test positives after 2012 come from the USA and
  Thailand collections.
- **UK only: rejected as an axis.** Only 1,439 strains, years 2001–2012; only AMC and CIP pass at
  any cut-off.
- **Norway only: chosen.** Train 2002–2013 (1,818 strains), test 2014–2017 (1,146 strains). The
  cut-off 2013 keeps 39% of Norway in test. Every cut-off from 2012 to 2014 gives the same set of
  antibiotics that pass, and 2013 is the middle one.

| antibiotic | train ≤ 2013 R / S | test 2014–17 R / S | train prev | test prev | tier |
|---|---:|---:|---:|---:|---|
| AMP | 647 / 1,171 | 518 / 628 | 0.356 | 0.452 | primary |
| CIP | 149 / 1,669 | 188 / 957 | 0.082 | 0.164 | primary |
| SXT | 449 / 1,369 | 324 / 822 | 0.247 | 0.283 | primary |
| GEN | 139 / 1,679 | 92 / 1,054 | 0.076 | 0.080 | primary |
| CTX | 69 / 1,553 | 82 / 1,061 | 0.043 | 0.072 | secondary (train R < 100) |
| CAZ | 68 / 1,750 | 80 / 1,065 | 0.037 | 0.070 | secondary (train R < 100) |
| AMC | 0 / 0 | 1 / 0 | — | — | not tested in Norway |

Unlike the lineage holdout, prevalence **rises** in the test period for CIP, CTX and CAZ. This is
a useful contrast: if the model's holdout drop were only a low-prevalence artefact, it should not
reappear here.

Lineage overlap: 93.8% of the test strains with a known ST have an ST seen in 2002–2013 (96 of the
152 test STs).

### Splits that are not possible

| axis | why |
|---|---|
| host | 24 non-human strains |
| any country other than UK / Norway as a test side | largest is USA, 227 strains, 98% AMP-R |
| leave-one-study-out | considered; the UK studies have different antibiotic panels (above), so most study × antibiotic cells are empty. Possible later for CIP, GEN, CTX, CAZ with Norway / 28720578 / 34485958 as the three folds |

## 4. Arcadia's "missing = reference" encoding

### What the code does (read, not assumed)

`code/notebooks/snp_matrix_from_vcf.ipynb` builds `arcadia_v2_snp_matrix.pkl` from Arcadia's
`filtered_output.vcf`:

```python
col = np.zeros(n_sel, dtype=np.int8)  # default 0 = ref
...
if a is None or a < 0:
    continue  # missing → ref (stays 0)
col[i] = a  # 0=ref, 1=alt1, 2=alt2, ...
```

So:

- A missing call becomes the reference allele.
- The 1% MAF filter counts missing calls as reference.
- Codes 1–3 are the index of the alternate allele (not dosages).
- The saved value distribution is 0: 3.10 × 10⁹, 1: 6.08 × 10⁸, 2: 1.49 × 10⁷, 3: 5.49 × 10⁵.
- How many of the zeros were missing calls was not recorded.

### What follows

A strain that lacks an accessory gene has no reads there, so every SNP of that gene reads as
"reference". In the SNP view, absence of a gene and the reference allele of that gene are the same
value. For accessory genes this makes the SNP view partly a presence/absence view, and SNP-level
"mutations" in accessory genes (e.g. inside a plasmid gene) are confounded with carriage. Core genes
such as gyrA and parC are unaffected in practice, because nearly all strains carry them.

### Check that needs Kaggle (the SNP matrix and VCF are only there)

Notebook `w0_snp_encoding` (CPU, about 3 h because the VCF is streamed once; account B), for every
kept SNP column:

1. Count missing calls in the VCF, the number the original build threw away. Record its distribution, and
   the share of columns where missing > 10% of strains.
2. Join to the gene's presence/absence column. Report, per column, the share of "0" codes that come
   from strains whose gene is absent.
3. Report how many of the 262,225 / 5,000-gene columns used by the models are in accessory
   genes (gene present in < 95% of strains).

Needs: `filtered_output.vcf.gz` (dataset `arcadia-presenceabsence-v2-ecoli-rawdownload`, already on
Kaggle), `arcadia_v2_snp_matrix.pkl`, `arcadia_v2_gene_presence_absence.pkl`.

### A second encoding problem found here: the presence/absence matrix is not "present = 1"

`arcadia_v2_gene_presence_absence.pkl` (32,441 columns: 18,494 CDS + 13,947 intergenic) records
`encoding_rule = "present = (PLINK_value == 0); A1=REF, --keep-allele-order"`. Compared with the
72 ECOR genomes the pan-genome was built from (`raw/pangenome/gene_presence_absence.csv`):

`results/data_inventory/pa_orientation_by_ecor_frequency.csv`

| share of the 72 ECOR genomes carrying the gene | CDS columns | mean "present" share in our 6,962 strains |
|---|---:|---:|
| ≤ 5% | 9,750 | 0.086 |
| 5–15% | 2,948 | 0.176 |
| 15–30% | 1,300 | 0.287 |
| 30–50% | 592 | 0.258 |
| 50–70% | 338 | 0.253 |
| 70–85% | 262 | 0.220 |
| 85–95% | 231 | 0.048 |
| 95–99% | 421 | 0.014 |
| ≥ 99% (core) | 2,652 | **0.000** |

- No column of the matrix is "present" in more than 50.3% of strains.
- 2,642 of the 2,652 ECOR core genes are "present" in under 1% of our strains.
- gyrA (`LMHECDEF_04343`) and parC (`LMHECDEF_01124`) are "present" in **0** of 6,962 strains.

A column of this matrix therefore marks the **minority state** of a gene, not its presence. For genes carried by most strains, 1 means *absent*.

**Most likely cause:** PLINK assigns A1 to the minor allele by default. If the original
`.bim` had A1 = the minor state, the rule `present = (value == 0)` holds only for genes carried by
fewer than half of the strains. This is a strong hypothesis, not proven: the raw `.bed`/`.bim` are
only on Kaggle.

What it does and does not affect:

- **Not affected:**
  - Tree models, logistic regression, kNN on Hamming distance and any model that treats a column
    and its complement alike.
  - Every predictive number of the paper is unchanged by this, as far as the encoding goes.
- **Affected:**
  - Lens B. It multiplies gene embeddings by `X_pa`, so for majority genes it attends to absence,
    and its "strain context" is the mean embedding of the minority states, not of the carried genes.
  - Every presence-based reading: "resistance genes found by attention", U_res/U_ST of "attended
    genes", "TEM present in 49% of isolates" (TEM's column is at 0.492, i.e. close to the flip point;
    its P(AMP-R | 1) = 0.84 suggests 1 = present for TEM, but that must be checked), the
    per-strain gene counts (min 2,141, max 8,333), and W4's mapping of AMRFinderPlus hits to columns.
  - Jaccard distances on P/A.

**Check (Kaggle, CPU, minutes; add to `w0_snp_encoding`):**

1. Read the `.bim` A1/A2 alleles of `LMHECDEF_04343` (gyrA) and of a rare gene.
2. Recompute presence from the `.bed` with the allele that means "present" in Arcadia's own
   documentation.
3. Confirm the flip column by column. Expected result: about 4,000 CDS columns and an unknown number
   of intergenic columns flip.

**Until it is checked:**

- W2's kNN uses Hamming distance, which does not care about orientation.
- W4 must orient columns before mapping genes.

Changing the matrix on `main` is the owner's decision (it changes Lens B, so the running re-run would describe the old encoding). See `docs/PROGRESS.md`.

## 5. What W0 needs on Kaggle

The `raw/` files are not yet in the Kaggle dataset. Notebooks that need them:

| file | needed by |
|---|---|
| `raw/strains_metadata.csv` | the country and time split builders (W1), and anything that recomputes the inventory |
| `raw/pangenome/pangenome_cds.fa` | W4 (AMRFinderPlus) |
| `raw/pangenome/gene_presence_absence.csv` | the P/A orientation fix (W0 / W4) |
| `processed/cds_eggNog.tsv`, `processed/genes.gff` | W4 |

The split files themselves (`country_split.pkl`, `time_split.pkl`) are built on CPU from
`strains_metadata.csv` and are small. They can also be built here and uploaded to the Kaggle dataset
next to `mlst_lineage_holdout_split.pkl`, so the Kaggle notebooks do not need the raw metadata.
This is the plan in `docs/PLAN.md`.

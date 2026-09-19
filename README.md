# Interpretable Sentiment Classification Across Two Technical Domains

CSE 4122 course project. The thesis: **domain-specific vocabulary breaks sentiment
models, and the failure mode generalises.** Two unrelated corpora, one shared
pipeline, plus cross-domain transfer experiments.

| Domain | Corpus | The vocabulary problem |
|---|---|---|
| `se` | Senti4SD gold standard (StackOverflow) | Harsh technical jargon that is functionally neutral — *fatal error*, *kill the process*, *crashed* |
| `health` | UCI Drug Reviews, Druglib.com (id 461) | Relief inversion — a clinically negative word naming a symptom that went away: *the nausea stopped*, *no more panic attacks* |

Full task breakdown and ownership: [project_work_division.md](project_work_division.md).

---

## Status

Phase 1 (corpus construction) and Phase 2's preprocessing module are done. The
health corpus is **blocked on hand-labelling**, which is the GATE 1 gate.

| | Status |
|---|---|
| SE corpus | **frozen** — 4,331 rows, Fleiss' κ = 0.759 |
| Health corpus | **incomplete** — 1,341 rows, 0 neutral. Needs 835 hand-labelled neutrals |
| Preprocessing module | **done** — 29 tests passing |
| Label validation | scorer **ready**, 100 rows awaiting hand-labelling |
| Splits, features, models, explainability | not started |

GATE 1 is not yet passed. See [Finishing the health corpus](#finishing-the-health-corpus).

---

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

Download both raw corpora (~700 MB, mostly the Senti4SD word-embedding model):

```bash
python data/download.py
```

**Run every command below from the repository root**, using `python -m`. The
modules import each other as `src.*`, so invoking a file by path will fail.

---

## Pipeline

```bash
python -m src.build_se          # Senti4SD  -> data/processed/se/
python -m src.build_health      # Druglib   -> data/processed/health/
python -m src.validate_health   # scores the label validation (see below)
python -m tools.make_examples   # -> report/preprocess_examples.md
python -m pytest -q             # 29 tests
```

`build_se` must run before `build_health`: the health corpus is balanced to the
SE class proportions, which `build_se` writes to
`data/processed/se/build_log.json`. Without it, `build_health` falls back to a
uniform target and says so.

Useful flags:

```bash
python -m src.build_health --inspect               # print rows each filter removed
python -m src.build_health --neutral-min-words 8   # see the length trade-off below
python -m src.validate_health --annotator2 FILE    # inter-annotator Cohen's κ
```

Every script writes a `build_log.json` next to its output holding every count it
reports. Those numbers go straight into the report — don't retype them by hand.

---

## Corpus statistics

Recorded from the build logs, not by eye. Both corpora use the project CSV
contract: `id;text;polarity`, semicolon-delimited, UTF-8 without BOM, labels
ordered `["negative", "neutral", "positive"]`, seed `42`.

### SE — `data/processed/se/clean.csv`

Source: `Senti4SD_GoldStandard_EmotionPolarity.xlsx`. The workbook is used rather
than Senti4SD's train/test partition CSVs because it holds the same 4,423 items
*plus* the `r1`/`r2`/`r3` rater columns, without which agreement cannot be
computed.

| | |
|---|---|
| Rows loaded | 4,423 |
| Dropped: duplicate text | 92 |
| Dropped: empty / bad label | 0 |
| **Final** | **4,331** |
| Fleiss' κ | **0.7586** over 4,359 items, 3 raters |
| Mean length | 29.97 words |

| Class | Count | Share |
|---|---|---|
| negative | 1,176 | 27.15% |
| neutral | 1,662 | 38.37% |
| positive | 1,493 | 34.47% |

The κ of 0.759 clears GATE 1's 0.75 threshold. Items where any rater was missing
or unparseable are excluded from κ (the rater columns are free text — mixed case
plus the typos `Postive`, `Poitive`, `Netural`, which are repaired first).

### Health — `data/processed/health/clean.csv`

Only `commentsReview` is used as training text. `benefitsReview` and
`sideEffectsReview` are positive- and negative-leaning *by construction*, so they
are useless as training data and are set aside to `bias_probe.csv` for the bias
probe in Task B5.3.

| Filter | Rows removed |
|---|---|
| Rows loaded | 4,143 |
| missing text | 13 |
| boilerplate (`see above`, `none`, `thanks`) | 25 |
| copy-pasted drug monograph | 24 |
| under 15 words | 776 |
| duplicate text | 61 |
| **After filtering** | **3,261** |

Rating bands (1–3 negative, 8–10 positive, 4–7 discarded):

| Band | Count |
|---|---|
| negative (1–3) | 591 |
| positive (8–10) | 1,823 |
| discarded (4–7) | 847 |

4–7 is **discarded, never mapped to neutral**. A mid rating means *mixed
feelings*; Senti4SD neutral means *absence of affect*. Conflating the two would
train the model to call emotionally intense text neutral and would silently
destroy every cross-domain comparison.

Current `clean.csv` holds **1,341 rows** (591 negative, 750 positive, **0
neutral**), mean 64.7 words — roughly twice the SE mean, which is worth noting
when comparing the two domains.

---

## The domain vocabulary problem

Real rows from the built corpora, for the report and the slides.

**SE — jargon that is functionally neutral.** 40 of the 1,662 gold-neutral rows
contain a word from the SE jargon lexicon:

- *i think that scope of 'killed' is ok.*
- *You don't know if threads are killed unless you catch a signal that tells you so!*
- *Try to add the complete failed message in the question!*
- *Here, I see `BAADF00D` (bad food), `BEEFCACE` (beef cake), `BAADCAB1E` (bad cable), `BADCAFE` (bad cafe), and `DEADDEAD` (dead dead). Is this intentional?*
- *Is GLUT dead for graphics programming?*

**Health — relief inversion.** **197 of 750** positive rows contain a symptom
word, and 50 of those pair it with relief framing (*no more*, *stopped*, *went
away*, *gone*):

- *The pain from the shots have practically disappeared.*
- *The depression went away shortly after starting the Premarin.*
- *After 4 days of the treatment, the swelling and the pain had almost gone.*
- *At times I have stopped taking it for 30-50 days and the depression returns.*
- *I cannot tolerate the pain without the Nortriptyline 50mg at bedtime.*

The health share (26% of positives carrying negative-sounding vocabulary) is much
higher than the SE one (2.4% of neutrals), so the two domains stress the model
by different amounts — that asymmetry belongs in the results discussion, not
hidden.

---

## Finishing the health corpus

Two hand-labelling jobs. Nothing downstream is valid until both are done.

### 1. Neutral class — 835 rows

```bash
python -m src.build_health      # writes 1,200 pre-scored candidates
```

Open `data/processed/health/neutral_candidates.csv`, fill the empty `polarity`
column with `negative` / `neutral` / `positive`, save it as
`data/processed/health/neutral_labelled.csv`, then re-run `build_health`.

Candidates are pre-scored on dosage, imperative and schedule signals with any
overt affect word disqualifying the row, which cuts the labelling work by about
an order of magnitude. Rows rated 4–7 are offered first because the binning
discards them anyway, so labelling them costs no other class any data. 1,343
candidates pass the prefilter.

**Why 835 and not the 400 the plan budgets.** Negative is capped at 591 rows, so
matching SE's 27.15% negative share caps the whole corpus at ~2,177 rows, and
38.37% of that is 835. The 400 figure assumed a 4,000-row corpus that the
negative class cannot support. `balance()` computes this and logs it as
`neutral_rows_needed_for_target` rather than silently dropping the class.

### 2. Label validation — 100 rows

Fill the `my_label` column in `data/processed/health/validation_sample.csv`,
then:

```bash
python -m src.validate_health
```

**Label from the text alone.** The file carries `rating` and `rating_derived`
columns for later diagnosis; reading them while labelling makes the agreement
figure worthless.

The scorer reports agreement and Cohen's κ for the current 1–3 / 8–10 binning
**and** the stricter 1–2 / 9–10 fallback side by side, plus a breakdown of
disagreements by rating and direction. If the current scheme falls below 80%
agreement, the output tells you whether the fallback rescues it or whether to
take B1.4's documented escape hatch (switch to the Drugs.com corpus, UCI id 462).

### GATE 1 checklist

- [x] Both corpora in `id;text;polarity` form
- [x] SE annotator agreement ≥ 0.75 — κ = 0.759
- [ ] ≥ 300 usable neutral health rows — currently 0
- [ ] Rating-agreement ≥ 80% on the validation sample
- [ ] Comparable size and class balance across the two corpora

---

## Preprocessing contract

`src/preprocess.py` is the single preprocessing path for every model in both
domains. It has no I/O and no pandas dependency, so it imports cleanly into
tests, notebooks, sklearn pipelines and the CLI.

```python
from src.preprocess import preprocess, tokenize, TOKEN_PATTERN

preprocess("Kill the process before restarting", "se")
preprocess("I take 600mg three times a day", "health")   # -> 'i take DOSE three times a day'
```

Shared rules, in order: `html.unescape` → lowercase → URLs `URL` → `@mentions`
`USER` → **domain rules** → emoticons `EMO_POS`/`EMO_NEG` → negation expansion →
collapse repeated characters and punctuation → squeeze whitespace.

Domain rules are the **only** permitted divergence. `se`: code tags, backtick
spans, `foo(bar)` calls and `a.b.c` paths → `CODE`. `health`: dosages → `DOSE`,
remaining bare numbers → `NUM`.

Deliberately **not** done: stopword removal and stemming. `kill`/`killed` and
`stop`/`stopped` are the phenomenon this project measures. Negation words are
never removed — in the health corpus, negation *is* the effect.

`TOKEN_PATTERN` is the single source of truth for tokenisation and must be
passed to `TfidfVectorizer(token_pattern=...)` in Task A3 and to
`LimeTextExplainer(split_expression=...)` in Task A6, so the three cannot drift
apart. A test enforces that it matches `tokenize()`.

`report/preprocess_examples.md` is generated from `describe_rules()`, so the
report's rule table cannot drift away from the code. Regenerate it, never edit
it by hand.

### Two documented deviations from the plan's rule order

Both are order-only; the set of rules is unchanged and still identical across
domains, so no cross-domain comparison is affected.

1. **Lowercasing moves from step 4 to step 2**, so every placeholder stays
   uppercase. The literal words *url* and *user* are common in StackOverflow
   text; lowercase placeholders would be indistinguishable from them and would
   silently corrupt the TF-IDF vocabulary.
2. **URL replacement runs before emoticon replacement.** `http://` contains
   `:/`, a sad-face emoticon; the reverse order turns every URL into
   `httpEMO_NEG/...`.

---

## Known issues and open decisions

**The 15-word filter fights the neutral class.** B1.2's minimum length removed
776 rows including *"Take pill once a day"* and *"400 mg every morning, 200 mg
early afternoon."* — exactly the absence-of-affect text B1.3 builds the neutral
class from. The threshold is left at 15 for all classes (no divergence).
`--neutral-min-words 8` raises the candidate pool from 1,343 to 1,543, but buys
those 200 neutrals at the cost of a length confound where the model can learn
"short = neutral" instead of learning sentiment. `mean_words_per_class` is logged
so the confound is measurable. **Undecided — this belongs in the report either
way.**

**Inter-annotator κ needs two people.** Working solo, only rating-agreement can
be reported; `validate_health` says so rather than inventing a number. This is
the project's main methodological weakness and must be stated plainly in
Limitations, alongside the label-provenance asymmetry: SE labels are
human-annotated by three raters with reported agreement, health labels are
rating-derived plus hand-labelled neutrals.

**Health text is ~2× longer than SE text** (64.7 vs 29.97 mean words). Relevant
to LIME/SHAP stability (Task A6.3) and to Word2Vec quality.

**`data/processed/health/clean.csv` contains Druglib review text verbatim**
(wording unchanged; only runs of whitespace are collapsed, so a row is one
line),
while `.gitignore` currently excludes only `data/raw/`. The Druglib licence is
research-only with no redistribution. **Decide whether to exclude the processed
health files from version control too.**

---

## Repository layout

```
data/
  download.py                     fetches both raw corpora
  raw/senti4sd/                   never edited; git-ignored
  raw/druglib/                    never edited; git-ignored (licence)
  processed/se/
    clean.csv                     id;text;polarity  (4,331 rows)
    build_log.json                filter counts, Fleiss' κ, class distribution
  processed/health/
    clean.csv                     id;text;polarity  (1,341 rows, no neutrals yet)
    bias_probe.csv                benefits/side-effects text for Task B5.3
    neutral_candidates.csv        1,200 pre-scored rows to hand-label
    validation_sample.csv         100 rows to hand-label for the κ check
    build_log.json                every filter and balance count
src/
  preprocess.py                   preprocess(), tokenize(), TOKEN_PATTERN
  build_se.py                     Senti4SD -> clean.csv  (Task A1)
  build_health.py                 Druglib  -> clean.csv  (Task B1)
  validate_health.py              scores the label validation (Task B1.4)
tests/test_preprocess.py          29 tests
tools/make_examples.py            generates the report's rule and example tables
report/preprocess_examples.md     generated; do not edit
```

Not yet written: `splits.py`, `evaluate.py`, `features_tfidf.py`,
`features_embed.py`, `models_tfidf.py`, `models_embed.py`, `transfer.py`,
`explain_lime.py`, `explain_shap.py`, `stress_test.py`, `cli/predict.py`,
`notebooks/demo.ipynb`.

**Next step after GATE 1:** Task B2 — `splits.py` (70/15/15 stratified, seed 42,
frozen afterwards) and `evaluate.py`. `clean.csv` is not model-ready until the
splits exist.

---

## Data licence

The Druglib donors require research-only use, no redistribution, and citation.
Raw Druglib files are git-ignored; `data/download.py` is committed instead. Cite:

> Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-Based
> Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data
> Learning. *Proceedings of the 2018 International Conference on Digital Health*,
> 121–125.

Senti4SD: Calefato, F., Lanubile, F., Maiorano, F., & Novielli, N. (2018).
Sentiment Polarity Detection for Software Development. *Empirical Software
Engineering*, 23(3), 1352–1382.

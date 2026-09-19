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

## The pipeline

Six stages, run **once per domain, one domain at a time**.

```
Stage 1  Data acquisition
Stage 2  Text extraction & clean-up
Stage 3  Preprocessing        (sentence segmentation, stemming, lemmatization)
Stage 4  Feature engineering
Stage 5  Model building
Stage 6  Evaluation
─────────────────────────────────────────────────────────────
Stage 7  Cross-domain integration  (transfer, demos, report)
```

**Order is fixed: finish SE completely, then start HEALTH.**

```
TRACK SE      Stage 1 → 2 → 3 → 4 → 5 → 6   ──▶ SE GATE
TRACK HEALTH  Stage 1 → 2 → 3 → 4 → 5 → 6   ──▶ HEALTH GATE
TRACK JOINT   Stage 7                        ──▶ SUBMISSION
```

Why: the health corpus is blocked on 835 rows of hand-labelling, and under the old
parallel plan that blocked the SE work too — which needs no hand-labelling at all.
The pipeline is domain-agnostic, so once it runs end-to-end on SE, the HEALTH track
is mostly configuration. The cost is that cross-domain transfer lands in the final
two weeks; the mitigation is a smoke-transfer check in Stage H-5.

The one sanctioned exception: **health neutral hand-labelling runs in the background
from Week 2**, because it is human time that cannot be compressed later. Nothing
else from the HEALTH track starts early.

---

## Status

**Current track: SE.** Stages 1, 2 and part of 3 are done. Stage 3's normalization
module is the next thing to write.

| Stage | Component | SE | HEALTH |
|---|---|---|---|
| 1 | Data acquisition | ✅ done | ✅ done |
| 2 | Corpus extraction | ✅ **frozen** — 4,331 rows, Fleiss' κ = 0.759 | ⚠️ 1,341 rows, **0 neutral** — needs 835 hand-labelled |
| 2 | String clean-up (`clean_text.py`) | ✅ done — 36 tests passing | ✅ same module, health branch |
| 2 | Label validation | — | ⏸ scorer ready, 100 rows awaiting hand-labelling |
| 3 | Segmentation / stemming / lemmatization | ❌ **not started** | ⏸ blocked on SE |
| 3 | Normalization ablation | ❌ **not started** | ⏸ blocked on SE |
| 3 | Splits | ✅ frozen — 3,031 / 650 / 650 | ⚠️ exist but **invalid** (built from the 0-neutral corpus) |
| 4 | Feature engineering | ❌ not started | ⏸ blocked |
| 5 | Model building | ❌ not started | ⏸ blocked |
| 6 | Evaluation framework | ✅ done | ✅ same module |
| 6 | Baselines | ✅ majority + VADER | ⚠️ provisional — must be re-run |
| 6 | LIME / SHAP / stress tests | ❌ not started | ⏸ blocked |
| 7 | Transfer, demos, report | ❌ not started | |

**Blocking issue — the environment does not match `requirements.txt`.** `.venv`
currently has only `pandas` installed, at **3.0.6** against a pinned **2.2.3**, so
`tests/test_evaluate.py` cannot even be collected (`No module named 'matplotlib'`).
Fix this before starting Stage 3:

```bash
.venv/Scripts/activate
pip install -r requirements.txt
python -m pytest -q          # must collect and pass cleanly
```

See [SE GATE](#-se-gate) for what still stands between here and the HEALTH track.

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
python -m src.acquisition.download
```

**Run every command from the repository root**, using `python -m`. The modules
import each other as `src.<stage>.<module>`, so invoking a file by path will fail.

---

## Running the pipeline

Stage by stage. `--domain` selects the corpus; **only run `health` after the SE
GATE is signed.**

```bash
# Stage 1 — acquisition
python -m src.acquisition.download

# Stage 2 — extraction & clean-up
python -m src.extraction.build_se              # -> data/processed/se/
python -m src.extraction.build_health          # -> data/processed/health/
python -m src.extraction.validate_health       # scores the label validation

# Stage 3 — preprocessing
python -m src.preprocessing.splits --domain se
# normalize.py and the ablation: not written yet

# Stage 6 — evaluation
python -m tools.run_baselines --domain se
python -m tools.make_examples                      # -> report/preprocess_examples.md

python -m pytest -q
```

`build_se` must run before `build_health`: the health corpus is balanced to the SE
class proportions, which `build_se` writes to
`data/processed/se/build_log.json`. Without it, `build_health` falls back to a
uniform target and says so.

Useful flags:

```bash
python -m src.extraction.build_health --inspect               # rows each filter removed
python -m src.extraction.build_health --neutral-min-words 8   # see the length trade-off
python -m src.extraction.validate_health --annotator2 FILE    # inter-annotator Cohen's κ
```

Every script writes a `build_log.json` next to its output holding every count it
reports. **Those numbers go straight into the report — don't retype them by hand.**

---

## Repository layout

```
src/
  acquisition/
    download.py                   fetches both raw corpora
  extraction/
    build_se.py                   Senti4SD .xlsx -> clean.csv        [A]
    build_health.py               Druglib .tsv   -> clean.csv        [B]
    validate_health.py            rating-vs-human agreement scorer   [B]
    clean_text.py                 clean(), tokenize(), TOKEN_PATTERN [A]
  preprocessing/
    normalize.py                  segmentation / stemming / lemma    [A]  NOT WRITTEN
    splits.py                     70/15/15 stratified, seed 42       [B]
  features/
    features_tfidf.py             unigram, (1,2), (1,3)              [A]  NOT WRITTEN
    features_embed.py             GloVe + per-domain Word2Vec        [B]  NOT WRITTEN
  modeling/
    models_tfidf.py               MNB, LR, LinearSVC                 [A]  NOT WRITTEN
    models_embed.py               GNB, LR, LinearSVC                 [B]  NOT WRITTEN
  evaluation/
    evaluate.py                   metrics, confusion, McNemar        [B]
    explain_lime.py                                                  [A]  NOT WRITTEN
    explain_shap.py                                                  [B]  NOT WRITTEN
    stress_test.py                                                   [B]  NOT WRITTEN
  integration/
    transfer.py                   the 2×2 transfer matrix            [A]  NOT WRITTEN
    compare_explainers.py         LIME vs SHAP agreement           [Joint] NOT WRITTEN

data/
  raw/senti4sd/                   never edited; git-ignored
  raw/druglib/                    never edited; git-ignored (licence)
  processed/se/
    clean.csv                     id;text;polarity  (4,331 rows)
    train.csv val.csv test.csv    frozen, seed 42
    build_log.json                filter counts, Fleiss' κ, class distribution
  processed/health/
    clean.csv                     id;text;polarity  (1,341 rows, no neutrals yet)
    bias_probe.csv                benefits/side-effects text for Stage H-6.6
    neutral_candidates.csv        1,200 pre-scored rows to hand-label
    validation_sample.csv         100 rows to hand-label for the κ check
    build_log.json                every filter and balance count
  stress_test/                    se_stress.csv, health_stress.csv     NOT WRITTEN
  lexicons/                       se_jargon.txt, health_symptoms.txt   NOT WRITTEN

cli/predict.py                    CLI demo with --compare            [A]  NOT WRITTEN
notebooks/demo.ipynb              notebook demo                      [B]  NOT WRITTEN
models/{se,health}/               saved .joblib pipelines; git-ignored
results/
  metrics/  metrics_test/  figures/  ablation/  explanations/  transfer/
report/preprocess_examples.md     generated; do not edit
tools/                            helper scripts, not part of the pipeline
tests/                            one module per stage
```

**Renamed in the stage restructure:** `src/preprocess.py` →
`src/extraction/clean_text.py`, and `preprocess(text, domain)` →
`clean(text, domain)`. The old name collided with Stage 3, which is the stage
actually called "preprocessing". What the module does — HTML unescape, placeholders,
lowercase, emoticons, negation expansion, character collapsing — is clean-up, which
is Stage 2. **The rules themselves are unchanged.**

---

## Stage 1–2 — Corpus statistics

Recorded from the build logs, not by eye. Both corpora use the project CSV contract:
`id;text;polarity`, semicolon-delimited, UTF-8 without BOM, labels ordered
`["negative", "neutral", "positive"]`, seed `42`.

### SE — `data/processed/se/clean.csv` ✅ frozen

Source: `Senti4SD_GoldStandard_EmotionPolarity.xlsx`. The workbook is used rather
than Senti4SD's train/test partition CSVs because it holds the same 4,423 items
*plus* the `r1`/`r2`/`r3` rater columns, without which agreement cannot be computed.

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

The κ of 0.759 clears the SE GATE's 0.75 threshold. Items where any rater was
missing or unparseable are excluded from κ (the rater columns are free text — mixed
case plus the typos `Postive`, `Poitive`, `Netural`, which are repaired first).

**Splits (frozen, seed 42):** 3,031 train / 650 val / 650 test. Class proportions
hold to within one row per class.

### Health — `data/processed/health/clean.csv` ⚠️ incomplete

Only `commentsReview` is used as training text. `benefitsReview` and
`sideEffectsReview` are positive- and negative-leaning *by construction*, so they are
useless as training data and are set aside to `bias_probe.csv` for Stage H-6.6.

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

4–7 is **discarded, never mapped to neutral**. A mid rating means *mixed feelings*;
Senti4SD neutral means *absence of affect*. Conflating the two would train the model
to call emotionally intense text neutral and would silently destroy every
cross-domain comparison.

Current `clean.csv` holds **1,341 rows** (591 negative, 750 positive, **0 neutral**),
mean 64.7 words — roughly twice the SE mean, which is worth noting when comparing the
two domains.

**The existing health splits are invalid** and must be regenerated once the corpus is
complete; they were built from this 0-neutral version.

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

**Health — relief inversion.** **197 of 750** positive rows contain a symptom word,
and 50 of those pair it with relief framing (*no more*, *stopped*, *went away*,
*gone*):

- *The pain from the shots have practically disappeared.*
- *The depression went away shortly after starting the Premarin.*
- *After 4 days of the treatment, the swelling and the pain had almost gone.*
- *At times I have stopped taking it for 30-50 days and the depression returns.*
- *I cannot tolerate the pain without the Nortriptyline 50mg at bedtime.*

The health share (26% of positives carrying negative-sounding vocabulary) is much
higher than the SE one (2.4% of neutrals), so the two domains stress the model by
different amounts — that asymmetry belongs in the results discussion, not hidden.

---

## Stage 2 — Clean-up contract

`src/extraction/clean_text.py` is the single clean-up path for every model in
both domains. It has no I/O and no pandas dependency, so it imports cleanly into
tests, notebooks, sklearn pipelines and the CLI.

```python
from src.extraction.clean_text import clean, tokenize, TOKEN_PATTERN

clean("Kill the process before restarting", "se")
clean("I take 600mg three times a day", "health")   # -> 'i take DOSE three times a day'
```

Shared rules, in order: `html.unescape` → lowercase → URLs `URL` → `@mentions`
`USER` → **domain rules** → emoticons `EMO_POS`/`EMO_NEG` → negation expansion →
collapse repeated characters and punctuation → squeeze whitespace.

Domain rules are the **only** permitted divergence. `se`: code tags, backtick spans,
`foo(bar)` calls and `a.b.c` paths → `CODE`. `health`: dosages → `DOSE`, remaining
bare numbers → `NUM`.

**Stopword removal is off, permanently.** Negation words appear in every standard
stopword list, and removing them would break both domains at once — in the health
corpus, negation *is* the effect.

`TOKEN_PATTERN` is the single source of truth for tokenisation and must be passed to
`TfidfVectorizer(token_pattern=...)` in Stage 4 and to
`LimeTextExplainer(split_expression=...)` in Stage 6, so the three cannot drift
apart. A test enforces that it matches `tokenize()`.

`report/preprocess_examples.md` is generated from `describe_rules()`, so the report's
rule table cannot drift away from the code. **Regenerate it, never edit it by hand.**

### Two documented deviations from the plan's rule order

Both are order-only; the set of rules is unchanged and still identical across
domains, so no cross-domain comparison is affected.

1. **Lowercasing moves from step 4 to step 2**, so every placeholder stays uppercase.
   The literal words *url* and *user* are common in StackOverflow text; lowercase
   placeholders would be indistinguishable from them and would silently corrupt the
   TF-IDF vocabulary.
2. **URL replacement runs before emoticon replacement.** `http://` contains `:/`, a
   sad-face emoticon; the reverse order turns every URL into `httpEMO_NEG/...`.

---

## Stage 3 — Preprocessing, and the stemming question

**Not written yet. This is the next task.**

Stage 3 adds sentence segmentation, stemming and lemmatization in
`src/preprocessing/normalize.py`:

```python
normalize(text, *, segment=False, mode="none")   # mode: "none" | "lemma" | "stem"
```

### Why this stage is a measurement, not a formality

Stemming and lemmatization collapse inflected forms:

| Original | Stemmed | Lemmatized |
|---|---|---|
| killed, kills, killing | `kill` | `kill` |
| stopped, stops, stopping | `stop` | `stop` |

Normally that helps — fewer distinct tokens, denser counts, better learning from a
small corpus. **Here it may destroy the finding**, because tense carries the
sentiment in both domains:

- HEALTH: *"the nausea **stopped**"* (positive — relief) vs *"the nausea won't
  **stop**"* (negative — ongoing). Stemmed, both are `stop`, and relief inversion
  becomes unmeasurable.
- SE: *"**killed** the process"* (neutral, routine) collapses into the emotional
  register of *kill*, hiding the jargon effect LIME is meant to expose.

### The resolution: implement it, switch it, measure it

Rather than asserting either position, Stage 3 runs an **ablation**: the same model
(TF-IDF (1,2) + Logistic Regression, defaults) trained three times on the SE training
split — `none`, `lemma`, `stem` — and scored on the SE **validation** split.

| Mode | Vocabulary size | Macro-F1 | neutral→negative rate |
|---|---|---|---|
| none | — | — | — |
| lemma | — | — | — |
| stem | — | — | — |

→ `results/ablation/normalization_se.csv`

**The winning mode by validation macro-F1 becomes the project default** and is used
unchanged for every later model in both domains. Whatever wins, the table is a report
finding: if stemming raises macro-F1 but also raises the neutral→negative confusion
rate, that is the accuracy/interpretability trade-off the whole project is about,
caught in one table.

Implementation notes:

- **Segmentation** — NLTK `punkt`, exposed as `segment(text) -> list[str]`. Used for
  per-sentence length statistics; it does **not** alter the text fed to the models.
- **Stemming** — `PorterStemmer`, with `SnowballStemmer("english")` as the alternate.
- **Lemmatization** — `WordNetLemmatizer` **with POS tags** from `nltk.pos_tag`.
  Without POS tags, `stopped` lemmatizes to `stopped`, not `stop`, and the comparison
  is meaningless.
- Stopword removal stays off in all three modes.

The ablation is re-run once on HEALTH in Stage H-3. **If a different mode wins there,
do not switch** — keep the SE choice for both and report the disagreement. Different
normalization per domain would confound every cross-domain comparison.

---

## Stage 6 — Baselines so far

| Domain | Model | Accuracy | Macro-F1 | SE neut→neg | HEALTH pos→neg |
|---|---|---|---|---|---|
| se | majority | 0.385 | 0.185 | — | — |
| se | vader | 0.715 | 0.708 | **0.252** | — |
| health | majority | 0.559 | 0.239 | — | — |
| health | vader | 0.485 | 0.350 | — | **0.416** |

The health rows are **provisional** — computed on the 0-neutral corpus and due to be
re-run in Stage H-6. They are kept because the VADER health number is already the
clearest single demonstration of the thesis: a general-purpose lexicon calls **41.6%**
of genuinely positive drug reviews negative.

Note the two different failure directions. SE's is gold-neutral → negative; HEALTH's
is gold-**positive** → negative. `evaluate()` reports both columns for every model.

---

## ✅ SE GATE

The HEALTH track does not open until every box is ticked.

- [x] `data/processed/se/clean.csv` in `id;text;polarity` form — 4,331 rows
- [x] Inter-rater agreement ≥ 0.75 — Fleiss' κ = **0.759**
- [x] Splits frozen and committed — 3,031 / 650 / 650, seed 42
- [x] `clean_text.py` passing its test suite — 36 tests
- [x] `evaluate.py` written; SE baselines recorded
- [ ] **Environment matches `requirements.txt`; `pytest` collects cleanly**
- [ ] `normalize.py` written and tested
- [ ] **Normalization ablation run; project default chosen and recorded**
- [ ] TF-IDF and embedding features built for SE
- [ ] 21 SE models trained, tuned and saved
- [ ] SE test evaluated **once**; error analysis tagged
- [ ] LIME and SHAP run on the 20 agreed SE instances; both bias tables produced
- [ ] SE stress set written, cross-annotated (κ recorded) and run
- [ ] **The full SE pipeline reruns end-to-end from `clean.csv` with one command**

That last box is the real gate. The HEALTH track's entire cost advantage depends on
the SE pipeline being *reproducible*, not merely *finished once*.

---

## Finishing the health corpus (Stage H-2, background work)

Two hand-labelling jobs. They are the only HEALTH-track work sanctioned to start
before the SE GATE, because they are human time that cannot be compressed later.
Budget ~30 rows per person per day.

### 1. Neutral class — 835 rows

```bash
python -m src.extraction.build_health      # writes 1,200 pre-scored candidates
```

Open `data/processed/health/neutral_candidates.csv`, fill the empty `polarity` column
with `negative` / `neutral` / `positive`, save it as
`data/processed/health/neutral_labelled.csv`, then re-run `build_health`.

Candidates are pre-scored on dosage, imperative and schedule signals, with any overt
affect word disqualifying the row, which cuts the labelling work by about an order of
magnitude. Rows rated 4–7 are offered first because the binning discards them anyway,
so labelling them costs no other class any data. 1,343 candidates pass the prefilter.

**Why 835 and not the 400 the plan originally budgeted.** Negative is capped at 591
rows, so matching SE's 27.15% negative share caps the whole corpus at ~2,177 rows, and
38.37% of that is 835. The 400 figure assumed a 4,000-row corpus the negative class
cannot support. `balance()` computes this and logs it as
`neutral_rows_needed_for_target` rather than silently dropping the class.

### 2. Label validation — 100 rows

Fill the `my_label` column in `data/processed/health/validation_sample.csv`, then:

```bash
python -m src.extraction.validate_health
```

**Label from the text alone.** The file carries `rating` and `rating_derived` columns
for later diagnosis; reading them while labelling makes the agreement figure
worthless.

The scorer reports agreement and Cohen's κ for the current 1–3 / 8–10 binning **and**
the stricter 1–2 / 9–10 fallback side by side, plus a breakdown of disagreements by
rating and direction. If the current scheme falls below 80% agreement, the output
tells you whether the fallback rescues it or whether to take the documented escape
hatch (switch to the Drugs.com corpus, UCI id 462).

### HEALTH GATE checklist

- [x] Corpus in `id;text;polarity` form
- [ ] ≥ 300 usable neutral health rows — **currently 0**
- [ ] Rating-agreement ≥ 80% on the validation sample
- [ ] Comparable size and class balance across the two corpora
- [ ] Splits **regenerated** from the completed corpus and re-frozen
- [ ] Health baselines re-run (the provisional rows overwritten)
- [ ] 21 health models trained; LIME, SHAP, bias probe, stress test complete
- [ ] **Smoke transfer shows degradation** — if it does not, it is a bug

---

## Known issues and open decisions

**The environment is broken.** `.venv` has only `pandas` 3.0.6 installed against a
pinned 2.2.3, and `tests/test_evaluate.py` cannot be collected. **Fix before Stage 3.**

**The 15-word filter fights the neutral class.** Stage H-2.2's minimum length removed
776 rows including *"Take pill once a day"* and *"400 mg every morning, 200 mg early
afternoon."* — exactly the absence-of-affect text the neutral class is built from. The
threshold is left at 15 for all classes (no divergence). `--neutral-min-words 8`
raises the candidate pool from 1,343 to 1,543, but buys those 200 neutrals at the cost
of a length confound where the model can learn "short = neutral" instead of learning
sentiment. `mean_words_per_class` is logged so the confound is measurable.
**Undecided — this belongs in the report either way.**

**Inter-annotator κ needs two people.** Working solo, only rating-agreement can be
reported; `validate_health` says so rather than inventing a number. This is the
project's main methodological weakness and must be stated plainly in Limitations,
alongside the label-provenance asymmetry: SE labels are human-annotated by three
raters with reported agreement, health labels are rating-derived plus hand-labelled
neutrals.

**Health text is ~2× longer than SE text** (64.7 vs 29.97 mean words). Relevant to
LIME/SHAP stability (Stage 6.5) and to Word2Vec quality.

**`data/processed/health/clean.csv` contains Druglib review text verbatim** (wording
unchanged; only runs of whitespace are collapsed, so a row is one line), while
`.gitignore` currently excludes only `data/raw/`. The Druglib licence is research-only
with no redistribution. **Decide whether to exclude the processed health files from
version control too — before the text enters the git history**, since removing it
afterwards requires a history rewrite.

**SE-first pushes cross-domain transfer to the final two weeks.** The mitigation is
the smoke-transfer check in Stage H-5: run one throwaway SE→HEALTH prediction the
moment the first health model is saved, so a structural problem surfaces in Week 7
rather than Week 8.

---

## Data licence

The Druglib donors require research-only use, no redistribution, and citation. Raw
Druglib files are git-ignored; `src/acquisition/download.py` is committed instead.
Cite:

> Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-Based
> Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.
> *Proceedings of the 2018 International Conference on Digital Health*, 121–125.

Senti4SD: Calefato, F., Lanubile, F., Maiorano, F., & Novielli, N. (2018). Sentiment
Polarity Detection for Software Development. *Empirical Software Engineering*, 23(3),
1352–1382.

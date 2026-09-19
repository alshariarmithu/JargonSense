# Interpretable Sentiment Classification in Software Engineering Communication

**CSE 4122 course project** · Al Shariar Hossain (2107066) · Hassan Mohammed Naquibul Hoque (2107077)

Sentiment models trained on general English misread software engineering text.
Words like *fatal error*, *kill the process* and *crashed* sound hostile but are
functionally neutral to a developer. This project builds a sentiment classifier for
StackOverflow posts and then uses **explainability methods to audit *why* it
succeeds or fails** on that vocabulary — not just to measure accuracy.

**The problem, already measured.** VADER, a standard general-purpose sentiment
lexicon, calls **25.2% of genuinely neutral StackOverflow posts negative**:

| Baseline | Accuracy | Macro-F1 | Gold-neutral called negative |
|---|---|---|---|
| Majority class | 0.385 | 0.185 | — |
| **VADER** | 0.715 | 0.708 | **25.2%** |

One in four. That is the problem this project investigates, quantified before a
single model was trained.

Full task breakdown, schedule and scope decisions: **[project_plan.md](project_plan.md)**

---

## The pipeline

Six stages. Each produces a file the next one consumes, so the whole project
re-runs from raw data with a handful of commands.

```
Stage 1  Data acquisition        Senti4SD gold standard
Stage 2  Text extraction         raw .xlsx  ->  clean.csv, plus clean()
Stage 3  Preprocessing           sentence segmentation, stemming, lemmatization
Stage 4  Feature engineering     TF-IDF (1,1)(1,2)(1,3), GloVe, Word2Vec
Stage 5  Model building          Naive Bayes, Logistic Regression, Linear SVM
Stage 6  Evaluation              metrics, LIME, SHAP, stress test, demo
```

---

## Status

| Stage | Component | Status |
|---|---|---|
| 1 | Data acquisition | ✅ done |
| 2 | Corpus extraction — 4,331 rows, Fleiss' κ = 0.759 | ✅ done |
| 2 | String clean-up (`clean_text.py`) | ✅ done — 36 tests |
| 3 | Segmentation / stemming / lemmatization | ✅ done — 23 tests |
| 3 | Normalization ablation | ✅ done — chose `none` |
| 3 | Splits — 3,031 / 650 / 650, frozen | ✅ done |
| 4 | TF-IDF features | ✅ done — 17 tests |
| 4 | GloVe + Word2Vec | ✅ done — 16 tests |
| 5 | 9 TF-IDF models | ✅ done — 19 tests |
| 5 | 6 embedding models | ✅ done |
| 6 | Evaluation framework | ✅ done |
| 6 | Baselines (majority, VADER) | ✅ done |
| 6 | Final test evaluation + error analysis | ✅ done |
| 6 | LIME / SHAP | ✅ done |
| 6 | Stress test (60 sentences) | ✅ done |
| 6 | Notebook demo | ✅ done |
| — | Report | ✅ [report/REPORT.md](report/REPORT.md) |

**123 tests passing.** The pipeline reproduces end to end from raw data.

---

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

Download the corpus:

```bash
python -m src.acquisition.download
```

**Run every command from the repository root**, using `python -m`. Modules import
each other as `src.<stage>.<module>`, so running a file by path will fail.

---

## Running the pipeline

```bash
# Stage 2 — build the corpus from the raw workbook
python -m src.extraction.build_se

# Stage 3 — create the frozen train/val/test splits
python -m src.preprocessing.splits --domain se

# Stage 6 — the two reference points every model must beat
python -m tools.run_baselines --domain se

# Stage 3 — decide stem / lemmatize / neither (writes the decision to disk)
python -m tools.run_ablation

# Stage 4 — TF-IDF vocabulary, compound-phrase and tuning analysis
python -m src.features.features_tfidf

# Stage 4 — GloVe vs self-trained Word2Vec (downloads GloVe once, ~128 MB)
python -m src.features.features_embed

# Stage 5 — train and tune all 15 models (~3 minutes)
python -m src.modeling.models_tfidf
python -m src.modeling.models_embed

# Stage 6 — spend the test split, once
python -m tools.run_final_evaluation

# Stage 6 — explainability and the stress test
python -m src.evaluation.explain_lime
python -m src.evaluation.explain_shap
python -m src.evaluation.stress_test

# regenerate the report's preprocessing rule table
python -m tools.make_examples

# tests
python -m pytest -q
```

Every script writes a `build_log.json` next to its output holding every count it
reports. **Those numbers go into the report — don't retype them by hand.**

---

## Repository layout

```
src/
  paths.py                  every path in the project, defined once
  acquisition/
    download.py             fetches the raw corpus
  extraction/
    build_se.py             raw .xlsx -> clean.csv, Fleiss' kappa
    clean_text.py           clean(), tokenize(), TOKEN_PATTERN
  preprocessing/
    splits.py               70/15/15 stratified, seed 42, frozen
    normalize.py            segmentation / stemming / lemmatization
  features/
    features_tfidf.py       unigram, (1,2), (1,3)
    features_embed.py       GloVe + self-trained Word2Vec
  modeling/
    models_tfidf.py         MNB, LR, LinearSVC
    models_embed.py         GNB, LR, LinearSVC
  evaluation/
    evaluate.py             metrics, confusion matrices, McNemar
    explain_lime.py         LIME: per-instance, aggregate, stability
    explain_shap.py         SHAP: exact linear + model-agnostic
    stress_test.py          the 60-sentence stress set

data/
  raw/senti4sd/             never edited; git-ignored
  processed/se/
    clean.csv               id;text;polarity  (4,331 rows)
    train.csv val.csv test.csv    frozen, seed 42
    build_log.json          filter counts, Fleiss' kappa, class distribution
  stress_test/se_stress.csv 60 hand-written sentences
  lexicons/se_jargon.txt    43 SE jargon words

results/
  metrics/                  one JSON per model + all_results.csv
  figures/                  confusion-matrix heatmaps, charts
  ablation/                 normalization comparison
  explanations/             LIME and SHAP output
models/se/                  saved .joblib pipelines; git-ignored
notebooks/demo.ipynb        interactive demo (+ demo.html, exported)
report/REPORT.md            the full write-up
report/preprocess_examples.md   generated; do not edit
tools/                      helper scripts, not part of the pipeline
tests/                      123 tests
```

---

## The corpus

`data/processed/se/clean.csv` — built by `src/extraction/build_se.py` from
`Senti4SD_GoldStandard_EmotionPolarity.xlsx`.

The workbook is used rather than Senti4SD's ready-made train/test CSVs because it
carries the `r1`/`r2`/`r3` rater columns, without which inter-rater agreement
cannot be computed.

| | |
|---|---|
| Rows loaded | 4,423 |
| Duplicates dropped | 92 |
| **Final** | **4,331** |
| Fleiss' κ | **0.7586** over 4,359 items, 3 raters |
| Mean length | 29.97 words |

| Class | Count | Share |
|---|---|---|
| negative | 1,176 | 27.15% |
| neutral | 1,662 | 38.37% |
| positive | 1,493 | 34.47% |

Splits: **3,031 train / 650 validation / 650 test**, stratified, seed 42, frozen.

Rater columns are free text with mixed case and the typos `Postive`, `Poitive`,
`Netural`; these are repaired before computing kappa, and items where any rater is
missing or unparseable are excluded.

---

## The vocabulary problem, in real data

40 of the 1,662 gold-**neutral** posts contain a word from the SE jargon lexicon:

- *i think that scope of 'killed' is ok.*
- *You don't know if threads are killed unless you catch a signal that tells you so!*
- *Try to add the complete failed message in the question!*
- *Is GLUT dead for graphics programming?*
- *Here, I see `BAADF00D` (bad food), `BEEFCACE` (beef cake), `BAADCAB1E` (bad cable), `BADCAFE` (bad cafe), and `DEADDEAD` (dead dead). Is this intentional?*

A general-purpose sentiment model sees *killed*, *failed*, *dead* and predicts
hostility. A developer sees ordinary technical description.

---

## Stage 2 — the clean-up contract

`src/extraction/clean_text.py` is the single clean-up path for every model. It has
no I/O and no pandas dependency, so it imports cleanly into tests, sklearn
pipelines, the notebook and the report generator.

```python
from src.extraction.clean_text import clean, tokenize, TOKEN_PATTERN

clean("Kill the process before restarting", "se")
clean("call foo(bar) and check a.b.c", "se")      # -> 'call CODE and check CODE'
```

Rules, in order: `html.unescape` → lowercase → URLs `URL` → `@mentions` `USER` →
domain rules → emoticons `EMO_POS`/`EMO_NEG` → negation expansion → collapse
repeated characters and punctuation → squeeze whitespace.

**Stopword removal is off, permanently.** Negation words appear in every standard
stopword list, and negation flips sentiment.

`TOKEN_PATTERN` is the single source of truth for tokenisation. It is passed to
`TfidfVectorizer(token_pattern=…)` in Stage 4 and
`LimeTextExplainer(split_expression=…)` in Stage 6, so the three cannot drift
apart. A test enforces that it matches `tokenize()`.

`report/preprocess_examples.md` is generated from `describe_rules()`, so the
report's rule table cannot drift from the code. **Regenerate it; never edit it by
hand.**

### Two deliberate order choices

1. **Lowercasing runs second, not fourth**, so placeholders stay uppercase. The
   literal words *url*, *user* and *code* are common in StackOverflow text, and
   Stage 4 confirms the risk was real — in the training split the placeholder
   `CODE` occurs in 174 posts while the ordinary English word *code* occurs in
   303. Lowercase placeholders would have merged them into one meaningless
   feature covering 477 posts.
2. **URLs are replaced before emoticons.** `http://` contains `:/`, a sad-face
   emoticon; the other order turns every URL into `httpEMO_NEG/...`.

---

## Stage 3 — preprocessing, and the stemming question

```python
from src.preprocessing.normalize import normalize, segment

normalize("the process was killed", mode="none")    # 'the process was killed'
normalize("the process was killed", mode="lemma")   # 'the process be kill'
normalize("the process was killed", mode="stem")    # 'the process wa kill'
```

Stemming and lemmatization collapse inflected forms:

| Original | Stemmed | Lemmatized |
|---|---|---|
| killed, kills, killing | `kill` | `kill` |

Normally this helps — fewer distinct tokens, denser counts on a small corpus.
**Here it may erase the effect being studied.** *"Killed the process"* is routine
technical usage; collapsing it into the emotional register of *kill* hides exactly
the confusion LIME is meant to expose.

So rather than asserting either position, the project **measures it**. The same
model (TF-IDF (1,2) + Logistic Regression) is trained three times and scored on
validation:

**Result** — `python -m tools.run_ablation`, scored on the validation split:

| Mode | Vocabulary | Accuracy | Macro-F1 | neutral→negative |
|---|---|---|---|---|
| **none** | 12,723 | 0.7938 | **0.7863** | 10.4% |
| lemma | 12,186 | 0.7831 | 0.7754 | 11.2% |
| stem | 12,371 | 0.7908 | 0.7844 | **10.0%** |

**Selected: `none`.** But the honest reading is that **normalization barely
matters on this corpus.** The gap between `none` and `stem` is 0.0019 macro-F1 —
about **2 documents out of 650**. That is noise, not evidence.

So the real justification for `none` is not that it scored highest. It is that
it keeps the inflections the project studies, and it is the simplest and fastest
option. The score merely fails to argue against it.

**One finding worth reporting:** the two metrics disagree. `none` wins on
macro-F1, but `stem` has the *lowest* neutral→negative rate (10.0% vs 10.4%) —
the project's actual failure metric. Accuracy and interpretability point in
different directions, exactly the trade-off this project is about.

The decision is written to `results/ablation/chosen_mode.json` and read by every
later stage via `load_chosen_mode()`, so the mode used downstream is
demonstrably the one this experiment produced rather than a hard-coded guess.

**Three implementation details that matter:**

- **Lemmatization is POS-aware.** `WordNetLemmatizer` assumes every word is a
  noun unless told otherwise, so `lemmatize("stopped")` returns `stopped`
  unchanged while `lemmatize("stopped", "v")` returns `stop`. Whole sentences
  are tagged at once, because a tagger needs surrounding words to tell a verb
  from a noun. A test enforces this.
- **Placeholders are never normalized.** `URL`, `CODE`, `EMO_POS` and the rest
  pass through untouched — stemming `CODE` to `code` would merge it with the
  ordinary English noun and corrupt the vocabulary.
- **All three modes share one tokenizer**, so the only difference between them
  is the word-normalization step. A test asserts they produce identical token
  counts; otherwise the ablation would be comparing two things at once.

Sentence segmentation (`segment()`) reports that SE posts average **2.33
sentences** (7,060 sentences across 3,031 training documents, longest 11). It is
a measurement tool for the report — it does not alter the text fed to models.

Run `python -m src.preprocessing.normalize` to see all of this demonstrated.

---

## Stage 4 — TF-IDF features

`python -m src.features.features_tfidf`

| Config | n-grams | Vocabulary | Sparsity | Features/doc |
|---|---|---|---|---|
| `tfidf11` | (1,1) | 3,449 | 0.9931 | 23.7 |
| `tfidf12` | (1,2) | 12,723 | 0.9968 | 40.7 |
| `tfidf13` | (1,3) | 17,902 | 0.9974 | 46.7 |

Fitted on the 3,031 training documents only.

### Most distinctive terms per class

Ranking by mean TF-IDF returns `i, the, is, to` for every class — common words
carry weight everywhere. Each term is instead scored by how much heavier it is
*inside* a class than outside it, which makes the table informative:

| Class | Top terms |
|---|---|
| negative | `EMO_NEG, sad, i, hate, horrible, afraid, not, terrible` |
| neutral | `?, URL, you, use, how, what, see, the` |
| positive | `!, excellent, thanks, great, EMO_POS, excellent !, thanks !, awesome` |

Neutral is dominated by **question words** — most neutral StackOverflow posts are
questions. Positive picks up the bigrams `excellent !` and `thanks !`, which the
unigram model cannot represent.

### ⚠️ The compound-phrase finding

The project proposal motivates n-grams with *fatal error*, *kill the process* and
*null pointer exception*. **None of those phrases occurs even once in the
Senti4SD corpus:**

| Phrase | In (1,1) | (1,2) | (1,3) | neg | neu | pos |
|---|---|---|---|---|---|---|
| fatal error | — | — | — | — | — | — |
| null pointer | — | — | — | — | — | — |
| kill the process | — | — | — | — | — | — |
| null pointer exception | — | — | — | — | — | — |
| **does not** | — | ✓ | ✓ | 46 | 52 | 26 |
| **not work** | — | ✓ | ✓ | 12 | 17 | 5 |
| **thank you** | — | ✓ | ✓ | 0 | 0 | 54 |
| **does not work** | — | — | ✓ | 5 | 11 | 5 |
| works fine | — | ✓ | ✓ | 1 | 2 | 7 |

9 of 14 tested phrases reached the (1,3) vocabulary. **The five that did not are
exactly the canonical SE-jargon examples.**

Two consequences, both of which belong in the report:

1. **The case for n-grams cannot rest on technical jargon here.** What bigrams
   actually capture in this corpus is *negation* (`does not`, `not work`) and
   *politeness* (`thank you`, 54 positive posts and 0 elsewhere).
2. **The Stage 6 stress test becomes load-bearing, not supplementary.** It is the
   only place the canonical phrases can be tested at all, because the corpus does
   not contain them.

### Tuning, and why the top scorer is not shipped

`min_df` ∈ {1, 2, 5} × `max_features` ∈ {None, 20000, 50000}, Logistic Regression
held at defaults, scored on validation.

The highest score was `tfidf13` with `min_df=1` at **0.7960** macro-F1 — using
**130,078 features for 3,031 documents** (43 per document) with a train/validation
gap of **0.196**. That is memorization, and its lead over the default setting is
+0.0129, about **8 of 650 validation documents**.

One standard error on a 650-document validation set is ≈0.016 macro-F1. So the
selection rule is stated explicitly in the code:

> among settings within one standard error of the best validation score, take the
> smallest vocabulary

This is the standard one-standard-error rule. It ships `min_df=5` for `tfidf11`
and `tfidf12`, and `min_df=2` for `tfidf13` — smaller models, statistically
indistinguishable scores, and explanations built from terms that appear in more
than one post.

**Every configuration has a train/validation gap of 0.13–0.20.** That is expected
for a linear model on 3,031 short documents, and it is why Stage 5 tunes
regularization (`C`) rather than trusting these defaults.

---

## Stage 4.2 — GloVe vs self-trained Word2Vec

`python -m src.features.features_embed`

| Embedding | Vocabulary | OOV tokens | Empty docs | Placeholders known |
|---|---|---|---|---|
| GloVe (pretrained) | 400,000 | 2.4% | 9 | none |
| Word2Vec (self-trained) | 3,752 | 3.9% | 0 | `CODE, URL, USER, EMO_POS, EMO_NEG` |

GloVe is lowercase general English, so it has **no vector for the placeholders**
Stage 2 inserts — every code span in a post is invisible to it. Word2Vec learned
them, because it was trained on the placeholder-bearing text.

### The nearest-neighbour table

The intended headline was: the same word, two different meanings, depending on
which corpus taught it. **Half of that worked.**

| Word | Times in training text | GloVe neighbours | Word2Vec neighbours |
|---|---|---|---|
| `kill` | **3** | killing, kills, destroy, shoot, attack, **poison** | activities, clarify, song, chain, exec |
| `crash` | 9 | accident, crashes, plane, **collision, airplane** | bus, modifying, press, broadcast, gc |
| `error` | 65 | errors, mistake, incorrect, fault | occurs, assertion, failed, failure, throws |
| `exception` | 12 | exceptions, except, instance, example | damn, typical, weird, onchange, scared |
| `fatal` | **0** | deadly, accident, deaths, **poisoning** | *out of vocabulary* |
| `hang` | **0** | hung, kong, hong, indices | *out of vocabulary* |
| `abort` | **0** | aborted, takeoff, **fetuses**, eject | *out of vocabulary* |

**GloVe delivers the argument completely.** It places `kill` next to *shoot* and
*poison*, `abort` next to *fetuses*, `crash` next to *airplane*. A general-English
model reading a StackOverflow post genuinely does see violence and aviation
disasters. That is the domain-shift claim, demonstrated rather than asserted.

**Word2Vec does not, and the frequency column says exactly why.** `kill` appears
**3 times** in 3,031 documents; `fatal`, `hang` and `abort` appear **zero times**.
Word2Vec needs hundreds of occurrences to place a word well. This is not a tuning
problem — it is the corpus-size limitation the proposal already lists under
Limitations, now measured instead of predicted.

Only `error` (65 occurrences) produces sensible domain neighbours — *assertion,
failed, failure, throws* — which is itself the proof: given enough examples, the
self-trained model does learn the technical sense.

### This confirms the Stage 4.1 finding

Two independent analyses now say the same thing:

| Stage | Finding |
|---|---|
| 4.1 | `fatal error`, `kill the process`, `null pointer exception` — **0 occurrences** |
| 4.2 | `fatal` 0×, `hang` 0×, `abort` 0×, `kill` 3× |

**The harsh SE vocabulary the proposal is built around is essentially absent from
Senti4SD.** The corpus is StackOverflow *discussion*, not error output. This is
the single most important thing to state in the report, and it makes the Stage 6
stress test the only place the central claim can be tested directly.

### Word2Vec settings

`vector_size` ∈ {50, 100} × CBOW vs skip-gram, judged by downstream validation
macro-F1 rather than by how the neighbour lists read:

| vector_size | Algorithm | Val macro-F1 |
|---|---|---|
| 50 | CBOW | 0.6751 |
| 50 | skip-gram | 0.6825 |
| **100** | **CBOW** | **0.7043** |
| 100 | skip-gram | 0.7042 |

Dimension matters more than the algorithm; CBOW and skip-gram are tied at 100d.
Total spread 0.029.

---

## Stage 5 — the nine TF-IDF models

`python -m src.modeling.models_tfidf` — tuned by 5-fold cross-validation inside
the training split, compared on validation. **The test split is untouched.**

| Model | Kind | CV F1 | Val F1 | neut→neg | neg recall |
|---|---|---|---|---|---|
| **tfidf13_svm** | discriminative | 0.8033 | **0.8058** | 14.1% | 71.8% |
| tfidf13_lr | discriminative | 0.8007 | 0.8007 | 14.5% | 70.6% |
| tfidf12_svm | discriminative | 0.8016 | 0.7993 | 14.5% | 69.5% |
| tfidf11_lr | discriminative | 0.8074 | 0.7978 | 14.1% | 72.3% |
| tfidf12_lr | discriminative | 0.8033 | 0.7939 | 16.1% | 73.5% |
| tfidf11_svm | discriminative | 0.8038 | 0.7929 | 13.2% | 66.7% |
| tfidf12_mnb | generative | 0.7570 | 0.7466 | 12.8% | 58.2% |
| tfidf13_mnb | generative | 0.7473 | 0.7305 | 14.9% | 58.8% |
| tfidf11_mnb | generative | 0.7326 | 0.7237 | 9.2% | 53.1% |

**Best: `tfidf13_svm`** at 0.8058 (`C=1`, `class_weight="balanced"`) — **+0.098
macro-F1 over the VADER baseline.**

### Proposal Outcome #2 — generative vs discriminative

| | Mean val macro-F1 | Best |
|---|---|---|
| Discriminative (LR, SVM) | **0.7984** | 0.8058 |
| Generative (Naive Bayes) | 0.7336 | 0.7466 |

A gap of **0.065** — far beyond the ~0.016 noise floor, and consistent across
all three feature sets. This one is answered cleanly.

### Outcome #1 — the n-gram range barely matters

`tfidf11_lr` (0.7978) versus `tfidf13_lr` (0.8007) is a difference of ~2
validation documents. That is consistent with the Stage 4 finding: the technical
compound phrases the n-grams were meant to capture **are not in this corpus**, so
widening the window buys almost nothing.

### ⚠️ A trap in the failure metric

`tfidf11_mnb` has the *lowest* neutral→negative rate at 9.2% — which looks like
the least jargon-biased model. It is not. It finds only **53% of genuinely
negative posts**, against 72% for the best model.

Its low rate is not restraint about jargon; it is reluctance to predict
"negative" at all. A model that never says negative would score a perfect 0% on
that metric and be worthless. **The two columns must always be reported
together**, which is why `val_negative_recall` sits beside it in the table.

### Sanity check on the saved model

```
Kill the process before restarting the server.   → neutral  (0.85)
How do I use this function?                      → neutral  (0.95)
Thanks, this works great!                        → positive (0.99)
This library is broken and the docs are useless. → negative (0.75)
The build failed with a fatal error on line 42.  → negative (0.93)
```

The trained model handles *"kill the process"* correctly where VADER does not.
The last line is the interesting one — a factual bug report read as hostility.
That is the failure Stage 6 exists to explain.

---

## Final results

**Full write-up: [report/REPORT.md](report/REPORT.md)** · **Demo: [notebooks/demo.ipynb](notebooks/demo.ipynb)**

### Test set — 15 models, read once

| Model | Representation | Kind | Test macro-F1 | neut→neg |
|---|---|---|---|---|
| **tfidf13_svm** | TF-IDF | discriminative | **0.8275** | 14.4% |
| tfidf13_lr | TF-IDF | discriminative | 0.8233 | 14.8% |
| tfidf12_lr | TF-IDF | discriminative | 0.8108 | 14.0% |
| tfidf12_mnb | TF-IDF | generative | 0.7601 | 12.0% |
| w2v_svm | Word2Vec | discriminative | 0.7059 | 11.6% |
| glove_lr | GloVe | discriminative | 0.7054 | 17.2% |
| glove_gnb | GloVe | generative | 0.5668 | 39.2% |
| *VADER baseline* | — | — | *0.7079* | *25.2%* |
| *majority baseline* | — | — | *0.1852* | — |

**Best: `tfidf13_svm`, 0.8275 — +0.12 macro-F1 over VADER.**

### The three proposal outcomes

**#1 — which representation?** TF-IDF, decisively: best 0.8275 vs Word2Vec 0.7059 and
GloVe 0.7054. The n-gram range barely matters, which follows from the corpus finding below.

**#2 — generative vs discriminative?** Discriminative wins: **0.7694** mean vs **0.6792**,
consistent across all five representations, McNemar *p* = 4.7e-9.

**#3 — domain cues or general-language bias?** The bias is real and its size tracks how
much general English the representation carries:

| Representation | False-alarm rate on harsh-but-neutral sentences |
|---|---|
| VADER (general lexicon) | **70%** |
| GloVe (general English) | **40%** |
| Word2Vec (this corpus) | **28%** |
| TF-IDF (this corpus) | **14%** |

### ⚠️ The finding that shaped the project

**Senti4SD contains almost none of the vocabulary this project is about.** Four independent
analyses agree:

| Stage | Evidence |
|---|---|
| 4.1 | `fatal error`, `kill the process`, `null pointer exception` — **0 occurrences** |
| 4.2 | `fatal` 0×, `hang` 0×, `abort` 0×, `kill` 3× in 3,031 posts |
| 6.5 | LIME: **no** lexicon word appears ≥3× in the neutral sample |
| 6.6 | SHAP: the Domain Bias Score cannot be computed for the same reason |

The corpus is StackOverflow *discussion*, not error output. So the central claim is
demonstrated on a purpose-built 60-sentence stress set instead — where it holds clearly.

### What LIME and SHAP found

Both agree (Spearman **0.872** over 80 shared tokens) that the model keys on **first-person
and affective language** — `afraid, hate, painful, extremely, me` — not on technical
vocabulary. LIME stability across five seeds: mean Jaccard **0.907**.

### The headline sentence

> VADER calls **70%** of harsh-but-neutral SE sentences negative.
> The best trained model calls **7.5%**, while still catching **75%** of real complaints.

---

## Out of scope

Deliberately excluded, recorded so the reasoning is visible:

| Excluded | Reason |
|---|---|
| Second domain (health / Druglib) | Not in the proposal; needs ~835 hand-labelled rows. Partial code and data remain in the repo but are unused. |
| Cross-domain transfer | Depends on the second domain. |
| BERT / transformers | Not in the proposal, which names three representation families. Recorded as Future Work. |
| McNemar tests, LIME stability study | Beyond the proposal's commitments. |
| CLI demo | The proposal says "Notebook/CLI" — one satisfies it. |

---

## Known issues

- **Word2Vec quality is limited by corpus size** (3,031 training documents). The
  proposal already lists this under Limitations. The nearest-neighbour table is
  reported regardless — a weak domain model beating GloVe on `kill` is the point.
- **Trigrams may be too sparse to help** at this corpus size. That is a finding,
  not a failure.
- **The stress set will be written by one person**, so it reflects one person's
  intuitions about what counts as harsh-but-neutral. Stated in Limitations.
- **LIME and SHAP are approximations** with known stability limits, as the
  proposal acknowledges.

---

## Licence and citation

Senti4SD: Calefato, F., Lanubile, F., Maiorano, F., & Novielli, N. (2018).
Sentiment Polarity Detection for Software Development. *Empirical Software
Engineering*, 23(3), 1352–1382.

The repository also contains an unused partial corpus from the UCI Drug Reviews
(Druglib.com, id 461) dataset, whose donors require research-only use, no
redistribution, and citation. Raw files are git-ignored.

> Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-Based
> Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.
> *Proceedings of the 2018 International Conference on Digital Health*, 121–125.

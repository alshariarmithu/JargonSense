# Project Plan — Interpretable Sentiment Classification in Software Engineering Communication

**Course:** CSE 4122
**Al Shariar Hossain** (Roll 2107066) · **Hassan Mohammed Naquibul Hoque** (Roll 2107077)

This plan tracks the project described in `project-proposal-nlp.pdf`. It is written
as a **single sequential task list**, because the implementation is being carried
out by one person rather than split between two.

---

## 1. What the proposal promised, and where it is delivered

This is the table to read first. Every commitment in the proposal maps to a stage
of the pipeline and a file in the repository.

| Proposal commitment | Stage | Delivered in | Status |
|---|---|---|---|
| Senti4SD, ~4,000 StackOverflow posts | 1–2 | `src/extraction/build_se.py` | ✅ 4,331 rows |
| TF-IDF with N-grams (1,2)/(1,3) | 4 | `src/features/features_tfidf.py` | ✅ |
| Pretrained (GloVe) vs self-trained (Word2Vec) | 4 | `src/features/features_embed.py` | ✅ |
| Naive Bayes (generative) vs LR/SVM (discriminative) | 5 | `src/modeling/` | ✅ |
| Evaluated across all feature sets | 5–6 | `results/metrics/master_results_se.csv` | ✅ |
| LIME/SHAP per-instance | 6 | `src/evaluation/explain_lime.py`, `explain_shap.py` | ✅ 40 HTML |
| LIME/SHAP aggregated across the test set | 6 | same | ✅ Spearman 0.872 |
| Hand-built stress-test set | 6 | `data/stress_test/se_stress.csv` | ✅ 60 sentences |
| P/R/F1 per class, confusion matrix | 6 | `src/evaluation/evaluate.py` | ✅ |
| Focus on negative-vs-neutral confusion | 6 | `se_neutral_to_negative_rate` | ✅ |
| Notebook/CLI demo with highlighted words | 6 | `notebooks/demo.ipynb` | ✅ + demo.html |

### Expected Outcomes → evidence

| # | Proposal outcome | Evidence that answers it |
|---|---|---|
| 1 | Which representation best handles SE vocabulary | Master results table + the per-representation comparison in Stage 6 |
| 2 | Generative vs discriminative comparison | NB vs LR/SVM across all 5 feature sets |
| 3 | Do models use domain-appropriate cues or general-language bias? | LIME/SHAP jargon-lexicon table + the stress-test false-alarm rate |
| 4 | A working, interpretable demo | `notebooks/demo.ipynb` |

---

## 2. The pipeline

Six stages, applied to one corpus.

```
Stage 1  Data acquisition
Stage 2  Text extraction & clean-up
Stage 3  Preprocessing        (sentence segmentation, stemming, lemmatization)
Stage 4  Feature engineering
Stage 5  Model building
Stage 6  Evaluation           (metrics, explainability, stress test, demo)
```

Each stage produces a file that the next stage consumes, so the project can be
re-run end to end and every number in the report traced back to a command.

---

## 3. The model grid — 15 models

| Representation | What it does | × Classifiers |
|---|---|---|
| `tfidf11` unigram | Counts single words. No context. The baseline. | MNB, LR, SVM |
| `tfidf12` (1,2) | Adds word pairs — catches `fatal error`, `no more` | MNB, LR, SVM |
| `tfidf13` (1,3) | Adds triples — catches `kill the process` | MNB, LR, SVM |
| GloVe | Pretrained word meanings from general English | GNB, LR, SVM |
| Word2Vec | Word meanings learned from the SE corpus itself | GNB, LR, SVM |

**9 TF-IDF + 6 embedding = 15 models**, plus 2 baselines (majority, VADER).

Three classifiers because the proposal commits to *"Naive Bayes (generative) vs.
Logistic Regression/SVM (discriminative)"*:

- **Naive Bayes** — generative: models what a negative post looks like.
- **Logistic Regression** — discriminative: models the boundary between classes.
- **Linear SVM** — margin-based: draws the widest separating gap.

Two implementation constraints worth stating in the report:

1. **`LinearSVC` has no `predict_proba`.** It is wrapped in
   `CalibratedClassifierCV(cv=5)` because LIME, SHAP and the demo all need
   probabilities.
2. **`MultinomialNB` cannot be used on embeddings** — GloVe and Word2Vec vectors
   contain negative values, and MNB requires non-negative counts. `GaussianNB` is
   used instead. This is a real constraint on generative models with dense
   representations, not a workaround to hide.

**Only 2 models get deep analysis:** the best TF-IDF model and the best embedding
model. LIME, SHAP, error analysis, the stress test and the demo all run on those
two. The other 13 are rows in the results table.

---

## 4. Contracts (fixed — changing these invalidates earlier results)

- `clean(text, domain) -> str` — `src/extraction/clean_text.py`
- `normalize(text, mode="none") -> str` and `segment(text) -> list[str]` — `src/preprocessing/normalize.py`
- `load_splits(domain) -> (train_df, val_df, test_df)` — `src/preprocessing/splits.py`
- `evaluate(name, domain, y_true, y_pred) -> dict` — `src/evaluation/evaluate.py`
- **Every saved model is a full pipeline** taking raw strings, exposing
  `predict()` and `predict_proba()`. Required by LIME, SHAP and the demo.
- Label order everywhere: `["negative", "neutral", "positive"]`
- CSV format: `id;text;polarity`, semicolon-delimited, UTF-8 no BOM
- Random seed: `42`
- Model files: `models/se/{features}_{classifier}.joblib`
- `TOKEN_PATTERN` is the one tokenisation rule, passed to both
  `TfidfVectorizer(token_pattern=…)` and `LimeTextExplainer(split_expression=…)`
  so the two cannot drift apart. A test enforces it.
- **Paths come from `src/paths.py`.** No module computes its own.

**The test split is spent exactly once**, in Stage 6. All tuning uses validation.

---

## 5. Stages in detail

### Stage 1 — Data acquisition ✅ done

Senti4SD gold standard, `Senti4SD_GoldStandard_EmotionPolarity.xlsx`, copied
unedited to `data/raw/senti4sd/`.

The workbook is used rather than Senti4SD's ready-made train/test CSVs because it
carries the `r1`/`r2`/`r3` rater columns, without which inter-rater agreement
cannot be computed.

**Deliverable:** `src/acquisition/download.py`

### Stage 2 — Text extraction & clean-up ✅ done

**2.1 Corpus extraction** — `src/extraction/build_se.py`

Keep `text` and `final`; standardise to `id`, `text`, `polarity`; drop duplicates
and empties; compute Fleiss' kappa; record the class distribution.

| | |
|---|---|
| Rows loaded | 4,423 |
| Duplicates dropped | 92 |
| **Final** | **4,331** |
| Fleiss' κ | **0.7586** over 4,359 items, 3 raters |
| Mean length | 29.97 words |

Rater columns are free text with mixed case and the typos `Postive`, `Poitive`,
`Netural`, repaired before scoring.

**2.2 String clean-up** — `src/extraction/clean_text.py`

`clean(text, domain)`. No I/O, no pandas, so it imports cleanly into tests,
sklearn pipelines and the notebook.

Rules in order: `html.unescape` → lowercase → URLs `URL` → `@mentions` `USER` →
domain rules → emoticons `EMO_POS`/`EMO_NEG` → negation expansion → collapse
repeated characters and punctuation → squeeze whitespace.

SE domain rule: code tags, backtick spans, `foo(bar)` and `a.b.c` → `CODE`.

**Two deliberate order choices, both documented in the report:**

1. **Lowercasing runs second, not fourth**, so placeholders stay uppercase. The
   literal words *url* and *user* are common in StackOverflow text; lowercase
   placeholders would be indistinguishable and would corrupt the vocabulary.
2. **URLs are replaced before emoticons.** `http://` contains `:/`, a sad-face
   emoticon; the other order turns every URL into `httpEMO_NEG/...`.

**Stopword removal is off permanently.** Negation words appear in every standard
stopword list, and negation flips sentiment.

**Deliverable:** 36 tests, plus `report/preprocess_examples.md` generated from
`describe_rules()` so the report's table cannot drift from the code.

### Stage 3 — Preprocessing

**3.1 `normalize.py`** ✅ done — 23 tests

```python
normalize(text, mode="none")      # mode: "none" | "lemma" | "stem"
segment(text) -> list[str]        # sentence splitting, for report statistics
```

- **Segmentation** — NLTK `punkt`, `segment(text) -> list[str]`. Used for
  per-sentence statistics in the report; does not alter text fed to models.
- **Stemming** — `PorterStemmer`, with `SnowballStemmer` as the alternate.
- **Lemmatization** — `WordNetLemmatizer` **with POS tags** from `nltk.pos_tag`.
  Without POS tags `stopped` lemmatizes to `stopped`, not `stop`, and the
  comparison below is meaningless.

**Why this stage is a measurement, not a formality.** Stemming collapses
`killed`/`kills`/`killing` to `kill`. Normally that helps — fewer distinct tokens,
denser counts on a small corpus. Here it may erase the effect being studied:
*"killed the process"* is routine technical usage, and collapsing it into the
emotional register of *kill* hides exactly the confusion LIME is meant to expose.

**3.2 Normalization ablation** ✅ done

Rather than assert either position, measure it. Same model (TF-IDF (1,2) + LR,
defaults), trained three times, scored on **validation**:

| Mode | Vocabulary | Accuracy | Macro-F1 | neutral→negative |
|---|---|---|---|---|
| **none** | 12,723 | 0.7938 | **0.7863** | 10.4% |
| lemma | 12,186 | 0.7831 | 0.7754 | 11.2% |
| stem | 12,371 | 0.7908 | 0.7844 | **10.0%** |

**Selected `none`** — but the margin over `stem` is 0.0019 macro-F1, roughly
**2 of 650 validation documents**. The defensible conclusion is that
normalization does not matter much here; `none` is preferred because it keeps
the inflections the project studies and is simplest, not because it won.

**The metrics disagree**, and that is a report finding: `none` wins on macro-F1
while `stem` has the lowest neutral→negative rate. Accuracy and the project's
failure metric point in opposite directions.

Sentence segmentation reports a mean of **2.33 sentences per post** (7,060
sentences over 3,031 documents, longest 11).

→ `results/ablation/normalization_se.csv`

The winner by validation macro-F1 becomes the project default for every later
model. **Whatever wins, the table goes in the report**: if stemming raises macro-F1
but also raises the neutral→negative rate, that is the accuracy/interpretability
trade-off this project is about, visible in one table.

**3.3 Splits** ✅ done — `src/preprocessing/splits.py`

70/15/15 stratified, `random_state=42` → 3,031 / 650 / 650. Frozen.

Splits are frozen before Stage 4 because every vectorizer, embedding and scaler is
fit on **train only**. Fitting on anything else leaks the test set.

### Stage 4 — Feature engineering ✅ done

**4.1 TF-IDF** ✅ done — 17 tests — `src/features/features_tfidf.py`

Three configs, each with `preprocessor=partial(clean, domain="se")` plus the
Stage-3 winning mode. Shared: `min_df=2`, `max_df=0.95`, `sublinear_tf=True`,
`token_pattern=TOKEN_PATTERN`. **Fit on train only.**

**Results.** Vocabulary at the shared defaults (`min_df=2`, fitted on train only):

| Config | n-grams | Vocabulary | Sparsity | Features/doc |
|---|---|---|---|---|
| `tfidf11` | (1,1) | 3,449 | 0.9931 | 23.7 |
| `tfidf12` | (1,2) | 12,723 | 0.9968 | 40.7 |
| `tfidf13` | (1,3) | 17,902 | 0.9974 | 46.7 |

**Top terms are ranked by distinctiveness, not raw mean TF-IDF**, which returns
`i, the, is, to` for every class. Distinctiveness = mean weight inside the class
minus mean weight outside it. Result: negative is `EMO_NEG, sad, hate, horrible,
terrible`; neutral is question words `?, URL, how, what, use`; positive is
`!, excellent, thanks, great` plus the bigrams `excellent !` and `thanks !`.

**⚠️ The compound-phrase finding.** The proposal motivates n-grams with *fatal
error*, *kill the process* and *null pointer exception*. **None of these occurs
even once in the corpus.** 9 of 14 tested phrases reached the (1,3) vocabulary,
and the 5 that did not are exactly the canonical jargon examples. What n-grams
actually capture here is negation (`does not` 124 posts, `not work` 34) and
politeness (`thank you`, 54 positive and 0 elsewhere).

Two consequences for the report:

1. The case for n-grams cannot rest on technical jargon in this corpus.
2. **Stage 6.7's stress test becomes load-bearing**, not supplementary — it is
   the only place the canonical phrases can be tested at all.

**Tuning and the selection rule.** `min_df` ∈ {1, 2, 5} × `max_features` ∈
{None, 20000, 50000}, LR at defaults, scored on validation. The top scorer was
`tfidf13` / `min_df=1` at 0.7960 — but with 130,078 features for 3,031 documents
and a train/validation gap of 0.196. Its lead is +0.0129, about 8 of 650
documents, while one standard error is ≈0.016.

The code therefore states its rule up front:

> among settings within one standard error of the best validation score, take
> the smallest vocabulary

the standard one-standard-error rule. Shipped: `min_df=5` for `tfidf11` and
`tfidf12`, `min_df=2` for `tfidf13`, written to
`results/features/tfidf_settings.json` and read by Stage 5.

**Every configuration overfits** (train/validation gap 0.13–0.20), which is why
Stage 5 tunes `C` rather than trusting these defaults.

**4.2 Embeddings** ✅ done — 16 tests — `src/features/features_embed.py`

- **GloVe** `glove.6B.100d` via gensim `KeyedVectors`. Document vector = mean of
  in-vocabulary token vectors; zero vector if none. Report the **OOV rate**
  overall and for the SE jargon lexicon and code tokens specifically.
- **Word2Vec** trained on the **training split only**: `vector_size=100, window=5,
  min_count=2, sg=1, negative=5, epochs=30, seed=42, workers=1`. Try
  `vector_size` ∈ {50, 100}, `sg` ∈ {0, 1}, pick by validation macro-F1.

**Results.**

| Embedding | Vocabulary | OOV tokens | Empty docs | Placeholders known |
|---|---|---|---|---|
| GloVe | 400,000 | 2.4% | 9 | none |
| Word2Vec | 3,752 | 3.9% | 0 | `CODE, URL, USER, EMO_POS, EMO_NEG` |

GloVe has no vector for the uppercase placeholders Stage 2 inserts, so every
code span is invisible to it. Word2Vec learned them.

**The nearest-neighbour table — half the intended result, plus a surprise.**

| Word | Times in train | GloVe | Word2Vec |
|---|---|---|---|
| `kill` | 3 | killing, destroy, shoot, attack, **poison** | activities, clarify, song, exec |
| `crash` | 9 | accident, plane, **collision, airplane** | bus, modifying, press, gc |
| `error` | 65 | errors, mistake, incorrect, fault | occurs, **assertion, failed, throws** |
| `exception` | 12 | exceptions, except, instance | damn, typical, weird, scared |
| `fatal` | 0 | deadly, deaths, **poisoning** | *out of vocabulary* |
| `hang` | 0 | hung, kong, hong, indices | *out of vocabulary* |
| `abort` | 0 | aborted, takeoff, **fetuses** | *out of vocabulary* |

**GloVe proves the domain-shift claim outright** — it places `kill` beside
*shoot* and *poison*, `abort` beside *fetuses*, `crash` beside *airplane*. A
general-English model reading StackOverflow genuinely sees violence and
aviation disasters. This half belongs on a slide.

**Word2Vec fails, and the frequency column explains why.** `kill` occurs 3
times in 3,031 documents; `fatal`, `hang` and `abort` occur zero times. This is
the corpus-size limitation the proposal already anticipated, now measured
rather than predicted. Only `error` (65×) yields sensible domain neighbours —
which proves the mechanism works given enough data.

**⚠️ This corroborates Stage 4.1.** Two independent analyses agree: the harsh SE
vocabulary the proposal is built around is essentially absent from Senti4SD.
The corpus is StackOverflow *discussion*, not error output. State this
prominently — and note that it makes the Stage 6.7 stress test the only place
the central claim can be tested directly.

**Settings search** (`vector_size` × CBOW/skip-gram, scored by downstream
validation macro-F1): 100d CBOW wins at 0.7043, 100d skip-gram ties at 0.7042,
50d loses ~0.03. Dimension matters more than the algorithm.

Wrapped as a sklearn transformer, `fit`/`transform` taking raw strings, so a
saved model is a complete pipeline for LIME, SHAP and the demo.

### Stage 5 — Model building — TF-IDF half ✅ done (19 tests)

`src/modeling/models_tfidf.py` (9) and `src/modeling/models_embed.py` (6).

Tuning by `GridSearchCV`, stratified 5-fold, `scoring="f1_macro"`:

- MNB: `alpha` ∈ {0.01, 0.1, 0.5, 1.0}
- GNB: `var_smoothing` ∈ {1e-9, 1e-7, 1e-5, 1e-3}
- LR: `C` ∈ {0.01, 0.1, 1, 10, 100}, `class_weight` ∈ {None, balanced}
- SVM: `C` ∈ {0.01, 0.1, 1, 10}, `class_weight` ∈ {None, balanced}

Saved as `models/se/{features}_{classifier}.joblib`.

**Results** (tuned on train by 5-fold CV, compared on validation; test untouched):

| Model | Kind | CV F1 | Val F1 | neut→neg | neg recall | Best params |
|---|---|---|---|---|---|---|
| **tfidf13_svm** | discriminative | 0.8033 | **0.8058** | 14.1% | 71.8% | C=1, balanced |
| tfidf13_lr | discriminative | 0.8007 | 0.8007 | 14.5% | 70.6% | C=10, balanced |
| tfidf12_svm | discriminative | 0.8016 | 0.7993 | 14.5% | 69.5% | C=1, None |
| tfidf11_lr | discriminative | 0.8074 | 0.7978 | 14.1% | 72.3% | C=1, balanced |
| tfidf12_lr | discriminative | 0.8033 | 0.7939 | 16.1% | 73.5% | C=1, balanced |
| tfidf11_svm | discriminative | 0.8038 | 0.7929 | 13.2% | 66.7% | C=0.1, balanced |
| tfidf12_mnb | generative | 0.7570 | 0.7466 | 12.8% | 58.2% | alpha=0.1 |
| tfidf13_mnb | generative | 0.7473 | 0.7305 | 14.9% | 58.8% | alpha=0.1 |
| tfidf11_mnb | generative | 0.7326 | 0.7237 | 9.2% | 53.1% | alpha=0.5 |

**Proposal Outcome #2 answered.** Discriminative models average 0.7984 against
0.7336 for the generative one — a 0.065 gap, far above the ~0.016 noise floor,
consistent across all three feature sets.

**Proposal Outcome #1, partial.** The n-gram range barely matters:
`tfidf11_lr` 0.7978 vs `tfidf13_lr` 0.8007, about 2 validation documents. This
follows directly from the Stage 4 finding that the technical compound phrases
are not in the corpus.

**Best model beats VADER by +0.098 macro-F1.**

⚠️ **A trap worth writing up.** `tfidf11_mnb` has the lowest neutral→negative
rate (9.2%) but finds only 53% of genuinely negative posts, against 72% for the
best model. Its low rate is reluctance to predict "negative" at all, not
restraint about jargon — a model that never predicts negative would score a
perfect 0%. `val_negative_recall` is reported beside it so the pair cannot be
misread.

### Stage 6 — Evaluation ✅ done

**6.1 Framework** ✅ done — `src/evaluation/evaluate.py`

Accuracy, macro-F1, weighted-F1, per-class P/R/F1, confusion matrix (counts and
row-normalised), heatmap PNG, JSON per model, and one row per model in
`results/metrics/all_results.csv`. Re-running a model replaces its row rather than
duplicating it.

**`se_neutral_to_negative_rate`** — the fraction of gold-neutral posts predicted
negative — is the project's headline metric, tracked for every model from the
start.

**6.2 Baselines** ✅ done

| Model | Accuracy | Macro-F1 | neutral→negative |
|---|---|---|---|
| majority | 0.385 | 0.185 | — |
| VADER | 0.715 | 0.708 | **25.2%** |

VADER — a general-purpose lexicon that has never seen a StackOverflow post — calls
**one in four** genuinely neutral posts negative. That is the problem statement,
measured, before any model is built.

**6.3 Test evaluation and results table** — 5 h

Retrain best configs, evaluate **once** on test. Master table sorted by macro-F1.
Three charts: macro-F1 by representation; generative vs discriminative average;
baseline vs best.

**6.4 Error analysis** — included above

Collect test cases where gold = neutral, predicted = negative. Tag each: *contains
SE jargon* / *sarcasm* / *annotation noise* / *other*. Counts plus 5 examples. For
LR, list the top 20 features pushing toward negative and flag jargon words.

**6.5 LIME** — 10 h — `src/evaluation/explain_lime.py`

Per-instance on **20 chosen posts** (5 correct negatives, 5 correct neutrals,
5 characteristic errors, 5 jargon-heavy), with the best TF-IDF and best embedding
model. HTML to `results/explanations/lime/`.

Aggregated across the test set: per token, total weight toward negative,
occurrences, mean weight.

**SE jargon lexicon** — `data/lexicons/se_jargon.txt`: `fatal, error, kill, killed,
crash, crashed, abort, exception, fail, failed, failure, dead, deadlock, bug,
broken, hang, panic, terminate, execute, garbage, deprecated, warning, exploit,
attack, destroy`.

For each lexicon word, the mean LIME weight toward negative **on gold-neutral
posts**. A strong positive value means the model reads technical vocabulary as
hostility. **This table is the direct answer to proposal Expected Outcome #3.**
Bar chart of the top 25 negative-driving tokens, lexicon words coloured separately.

**6.6 SHAP** — 8 h — `src/evaluation/explain_shap.py`

Same 20 instances. TF-IDF LR via `shap.LinearExplainer` with 100–200 background
rows, mapping indices back to n-gram names. Other models via
`shap.Explainer(pipeline.predict_proba, shap.maskers.Text(TOKEN_PATTERN))`.
Global bar plot and beeswarm for the negative class. Token aggregation in the same
format as LIME so the two are directly comparable.

**6.7 Stress test** — 8 h — `src/evaluation/stress_test.py`

- **40 harsh-but-neutral** sentences: *"Kill the process before restarting the
  server."* / *"The build failed with a fatal error on line 42."*
- **20 genuinely negative**, of which **10 contain no jargon at all** — this
  separates jargon from real hostility.
- **10 minimal pairs:** *"The app crashed after the update."* vs *"The app crashed
  again, this update is useless."*
- ≥20 lexicon words, max 5 sentences each, lengths 5–30 words.
- `data/stress_test/se_stress.csv` — `id, text, intended_label, lexicon_word, pair_id`

Metrics per model: **false-alarm rate** (% of jargon-heavy non-negative sentences
called negative, lower better), **negative recall** on genuine complaints (higher
better), recall on negative-with-jargon vs without, and **minimal-pair accuracy**.
Scatter plot: x = false-alarm rate, y = negative recall (ideal = top-left).

**6.8 Demo** — 6 h — `notebooks/demo.ipynb`

Text box, model dropdown, Predict button. Output: predicted label, all three class
probabilities, and the sentence rendered with per-word background colouring (red
negative, green positive, intensity = weight). Example gallery of 8 preloaded
buttons. Final cells: master results table, charts, stress scatter plot.

Must run top-to-bottom under "Restart and Run All"; export static HTML for
submission.

The proposal says *"Notebook/CLI"* — the notebook satisfies it and doubles as the
figure source for the report.

---

## 6. Schedule

**Status: complete.** All stages implemented, 123 tests passing, report written.
Remaining: slides.

| Week | Work | Hours |
|---|---|---|
| 1 | ✅ Stages 1–2, splits, evaluation framework, baselines | done |
| 2 | ✅ Stage 3 complete — `normalize.py` + ablation | done |
| 3 | ✅ Stage 4.1 — TF-IDF features, vocabulary and compound-phrase analysis | done |
| 4 | ✅ Stage 5 — 9 TF-IDF models tuned and saved | done |
| 5 | ✅ Stage 4.2 — embeddings; Stage 5.2 — 6 embedding models | done |
| 6 | ✅ Stage 6.3–6.4 — test evaluation, results table, error analysis | done |
| 7 | ✅ Stage 6.5–6.6 — LIME + SHAP | done |
| 8 | ✅ Stage 6.7–6.8 — stress test + notebook | done |
| 9 | ✅ Report written · ⬜ slides | 5 |
| | **Remaining** | **slides only** |

---

## 7. Deliverables checklist

- [x] `src/acquisition/download.py`
- [x] `src/extraction/build_se.py` + `clean.csv` + Fleiss' κ
- [x] `src/extraction/clean_text.py` + 36 tests + rule table
- [x] `src/preprocessing/splits.py` + frozen splits
- [x] `src/evaluation/evaluate.py` + tests (65 tests total)
- [x] Two baselines recorded
- [x] `src/preprocessing/normalize.py` + 23 tests
- [x] Normalization ablation table + recorded decision
- [x] `src/features/features_tfidf.py` + vocabulary, compound-phrase and tuning tables
- [x] `src/features/features_embed.py` + OOV, nearest-neighbour and settings tables
- [x] 15 trained models
- [x] Master results table + charts
- [x] Error-analysis table
- [x] `data/lexicons/se_jargon.txt` — 43 words
- [x] LIME: 20 explanations, aggregate table, stability 0.907
- [x] SHAP: 20 explanations, exact linear values, LIME agreement 0.872
- [x] 60 stress sentences + results + scatter plot
- [x] `notebooks/demo.ipynb` + exported HTML (runs clean top-to-bottom)
- [x] Report — [report/REPORT.md](report/REPORT.md) · ⬜ slides

---

## 8. Out of scope, and why

These were considered and deliberately excluded. Recording them here so the
reasoning is visible rather than looking like oversights.

| Excluded | Reason |
|---|---|
| **Second domain (health / Druglib)** | Not in the proposal. Requires ~835 rows of hand-labelling, which is not feasible alongside the committed work. Code and partial corpus remain in the repository but are not used. |
| **Cross-domain transfer experiments** | Depends on the second domain. |
| **BERT / transformers** | Not in the proposal, which names exactly three representation families. Adding a fourth risks the central finding — a contextual model may fix the problem, which is a different project. Recorded as Future Work. |
| **McNemar significance testing** | Useful but not committed to; 15 models make the ranking clear enough. |
| **LIME stability analysis** (5 seeds × 3 sample sizes) | The proposal acknowledges LIME/SHAP instability in Limitations; measuring it in depth is beyond a course project. |
| **LIME vs SHAP rank correlation** | Both methods are reported; a formal agreement study is beyond the proposal. |
| **CLI demo** | The proposal says "Notebook/CLI" — one satisfies it. |

---

## 9. Risks

| Risk | When it shows | Response |
|---|---|---|
| Stemming erases the jargon signal | Stage 3.2 ablation | Measured, not guessed. If stemming wins on macro-F1 but worsens the neutral→negative rate, pick by confusion rate and report the trade-off. |
| Word2Vec quality poor at 3,031 training docs | Stage 4.2 | Expected; the proposal already lists it under Limitations. Report the nearest-neighbour table anyway — even a weak domain model beating GloVe on `kill` is the point. |
| Trigrams too sparse to help | Stage 4.1 | Likely at this corpus size. A finding, not a failure. `tfidf11` and `tfidf12` carry the compound-phrase argument. |
| LIME slow on the full test set | Stage 6.5 | Fall back to a stratified sample of ≥300 and say so. |
| Stress set reflects author bias | Stage 6.7 | Single-annotator limitation; state plainly in Limitations. |
| Test set touched more than once | Stage 6.3 | All tuning on validation. The test split is read once. |

---

## 10. Limitations to state in the report

- Small corpus (4,331 posts) limits self-trained embedding quality — as the
  proposal anticipated.
- LIME and SHAP are approximations with known instability; explanations are
  indicative, not exact.
- The stress set is written by one person, so it reflects one person's intuitions
  about what counts as harsh-but-neutral.
- Findings are demonstrated on one domain and one corpus. Whether the failure mode
  generalises to other technical domains is untested here.
- Contextual models (BERT and similar) are not evaluated; they may substantially
  reduce the lexical bias measured here. This is the natural next experiment.

---

## 11. Data licence

Senti4SD: Calefato, F., Lanubile, F., Maiorano, F., & Novielli, N. (2018).
Sentiment Polarity Detection for Software Development. *Empirical Software
Engineering*, 23(3), 1352–1382.

The repository also contains an unused partial corpus built from the UCI Drug
Reviews (Druglib.com, id 461) dataset. Its donors require research-only use, no
redistribution, and citation. Raw files are git-ignored.

> Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-Based
> Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.
> *Proceedings of the 2018 International Conference on Digital Health*, 121–125.

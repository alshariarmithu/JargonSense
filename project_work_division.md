# Work Division Plan: Interpretable Sentiment Classification Across Two Technical Domains

**Course:** CSE 4122
**Member A:** Al Shariar Hossain (Roll 2107066)
**Member B:** Hassan Mohammed Naquibul Hoque (Roll 2107077)

---

## 0. How to Read This Document

The project is organised along a **six-stage NLP pipeline**, and the pipeline is
run **once per domain, one domain at a time**.

```
Stage 1  Data acquisition
Stage 2  Text extraction & clean-up
Stage 3  Preprocessing        (sentence segmentation, stemming, lemmatization)
Stage 4  Feature engineering
Stage 5  Model building
Stage 6  Evaluation
─────────────────────────────────────────────────────────────────
Stage 7  Cross-domain integration   (only exists because there are two domains)
```

**Track order is fixed: SE first, completely, then HEALTH.**

```
TRACK SE        Stage 1 → 2 → 3 → 4 → 5 → 6        ──▶ SE GATE
TRACK HEALTH    Stage 1 → 2 → 3 → 4 → 5 → 6        ──▶ HEALTH GATE
TRACK JOINT     Stage 7: transfer, demos, report   ──▶ SUBMISSION
```

Nothing in the HEALTH track begins until the SE GATE is signed off, with **one
deliberate exception** recorded in Section 2.3 (the health neutral hand-labelling,
which is slow human work and cannot be parallelised later).

Within a track, **both members work at the same time on the same stage**. The
method split from the original plan is unchanged: **A owns TF-IDF + LIME, B owns
embeddings + SHAP.** What changed is the *order*: we no longer build both
corpora in parallel.

---

## 1. Why SE First

The old plan ran both domains in lockstep. Three problems with that, all of which
we already hit:

1. **The health corpus is blocked on 835 rows of hand-labelling.** Under the old
   plan that blocked GATE 1, which blocked everything — including all the SE work
   that needs no hand-labelling at all. SE is fully unblocked today and was sitting
   idle behind a gate it had already passed.
2. **Debugging two corpora at once doubles the surface area.** Every bug in
   `features_tfidf.py` had to be diagnosed against two datasets before we knew
   whether it was a code bug or a data bug.
3. **The pipeline is domain-agnostic by design.** Once it is proven end-to-end on
   SE, running HEALTH through it is mostly configuration, not new code. The second
   track is genuinely cheaper than the first — but only if the first one is
   *finished*, not half-finished.

**What SE-first costs us, stated honestly.** Cross-domain transfer (Stage 7) is
the project's central novelty and it cannot start until both tracks are done. That
pushes the riskiest experiment to the last two weeks with no slack. Section 6
lists the mitigation: a throwaway smoke-transfer run the moment HEALTH Stage 5
produces its first model, so a catastrophic surprise surfaces in Week 7, not
Week 8.

### The thesis is unchanged

| Domain | Corpus | The vocabulary problem |
|---|---|---|
| Software engineering (`se`) | Senti4SD gold standard (StackOverflow) | Harsh technical jargon that is functionally neutral: *fatal error*, *kill the process*, *crashed* |
| Healthcare (`health`) | UCI Drug Reviews (Druglib.com), id 461 | Relief inversion: a clinically negative word naming a symptom that went away — *the nausea stopped*, *no more panic attacks* |

**"Domain-specific vocabulary breaks sentiment models, and the failure mode
generalises."** A finding confirmed in two unrelated corpora is a result; the same
finding in one corpus is an anecdote.

---

## 2. Stage Map and Ownership

### 2.1 The stage-to-file map

```
src/
  acquisition/
    download.py              # Joint — fetches both raw corpora
  extraction/
    build_se.py              # A — Senti4SD .xlsx  -> data/processed/se/clean.csv
    build_health.py          # B — Druglib .tsv    -> data/processed/health/clean.csv
    validate_health.py       # B — rating-vs-human agreement scorer
    clean_text.py            # A — clean(text, domain): the string-level clean-up
  preprocessing/
    normalize.py             # A — segmentation, stemming, lemmatization (switchable)
    splits.py                # B — 70/15/15 stratified, seed 42, frozen after
  features/
    features_tfidf.py        # A — unigram / (1,2) / (1,3)
    features_embed.py        # B — GloVe + per-domain Word2Vec
  modeling/
    models_tfidf.py          # A — MNB, LR, LinearSVC
    models_embed.py          # B — GNB, LR, LinearSVC
  evaluation/
    evaluate.py              # B — metrics, confusion matrices, McNemar, baselines
    explain_lime.py          # A
    explain_shap.py          # B
    stress_test.py           # B — runs both stress sets
  integration/
    transfer.py              # A writes, B reviews, each runs their own families
    compare_explainers.py    # Joint — LIME vs SHAP agreement
cli/predict.py               # A
notebooks/demo.ipynb         # B
tools/                       # helper scripts, not part of the pipeline
tests/                       # one test module per stage
```

**Rename in this restructure:** `src/preprocess.py` → `src/extraction/clean_text.py`,
and the function `preprocess(text, domain)` → `clean(text, domain)`. The old name
collided with Stage 3, which is the stage actually called "preprocessing". What the
module *does* — HTML unescape, URL/mention placeholders, lowercase, emoticons,
negation expansion, character collapsing — is clean-up, which is Stage 2. The rules
themselves are unchanged.

### 2.2 Who owns what, by stage

| Stage | Member A (Al Shariar, 2107066) | Member B (Hassan, 2107077) |
|---|---|---|
| 1 · Acquisition | Senti4SD download + integrity check | Druglib download + licence handling |
| 2 · Extraction & clean-up | `build_se.py`, `clean_text.py` | `build_health.py`, `validate_health.py` |
| 3 · Preprocessing | `normalize.py` — segmentation, stemming, lemmatization | `splits.py`; runs the three-way normalization ablation |
| 4 · Feature engineering | TF-IDF: unigram, (1,2), (1,3) | GloVe + self-trained Word2Vec |
| 5 · Model building | MNB, LR, Linear SVM on TF-IDF | GNB, LR, Linear SVM on embeddings |
| 6 · Evaluation | LIME: per-instance, aggregated, stability | `evaluate.py`, baselines, SHAP, bias score, stress tests |
| 7 · Integration | `transfer.py`, CLI demo | Notebook demo, transfer runs for embedding families |
| Lexicons | SE jargon lexicon | Health symptom lexicon |
| Stress sets | 60 SE sentences | 60 health sentences + the evaluation harness |
| Report | Intro, datasets (SE), preprocessing, TF-IDF results, LIME | Evaluation method, datasets (health), embedding results, SHAP, stress tests |
| Joint | Setup, cross-domain discussion, LIME vs SHAP, slides | Setup, cross-domain discussion, LIME vs SHAP, slides |

### 2.3 The one permitted violation of SE-first

**Health neutral hand-labelling (835 rows) starts in Week 1 and runs in the
background throughout the SE track.**

It is the only task on the critical path that is pure human time and cannot be
compressed by having a working pipeline. Everything else in the HEALTH track waits.
Budget ~30 rows per person per day; at that rate 835 rows takes about two weeks of
background effort, finishing comfortably before the HEALTH track opens in Week 6.

This exception is recorded here so it does not look like the plan was quietly
abandoned. Nothing else from the HEALTH track may start early.

---

## 3. Shared Contracts (Fixed — Do Not Change Unilaterally)

### 3.1 Function signatures

- `clean(text: str, domain: str) -> str` — `src/extraction/clean_text.py` (A).
  `domain` is `"se"` or `"health"` and controls **only** domain-specific rules
  (code spans for SE, dosage patterns for HEALTH). Everything else is identical
  across domains.
- `normalize(text: str, *, segment: bool = False, mode: str = "none") -> str` —
  `src/preprocessing/normalize.py` (A). `mode` is `"none" | "lemma" | "stem"`.
- `tokenize(text: str) -> list[str]` and `TOKEN_PATTERN` — `clean_text.py` (A).
- `load_splits(domain: str) -> (train_df, val_df, test_df)` —
  `src/preprocessing/splits.py` (B). Columns: `id`, `text`, `polarity`.
- `evaluate(name, domain, y_true, y_pred, out_dir="results/metrics") -> dict` —
  `src/evaluation/evaluate.py` (B).
- **Every saved model is a full pipeline** exposing `predict(list[str])` and
  `predict_proba(list[str])` on **raw** strings. Required by LIME, SHAP, the CLI,
  the notebook and the transfer experiments.

### 3.2 Constants

- Label order everywhere: `["negative", "neutral", "positive"]`.
- CSV format everywhere: `id;text;polarity`, semicolon-delimited, UTF-8 no BOM.
  Semicolons because review text is full of commas.
- Random seed everywhere: `42`.
- Model naming: `models/{domain}/{features}_{classifier}.joblib`, e.g.
  `models/se/tfidf13_lr.joblib`, `models/health/glove_svm.joblib`.
- `TOKEN_PATTERN` is the single source of truth for tokenisation. It must be
  passed to `TfidfVectorizer(token_pattern=...)` in Stage 4 and to
  `LimeTextExplainer(split_expression=...)` in Stage 6, so the three cannot drift.
  A test enforces this.

### 3.3 The non-negotiable rule

**Both corpora must be acquired, cleaned, preprocessed, featurised, modelled and
evaluated identically.** The entire project rests on comparing two domains. Any
asymmetry in preprocessing, split ratio, class balance or tuning protocol becomes a
confound an examiner will find immediately.

SE-first makes this *easier* to enforce, not harder: the HEALTH track reruns the
exact code the SE track produced. **Any code change made during the HEALTH track
for HEALTH's benefit requires re-running the affected SE stage.** No exceptions —
this is the single easiest way to break the project.

---

# TRACK SE

Run stages 1–6 to completion on the SE corpus before opening the HEALTH track.

---

## Stage SE-1 — Data Acquisition

**Owner:** Joint · **Status: DONE**

1. `git lfs clone https://github.com/collab-uniba/Senti4SD.git`; locate
   `Senti4SD_GoldStandard_EmotionPolarity.xlsx` (4,423 rows; columns `study`,
   `label`, `so.id`, `post.type`, `text`, `final`, `r1`, `r2`, `r3`).
2. Copy to `data/raw/senti4sd/` **unchanged**. Never edit raw files.
3. The workbook is used rather than Senti4SD's train/test partition CSVs because it
   holds the same items *plus* the `r1`/`r2`/`r3` rater columns, without which
   inter-rater agreement cannot be computed.
4. Record: total rows, class distribution, mean text length, and 5 examples of the
   vocabulary problem. These go straight into the report.

**Deliverable:** `src/acquisition/download.py`, raw files in place, `.gitignore`
excluding `data/raw/`.

---

## Stage SE-2 — Text Extraction & Clean-up

### SE-2.1 Corpus extraction (Member A) — **DONE**

1. Load the gold standard; keep `text` and `final`.
2. Standardise columns to `id`, `text`, `polarity`; lowercase the labels.
3. Remove exact duplicate texts (keep first); log the count.
4. Drop empty or whitespace-only rows.
5. Compute **Fleiss' kappa** from `r1`/`r2`/`r3`. The rater columns are free text
   with mixed case and the typos `Postive`, `Poitive`, `Netural` — repair before
   scoring, and exclude items where any rater is missing or unparseable.
6. Record the class distribution. **This becomes the target distribution the
   HEALTH track matches in Stage H-2.**
7. Save to `data/processed/se/clean.csv`.

**Result:** 4,331 rows (4,423 loaded, 92 duplicates dropped), Fleiss' κ = **0.7586**
over 4,359 items, mean 29.97 words.

### SE-2.2 String-level clean-up (Member A) — **DONE, needs renaming**

`clean(text, domain)` in `src/extraction/clean_text.py`. No I/O, no pandas, so
it imports cleanly into tests, notebooks, sklearn pipelines and the CLI.

Shared rules, in order:

1. `html.unescape`
2. **Lowercase**
3. URLs → `URL` (`https?://\S+|www\.\S+`)
4. `@mentions` → `USER`
5. **Domain rules** (the only permitted divergence)
6. Emoticons → `EMO_POS` / `EMO_NEG`
7. Negation expansion: `can't` → `can not`, `n't` → ` not`
8. Collapse repeated characters (`sooooo` → `soo`) and punctuation (`!!!!` → `!!`)
9. Squeeze whitespace

Domain rules — `se`: code tags, backtick spans, `foo(bar)` calls and `a.b.c` paths
→ `CODE`. `health`: dosages → `DOSE`, remaining bare numbers → `NUM`.

**Two documented deviations from the original rule order.** Both are order-only;
the rule *set* is unchanged and still identical across domains, so no cross-domain
comparison is affected.

1. **Lowercasing moved from step 4 to step 2**, so every placeholder stays
   uppercase. The literal words *url* and *user* are common in StackOverflow text;
   lowercase placeholders would be indistinguishable and would silently corrupt the
   TF-IDF vocabulary.
2. **URL replacement runs before emoticon replacement.** `http://` contains `:/`,
   a sad-face emoticon; the reverse order turns every URL into `httpEMO_NEG/...`.

**Never remove negation words.** They flip sentiment, and in the health corpus
("no more panic attacks") negation *is* the entire phenomenon.

**Deliverables:** `clean_text.py`, `tests/test_clean_text.py` (36 tests passing),
`report/preprocess_examples.md` generated from `describe_rules()` so the report's
rule table cannot drift from the code.

---

## Stage SE-3 — Preprocessing

### SE-3.1 Sentence segmentation, stemming, lemmatization (Member A) — **NOT STARTED**

New module `src/preprocessing/normalize.py`. This is the stage the course
pipeline names explicitly, and it is **not** a formality — it is where the project's
sharpest methodological tension lives, so it is implemented and *measured* rather
than asserted either way.

**The tension.** Stemming and lemmatization collapse inflected forms:

| Original | Stemmed | Lemmatized |
|---|---|---|
| killed, kills, killing | `kill` | `kill` |
| stopped, stops, stopping | `stop` | `stop` |

Normally this helps: fewer distinct tokens, denser counts, better learning from a
small corpus. **For this project it may destroy the finding.** Tense carries the
sentiment in both domains:

- HEALTH: *"the nausea **stopped**"* (positive — relief) vs *"the nausea won't
  **stop**"* (negative — ongoing). Stemmed, both are `stop`, and relief inversion
  becomes unmeasurable.
- SE: *"**killed** the process"* (neutral — routine work) vs the emotional register
  of *kill*. Collapsing them hides the jargon effect LIME is supposed to expose.

**The resolution: implement it, switch it, measure it.**

1. **Sentence segmentation** — NLTK `punkt`. Exposed as `segment(text) -> list[str]`.
   Used for the per-sentence length statistics in the report and required by the
   health corpus's longer reviews. It does **not** alter the text fed to the models.
2. **Stemming** — `nltk.stem.PorterStemmer` (and `SnowballStemmer("english")` as the
   alternate), behind `mode="stem"`.
3. **Lemmatization** — `nltk.stem.WordNetLemmatizer` with POS tags from
   `nltk.pos_tag`, behind `mode="lemma"`. POS tagging matters: without it,
   `stopped` lemmatizes to `stopped`, not `stop`, and the comparison is meaningless.
4. **Stopword removal stays off in all three modes.** Negation words are stopwords
   in every standard list, and removing them would break both domains at once. Note
   this explicitly in the report.
5. Signature: `normalize(text, *, segment=False, mode="none")`, `mode` in
   `{"none", "lemma", "stem"}`. Default `"none"`.

**Deliverables:** `normalize.py`, `tests/test_normalize.py` (≥10 tests, including
`stopped`/`stop` and `killed`/`kill` in all three modes), a before/after table for
the report.

### SE-3.2 The normalization ablation (Member B) — **NOT STARTED**

The evidence that decides which mode the rest of the project uses.

1. Train the **same** model — TF-IDF (1,2) + Logistic Regression, defaults — three
   times on the SE training split: `mode="none"`, `mode="lemma"`, `mode="stem"`.
2. Score all three on the SE **validation** split (not test — test is spent once,
   in Stage SE-6).
3. Record for each mode: vocabulary size, macro-F1, and the
   **neutral→negative confusion rate** (the SE failure metric).
4. Report all three rows in one table in `results/ablation/normalization_se.csv`.
5. **The winning mode by validation macro-F1 becomes the project default** and is
   used unchanged for every later model, in both domains.
6. **Whatever wins, the table is a report finding.** If stemming raises macro-F1 but
   also raises the neutral→negative rate, say so plainly — that is the accuracy/
   interpretability trade-off the whole project is about, caught in one table.
7. Re-run this ablation once on HEALTH in Stage H-3 to confirm the choice transfers.
   If it does not, that disagreement is itself a result.

**Deliverables:** ablation table, one paragraph in the report, a recorded decision.

### SE-3.3 Splits (Member B) — **DONE**

1. `train_test_split` twice with `stratify=polarity`, `random_state=42` →
   **70% train / 15% val / 15% test**.
2. Verify class proportions match across all three splits; print the table.
3. Commit the splits — **frozen from this point**.
4. Implement `load_splits(domain)`.

**Result:** 3,031 train / 650 val / 650 test. Class proportions hold to within one
row per class.

**Splits are frozen before Stage 4 for a reason:** every vectorizer, embedding and
scaler is fit on **train only**. Fitting on anything else leaks the test set.

---

## Stage SE-4 — Feature Engineering

### SE-4.1 TF-IDF (Member A) — **NOT STARTED**

1. Three configs, each with `preprocessor=partial(clean, domain="se")` and the
   Stage-3 winning normalization mode:
   - `tfidf11`: `ngram_range=(1,1)` — baseline
   - `tfidf12`: `ngram_range=(1,2)`
   - `tfidf13`: `ngram_range=(1,3)`
2. Shared settings: `min_df=2`, `max_df=0.95`, `sublinear_tf=True`,
   `token_pattern=TOKEN_PATTERN`.
3. **Fit on train only**; transform val and test.

**Vocabulary analysis:**

4. Vocabulary size and sparsity per config.
5. Top 30 n-grams by mean TF-IDF weight per class.
6. **Compound phrase check.** Do `fatal error`, `kill the process`,
   `null pointer exception` survive into the (1,2)/(1,3) vocabularies? Count
   occurrences per class. The HEALTH track repeats this with `no side effects`,
   `pain went away`, `no more`, `stopped working`, and the two tables are compared
   in Stage 7.

**Light tuning:** LR at default `C`; try `min_df` in {1, 2, 5} and `max_features` in
{None, 20000, 50000} on validation. Record the choice — HEALTH uses the **same
protocol**, not necessarily the same value.

**Deliverables:** `features_tfidf.py`, vocabulary stats, SE compound-phrase table.

### SE-4.2 Embeddings (Member B) — **NOT STARTED**

**Pretrained GloVe.** `glove.6B.100d.txt` via gensim `KeyedVectors`. Tokenize with
A's `clean()` + `tokenize()`. Document vector = mean of in-vocabulary token vectors;
zero vector if none. Variant: TF-IDF-weighted mean (IDF from a unigram TF-IDF fitted
on train). **Cut this variant first if time runs short.**

Report the **OOV rate** overall and specifically for the SE jargon lexicon and code
tokens.

**Self-trained Word2Vec.** Train on the **SE training split only**:
`vector_size=100, window=5, min_count=2, sg=1, negative=5, epochs=30, seed=42,
workers=1`. Try `vector_size` in {50, 100}, `sg` in {0, 1}; pick by validation
macro-F1 with LR. Save to `models/se/w2v.model`.

**The headline table.** Probe words `kill, fatal, crash, error, exception, hang,
abort`. List the 10 nearest neighbours in GloVe and in SE-Word2Vec side by side.
**The expected result is the most quotable thing in the project:** GloVe places
`kill` near `murder`; SE-Word2Vec places it near `process` / `terminate`. Put this
on a slide. Optional 2D PCA/t-SNE panel.

Note that ~4k documents limits Word2Vec quality — and that the HEALTH corpus will be
limited *equally*, which makes it a controlled comparison rather than a flaw.

**Wrap as sklearn transformers:** `MeanEmbeddingVectorizer` and
`TfidfEmbeddingVectorizer`, `fit`/`transform` taking raw strings.

**Deliverables:** `features_embed.py`, OOV table, nearest-neighbour table.

---

## Stage SE-5 — Model Building

### SE-5.1 TF-IDF models (Member A) — **NOT STARTED**

**9 experiment rows** (3 feature sets × 3 classifiers).

1. `MultinomialNB` — generative.
2. `LogisticRegression(max_iter=2000)` — discriminative.
3. `LinearSVC` wrapped in `CalibratedClassifierCV(cv=5)` — **LinearSVC has no
   `predict_proba`**, which LIME, SHAP and both demos require.

**Tuning** — validation set or `GridSearchCV` stratified 5-fold,
`scoring="f1_macro"`:

- MNB: `alpha` in {0.01, 0.1, 0.5, 1.0}
- LR: `C` in {0.01, 0.1, 1, 10, 100}, `class_weight` in {None, "balanced"}
- SVM: `C` in {0.01, 0.1, 1, 10}, `class_weight` in {None, "balanced"}

Retrain the best configs, save to `models/se/tfidf1X_{mnb|lr|svm}.joblib`, write
predictions to `results/metrics/se_tfidf_predictions.csv`.

### SE-5.2 Embedding models (Member B) — **NOT STARTED**

**12 experiment rows** (2 embeddings × 2 pooling × 3 classifiers), or 6 if the
weighted variant is cut.

1. **Gaussian NB** — `MultinomialNB` is *impossible* on embeddings because they
   contain negative values. This is a substantive generative-model constraint and
   belongs in the report as a finding, not a footnote.
2. LR with `StandardScaler`.
3. `LinearSVC` with `StandardScaler`, wrapped in `CalibratedClassifierCV(cv=5)`.

**Tuning** — same protocol as SE-5.1:

- GNB: `var_smoothing` in {1e-9, 1e-7, 1e-5, 1e-3}
- LR / SVM: `C` in {0.01, 0.1, 1, 10}, `class_weight` in {None, "balanced"}

Save as `models/se/{glove|w2v}{mean|wt}_{gnb|lr|svm}.joblib`.

---

## Stage SE-6 — Evaluation

### SE-6.1 Evaluation framework (Member B) — **DONE**

1. `evaluate(name, domain, y_true, y_pred, out_dir)` computes accuracy, macro-F1,
   weighted-F1, per-class P/R/F1, and the confusion matrix (counts and
   row-normalised).
2. **Domain-specific confusion rates.** For SE: fraction of gold-neutral predicted
   negative. For HEALTH: fraction of gold-**positive** predicted negative. These are
   *different error directions* and both columns exist in the results table from the
   start.
3. Confusion-matrix heatmap → `results/figures/cm_{domain}_{name}.png`.
4. All numbers → `results/metrics/{domain}_{name}.json`.
5. `append_to_results_table(...)` → one row in `results/metrics/all_results.csv`
   with a `domain` column.
6. McNemar's test helper for comparing two models' predictions.
7. Five unit tests with hand-computed values.

### SE-6.2 Baselines (Member B) — **DONE for SE**

Majority-class and VADER, both domains.

| Domain | Model | Accuracy | Macro-F1 | SE neut→neg | HEALTH pos→neg |
|---|---|---|---|---|---|
| se | majority | 0.385 | 0.185 | — | — |
| se | vader | 0.715 | 0.708 | **0.252** | — |
| health | majority | 0.559 | 0.239 | — | — |
| health | vader | 0.485 | 0.350 | — | **0.416** |

The health rows are **provisional** — they were computed on a corpus with zero
neutral rows and must be re-run in Stage H-6. They are kept here only because the
VADER health number is already the clearest single demonstration of the thesis: a
general-purpose lexicon calls **41.6%** of genuinely positive drug reviews negative.

### SE-6.3 Final test evaluation (Both) — **NOT STARTED**

Retrain best configs, evaluate **once** on the SE test split through `evaluate()`.
**The test split is spent here. Do not touch it again during the SE track.**

### SE-6.4 Error analysis (Both) — **NOT STARTED**

1. Collect test cases where gold = neutral, predicted = negative. Tag each:
   *contains SE jargon* / *sarcasm* / *annotation noise* / *other*.
2. Report counts plus 5 representative examples.
3. For LR, list the top 20 features pushing toward negative and flag jargon.
4. **Overlap of errors** between the best TF-IDF and the best embedding model —
   shared vs unique.
5. McNemar's tests: best TF-IDF vs best embedding; best NB vs best discriminative.

### SE-6.5 LIME (Member A) — **NOT STARTED**

**Per-instance.** `LimeTextExplainer(class_names=["negative","neutral","positive"],
split_expression=TOKEN_PATTERN, random_state=42)`. Explain **20 agreed SE
instances** — 5 correct negatives, 5 correct neutrals, 5 characteristic errors,
5 jargon-heavy cases — with both the best TF-IDF model and B's best embedding model.
`explain_instance(text, pipeline.predict_proba, num_features=10, num_samples=2000,
labels=[0,1,2])`. HTML → `results/explanations/lime/se/`. Two to three sentences of
observation per case.

**Aggregated.** Run LIME across the full SE test split (or a stratified ≥300 sample
if runtime bites). Accumulate per token: total weight toward negative, occurrences,
mean weight.

**SE jargon lexicon** (A owns, `data/lexicons/se_jargon.txt`): `fatal, error, kill,
killed, crash, crashed, abort, exception, fail, failed, failure, dead, deadlock,
bug, broken, hang, panic, terminate, execute, garbage, deprecated, warning, exploit,
attack, destroy`.

For each lexicon word, mean LIME weight toward negative **on gold-neutral SE
instances**. A strong positive value means the model reads domain vocabulary as
hostility. Horizontal bar chart of the top 25 negative-driving tokens, lexicon words
coloured differently.

**Stability check.** For the 20 fixed instances, run LIME with 5 `random_state`
values; report Jaccard overlap of top-5 and top-10 token sets. Repeat with
`num_samples` in {500, 2000, 5000}. Report mean and standard deviation. The HEALTH
track repeats this, and the comparison matters — health reviews are ~2× longer, so
stability plausibly differs.

### SE-6.6 SHAP (Member B) — **NOT STARTED**

**Per-instance.** TF-IDF LR: `shap.LinearExplainer(clf, background)` with 100–200
transformed training rows; map indices back to n-gram names. Embedding and SVM
models: `shap.Explainer(pipeline.predict_proba, shap.maskers.Text(TOKEN_PATTERN))`.
**Same 20 fixed instances as A.** `shap.plots.text` HTML and waterfall PNGs →
`results/explanations/shap/se/`.

**Global.** SHAP over the full SE test split (or stratified 300 for the Partition
explainer). Global bar plot (mean |SHAP|) and beeswarm for the negative class.
Token-level aggregation **in A's LIME format** so the two are directly comparable in
Stage 7.

**Domain Bias Score.** For each SE jargon lexicon word, mean SHAP contribution
toward negative on instances whose gold label is **not** negative. The score per
model is the mean over lexicon words appearing ≥3 times. Compare across TF-IDF,
GloVe and Word2Vec in one table and bar chart. The HEALTH track produces the
matching table with its own lexicon, and the pair is the core evidence that lexical
bias is general rather than a Senti4SD artefact.

### SE-6.7 SE stress test (Member A writes, Member B runs) — **NOT STARTED**

1. **40 harsh-but-neutral** sentences: *"Kill the process before restarting the
   server."* / *"The build failed with a fatal error on line 42."* /
   *"Garbage collection destroys unused objects."*
2. **20 genuinely negative** SE sentences, of which **10 contain no jargon at all** —
   this separates jargon from real hostility.
3. Cover ≥20 lexicon words, max 5 sentences per word. Vary length (5–30 words) and
   style (question, instruction, report).
4. **10 minimal pairs:** *"The app crashed after the update."* vs *"The app crashed
   again, this update is useless."*
5. Save to `data/stress_test/se_stress.csv` with columns `id, text, intended_label,
   lexicon_word, pair_id`.
6. **Blind cross-annotation:** shuffle, strip `intended_label`, B labels A's set.
   Cohen's kappa between intended and independent label. Discuss disagreements
   together; drop what you cannot agree on; record the count.
7. **Run** all saved SE models plus VADER. Metrics:
   - **False-alarm rate** — % of jargon-heavy non-negative sentences predicted
     negative (lower is better)
   - **Negative recall** on genuine complaints (higher is better)
   - Recall on negative-with-jargon vs negative-without
   - **Minimal-pair accuracy** — % of pairs where both halves are correct
8. Scatter plot: x = false-alarm rate, y = negative recall, one point per model
   (ideal = top-left).
9. LIME and SHAP on the 10 worst-handled sentences.

---

## ✅ SE GATE — Do Not Open the HEALTH Track Until

- [x] `data/processed/se/clean.csv` exists in `id;text;polarity` form — 4,331 rows
- [x] Inter-rater agreement ≥ 0.75 — Fleiss' κ = **0.759**
- [x] Splits frozen and committed — 3,031 / 650 / 650, seed 42
- [x] `clean_text.py` passing its test suite — 36 tests
- [x] `evaluate.py` written, baselines recorded for SE
- [ ] `normalize.py` written and tested; **normalization ablation run and the
      project default chosen**
- [ ] TF-IDF and embedding feature matrices built for SE
- [ ] All 21 SE models trained, tuned and saved
- [ ] SE test evaluated **once**; error analysis tagged
- [ ] LIME and SHAP run on the 20 agreed SE instances; both bias tables produced
- [ ] SE stress set written, cross-annotated (κ recorded), and run
- [ ] **The full SE pipeline reruns end-to-end from `clean.csv` with one command**

That last box is the real gate. The HEALTH track's entire cost advantage depends on
the SE pipeline being *reproducible*, not merely *finished once*.

---

# TRACK HEALTH

**Do not begin until the SE GATE is signed.** Exception: the neutral hand-labelling
in Stage H-2.3, which runs in the background from Week 1 (Section 2.3).

Every stage below reruns SE code with `domain="health"`. **If you find yourself
writing a new module, stop** — either it belongs in the shared pipeline (in which
case add it and re-run SE), or it is an unjustified asymmetry.

---

## Stage H-1 — Data Acquisition

**Owner:** Member B · **Status: DONE**

1. `pip install ucimlrepo`, then `fetch_ucirepo(id=461)`. Copy raw `.tsv` files to
   `data/raw/druglib/`.
2. **Licence.** The Druglib donors require research-only use, no redistribution and
   citation. **Raw Druglib files are never committed.** `src/acquisition/download.py` is
   committed instead. Cite Gräßer, Kallumadi, Malberg & Zaunseder (2018).
3. **Open decision:** `data/processed/health/clean.csv` contains Druglib review text
   verbatim while `.gitignore` excludes only `data/raw/`. Decide whether the
   processed health files are also excluded from version control. **Resolve this
   before the HEALTH track opens, not after the text is in the git history** — once
   committed, removing it requires a history rewrite.

---

## Stage H-2 — Text Extraction & Clean-up (Member B)

Druglib gives **ratings, not sentiment labels**, and they disagree with the text
constantly. A row rated 10 can have a side-effects field reading *"heavier bleeding
and clotting than normal."* Handle this explicitly.

### H-2.1 Field selection — **DONE**

1. Use **`commentsReview` only** as training text — the free-form overall statement,
   functionally closest to a StackOverflow post.
2. **Set `benefitsReview` and `sideEffectsReview` aside entirely.** They are
   positive- and negative-leaning *by construction*, so they are useless as training
   data — but they become the bias probe in Stage H-6.6. Saved to
   `data/processed/health/bias_probe.csv`, keyed by row id.

### H-2.2 Filtering — **DONE**

| Filter | Rows removed |
|---|---|
| Rows loaded | 4,143 |
| missing text | 13 |
| boilerplate (`see above`, `none`, `thanks`) | 25 |
| copy-pasted drug monograph | 24 |
| under 15 words | 776 |
| duplicate text | 61 |
| **After filtering** | **3,261** |

**Report these counts.** Unexplained data loss looks like cherry-picking.

### H-2.3 Label assignment — **BLOCKED on hand-labelling**

**Binning:** rating **1–3 → negative**, **8–10 → positive**, **4–7 → discarded**.

| Band | Count |
|---|---|
| negative (1–3) | 591 |
| positive (8–10) | 1,823 |
| discarded (4–7) | 847 |

**4–7 is discarded, never mapped to neutral.** A mid rating means *mixed feelings*;
Senti4SD neutral means *absence of affect*. Conflating the two would train the model
to call emotionally intense text neutral and would silently destroy every
cross-domain comparison in Stage 7.

**Neutral is built by hand.** `build_health.py` writes 1,200 pre-scored candidates to
`neutral_candidates.csv`, scored on dosage, imperative and schedule signals, with any
overt affect word disqualifying the row. 1,343 candidates pass the prefilter. Rows
rated 4–7 are offered first because the binning discards them anyway, so labelling
them costs no other class any data.

Fill the empty `polarity` column, save as `neutral_labelled.csv`, re-run
`build_health`.

**Why 835 rows and not the 400 originally budgeted.** Negative is capped at 591 rows,
so matching SE's 27.15% negative share caps the whole corpus at ~2,177 rows, and
38.37% of that is 835. The 400 figure assumed a 4,000-row corpus the negative class
cannot support. `balance()` computes this and logs it as
`neutral_rows_needed_for_target` rather than silently dropping the class.

**Current state: 1,341 rows — 591 negative, 750 positive, 0 neutral.** Mean 64.7
words, roughly 2× the SE mean.

### H-2.4 Label validation — **BLOCKED on hand-labelling**

1. Both members independently hand-label the **same 100** `commentsReview` rows in
   `validation_sample.csv`. **Label from the text alone** — the file carries `rating`
   and `rating_derived` columns for later diagnosis, and reading them while
   labelling makes the agreement figure worthless.
2. `python -m src.extraction.validate_health` reports agreement and Cohen's κ for
   the current 1–3 / 8–10 binning **and** the stricter 1–2 / 9–10 fallback side by
   side, plus disagreements broken down by rating and direction.
3. **If agreement with the rating-derived labels is below 80%, the binning is
   wrong.** Switch to 1–2 / 9–10 and re-measure.
4. **Fallback:** if agreement stays poor, switch to the Drugs.com corpus (UCI id 462)
   sampled to ~4,000. **Decide this the week the HEALTH track opens, not later.**

### H-2.5 Balance — **BLOCKED**

Sample to match Senti4SD's class proportions from SE-2.1. Save to
`data/processed/health/clean.csv`.

### H-2.6 Clean-up

**No new code.** `clean(text, "health")` already exists and is already tested. The
health branch replaces dosages with `DOSE` and bare numbers with `NUM`.

---

## Stage H-3 — Preprocessing (Member B)

1. **No new code.** Run `normalize.py` in the mode chosen by the SE-3.2 ablation.
2. **Re-run the ablation once on HEALTH** (`results/ablation/normalization_health.csv`)
   to confirm the choice transfers. If a different mode wins on HEALTH, **do not
   switch** — keep the SE choice for both, and report the disagreement. Using
   different normalization per domain would confound every cross-domain comparison in
   Stage 7. The disagreement is the finding.
3. **Splits.** Re-run `splits.py --domain health` on the *completed* corpus. The
   existing `data/processed/health/` splits were generated from the 0-neutral corpus
   and **must be regenerated and re-frozen**.
4. Report the per-sentence statistics from `segment()`: health reviews average ~2×
   the SE word count, which is relevant to LIME/SHAP stability (H-6.5) and Word2Vec
   quality.

---

## Stage H-4 — Feature Engineering

**No new code.** Re-run SE-4.1 (A) and SE-4.2 (B) with `domain="health"`.

- **A:** three TF-IDF configs, vocabulary stats, and the health compound-phrase
  check — do `no side effects`, `pain went away`, `no more`, `stopped working`
  survive into the (1,2)/(1,3) vocabularies? Same table format as SE, with a
  `domain` column.
- **B:** GloVe + a **separate** health Word2Vec trained on the health training split
  only. Probe words `nausea, pain, attack, depression, stopped, gone, relief`.
  Expect drug names to be heavily OOV in GloVe — **that is a finding, report the
  rate.** The predicted headline: GloVe places `attack` near `assault`;
  HEALTH-Word2Vec places it near `episode` / `migraine`.
- Tuning: **same protocol**, independently chosen values. Record both.

---

## Stage H-5 — Model Building

**No new code.** Re-run SE-5.1 (A) and SE-5.2 (B) with `domain="health"`. 9 TF-IDF
rows + 12 embedding rows = **21 health models**, saved to `models/health/`.

**Smoke-transfer checkpoint (Section 1's mitigation).** The moment the first health
model is saved, run one throwaway cross-domain prediction — SE-trained TF-IDF LR on
100 health test rows — and eyeball the macro-F1. You are not reporting this number;
you are checking that Stage 7 will not blow up in the final week. **If it does not
degrade, treat it as a bug immediately** (see GATE 2).

---

## Stage H-6 — Evaluation

**No new code** except the bias probe in H-6.6.

1. **H-6.1 Baselines** — re-run majority and VADER on the *completed* corpus. The
   current health baseline rows in `all_results.csv` are provisional and must be
   overwritten.
2. **H-6.2 Final test evaluation** — once, through `evaluate()`.
3. **H-6.3 Error analysis** — collect gold = positive, predicted = negative. Tag
   each: *relief inversion* / *mixed review* / *binning error* / *other*. Note this
   is a **different error direction** from SE's neutral→negative. Counts plus 5
   examples; top 20 LR features pushing negative, domain vocabulary flagged.
4. **H-6.4 LIME (A)** — 20 health instances, aggregated run, stability check.
   **Health symptom lexicon** (B owns, `data/lexicons/health_symptoms.txt`):
   `nausea, pain, cramps, depression, anxiety, attack, attacks, panic, migraine,
   headache, seizure, insomnia, fatigue, dizziness, swelling, rash, bleeding,
   vomiting, wheezing, itching`. Mean LIME weight toward negative on
   **gold-positive** health instances.
5. **H-6.5 SHAP (B)** — same structure; Domain Bias Score using the health lexicon.
6. **H-6.6 The bias probe — unique to the health corpus (B).** Take
   `bias_probe.csv`. Classify the `sideEffectsReview` text from rows rated **8–10** —
   patients who *liked* the drug, describing its side effects — and count how often
   the model predicts negative. **A clean, single-number bias measurement with no
   hand-labelling involved.** Senti4SD offers no equivalent. Report it prominently.
7. **H-6.7 Health stress set (B writes, A blind-annotates).**
   - **40 relief-framed positives:** *"The dizziness finally stopped after a week."* /
     *"No more panic attacks since I started this."*
   - **20 genuine complaints**, of which 10 contain no symptom vocabulary.
   - **10 minimal pairs using the same symptom word in both halves** — sharper than
     the SE version and worth highlighting: *"The nausea finally stopped."* vs
     *"The nausea never stopped."*
   - Save to `data/stress_test/health_stress.csv`, same columns as SE.
   - Blind cross-annotation, Cohen's κ, same metrics and scatter plot as SE-6.7.

---

## ✅ HEALTH GATE — Do Not Open Stage 7 Until

- [ ] ≥ 300 usable neutral health rows — **currently 0**
- [ ] Rating-agreement ≥ 80% on the 100-row validation sample, or the documented
      fallback taken
- [ ] Comparable size and class balance across the two corpora
- [ ] Health splits **regenerated** from the completed corpus and re-frozen
- [ ] All 21 health models trained and saved
- [ ] Health baselines **re-run** (the provisional rows overwritten)
- [ ] LIME, SHAP, bias score and bias probe complete
- [ ] Health stress set written, cross-annotated and run
- [ ] **Smoke transfer shows degradation** (H-5)

---

# TRACK JOINT — Stage 7: Cross-Domain Integration

This is what elevates the project from "sentiment classification, twice" to a
transferability study. Neither member owns it alone.

### 7.1 Transfer matrix (A writes `transfer.py`, B reviews; each runs their own families)

Four conditions per model family:

| Train | Test | Meaning |
|---|---|---|
| SE | SE | in-domain baseline |
| HEALTH | HEALTH | in-domain baseline |
| SE | HEALTH | cross-domain |
| HEALTH | SE | cross-domain |

1. **A runs the TF-IDF families, B runs the embedding families.** Report macro-F1
   for all four cells per family.
2. **Handle the vocabulary problem honestly.** A TF-IDF model trained on SE has no
   vocabulary for health text, so cross-domain TF-IDF will collapse largely through
   OOV. **Report the test-set OOV rate alongside every cross-domain number**, or the
   result is uninterpretable. Embeddings degrade more gracefully — that contrast is
   itself a finding and is the strongest argument for the embedding half.
3. **Per-class degradation.** Which class collapses? Predicted direction: SE-trained
   models over-predict negative on health text (relief sentences read as complaints);
   health-trained models over-predict negative on SE jargon. Confirm or refute with
   the confusion matrices.
4. Heatmap of the transfer matrix. Save everything to `results/transfer/`.

### ⚠️ GATE 2 — Sanity Check

**If cross-domain performance does not degrade, something is wrong.** Check for
leakage, label mapping errors, or a test set that is accidentally trivial before
reporting a surprising result. **A suspiciously good number is a bug until proven
otherwise.**

### 7.2 Cross-domain stress test (B)

Run SE-trained models on the health stress set and vice versa. **This is the most
legible result in the whole project for a non-CS audience** — watch an SE-trained
model call *"no more panic attacks"* negative.

### 7.3 LIME vs SHAP agreement (split evenly)

1. **A:** export LIME top-10 token rankings for all 40 instances in a common CSV.
2. **B:** export SHAP top-10 in the same format.
3. **Together:** Spearman rank correlation and top-5 overlap per instance.
   **Compare agreement between domains** — if the two methods agree more on SE than
   on health, that is worth a paragraph. Link it to Limitations.

### 7.4 CLI demo (A)

`cli/predict.py` with `argparse`:

- `--domain se|health` — selects the trained model
- `--model` (default: best for that domain; choices = files in `models/{domain}/`)
- `--compare` — **runs both domains' best models on the same input, side by side**
- `--text "..."` or `--file inputs.txt`
- `--top-k` highlighted words (default 8)
- `--explainer lime|coef`
- `--list-models` — prints available models with test macro-F1 from B's metrics JSON

Print the predicted label and all three class probabilities. Colour each word with
`colorama`: red toward negative, green toward positive, grey neutral; show weights
for the top-k. Handle missing model file, empty text, unknown option. README usage
section plus 5 screenshots: SE jargon-neutral, SE negative, health relief-positive,
health complaint, **and one `--compare` run showing the cross-domain failure**.

### 7.5 Notebook demo (B)

`notebooks/demo.ipynb`: Setup → Load Models → Interactive Prediction →
**Cross-Domain Comparison** → Example Gallery → Results Summary.

`ipywidgets`: text area, domain dropdown, model dropdown, explainer toggle
(LIME / SHAP), Predict button. On click: probability bar chart plus the sentence
rendered as HTML with per-word background colouring (red negative, green positive,
intensity = weight).

**Cross-domain panel:** one input, both domains' models, two highlight renderings
side by side. **This is the centrepiece.** Example gallery: 8 preloaded buttons —
2 SE jargon-neutral, 2 SE negative, 2 health relief-positive, 2 health complaint.
Final cells: master results table, transfer heatmap, both stress scatter plots.

Must run top-to-bottom under "Restart and Run All" on a fresh environment; export
static HTML for submission.

**Both demos must let the user switch the trained model while keeping the input
fixed.** That switch is the whole point: the same sentence, two models, two
different answers, with highlighted words explaining each. It is what makes the
project legible to a non-CS faculty member, and it should be the first thing shown
in the presentation.

### 7.6 Master results table (B)

1. Combine all rows (4 baselines + 18 TF-IDF + 24 embedding) sorted by macro-F1,
   with a `domain` column.
2. Three charts: macro-F1 by representation **grouped by domain**; generative vs
   discriminative average per representation per domain; baseline vs best per domain.
3. **Key question to answer explicitly:** does the same representation win in both
   domains? If TF-IDF (1,3) wins on SE but GloVe wins on HEALTH, that is a result
   about compound technical phrases, not a nuisance.

### 7.7 Integration testing (split evenly)

A runs B's notebook on a fresh clone; B runs A's CLI on a fresh clone. File issues;
the owner fixes. Verify every model in both domain folders loads and works in both
demos.

### 7.8 Final report

| Section | Owner |
|---|---|
| Abstract | Joint, written last |
| Introduction and motivation (domain shift, both domains) | A |
| Related work | A (SE sentiment) + B (health/clinical sentiment) |
| Stage 1–2: Datasets — Senti4SD | A |
| Stage 1–2: Datasets — Druglib construction, binning, validation | B |
| Stage 2: Clean-up rules and the two order deviations | A |
| **Stage 3: Preprocessing and the normalization ablation** | A (module) + B (ablation table) |
| Stage 6: Evaluation methodology, baselines | B |
| Stage 4–5: TF-IDF experiments, both domains | A |
| Stage 4–5: Embedding experiments, both domains | B |
| **Stage 7: Cross-domain transfer results** | A (TF-IDF half) + B (embedding half), joint discussion |
| Generative vs discriminative comparison | B |
| LIME analysis | A |
| SHAP analysis and bias scores | B |
| Stress-test results | B |
| Demo description | A (CLI) + B (notebook) |
| Discussion of expected outcomes | A: 1, 4 · B: 2, 3 · both review |
| Limitations | Joint |
| Conclusion and future work | Joint |

**Limitations must include:** small corpora in both domains; Word2Vec quality at ~4k
documents; LIME/SHAP instability with numbers from SE-6.5 and 7.3; stress-set size
and author bias; **the label-provenance asymmetry** — Senti4SD labels are
human-annotated by three raters with reported agreement (κ = 0.759), while health
labels are rating-derived plus hand-labelled neutrals, state the validation κ from
H-2.4 and be upfront that this is the project's main methodological weakness; **and
the sequential track order** — SE-first means the SE pipeline was designed before the
health corpus was fully seen, so any health-specific need discovered late was
absorbed into shared code rather than given its own treatment. An examiner will find
these; find them first.

### 7.9 Presentation

1. **Open with the cross-domain demo**, not with methodology. One sentence, two
   models, two answers. Thirty seconds, and the non-CS members of the audience
   understand the entire project.
2. Split: **A** presents the problem, both datasets, preprocessing, TF-IDF, LIME,
   CLI. **B** presents embeddings, model comparison, the transfer matrix, SHAP,
   stress tests, notebook.
3. The nearest-neighbour table (SE-4.2 / H-4) and the transfer heatmap (7.1) are the
   two strongest slides.
4. Rehearse twice; each member must be able to answer questions about the other's
   half.

---

## 4. Timeline (8 Weeks)

| Week | Track | Member A | Member B | Checkpoint |
|---|---|---|---|---|
| 1 | SE | Stage 1–2: Senti4SD extraction, `clean_text.py` | Stage 1: Druglib download; Stage 3: `splits.py`, `evaluate.py` | Both **done** — corpus frozen, κ = 0.759, splits frozen, baselines in |
| 2 | SE | Stage 3: `normalize.py` + tests | Stage 3: normalization ablation; **health hand-labelling starts (background)** | Normalization mode chosen and recorded |
| 3 | SE | Stage 4: TF-IDF features + vocabulary analysis | Stage 4: GloVe + SE Word2Vec, OOV table, neighbour table | Feature matrices saved for SE |
| 4 | SE | Stage 5: 9 TF-IDF models + tuning | Stage 5: 12 embedding models + tuning | 21 SE models saved |
| 5 | SE | Stage 6: LIME, SE stress set, error analysis | Stage 6: test eval, SHAP, bias score, stress run | **SE GATE** |
| 6 | HEALTH | Stage 2–4: health TF-IDF features | Stage 2–3: finish corpus, validation κ, regenerate splits; Stage 4: health embeddings | Health corpus frozen; **HEALTH corpus gate** |
| 7 | HEALTH | Stage 5–6: health TF-IDF models, LIME | Stage 5–6: health embedding models, SHAP, bias probe, stress | **HEALTH GATE** + smoke transfer |
| 8 | JOINT | Stage 7: `transfer.py`, CLI, report sections | Stage 7: transfer runs, notebook, master table, report sections | **GATE 2**, final report + slides |

Week 6–7 is deliberately tight and that is the design: by then the pipeline is code
that already runs, and the health track is configuration plus hand-labelled data
that was prepared in the background from Week 2.

---

## 5. Estimated Workload

| Stage | Member A (h) | Member B (h) |
|---|---|---|
| Setup (shared) | 3 | 3 |
| SE-1 Acquisition | 2 | 2 |
| SE-2 Extraction & clean-up | 7 | 3 |
| SE-3 Preprocessing (normalize + ablation + splits) | 7 | 8 |
| SE-4 Feature engineering | 8 | 9 |
| SE-5 Model building | 9 | 9 |
| SE-6 Evaluation (metrics, LIME/SHAP, stress) | 13 | 15 |
| H-1 Acquisition | 0 | 2 |
| H-2 Extraction, labelling, validation | 4 | 16 |
| H-3 Preprocessing | 1 | 3 |
| H-4 Feature engineering | 4 | 5 |
| H-5 Model building | 4 | 4 |
| H-6 Evaluation | 7 | 10 |
| Stage 7 Integration, demos, report, slides | 18 | 16 |
| **Total** | **87** | **105** |

Member B carries more because building a corpus from raw ratings is genuinely harder
than cleaning a pre-annotated one, and B owns the evaluation framework the whole
project depends on. This is partly offset by A owning the n-gram vocabulary analysis
and `normalize.py`.

**If the total runs long:** cut the TF-IDF-weighted embedding variant (SE-4.2) first —
that halves B's model grid from 12 to 6 per domain with minimal loss. Cut `tfidf11`
last; it is the baseline.

---

## 6. Risk Register

| Risk | Detection point | Response |
|---|---|---|
| **SE-first delays cross-domain transfer to Week 8** | Week 7 | Run the smoke transfer the moment the first health model saves (H-5). Do not wait for the full matrix to discover a structural problem. |
| **Health hand-labelling not finished when Week 6 arrives** | Weekly sync from Week 2 | It is the one task started early (Section 2.3). If it slips past Week 5, cut the neutral target from 835 to 300 (the GATE minimum) and downsample negative and positive to match. |
| A health-specific need forces a change to shared code | Any time in the HEALTH track | Make the change, then **re-run the affected SE stage**. Never branch the pipeline per domain. |
| Stemming destroys the relief-inversion signal | Stage SE-3.2 ablation | Expected and *measured*, not guessed. The ablation table decides; if stemming wins on macro-F1 but worsens the confusion rate, report the trade-off and pick by confusion rate. |
| Druglib binning disagrees with human judgement | Stage H-2.4 | Adjust thresholds to 1–2 / 9–10; if still poor, switch to Drugs.com (id 462) sampled to 4k. Decide the week the track opens. |
| Too few usable neutral health rows | Stage H-2.3 | Widen the imperative/dosage prefilter (`--neutral-min-words 8` raises the pool from 1,343 to 1,543); accept a smaller neutral class and downsample the others to match. |
| The 15-word filter fights the neutral class | Stage H-2.2 | Known: it removed *"Take pill once a day"* — exactly the absence-of-affect text the neutral class needs. `--neutral-min-words 8` buys ~200 neutrals at the cost of a length confound the model can learn instead of sentiment. `mean_words_per_class` is logged so the confound is measurable. **Undecided — belongs in the report either way.** |
| Cross-domain transfer shows no degradation | GATE 2, Week 8 | Treat as a bug: check leakage, label mapping, test-set construction before reporting. |
| Cross-domain TF-IDF collapses entirely through OOV | Week 8 | Expected — report OOV rate alongside; the embedding contrast is the real result. |
| Word2Vec quality poor in both domains | Stage SE-4.2 | Expected at ~4k documents; report as a controlled limitation, not a failure. Both domains are limited equally, which makes it a fair comparison. |
| Inter-annotator κ needs two people | Throughout | Working solo, only rating-agreement can be reported; `validate_health` says so rather than inventing a number. State plainly in Limitations. |
| Environment drift between members | Any time | `.venv` currently has only pandas 3.0.6 installed against a pinned 2.2.3, and `tests/test_evaluate.py` cannot be collected. Fix before Stage SE-3 and add a CI check that `pytest` collects cleanly. |

---

## 7. Collaboration Rules

1. Short sync twice a week (15 minutes): done, next, blocked.
2. Never push directly to `main`; every PR reviewed by the other member.
3. Branch naming: `a/features-tfidf`, `b/preprocessing-ablation` — member prefix, then stage folder.
4. **Never change** frozen splits, label order, the seed, function signatures, the
   chosen normalization mode, or either corpus after its gate, without agreement.
5. Log every experiment (config, domain, date, result) in
   `results/metrics/all_results.csv`.
6. **Any processing step applied to one domain must be applied to the other**, or
   documented as a deliberate, justified exception.
7. **Do not start HEALTH work early.** The only sanctioned exception is the neutral
   hand-labelling (Section 2.3). Starting health modelling before the SE GATE
   reintroduces the exact problem this restructure removed.
8. Commit history on GitHub serves as evidence of equal contribution.

---

## 8. Final Deliverables Checklist

| Deliverable | Owner | Stage |
|---|---|---|
| `download.py` + both raw corpora | Joint | 1 |
| `build_se.py` + aligned SE corpus + Fleiss' κ | A | SE-2 |
| `clean_text.py` + 36 tests + rule table | A | SE-2 |
| `normalize.py` + tests (segmentation, stemming, lemmatization) | A | SE-3 |
| **Normalization ablation table (none / lemma / stem)** | B | SE-3 |
| `splits.py` + frozen splits, both domains | B | SE-3 / H-3 |
| `evaluate.py` + tests + 4 baseline rows | B | SE-6 |
| TF-IDF features + 18 models (both domains) | A | 4–5 |
| Embedding features + 24 models (both domains) | B | 4–5 |
| `build_health.py` + health corpus + bias probe | B | H-2 |
| Label validation (κ, rating agreement) | B, annotated by both | H-2 |
| LIME explanations, aggregation, stability (both domains) | A | 6 |
| SHAP explanations, global plots, bias scores, bias probe | B | 6 |
| SE jargon lexicon | A | 6 |
| Health symptom lexicon | B | 6 |
| 60 SE stress sentences | A | SE-6 |
| 60 health stress sentences + evaluation harness | B | H-6 |
| `transfer.py` + transfer matrix + OOV-adjusted analysis | A writes, both run | 7 |
| LIME vs SHAP agreement analysis | A + B | 7 |
| CLI demo with `--compare` | A | 7 |
| Notebook demo with cross-domain panel | B | 7 |
| Final report + slides | A + B | 7 |

---

## 9. Data Licence

The Druglib donors require research-only use, no redistribution, and citation. Raw
Druglib files are git-ignored; `src/acquisition/download.py` is committed instead.

> Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-Based
> Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.
> *Proceedings of the 2018 International Conference on Digital Health*, 121–125.

Senti4SD: Calefato, F., Lanubile, F., Maiorano, F., & Novielli, N. (2018). Sentiment
Polarity Detection for Software Development. *Empirical Software Engineering*,
23(3), 1352–1382.

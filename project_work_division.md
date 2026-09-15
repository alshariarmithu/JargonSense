# Work Division Plan: Interpretable Sentiment Classification in SE Communication

**Course:** CSE 4122
**Member A:** Al Shariar Hossain (Roll 2107066)
**Member B:** Hassan Mohammed Naquibul Hoque (Roll 2107077)

---

## 1. How the Work Is Split

The project is divided along two parallel "tracks" that mirror each other, so both members do the same kind of work (features + models + explainability + demo + report) on different parts of the system. A small shared phase at the start and a joint integration phase at the end make sure the two halves fit together.

| Area | Member A (Al Shariar, 2107066) | Member B (Hassan, 2107077) |
|---|---|---|
| Foundation | Data loading, cleaning, preprocessing module | Fixed data splits, evaluation framework |
| Features | TF-IDF: unigram, (1,2), (1,3) | GloVe (pretrained) + Word2Vec (self-trained) |
| Models | Multinomial NB, Logistic Regression, Linear SVM on TF-IDF | Gaussian NB, Logistic Regression, Linear SVM on embeddings |
| Explainability | LIME (per-instance + aggregated + stability check) | SHAP (per-instance + global + lexical bias score) |
| Stress test | Write 60 harsh-but-neutral sentences; annotate B's set | Write 60 genuinely negative sentences; annotate A's set; run evaluation |
| Demo | CLI tool (terminal, colored word highlights) | Jupyter notebook demo (widgets, HTML highlights) |
| Report | Intro, dataset, TF-IDF results, LIME analysis | Embedding results, SHAP analysis, stress-test results |
| Joint | Environment setup, LIME vs SHAP comparison, discussion, slides | Environment setup, LIME vs SHAP comparison, discussion, slides |

### Estimated Workload

| Phase | Member A (hours) | Member B (hours) |
|---|---|---|
| Phase 0: Shared setup | 3 | 3 |
| Phase 1: Foundation | 8 | 8 |
| Phase 2: Features | 8 | 9 |
| Phase 3: Models + tuning | 9 | 9 |
| Phase 4: Explainability | 12 | 12 |
| Phase 5: Stress test | 7 | 8 |
| Phase 6: Demo | 7 | 7 |
| Phase 7: Integration + report | 12 | 12 |
| **Total** | **66** | **68** |

---

## 2. Suggested Timeline (8 Weeks)

| Week | Member A | Member B | Checkpoint |
|---|---|---|---|
| 1 | Phase 0 (joint), A1 preprocessing | Phase 0 (joint), B1 splits + evaluation | Splits frozen, `preprocess()` and `evaluate()` working |
| 2 | A2 TF-IDF features | B2 GloVe + Word2Vec features | Feature matrices saved |
| 3 | A3 TF-IDF models + tuning | B3 embedding models + tuning | All 21 experiment rows in results table |
| 4 | A4 LIME | B4 SHAP | Per-instance explanations for same 20 test examples |
| 5 | A4 aggregation + stability | B4 global analysis + bias score | Bias tables for both methods |
| 6 | A5 stress sentences + annotation | B5 stress sentences + annotation + evaluation | Stress-test results |
| 7 | A6 CLI demo | B6 notebook demo | Both demos run on all saved models |
| 8 | Phase 7 joint + own report sections | Phase 7 joint + own report sections | Final report + slides |

---

## 3. Shared Contracts (Agree on These First)

Because both members work in parallel, these interfaces must be fixed in Week 1 and not changed without telling the other person.

### 3.1 Repository Structure

```
senti-se/
  data/
    raw/                  # original Senti4SD files (never edited)
    processed/            # train.csv, val.csv, test.csv
    stress_test/          # stress_test.csv
  src/
    preprocess.py         # Member A
    splits.py             # Member B
    evaluate.py           # Member B
    features_tfidf.py     # Member A
    features_embed.py     # Member B
    models_tfidf.py       # Member A
    models_embed.py       # Member B
    explain_lime.py       # Member A
    explain_shap.py       # Member B
    stress_test.py        # Member B (sentences from both)
  cli/predict.py          # Member A
  notebooks/demo.ipynb    # Member B
  models/                 # saved .joblib pipelines
  results/
    metrics/  figures/  explanations/
  report/
  requirements.txt
  README.md
```

### 3.2 Function Signatures

- `preprocess(text: str) -> str` in `src/preprocess.py` (A) — used by everyone.
- `load_splits() -> (train_df, val_df, test_df)` in `src/splits.py` (B) — columns: `id`, `text`, `label`.
- `evaluate(name, y_true, y_pred, out_dir="results/metrics") -> dict` in `src/evaluate.py` (B) — returns macro-F1, per-class P/R/F1, saves confusion matrix PNG and JSON.
- **Every saved model must be a full pipeline** with `predict(list_of_raw_strings)` and `predict_proba(list_of_raw_strings)`. This is required by LIME, SHAP, the CLI, and the notebook.
- Model file naming: `models/{features}_{classifier}.joblib`, e.g. `tfidf13_lr.joblib`, `glove_svm.joblib`.
- Label order everywhere: `["negative", "neutral", "positive"]`.
- Random seed everywhere: `42`.

---

## Phase 0 — Shared Setup (Both Members, Together)

### Task 0.1: Environment and Repository

1. Create a GitHub repository; add both members as collaborators.
2. Create the folder structure from Section 3.1.
3. Create a virtual environment (Python 3.10 or 3.11).
4. Install and pin packages in `requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `gensim`, `lime`, `shap`, `matplotlib`, `seaborn`, `nltk`, `joblib`, `colorama`, `ipywidgets`, `jupyter`, `scipy`.
5. Add `.gitignore` for `models/`, large embedding files, `__pycache__`, `.ipynb_checkpoints`.
6. Agree on a branch workflow: each member works on their own branch (`a/...`, `b/...`) and opens a pull request that the other reviews before merging.

### Task 0.2: Download and Inspect the Dataset

1. Clone `collab-uniba/Senti4SD` and locate the gold-standard labeled CSV (check the README for the exact file name and column names).
2. Copy it into `data/raw/` unchanged.
3. Together, read 30 random rows from each class to get a feel for the data.
4. Record: total rows, class counts, average text length, and 5 example posts containing SE jargon (e.g., "crash", "kill", "fatal", "exception").
5. Write these numbers into `README.md` — they go into the report later.

**Done when:** both members can run `python -c "import pandas; pandas.read_csv('data/raw/...')"` from the same repo.

---

## MEMBER A — Al Shariar Hossain (2107066)

### Task A1: Data Cleaning and Preprocessing Module

**Goal:** a single `preprocess()` function that every model uses.

#### A1.1 Data cleaning
1. Load the raw CSV; standardize column names to `id`, `text`, `label`.
2. Map labels to lowercase strings `negative`, `neutral`, `positive`.
3. Remove exact duplicate texts (keep first); log how many were removed.
4. Drop rows with empty or whitespace-only text.
5. Save the cleaned file to `data/processed/clean.csv` and hand it to Member B for splitting.

#### A1.2 Preprocessing function
1. Decode HTML entities (`&amp;` to `&`, etc.) with `html.unescape`.
2. Replace URLs with the token `URL` (regex `https?://\S+|www\.\S+`).
3. Replace inline code or code-looking spans (backticks, `foo()`, `a.b.c`) with `CODE` — decide rules together with B and document them.
4. Replace user mentions (`@name`) with `USER`.
5. Replace standalone numbers with `NUM`.
6. Lowercase everything.
7. Expand negation contractions: `can't` to `can not`, `n't` to ` not` — **do not remove negation words**, they flip sentiment.
8. Keep emoticons such as `:)`, `:(`, `:D` as tokens (map them to `EMO_POS` / `EMO_NEG`).
9. Collapse repeated characters (`sooooo` to `soo`) and repeated punctuation (`!!!!` to `!!`).
10. **Do not** remove stopwords by default and **do not** stem — words like "kill" vs "killed" matter for the analysis. (Optionally test stopword removal later as an ablation.)
11. Tokenizer helper: `tokenize(text) -> list[str]` using a regex that keeps `!`, `?`, and emoticon tokens.

#### A1.3 Testing and documentation
1. Write 10 unit tests (`tests/test_preprocess.py`) with input/expected-output pairs, including the jargon sentences "The process was killed" and "Fatal error on line 3".
2. Add a docstring listing every rule in order.
3. Show 10 before/after examples in a markdown table for the report.

**Deliverables:** `src/preprocess.py`, `tests/test_preprocess.py`, `data/processed/clean.csv`.

---

### Task A2: TF-IDF Feature Engineering

**Goal:** three TF-IDF representations that capture compound technical phrases.

#### A2.1 Build vectorizers
1. Load splits using B's `load_splits()`.
2. Create three `TfidfVectorizer` configurations with `preprocessor=preprocess`:
   - `tfidf11`: `ngram_range=(1,1)` — baseline
   - `tfidf12`: `ngram_range=(1,2)`
   - `tfidf13`: `ngram_range=(1,3)`
3. Shared settings: `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `token_pattern` matching the A1 tokenizer.
4. **Fit on training data only**; transform val and test (avoids data leakage).

#### A2.2 Analyze the vocabularies
1. Record vocabulary size and matrix sparsity for each configuration.
2. List the top 30 n-grams by mean TF-IDF weight per class.
3. Check whether compound phrases like `fatal error`, `kill the process`, `null pointer exception` exist in the (1,2) and (1,3) vocabularies; count how often each appears per class.
4. Save a table: phrase, count in negative, count in neutral, count in positive.

#### A2.3 Tune feature settings (light)
1. Using LR with default C, try `min_df` in {1, 2, 5} and `max_features` in {None, 20000, 50000} on the validation set.
2. Keep the best settings fixed for all TF-IDF experiments; record the choice.

**Deliverables:** `src/features_tfidf.py`, vocabulary stats table, jargon-phrase frequency table.

---

### Task A3: Models on TF-IDF Features

**Goal:** 9 experiment rows (3 feature sets x 3 classifiers).

#### A3.1 Build pipelines
For each of `tfidf11`, `tfidf12`, `tfidf13`, build `sklearn.pipeline.Pipeline([("tfidf", ...), ("clf", ...)])` with:
1. `MultinomialNB` (generative).
2. `LogisticRegression(max_iter=2000, solver="saga" or "lbfgs", multi_class="auto")` (discriminative).
3. `LinearSVC` wrapped in `CalibratedClassifierCV(cv=5)` — **LinearSVC has no `predict_proba`**, which LIME and the demos need.

#### A3.2 Hyperparameter tuning
1. Use the fixed validation set (or `GridSearchCV` with stratified 5-fold on train) with `scoring="f1_macro"`.
2. Grids:
   - MNB: `alpha` in {0.01, 0.1, 0.5, 1.0}
   - LR: `C` in {0.01, 0.1, 1, 10, 100}, `class_weight` in {None, "balanced"}
   - SVM: `C` in {0.01, 0.1, 1, 10}, `class_weight` in {None, "balanced"}
3. Record the best parameters for all 9 combinations.

#### A3.3 Final training and evaluation
1. Retrain each best configuration on train (+ val if the team agrees; be consistent with B).
2. Evaluate once on the test set using B's `evaluate()`.
3. Save each pipeline to `models/tfidf1X_{mnb|lr|svm}.joblib`.
4. Save predictions to `results/metrics/tfidf_predictions.csv` (id, text, gold, pred per model) — needed for error analysis.

#### A3.4 Error analysis
1. From the best TF-IDF model, collect all test cases where gold = neutral but predicted = negative.
2. Manually tag each: "contains SE jargon" / "sarcasm" / "annotation noise" / "other".
3. Report counts and 5 representative examples.
4. For LR, list the top 20 positive-weight features for the negative class (`clf.coef_`) and highlight any technical terms.

**Deliverables:** `src/models_tfidf.py`, 9 saved models, results table, error-analysis table.

---

### Task A4: LIME Explainability

**Goal:** explain individual predictions and aggregate them to detect lexical bias.

#### A4.1 Per-instance explanations
1. Create `LimeTextExplainer(class_names=["negative","neutral","positive"], split_expression=<A1 tokenizer regex>, random_state=42)`.
2. Agree with B on a fixed list of **20 test instances** (both explain the same ones): 5 correct negatives, 5 correct neutrals, 5 neutral-predicted-as-negative errors, 5 jargon-heavy neutrals.
3. For each instance and for the best TF-IDF model and best embedding model (load B's saved pipeline), run `explain_instance(text, pipeline.predict_proba, num_features=10, num_samples=2000, labels=[0,1,2])`.
4. Save each as HTML (`exp.save_to_file`) to `results/explanations/lime/`.
5. Write a short observation (2–3 sentences) for each of the 20 cases.

#### A4.2 Aggregated explanations over the test set
1. Run LIME on the whole test set (or a stratified sample of at least 300 if runtime is too long) for the best TF-IDF model and the best embedding model.
2. For each token, accumulate: total weight toward negative, number of occurrences, mean weight.
3. Build a **SE jargon lexicon** (shared with B): `fatal, error, kill, killed, crash, crashed, abort, exception, fail, failed, failure, dead, deadlock, bug, broken, hang, panic, terminate, execute, garbage, deprecated, warning, exploit, attack, destroy`.
4. Compute, for each jargon word, its mean LIME weight toward the negative class **only on gold-neutral instances**. A strong positive value means the model treats jargon as hostility.
5. Plot a horizontal bar chart of the top 25 negative-driving tokens overall, coloring jargon words differently.

#### A4.3 Stability check (addresses the Limitations section)
1. For the 20 fixed instances, run LIME with 5 different `random_state` values.
2. Compute the Jaccard overlap of the top-5 and top-10 token sets across runs.
3. Repeat with `num_samples` in {500, 2000, 5000} to show how stability changes.
4. Report mean and standard deviation of overlap in a table.

**Deliverables:** `src/explain_lime.py`, 20 HTML explanations, aggregated bias table + bar chart, stability table.

---

### Task A5: Stress Test — Harsh-but-Neutral Sentences

1. Write **60 sentences** that sound harsh but are functionally neutral, e.g.:
   - "Kill the process before restarting the server."
   - "The build failed with a fatal error on line 42."
   - "Garbage collection destroys unused objects."
   - "The thread hangs until the deadlock is resolved."
2. Cover at least 20 different jargon words from the shared lexicon (max 5 sentences per word).
3. Vary length (5–30 words) and style (question, instruction, report).
4. Write about **10 minimal pairs**: a neutral jargon sentence and a lightly edited negative version, e.g., "The app crashed after the update." vs. "The app crashed again, this update is useless."
5. Save to `data/stress_test/a_sentences.csv` with columns `id, text, intended_label, jargon_word, pair_id`.
6. **Blind annotation:** label all of B's 60 sentences without seeing B's intended labels (see B5.2).

**Deliverables:** 60 sentences + labels for B's set.

---

### Task A6: CLI Demo

1. Create `cli/predict.py` using `argparse` with options:
   - `--model` (default: best model; choices = all files in `models/`)
   - `--text "..."` for a single input, or `--file inputs.txt` for batch mode
   - `--top-k` number of highlighted words (default 8)
   - `--explainer lime|coef` (coef = fast linear weights for TF-IDF LR)
2. Load the pipeline with `joblib.load`.
3. Print predicted label and probability for all three classes.
4. Run LIME (or read LR coefficients) and print the original sentence with each word colored using `colorama`: red for pushing toward negative, green toward positive, gray neutral; show the weight next to the top-k words.
5. Add a `--list-models` flag printing available models and their test macro-F1 (read from B's metrics JSON).
6. Handle errors: missing model file, empty text, unknown option.
7. Add usage examples to `README.md` and take 3 screenshots (jargon neutral, genuinely negative, positive) for the report.

**Deliverables:** `cli/predict.py`, README section, screenshots.

---

### Task A7: Report Sections (Member A)

1. **Introduction and motivation** — domain shift problem, examples of jargon.
2. **Dataset and preprocessing** — statistics from Task 0.2, cleaning steps, preprocessing rules table.
3. **TF-IDF experiments** — feature stats, 9-row results, n-gram effect, top LR features.
4. **LIME analysis** — per-instance examples, aggregated bias chart, stability results.

---

## MEMBER B — Hassan Mohammed Naquibul Hoque (2107077)

### Task B1: Data Splits and Evaluation Framework

**Goal:** identical splits and identical metrics for everyone.

#### B1.1 Splits
1. Take `data/processed/clean.csv` from A.
2. Use `train_test_split` twice with `stratify=label`, `random_state=42` to produce **70% train / 15% val / 15% test**.
3. Verify class proportions are the same in all three splits (print a table).
4. Save `train.csv`, `val.csv`, `test.csv` and commit them — **splits are frozen after this**.
5. Implement `load_splits()` in `src/splits.py`.

#### B1.2 Evaluation module
1. `evaluate(name, y_true, y_pred, out_dir)` computes:
   - Accuracy, macro-F1, weighted-F1
   - Per-class precision, recall, F1 (`classification_report(output_dict=True)`)
   - Confusion matrix (raw counts and row-normalized)
   - **Negative-vs-neutral confusion rate**: fraction of gold-neutral predicted negative, and gold-negative predicted neutral
2. Save a confusion-matrix heatmap (`seaborn.heatmap`) to `results/figures/cm_{name}.png`.
3. Save all numbers to `results/metrics/{name}.json`.
4. `append_to_results_table(name, metrics)` writes one row to `results/metrics/all_results.csv`.
5. Add a **significance test** helper: McNemar's test between two models' predictions (`statsmodels` or manual chi-square) to check whether differences are real.
6. Add a **baseline**: majority-class classifier and a general-purpose lexicon model (e.g., VADER from `nltk`) — the VADER result directly shows the domain-shift problem from the proposal.
7. Write 5 unit tests with hand-computed values.

**Deliverables:** frozen splits, `src/splits.py`, `src/evaluate.py`, baseline rows in the results table.

---

### Task B2: Embedding Features (GloVe vs. Word2Vec)

**Goal:** document vectors from pretrained and domain-trained embeddings.

#### B2.1 Pretrained GloVe
1. Download `glove.6B.100d.txt` (optionally also 300d); load into a dict or via `gensim` `KeyedVectors.load_word2vec_format(..., no_header=True)`.
2. Tokenize text with A's `preprocess()` + `tokenize()`.
3. Document vector = mean of word vectors of in-vocabulary tokens; zero vector if none.
4. Second variant: **TF-IDF-weighted mean** (use IDF values from a unigram TF-IDF fitted on train).
5. Compute **OOV rate** overall and specifically for jargon lexicon words and code-related tokens.

#### B2.2 Self-trained Word2Vec
1. Train `gensim.models.Word2Vec` on **training split texts only** (no leakage):
   `vector_size=100, window=5, min_count=2, sg=1, negative=5, epochs=30, seed=42, workers=1` (workers=1 for reproducibility).
2. Try `vector_size` in {50, 100} and `sg` in {0, 1}; pick by validation macro-F1 with LR.
3. Build mean and TF-IDF-weighted document vectors as in B2.1.
4. Save the model to `models/w2v.model`.

#### B2.3 Qualitative embedding analysis
1. For the words `kill`, `fatal`, `crash`, `error`, `exception`, `hang`, `abort`, list the 10 nearest neighbors in GloVe and in Word2Vec.
2. Put them side-by-side in a table: does GloVe place "kill" near "murder" while Word2Vec places it near "process"/"terminate"?
3. Optionally, a 2D t-SNE/PCA plot of jargon words plus clearly emotional words for both embeddings.
4. Note in the report that the small corpus (~4k posts) limits Word2Vec quality (link to Limitations).

#### B2.4 Wrap as sklearn transformers
1. Implement `MeanEmbeddingVectorizer` and `TfidfEmbeddingVectorizer` classes with `fit`/`transform` so they plug into a `Pipeline` taking raw strings.

**Deliverables:** `src/features_embed.py`, OOV table, nearest-neighbor table, optional plot.

---

### Task B3: Models on Embedding Features

**Goal:** 12 experiment rows (GloVe/W2V x mean/weighted x 3 classifiers).

#### B3.1 Build pipelines
1. **Gaussian NB** — `MultinomialNB` cannot be used because embeddings have negative values; note this in the report as a key generative-model difference.
2. Logistic Regression with `StandardScaler` before it.
3. `LinearSVC` (with `StandardScaler`) wrapped in `CalibratedClassifierCV(cv=5)` for probabilities.

#### B3.2 Hyperparameter tuning
1. Same protocol as A3.2 (`f1_macro`, same validation approach).
2. Grids:
   - GNB: `var_smoothing` in {1e-9, 1e-7, 1e-5, 1e-3}
   - LR: `C` in {0.01, 0.1, 1, 10}, `class_weight` in {None, "balanced"}
   - SVM: `C` in {0.01, 0.1, 1, 10}, `class_weight` in {None, "balanced"}

#### B3.3 Final training and evaluation
1. Retrain best configs, evaluate once on test with `evaluate()`.
2. Save pipelines as `models/{glove|w2v}{mean|wt}_{gnb|lr|svm}.joblib`.
3. Save predictions to `results/metrics/embed_predictions.csv`.

#### B3.4 Error analysis
1. Same procedure as A3.4 on the best embedding model (tag neutral-to-negative errors).
2. Compute the overlap of errors between the best TF-IDF and best embedding models: which errors are shared, which are unique.

#### B3.5 Master results table (for Expected Outcomes 1 and 2)
1. Combine all rows (baselines + 9 TF-IDF + 12 embedding) into one table sorted by macro-F1.
2. Make two summary charts: macro-F1 grouped by representation; generative vs. discriminative average per representation.
3. Run McNemar's test between: best TF-IDF vs. best embedding; best NB vs. best discriminative.

**Deliverables:** `src/models_embed.py`, 12 saved models, master results table, 2 charts, significance results.

---

### Task B4: SHAP Explainability

**Goal:** a second, theoretically grounded explanation method to cross-check LIME.

#### B4.1 Per-instance explanations
1. For **TF-IDF Logistic Regression**: `shap.LinearExplainer(clf, background)` where background = 100–200 transformed training rows; map feature indices back to n-gram names.
2. For **embedding models and SVM** (black-box from the text's point of view): `shap.Explainer(pipeline.predict_proba, shap.maskers.Text(tokenizer_regex))` (Partition explainer), which gives word-level attributions on raw text.
3. Explain the **same 20 fixed instances** agreed with A.
4. Save `shap.plots.text` outputs as HTML and waterfall plots as PNG to `results/explanations/shap/`.
5. Write 2–3 sentences of observations per instance.

#### B4.2 Global analysis
1. Compute SHAP values on the full test set (or stratified sample of 300 for the Partition explainer).
2. Produce a global bar plot (mean |SHAP|) and beeswarm plot for the negative class (TF-IDF LR).
3. Aggregate token-level SHAP values for text-masker models the same way A aggregates LIME weights (sum, count, mean per token).

#### B4.3 Lexical bias score
1. Using the shared jargon lexicon, compute for each word its mean SHAP contribution toward negative **on gold-neutral instances**.
2. Define an overall **Jargon Bias Score** per model = mean of those values over all lexicon words that appear at least 3 times.
3. Compare the bias score across TF-IDF, GloVe, and Word2Vec models in one table and bar chart — this is the core evidence for Expected Outcome 3.

**Deliverables:** `src/explain_shap.py`, 20 explanations, global plots, bias score table + chart.

---

### Task B5: Stress Test — Genuinely Negative Sentences + Evaluation

#### B5.1 Write sentences
1. Write **60 genuinely negative** SE sentences expressing frustration, hostility, or complaint, e.g.:
   - "This documentation is useless and wasted my whole day."
   - "I am sick of this library breaking every single release."
2. Include **20 without any jargon words** and **40 that include jargon** (e.g., "This stupid crash is driving me crazy") — so the test separates jargon from real hostility.
3. Write the negative halves of about 10 minimal pairs, coordinating with A.
4. Save to `data/stress_test/b_sentences.csv` with the same columns as A.

#### B5.2 Blind cross-annotation
1. Shuffle A's and B's sentences separately and remove `intended_label`.
2. A labels B's set, B labels A's set (negative/neutral/positive).
3. Compute Cohen's kappa between the author's intended label and the other member's label (`sklearn.metrics.cohen_kappa_score`).
4. Discuss disagreements together; drop sentences you cannot agree on; record how many were dropped.
5. Merge into `data/stress_test/stress_test.csv`.

#### B5.3 Run the stress test
1. Run all saved models (21 + VADER baseline) on the stress set.
2. Metrics per model:
   - **Jargon false-alarm rate** = % of harsh-but-neutral sentences predicted negative (lower is better)
   - **Negative recall** on genuinely negative sentences (higher is better)
   - Recall on negative-with-jargon vs. negative-without-jargon
   - **Minimal-pair accuracy** = % of pairs where both sentences are classified correctly
3. Plot a scatter: x = jargon false-alarm rate, y = negative recall, one point per model (ideal = top-left).
4. Run LIME (from A's module) and SHAP on the 10 worst-handled stress sentences for the best model.

**Deliverables:** 60 sentences, annotations + kappa, `src/stress_test.py`, stress-test table + scatter plot.

---

### Task B6: Notebook Demo

1. Create `notebooks/demo.ipynb` with sections: Setup, Load Models, Interactive Prediction, Example Gallery, Stress-Test Summary.
2. Interactive widget (`ipywidgets`): text area, model dropdown (all files in `models/`), explainer toggle (LIME / SHAP), "Predict" button.
3. On click: show a probability bar chart for the three classes and the input sentence rendered as HTML with each word's background colored by its contribution (red = negative, green = positive, intensity = weight).
4. Example gallery: preloaded buttons for 6 sentences (2 jargon-neutral, 2 negative, 2 positive).
5. Final cell displays the master results table and the stress-test scatter plot.
6. Make sure it runs top-to-bottom with "Restart and Run All" on a fresh environment; export a static HTML copy for submission.

**Deliverables:** `notebooks/demo.ipynb`, `demo.html` export.

---

### Task B7: Report Sections (Member B)

1. **Evaluation methodology** — splits, metrics, significance test, baselines (VADER).
2. **Embedding experiments** — GloVe vs. Word2Vec, OOV, nearest neighbors, 12-row results.
3. **Generative vs. discriminative comparison** — master table and charts.
4. **SHAP analysis** — per-instance examples, global plots, Jargon Bias Score.
5. **Stress-test results** — construction, kappa, metrics, scatter plot.

---

## Phase 7 — Joint Integration and Final Report (Both Members)

### Task J1: LIME vs. SHAP Agreement (split evenly)
1. **A:** for the 20 fixed instances, export LIME top-10 token rankings in a common CSV format.
2. **B:** export SHAP top-10 rankings in the same format.
3. **Together:** compute Spearman rank correlation and top-5 overlap per instance; discuss where they disagree and why (both are approximations — link to Limitations).

### Task J2: Integration Testing (split evenly)
1. **A** runs B's notebook on a fresh clone; **B** runs A's CLI on a fresh clone.
2. Each files issues for anything broken; the owner fixes it.
3. Verify every saved model loads and works in both demos.

### Task J3: Final Report (Joint Sections)
1. **Results discussion** answering the four Expected Outcomes directly — draft by A for outcomes 1 and 4, draft by B for outcomes 2 and 3; each reviews the other's draft.
2. **Limitations** — small corpus, Word2Vec quality, LIME/SHAP instability (with numbers from A4.3 and J1), stress-test size and author bias.
3. **Conclusion and future work** (e.g., transformer models such as BERTweet or SE-specific BERT).
4. Abstract written together last.
5. Each member proofreads the other's sections.

### Task J4: Presentation
1. Slides split in half: A presents problem, data, TF-IDF, LIME, CLI demo; B presents embeddings, model comparison, SHAP, stress test, notebook demo.
2. Rehearse together twice; each member should be able to answer questions about the other's half.

---

## 4. Final Deliverables Checklist

| Deliverable | Owner |
|---|---|
| `src/preprocess.py` + tests | A |
| `src/splits.py`, `src/evaluate.py` + tests | B |
| TF-IDF features + 9 models | A |
| Embedding features + 12 models | B |
| LIME explanations, aggregation, stability | A |
| SHAP explanations, global plots, bias score | B |
| 60 harsh-but-neutral stress sentences | A |
| 60 genuinely negative stress sentences + evaluation | B |
| CLI demo | A |
| Notebook demo | B |
| LIME vs SHAP agreement analysis | A + B |
| Final report + slides | A + B |

## 5. Collaboration Rules

1. Short sync twice a week (15 minutes): what was done, what is next, any blocked items.
2. Never push directly to `main`; every PR is reviewed by the other member.
3. Never change frozen splits, label order, the seed, or function signatures without agreement.
4. Log every experiment (config, date, result) in `results/metrics/all_results.csv` so nothing is lost.
5. Commit history on GitHub serves as evidence of equal contribution.

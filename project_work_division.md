# Work Division Plan: Interpretable Sentiment Classification Across Two Technical Domains

**Course:** CSE 4122
**Member A:** Al Shariar Hossain (Roll 2107066)
**Member B:** Hassan Mohammed Naquibul Hoque (Roll 2107077)

---

## 0. What Changed and Why

The project now runs on **two corpora instead of one**:

| Domain | Corpus | Size | Vocabulary problem |
|---|---|---|---|
| Software engineering (SE) | Senti4SD gold standard (StackOverflow) | ~4,400 | Harsh technical jargon that is functionally neutral: *fatal error*, *kill the process*, *crashed* |
| Healthcare (HEALTH) | UCI Drug Reviews (Druglib.com), id 461 | ~4,100 raw | Relief inversion: clinically negative words naming a symptom that went away — *the nausea stopped*, *no more panic attacks* |

**This is not a demo add-on.** The thesis is no longer "SE vocabulary breaks sentiment models" but **"domain-specific vocabulary breaks sentiment models, and the failure mode generalises."** Two independent domains, one shared pipeline, plus cross-domain transfer experiments. A finding confirmed in two unrelated corpora is a result; the same finding in one corpus is an anecdote.

Secondary benefit, and the reason the teacher asked: the health corpus is legible to any non-CS reader. They can type a sentence, judge the correct answer themselves, and evaluate whether the highlighted words make sense — which is impossible with StackOverflow text unless you are already a developer.

**Division principle.** The method split from the original plan is kept (A = TF-IDF + LIME, B = embeddings + SHAP). Both members now run their method on **both** corpora. The pipeline is written domain-agnostic once, and the corpus becomes a command-line argument. Nobody writes anything twice.

**Dataset ownership** is the one thing split by domain: A owns Senti4SD preparation, B owns Druglib construction. After Week 1 both corpora are frozen and shared.

---

## 1. How the Work Is Split

| Area | Member A (Al Shariar, 2107066) | Member B (Hassan, 2107077) |
|---|---|---|
| Corpus ownership | Senti4SD cleaning and alignment | Druglib construction, binning, neutral labelling |
| Foundation | Preprocessing module (domain-aware) | Fixed splits (both domains), evaluation framework |
| Features | TF-IDF: unigram, (1,2), (1,3) — both corpora | GloVe + Word2Vec (self-trained per domain) |
| Models | MNB, LR, Linear SVM on TF-IDF — both corpora | GNB, LR, Linear SVM on embeddings — both corpora |
| Cross-domain | Runs SE→HEALTH transfer for TF-IDF models | Runs HEALTH→SE transfer for embedding models |
| Explainability | LIME (per-instance, aggregated, stability) | SHAP (per-instance, global, bias score) |
| Lexicons | SE jargon lexicon | Health symptom lexicon |
| Stress test | SE set: 40 harsh-but-neutral + 20 negative | HEALTH set: 40 relief-framed + 20 complaint-framed; runs evaluation for both |
| Demo | CLI tool with `--domain` switch | Notebook demo with side-by-side domain comparison |
| Report | Intro, datasets, TF-IDF results, LIME analysis | Evaluation method, embedding results, SHAP analysis, stress tests |
| Joint | Setup, cross-domain discussion, LIME vs SHAP, slides | Setup, cross-domain discussion, LIME vs SHAP, slides |

### Estimated Workload

| Phase | Member A (hours) | Member B (hours) |
|---|---|---|
| Phase 0: Shared setup | 3 | 3 |
| Phase 1: Corpus construction | 9 | 14 |
| Phase 2: Foundation (preprocess / splits / evaluate) | 8 | 8 |
| Phase 3: Features (both domains) | 10 | 11 |
| Phase 4: Models + tuning (both domains) | 12 | 12 |
| Phase 5: Cross-domain transfer | 6 | 6 |
| Phase 6: Explainability (both domains) | 14 | 14 |
| Phase 7: Stress tests | 8 | 10 |
| Phase 8: Demo | 7 | 7 |
| Phase 9: Integration + report | 13 | 13 |
| **Total** | **90** | **98** |

Member B carries more in Phase 1 because building a corpus from raw ratings is genuinely harder than cleaning a pre-annotated one. This is offset by A owning the heavier n-gram vocabulary analysis in Phase 3. **If the total runs long, cut the TF-IDF-weighted embedding variant (Section 6.2, B3) — that halves B's model grid from 12 to 6 per domain with minimal loss.**

---

## 2. Suggested Timeline (8 Weeks)

| Week | Member A | Member B | Checkpoint |
|---|---|---|---|
| 1 | Phase 0 joint; Senti4SD cleaning; hand-label 50 Druglib rows for B's validation | Phase 0 joint; Druglib filtering, binning, neutral labelling; hand-label 50 rows | **GATE 1** — both corpora frozen in `id;text;polarity` form; annotator agreement ≥ 0.75 |
| 2 | A2 preprocessing module | B1 splits + evaluation + baselines | `preprocess()`, `load_splits(domain)`, `evaluate()` working on both domains |
| 3 | A3 TF-IDF features, both domains | B2 GloVe + Word2Vec, both domains | Feature matrices saved for both corpora |
| 4 | A4 TF-IDF models + tuning | B3 embedding models + tuning | 42 in-domain experiment rows complete |
| 5 | A5 cross-domain transfer (TF-IDF) | B4 cross-domain transfer (embeddings) | **GATE 2** — transfer matrix complete; degradation confirmed |
| 6 | A6 LIME, both domains | B5 SHAP, both domains | Bias tables for both methods, both domains |
| 7 | A7 SE stress set + annotation; A8 CLI demo | B6 HEALTH stress set + annotation + evaluation; B7 notebook demo | Stress results; both demos run on all saved models |
| 8 | Phase 9 joint + own report sections | Phase 9 joint + own report sections | Final report + slides |

---

## 3. Shared Contracts (Agree on These First)

Both members work in parallel, so these interfaces are fixed in Week 1 and not changed without telling the other person.

### 3.1 Repository Structure

```
senti-crossdomain/
  data/
    raw/
      senti4sd/           # original gold standard (never edited)
      druglib/            # original .tsv files (never edited)
    processed/
      se/                 # train.csv, val.csv, test.csv
      health/             # train.csv, val.csv, test.csv
    stress_test/
      se_stress.csv
      health_stress.csv
    lexicons/
      se_jargon.txt       # Member A
      health_symptoms.txt # Member B
  src/
    build_se.py           # Member A
    build_health.py       # Member B
    preprocess.py         # Member A
    splits.py             # Member B
    evaluate.py           # Member B
    features_tfidf.py     # Member A
    features_embed.py     # Member B
    models_tfidf.py       # Member A
    models_embed.py       # Member B
    transfer.py           # Joint (A writes, B reviews)
    explain_lime.py       # Member A
    explain_shap.py       # Member B
    stress_test.py        # Member B
  cli/predict.py          # Member A
  notebooks/demo.ipynb    # Member B
  models/
    se/  health/          # saved .joblib pipelines, one folder per domain
  results/
    metrics/  figures/  explanations/  transfer/
  report/
  requirements.txt
  README.md
```

### 3.2 Function Signatures

- `preprocess(text: str, domain: str) -> str` in `src/preprocess.py` (A). The `domain` argument is `"se"` or `"health"` and controls only domain-specific rules (code spans for SE, dosage patterns for HEALTH). **Everything else is identical across domains** — differing preprocessing would confound every comparison.
- `load_splits(domain: str) -> (train_df, val_df, test_df)` in `src/splits.py` (B) — columns: `id`, `text`, `label`.
- `evaluate(name, domain, y_true, y_pred, out_dir="results/metrics") -> dict` in `src/evaluate.py` (B).
- **Every saved model must be a full pipeline** with `predict(list_of_raw_strings)` and `predict_proba(list_of_raw_strings)`. Required by LIME, SHAP, the CLI, the notebook, and the transfer experiments.
- Model naming: `models/{domain}/{features}_{classifier}.joblib`, e.g. `models/se/tfidf13_lr.joblib`, `models/health/glove_svm.joblib`.
- Label order everywhere: `["negative", "neutral", "positive"]`.
- CSV format everywhere: `id;text;polarity`, semicolon-delimited, UTF-8 without BOM. Semicolons because review text is full of commas and the Senti4SD tooling expects them.
- Random seed everywhere: `42`.

### 3.3 The Non-Negotiable Rule

**Both corpora must be processed, split, modelled, and evaluated identically.** The entire project rests on comparing two domains. Any asymmetry in preprocessing, split ratio, class balance, or tuning protocol becomes a confound an examiner will find immediately. When in doubt, do the same thing to both.

---

## Phase 0 — Shared Setup (Both Members, Together)

### Task 0.1: Environment and Repository

1. Create a GitHub repository; add both members as collaborators.
2. Create the folder structure from Section 3.1.
3. Virtual environment, Python 3.10 or 3.11.
4. Pin in `requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `gensim`, `lime`, `shap`, `matplotlib`, `seaborn`, `nltk`, `joblib`, `colorama`, `ipywidgets`, `jupyter`, `scipy`, `statsmodels`, `ucimlrepo`.
5. `.gitignore` for `models/`, large embedding files, `__pycache__`, `.ipynb_checkpoints`, **and `data/raw/druglib/`** — see the licence note below.
6. Branch workflow: each member works on `a/...` or `b/...` branches and opens a PR the other reviews before merging.

### Task 0.2: Download Both Datasets

1. **Senti4SD:** `git lfs clone https://github.com/collab-uniba/Senti4SD.git`, locate the gold-standard CSV (4,423 rows, 9 columns: `study`, `label`, `so.id`, `post.type`, `text`, `final`, `r1`, `r2`, `r3`). Copy to `data/raw/senti4sd/` unchanged.
2. **Druglib:** `pip install ucimlrepo`, then `fetch_ucirepo(id=461)`. Copy the raw `.tsv` files to `data/raw/druglib/`.
3. **Licence note:** the Druglib donors require research-only use, no redistribution, and citation. **Do not commit the raw Druglib files to GitHub.** Add a download script instead. Cite Gräßer, Kallumadi, Malberg & Zaunseder (2018).
4. Together, read 30 random rows from each corpus and record: total rows, class or rating distribution, average text length, and 5 examples of the domain vocabulary problem from each.
5. Write these into `README.md` — they go into the report.

**Done when:** both members can load both raw datasets from the same repo.

---

## Phase 1 — Corpus Construction (Week 1, the Critical Gate)

This phase decides whether the project works. Do it first and do it properly.

### Task A1: Senti4SD Alignment (Member A)

1. Load the gold standard; keep `text` and `final`.
2. Standardise column names to `id`, `text`, `polarity`; map labels to lowercase `negative`, `neutral`, `positive`.
3. Remove exact duplicate texts (keep first); log the count.
4. Drop empty or whitespace-only rows.
5. Record inter-rater agreement from the `r1`, `r2`, `r3` columns (Fleiss' kappa) — **this is a number the health corpus cannot match, and it belongs in the report as an honest asymmetry.**
6. Record class distribution; this becomes the target distribution B matches in Task B1.
7. Save to `data/processed/se/clean.csv`.

**Deliverable:** `src/build_se.py`, `clean.csv`, agreement statistic, class distribution.

### Task B1: Druglib Construction (Member B)

This is the harder half. Druglib gives you **ratings, not sentiment labels**, and they disagree with the text constantly. A row rated 10 can have a side-effects field reading "heavier bleeding and clotting than normal." A row rated 1 can have a positive benefits field. Handle this explicitly.

#### B1.1 Field selection
1. Use **`commentsReview` only** as training text. It is the free-form overall statement and is functionally closest to a StackOverflow post.
2. **Set `benefitsReview` and `sideEffectsReview` aside entirely.** They are positive-leaning and negative-leaning *by construction*, so they are useless as training data — but they become the bias probe in Task B5.3. Save them to `data/processed/health/bias_probe.csv` keyed by row id.

#### B1.2 Filtering
1. Drop rows shorter than 15 words.
2. Drop boilerplate: `^(thanks|see above|none|n/?a|same as above|don'?t know yet)\.?$` and near-variants.
3. Drop copy-pasted drug monograph text (detectable by clinical listing style and absence of first-person pronouns — inspect and tune the rule by hand).
4. Log how many rows each filter removed. **Report these counts**; unexplained data loss looks like cherry-picking.

#### B1.3 Label assignment
1. Bin: rating **1–3 → negative**, **8–10 → positive**, **4–7 → discard**.
2. **Do not map 4–7 to neutral.** A mid-range rating means *mixed feelings*, but Senti4SD neutral means *absence of affect*. Conflating them would train the model to call emotionally intense text neutral and would silently destroy every cross-domain comparison.
3. Build neutral separately by hand: sample ~500 candidates by filtering for imperative and dosage patterns ("I take 600mg three times a day", "one puff in the morning"), then hand-label ~400 as neutral. Prefilter first; it cuts the labelling work by an order of magnitude.

#### B1.4 Validation (mandatory)
1. Both members independently hand-label the **same 100** `commentsReview` rows.
2. Compute Cohen's kappa between the two annotators, and agreement between each annotator and the rating-derived label.
3. **If agreement with the rating-derived labels is below 80%, the binning is wrong.** Adjust the thresholds (try 1–2 / 9–10) and re-measure before proceeding.
4. **Fallback:** if agreement stays poor, switch to the Drugs.com corpus (UCI id 462) sampled down to ~4,000. Decide this by the end of Week 1, not later.

#### B1.5 Balance
1. Sample to roughly 4,000 rows matching Senti4SD's class proportions from A1.6.
2. Save to `data/processed/health/clean.csv`.

**Deliverable:** `src/build_health.py`, `clean.csv`, `bias_probe.csv`, filter log, kappa and agreement figures.

### GATE 1 — Do Not Proceed Until

- Both corpora exist in `id;text;polarity` form at comparable size and class balance.
- Annotator kappa ≥ 0.75 and rating-agreement ≥ 80%, or the documented fallback has been taken.
- At least 300 usable neutral health rows exist. **Neutral is the scarcest class and the most likely thing to block the project.**

---

## Phase 2 — Foundation

### Task A2: Preprocessing Module (Member A)

**Goal:** one `preprocess()` used by every model in both domains.

#### A2.1 Shared rules (identical for both domains)
1. Decode HTML entities with `html.unescape`.
2. Replace URLs with `URL` (`https?://\S+|www\.\S+`).
3. Replace `@name` mentions with `USER`.
4. Lowercase everything.
5. Expand negation contractions: `can't` → `can not`, `n't` → ` not`. **Never remove negation words** — they flip sentiment, and in the health corpus ("no more panic attacks") negation is the entire phenomenon.
6. Keep emoticons as tokens (`EMO_POS` / `EMO_NEG`).
7. Collapse repeated characters (`sooooo` → `soo`) and punctuation (`!!!!` → `!!`).
8. **Do not** remove stopwords and **do not** stem. "kill" vs "killed", "stopped" vs "stop" both matter for the analysis.
9. `tokenize(text) -> list[str]` keeping `!`, `?`, and emoticon tokens.

#### A2.2 Domain-specific rules (the only permitted divergence)
- `domain="se"`: replace inline code and code-like spans (backticks, `foo()`, `a.b.c`) with `CODE`.
- `domain="health"`: replace dosage expressions (`600mg`, `10 mg`, `1/4 packet`) with `DOSE`, and standalone numbers with `NUM`.
- Document both rule sets side by side in the report and justify why they cannot be identical.

#### A2.3 Testing
1. Ten unit tests in `tests/test_preprocess.py` including `"The process was killed"`, `"Fatal error on line 3"`, `"The nausea finally stopped"`, `"I take 600mg three times a day"`.
2. Docstring listing every rule in order.
3. Ten before/after examples per domain as a markdown table for the report.

**Deliverables:** `src/preprocess.py`, tests, example tables.

### Task B2: Splits and Evaluation (Member B)

#### B2.1 Splits
1. For **each** domain: `train_test_split` twice with `stratify=label`, `random_state=42` → 70% train / 15% val / 15% test.
2. Verify class proportions match across all three splits in both domains; print the table.
3. Commit the splits — **frozen after this**.
4. Implement `load_splits(domain)`.

#### B2.2 Evaluation module
1. `evaluate(name, domain, y_true, y_pred, out_dir)` computes accuracy, macro-F1, weighted-F1, per-class P/R/F1, confusion matrix (counts and row-normalised).
2. **Domain-specific confusion rates:** for SE, fraction of gold-neutral predicted negative. For HEALTH, fraction of gold-**positive** predicted negative — that is the relief-inversion failure, and it is a *different* error direction from the SE one. Report both.
3. Confusion-matrix heatmap to `results/figures/cm_{domain}_{name}.png`.
4. All numbers to `results/metrics/{domain}_{name}.json`.
5. `append_to_results_table(name, domain, metrics)` → one row in `results/metrics/all_results.csv` (with a `domain` column).
6. McNemar's test helper for comparing two models' predictions.
7. **Baselines, per domain:** majority-class and VADER. The VADER result on the health corpus is the single clearest demonstration of the domain-shift problem for a non-CS audience — a general-purpose lexicon reading "no more panic attacks" as negative needs no explanation at all.
8. Five unit tests with hand-computed values.

**Deliverables:** frozen splits for both domains, `src/splits.py`, `src/evaluate.py`, four baseline rows.

---

## Phase 3 — Features

### Task A3: TF-IDF Features, Both Domains (Member A)

#### A3.1 Build vectorizers
1. For each domain, three configs with `preprocessor=partial(preprocess, domain=d)`:
   - `tfidf11`: `ngram_range=(1,1)` baseline
   - `tfidf12`: `ngram_range=(1,2)`
   - `tfidf13`: `ngram_range=(1,3)`
2. Shared settings: `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `token_pattern` matching the A2 tokenizer.
3. **Fit on training data only**; transform val and test.

#### A3.2 Vocabulary analysis
1. Vocabulary size and sparsity per config per domain.
2. Top 30 n-grams by mean TF-IDF weight per class per domain.
3. **Compound phrase check.** SE: does `fatal error`, `kill the process`, `null pointer exception` survive into the (1,2)/(1,3) vocabularies? HEALTH: does `no side effects`, `pain went away`, `no more`, `stopped working` survive? Count occurrences per class.
4. **This is a direct cross-domain comparison** — do compound phrases matter more in one domain than the other? Save as one table with a domain column.

#### A3.3 Light tuning
1. LR with default C; try `min_df` in {1, 2, 5} and `max_features` in {None, 20000, 50000} on validation.
2. **Tune per domain, but keep the protocol identical.** Record both choices.

**Deliverables:** `src/features_tfidf.py`, vocabulary stats, compound-phrase table covering both domains.

### Task B3: Embedding Features, Both Domains (Member B)

#### B3.1 Pretrained GloVe
1. `glove.6B.100d.txt` via gensim `KeyedVectors`.
2. Tokenize with A's `preprocess()` + `tokenize()`.
3. Document vector = mean of in-vocabulary token vectors; zero vector if none.
4. Variant: TF-IDF-weighted mean (IDF from a unigram TF-IDF fitted on train). **Cut this variant first if time runs short.**
5. **OOV rate** overall, and specifically for the SE jargon lexicon, the health symptom lexicon, code tokens, and drug names. Expect drug names to be heavily OOV — that is a finding.

#### B3.2 Self-trained Word2Vec, one model per domain
1. Train on **that domain's training split only**: `vector_size=100, window=5, min_count=2, sg=1, negative=5, epochs=30, seed=42, workers=1`.
2. Try `vector_size` in {50, 100}, `sg` in {0, 1}; pick by validation macro-F1 with LR.
3. Save to `models/{domain}/w2v.model`.

#### B3.3 Qualitative analysis — the headline table
1. SE probes: `kill`, `fatal`, `crash`, `error`, `exception`, `hang`, `abort`. HEALTH probes: `nausea`, `pain`, `attack`, `depression`, `stopped`, `gone`, `relief`.
2. List the 10 nearest neighbours in GloVe and in each domain's Word2Vec.
3. **The expected result is the most quotable thing in the project:** GloVe places `kill` near `murder` and `attack` near `assault`, while SE-Word2Vec places `kill` near `process` / `terminate` and HEALTH-Word2Vec places `attack` near `episode` / `migraine`. Put this table on a slide.
4. Optional 2D PCA/t-SNE of probe words plus clearly emotional words, one panel per embedding.
5. Note that ~4k documents limits Word2Vec quality **in both domains equally** — which is itself a controlled comparison.

#### B3.4 Wrap as sklearn transformers
`MeanEmbeddingVectorizer` and `TfidfEmbeddingVectorizer` with `fit`/`transform` taking raw strings.

**Deliverables:** `src/features_embed.py`, OOV table, nearest-neighbour table, optional plot.

---

## Phase 4 — Models

### Task A4: TF-IDF Models (Member A)

**18 experiment rows** (3 feature sets × 3 classifiers × 2 domains).

#### A4.1 Pipelines
1. `MultinomialNB` (generative).
2. `LogisticRegression(max_iter=2000)` (discriminative).
3. `LinearSVC` wrapped in `CalibratedClassifierCV(cv=5)` — **LinearSVC has no `predict_proba`**, which LIME, SHAP, and both demos require.

#### A4.2 Tuning
Validation set or `GridSearchCV` stratified 5-fold, `scoring="f1_macro"`, **same protocol both domains**:
- MNB: `alpha` in {0.01, 0.1, 0.5, 1.0}
- LR: `C` in {0.01, 0.1, 1, 10, 100}, `class_weight` in {None, "balanced"}
- SVM: `C` in {0.01, 0.1, 1, 10}, `class_weight` in {None, "balanced"}

#### A4.3 Final training and evaluation
1. Retrain best configs; evaluate once on test via `evaluate()`.
2. Save to `models/{domain}/tfidf1X_{mnb|lr|svm}.joblib`.
3. Predictions to `results/metrics/{domain}_tfidf_predictions.csv`.

#### A4.4 Error analysis, both domains
1. SE: collect test cases where gold = neutral, predicted = negative. Tag each: *contains SE jargon* / *sarcasm* / *annotation noise* / *other*.
2. HEALTH: collect gold = positive, predicted = negative. Tag each: *relief inversion* / *mixed review* / *binning error* / *other*.
3. Report counts and 5 representative examples per domain.
4. For LR in each domain, list the top 20 features pushing toward negative and flag domain vocabulary.

**Deliverables:** `src/models_tfidf.py`, 18 models, results table, two error-analysis tables.

### Task B4: Embedding Models (Member B)

**24 experiment rows** (2 embeddings × 2 pooling × 3 classifiers × 2 domains), or 12 if the weighted variant is cut.

#### B4.1 Pipelines
1. **Gaussian NB** — `MultinomialNB` is impossible on embeddings because they contain negative values. Note this in the report as a substantive generative-model constraint, not a footnote.
2. LR with `StandardScaler`.
3. `LinearSVC` with `StandardScaler`, wrapped in `CalibratedClassifierCV(cv=5)`.

#### B4.2 Tuning
Same protocol as A4.2:
- GNB: `var_smoothing` in {1e-9, 1e-7, 1e-5, 1e-3}
- LR / SVM: `C` in {0.01, 0.1, 1, 10}, `class_weight` in {None, "balanced"}

#### B4.3 Final training and evaluation
Retrain, evaluate once on test, save as `models/{domain}/{glove|w2v}{mean|wt}_{gnb|lr|svm}.joblib`.

#### B4.4 Error analysis
1. Same tagging procedure as A4.4 on the best embedding model per domain.
2. Overlap of errors between the best TF-IDF and best embedding model — shared vs unique, per domain.

#### B4.5 Master results table
1. Combine all rows (4 baselines + 18 TF-IDF + 24 embedding) sorted by macro-F1, with a `domain` column.
2. Three charts: macro-F1 by representation **grouped by domain**; generative vs discriminative average per representation per domain; baseline vs best per domain.
3. McNemar's tests: best TF-IDF vs best embedding (each domain); best NB vs best discriminative (each domain).
4. **Key question to answer explicitly:** does the same representation win in both domains? If TF-IDF (1,3) wins on SE but GloVe wins on HEALTH, that is a result about compound technical phrases, not a nuisance.

**Deliverables:** `src/models_embed.py`, 24 models, master table, 3 charts, significance results.

---

## Phase 5 — Cross-Domain Transfer (The New Core Experiment)

This is what elevates the project from "sentiment classification, twice" to a transferability study. Neither member owns it alone.

### Task J-T: Transfer Matrix (A writes `src/transfer.py`, B reviews; each runs their own models)

1. Four conditions per model family:

   | Train | Test | Meaning |
   |---|---|---|
   | SE | SE | in-domain baseline |
   | HEALTH | HEALTH | in-domain baseline |
   | SE | HEALTH | cross-domain |
   | HEALTH | SE | cross-domain |

2. **A runs the TF-IDF families, B runs the embedding families.** Report macro-F1 for all four cells per family.
3. **Handle the vocabulary problem honestly.** A TF-IDF model trained on SE has no vocabulary for the health corpus, so cross-domain TF-IDF will collapse largely through OOV. Report the **test-set OOV rate** alongside every cross-domain number, or the result is uninterpretable. Embeddings degrade more gracefully — that contrast is itself a finding, and it is the strongest argument for the embedding half of the project.
4. **Per-class degradation.** Which class collapses? Predicted direction: SE-trained models over-predict negative on health text (relief sentences read as complaints); health-trained models over-predict negative on SE jargon. Confirm or refute with the confusion matrices.
5. Heatmap of the transfer matrix for the report.
6. Save everything to `results/transfer/`.

### GATE 2 — Sanity Check

**If cross-domain performance does not degrade, something is wrong.** Check for leakage, label mapping errors, or a test set that is accidentally trivial before reporting a surprising result. A suspiciously good number is a bug until proven otherwise.

**Deliverables:** `src/transfer.py`, transfer matrix table, OOV-adjusted analysis, per-class degradation tables, heatmap.

---

## Phase 6 — Explainability

Both members explain **the same 20 instances per domain** (40 total), agreed jointly: 5 correct negatives, 5 correct neutrals, 5 characteristic errors, 5 domain-vocabulary-heavy cases.

### Task A6: LIME (Member A)

#### A6.1 Per-instance
1. `LimeTextExplainer(class_names=["negative","neutral","positive"], split_expression=<A2 tokenizer regex>, random_state=42)`.
2. For each of the 40 instances, explain with the best TF-IDF model and the best embedding model (load B's pipeline) **for that domain**.
3. `explain_instance(text, pipeline.predict_proba, num_features=10, num_samples=2000, labels=[0,1,2])`.
4. Save HTML to `results/explanations/lime/{domain}/`.
5. Two to three sentences of observation per case.

#### A6.2 Aggregated
1. Run LIME across each domain's full test set (or a stratified sample of ≥300 if runtime is a problem) for the best TF-IDF and best embedding model.
2. Accumulate per token: total weight toward negative, occurrences, mean weight.
3. **SE jargon lexicon** (A owns, saved to `data/lexicons/se_jargon.txt`): `fatal, error, kill, killed, crash, crashed, abort, exception, fail, failed, failure, dead, deadlock, bug, broken, hang, panic, terminate, execute, garbage, deprecated, warning, exploit, attack, destroy`.
4. For each lexicon word, mean LIME weight toward negative **on gold-neutral SE instances** and **on gold-positive HEALTH instances** using B's health lexicon. A strong positive value means the model reads domain vocabulary as hostility.
5. Horizontal bar chart of the top 25 negative-driving tokens per domain, lexicon words coloured differently.

#### A6.3 Stability check
1. For the 40 fixed instances, run LIME with 5 different `random_state` values.
2. Jaccard overlap of top-5 and top-10 token sets across runs.
3. Repeat with `num_samples` in {500, 2000, 5000}.
4. **Report mean and standard deviation per domain.** If stability differs between domains — plausible, since health reviews are longer than StackOverflow posts — that is worth discussing.

**Deliverables:** `src/explain_lime.py`, 40 HTML explanations, per-domain bias tables and charts, stability table.

### Task B5: SHAP (Member B)

#### B5.1 Per-instance
1. TF-IDF LR: `shap.LinearExplainer(clf, background)` with 100–200 transformed training rows; map indices back to n-gram names.
2. Embedding and SVM models: `shap.Explainer(pipeline.predict_proba, shap.maskers.Text(tokenizer_regex))`.
3. Same 40 fixed instances as A.
4. `shap.plots.text` HTML and waterfall PNGs to `results/explanations/shap/{domain}/`.
5. Two to three sentences per instance.

#### B5.2 Global
1. SHAP values over each domain's full test set (or stratified 300 for the Partition explainer).
2. Global bar plot (mean |SHAP|) and beeswarm for the negative class, per domain.
3. Token-level aggregation matching A's LIME format so the two are directly comparable.

#### B5.3 Lexical bias score — the central metric
1. **Health symptom lexicon** (B owns, `data/lexicons/health_symptoms.txt`): `nausea, pain, cramps, depression, anxiety, attack, attacks, panic, migraine, headache, seizure, insomnia, fatigue, dizziness, swelling, rash, bleeding, vomiting, wheezing, itching`.
2. For each lexicon word, mean SHAP contribution toward negative on instances where the gold label is **not** negative.
3. **Domain Bias Score** per model = mean over lexicon words appearing ≥3 times. Compute for both domains with each domain's own lexicon.
4. Compare across TF-IDF, GloVe, and Word2Vec models in one table and bar chart, **grouped by domain**. This is the core evidence that lexical bias is a general phenomenon and not an artefact of one corpus.
5. **The bias probe (unique to the health corpus).** Take `bias_probe.csv` from B1.1. Classify the `sideEffectsReview` text from rows rated 8–10 — patients who *liked* the drug describing its side effects — and count how often the model predicts negative. This is a clean, single-number bias measurement with no hand-labelling involved, and Senti4SD offers no equivalent. Report it prominently.

**Deliverables:** `src/explain_shap.py`, 40 explanations, global plots, bias score table and chart, bias-probe result.

---

## Phase 7 — Stress Tests

Two sets, one per domain, same structure so the results are comparable.

### Task A7: SE Stress Set (Member A)

1. **40 harsh-but-neutral** sentences: "Kill the process before restarting the server." / "The build failed with a fatal error on line 42." / "Garbage collection destroys unused objects."
2. **20 genuinely negative** SE sentences, of which 10 contain no jargon at all — this separates jargon from real hostility.
3. Cover ≥20 lexicon words, max 5 sentences per word. Vary length (5–30 words) and style (question, instruction, report).
4. **10 minimal pairs:** "The app crashed after the update." vs "The app crashed again, this update is useless."
5. Save to `data/stress_test/se_stress.csv` with `id, text, intended_label, lexicon_word, pair_id`.
6. Blind-annotate B's health set (see B6.2).

### Task B6: Health Stress Set + Evaluation (Member B)

#### B6.1 Write sentences
1. **40 relief-framed positives** using symptom vocabulary: "The dizziness finally stopped after a week." / "No more panic attacks since I started this."
2. **20 genuine complaints**, of which 10 contain no symptom vocabulary.
3. **10 minimal pairs using the same symptom word in both halves** — this is sharper than the SE version and should be highlighted: "The nausea finally stopped." / "The nausea never stopped."
4. Save to `data/stress_test/health_stress.csv`, same columns.

#### B6.2 Blind cross-annotation
1. Shuffle each set, strip `intended_label`.
2. A labels B's set, B labels A's set.
3. Cohen's kappa between intended and independent label, **per domain**.
4. Discuss disagreements together; drop what you cannot agree on; record the count.

#### B6.3 Run
1. All saved models plus VADER on both stress sets.
2. Metrics per model per domain:
   - **False-alarm rate** — % of vocabulary-heavy non-negative sentences predicted negative (lower is better)
   - **Negative recall** on genuine complaints (higher is better)
   - Recall on negative-with-vocabulary vs negative-without
   - **Minimal-pair accuracy** — % of pairs where both halves are classified correctly
3. Scatter plot per domain: x = false-alarm rate, y = negative recall, one point per model (ideal = top-left).
4. **Cross-domain stress test:** run SE-trained models on the health stress set and vice versa. This is the most legible result in the whole project for a non-CS audience — watch an SE-trained model call "no more panic attacks" negative.
5. LIME and SHAP on the 10 worst-handled sentences per domain.

**Deliverables:** 120 stress sentences total, annotations and kappa, `src/stress_test.py`, two tables, two scatter plots, cross-domain stress results.

---

## Phase 8 — Demos

**Both demos must let the user switch the trained model while keeping the input fixed.** That switch is the whole point: the same sentence, two models, two different answers, with highlighted words explaining each. It is what makes the project legible to a non-CS faculty member, and it should be the first thing shown in the presentation.

### Task A8: CLI (Member A)

1. `cli/predict.py` with `argparse`:
   - `--domain se|health` — selects which trained model to use
   - `--model` (default: best for that domain; choices = files in `models/{domain}/`)
   - `--compare` — **runs both domains' best models on the same input and prints results side by side**
   - `--text "..."` or `--file inputs.txt`
   - `--top-k` highlighted words (default 8)
   - `--explainer lime|coef`
   - `--list-models` — prints available models and test macro-F1 from B's metrics JSON
2. Print predicted label and all three class probabilities.
3. Colour each word with `colorama`: red toward negative, green toward positive, grey neutral; show weights for the top-k.
4. Handle missing model file, empty text, unknown option.
5. README usage section plus 5 screenshots: SE jargon-neutral, SE negative, health relief-positive, health complaint, **and one `--compare` run showing the cross-domain failure.**

### Task B7: Notebook (Member B)

1. `notebooks/demo.ipynb`: Setup, Load Models, Interactive Prediction, **Cross-Domain Comparison**, Example Gallery, Results Summary.
2. `ipywidgets`: text area, domain dropdown, model dropdown, explainer toggle (LIME / SHAP), Predict button.
3. On click: probability bar chart plus the sentence rendered as HTML with per-word background colouring (red negative, green positive, intensity = weight).
4. **Cross-domain panel:** one input, both domains' models, two highlight renderings side by side. This is the centrepiece.
5. Example gallery: 8 preloaded buttons — 2 SE jargon-neutral, 2 SE negative, 2 health relief-positive, 2 health complaint.
6. Final cells: master results table, transfer heatmap, both stress scatter plots.
7. Must run top-to-bottom under "Restart and Run All" on a fresh environment; export static HTML for submission.

---

## Phase 9 — Joint Integration and Final Report

### Task J1: LIME vs SHAP Agreement (split evenly)
1. **A:** export LIME top-10 token rankings for all 40 instances in a common CSV format.
2. **B:** export SHAP top-10 in the same format.
3. **Together:** Spearman rank correlation and top-5 overlap per instance; **compare agreement between domains** — if the two methods agree more on SE than on health (or vice versa), that is worth a paragraph. Link to Limitations.

### Task J2: Integration Testing (split evenly)
1. A runs B's notebook on a fresh clone; B runs A's CLI on a fresh clone.
2. File issues; the owner fixes.
3. Verify every model in both domain folders loads and works in both demos.

### Task J3: Final Report

Section ownership:

| Section | Owner |
|---|---|
| Abstract | Joint, written last |
| Introduction and motivation (domain shift, both domains) | A |
| Related work | A (SE sentiment) + B (health/clinical sentiment) |
| Datasets: Senti4SD | A |
| Datasets: Druglib construction, binning, validation | B |
| Preprocessing | A |
| Evaluation methodology, baselines | B |
| TF-IDF experiments, both domains | A |
| Embedding experiments, both domains | B |
| **Cross-domain transfer results** | A (TF-IDF half) + B (embedding half), joint discussion |
| Generative vs discriminative comparison | B |
| LIME analysis | A |
| SHAP analysis and bias scores | B |
| Stress-test results | B |
| Demo description | A (CLI) + B (notebook) |
| Discussion of Expected Outcomes | A: 1, 4 · B: 2, 3 · both review |
| Limitations | Joint |
| Conclusion and future work | Joint |

**Limitations must now include:** small corpora in both domains; Word2Vec quality at 4k documents; LIME/SHAP instability with numbers from A6.3 and J1; stress-set size and author bias; **and the label-provenance asymmetry — Senti4SD labels are human-annotated with three raters and reported agreement, while health labels are rating-derived plus hand-labelled neutrals. State the validation kappa from B1.4 and be upfront that this is the project's main methodological weakness.** An examiner will find it; find it first.

### Task J4: Presentation

1. **Open with the cross-domain demo**, not with methodology. One sentence, two models, two answers. Thirty seconds, and the non-CS members of the audience understand the entire project.
2. Split: A presents problem, both datasets, TF-IDF, LIME, CLI. B presents embeddings, model comparison, transfer matrix, SHAP, stress tests, notebook.
3. The nearest-neighbour table from B3.3 and the transfer heatmap from Phase 5 are the two strongest slides.
4. Rehearse twice; each member must be able to answer questions about the other's half.

---

## 4. Final Deliverables Checklist

| Deliverable | Owner |
|---|---|
| `src/build_se.py` + aligned SE corpus | A |
| `src/build_health.py` + aligned health corpus + bias probe | B |
| Label validation (kappa, rating agreement) | B, annotated by both |
| `src/preprocess.py` + tests | A |
| `src/splits.py`, `src/evaluate.py` + tests | B |
| TF-IDF features + 18 models (both domains) | A |
| Embedding features + 24 models (both domains) | B |
| `src/transfer.py` + transfer matrix | A writes, both run |
| LIME explanations, aggregation, stability (both domains) | A |
| SHAP explanations, global plots, bias scores, bias probe | B |
| SE jargon lexicon | A |
| Health symptom lexicon | B |
| 60 SE stress sentences | A |
| 60 health stress sentences + evaluation | B |
| CLI demo with `--compare` | A |
| Notebook demo with cross-domain panel | B |
| LIME vs SHAP agreement analysis | A + B |
| Final report + slides | A + B |

---

## 5. Collaboration Rules

1. Short sync twice a week (15 minutes): done, next, blocked.
2. Never push directly to `main`; every PR reviewed by the other member.
3. Never change frozen splits, label order, the seed, function signatures, or **either corpus** after Week 1 without agreement.
4. Log every experiment (config, domain, date, result) in `results/metrics/all_results.csv`.
5. **Any processing step applied to one domain must be applied to the other**, or documented as a deliberate, justified exception. This is the single easiest way to break the project.
6. Commit history on GitHub serves as evidence of equal contribution.

---

## 6. Risk Register

| Risk | Detection point | Response |
|---|---|---|
| Druglib binning disagrees with human judgement | Gate 1, Week 1 | Adjust thresholds to 1–2 / 9–10; if still poor, switch to Drugs.com (id 462) sampled to 4k |
| Too few usable neutral health rows | Gate 1, Week 1 | Widen the imperative/dosage prefilter; accept a smaller neutral class and downsample the other two to match |
| Cross-domain transfer shows no degradation | Gate 2, Week 5 | Treat as a bug: check leakage, label mapping, test-set construction before reporting |
| Cross-domain TF-IDF collapses entirely through OOV | Week 5 | Expected — report OOV rate alongside; the embedding contrast is the real result |
| Workload overrun | Week 4 | Cut the TF-IDF-weighted embedding variant (24 → 12 models); cut `tfidf11` last, it is the baseline |
| Word2Vec quality poor in both domains | Week 3 | Expected at 4k documents; report as a controlled limitation, not a failure |
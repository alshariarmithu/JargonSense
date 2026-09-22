# Interpretable Sentiment Classification in Software Engineering Communication

**Khulna University of Engineering & Technology** · Department of Computer Science and Engineering

**Course No:** CSE 4122 · **Course Title:** Natural Language Processing Laboratory ·
**Submission Date:** 23 September, 2026

**Submitted to:** Dr. K. M. Azharul Hasan (Professor, Dept. of CSE, KUET) ·
Md Nazirulhasan Shawon (Assistant Professor, Dept. of CSE, KUET)

**Submitted by:** Al Shariar Hossain (2107066) · Hassan Mohammed Naquibul Hoque (2107077)

> Every number in this report is produced by a script in the repository and written to
> `results/`. Nothing is typed by hand. The command that produces each table is named
> beside it.

---

## Abstract

Sentiment models trained on general English misread software engineering text: words such
as *fatal error*, *kill the process* and *crashed* sound hostile but are functionally
neutral to a developer. We build sentiment classifiers for the Senti4SD StackOverflow
corpus across five representations and three classifiers, and use LIME and SHAP to audit
*why* they succeed or fail rather than only measuring accuracy.

The best model, TF-IDF (1,3) with a calibrated linear SVM, reaches **0.8275 test macro-F1**
against **0.7079** for the VADER lexicon baseline. Discriminative classifiers beat the
generative one decisively (0.7694 vs 0.6792 mean macro-F1, McNemar *p* < 1e-8).

Our central finding is negative and then positive. **The Senti4SD corpus contains almost
none of the vocabulary the problem is about** — *fatal*, *hang* and *abort* occur zero
times in 3,031 training posts, *kill* three times — so the corpus itself cannot answer the
research question. On a hand-written 60-sentence stress set it can be answered clearly:
VADER labels **70%** of harsh-but-neutral SE sentences negative, while the false-alarm rate
falls monotonically with how much general English the representation carries — GloVe 40%,
Word2Vec 28%, TF-IDF 14%. Lexical bias is real, measurable, and inherited from pretraining.

---

## 1. Introduction

Sentiment analysis is applied to developer platforms to track team morale, triage hostile
code review, and measure community health. These tools are usually trained on product
reviews or social media, where *crashed*, *killed* and *failed* reliably signal
displeasure. In software engineering they usually signal nothing of the kind.

This is **domain shift** in its most concrete form: the same token carries different
sentiment in different communities. A model that has not learned the difference will
systematically mistake technical description for hostility.

Accuracy alone cannot detect this. A model can score well overall while being wrong for
exactly the reason that matters. So this project pairs classification with
**explainability**: LIME and SHAP identify which words drive each prediction, turning "the
model is 83% accurate" into "the model treats *afraid* and *hate* as negative evidence and
does not key on technical vocabulary".

### Contributions

1. A reproducible six-stage pipeline over Senti4SD, 15 models, one command per stage.
2. Measurement — not assumption — of whether stemming and lemmatization help (§4).
3. Evidence that the motivating vocabulary is largely **absent** from the standard SE
   sentiment corpus, from four independent analyses (§5, §8).
4. A 60-sentence stress set that isolates technical vocabulary from hostility, and a clean
   ordering of lexical bias across representations (§9).

---

## 2. Related work

**SE sentiment.** Senti4SD (Calefato et al., 2018) established that general-purpose
sentiment tools transfer poorly to developer communication and released the gold standard
used here. Later work (SentiCR, SentiSE) repeats the finding on code review. Our
contribution is not a better classifier but an *audit*: we ask which words the model
actually uses.

**Explainability.** LIME (Ribeiro et al., 2016) fits a local linear surrogate to
perturbations of one input. SHAP (Lundberg & Lee, 2017) allocates the prediction among
features by Shapley value, exact for linear models. Both are standard for text; their
known instability is treated as a measured quantity here (§8.3), not a caveat.

---

## 3. Dataset and preprocessing

### 3.1 Corpus

`Senti4SD_GoldStandard_EmotionPolarity.xlsx`, chosen over Senti4SD's ready-made train/test
CSVs because it carries the `r1`/`r2`/`r3` rater columns needed for agreement.

| | |
|---|---|
| Rows loaded | 4,423 |
| Duplicate texts dropped | 92 |
| **Final corpus** | **4,331** |
| Fleiss' κ | **0.7586** (4,359 items, 3 raters) |
| Mean length | 29.97 words · 2.33 sentences |

| Class | Count | Share |
|---|---|---|
| negative | 1,176 | 27.15% |
| neutral | 1,662 | 38.37% |
| positive | 1,493 | 34.47% |

Splits: **3,031 / 650 / 650**, stratified, seed 42, frozen before any model was built.
Rater columns are free text containing the typos `Postive`, `Poitive`, `Netural`; these are
repaired before computing κ and items with an unparseable rater are excluded.

> `python -m src.extraction.build_se` · `python -m src.preprocessing.splits --domain se`

### 3.2 Cleaning

One function, `clean(text, domain)`, applied identically to every model:
`html.unescape` → lowercase → URLs→`URL` → mentions→`USER` → code spans→`CODE` →
emoticons→`EMO_POS`/`EMO_NEG` → negation expansion → collapse repeats → squeeze whitespace.

**Two order decisions, both load-bearing.**

*Lowercasing runs second, not fourth*, so placeholders stay uppercase and remain
distinguishable from the literal words. This was not hypothetical: in the training split the
placeholder `CODE` occurs in 174 posts and the ordinary English word *code* in 303. Merging
them would have produced one meaningless feature spanning 477 posts.

*URLs are replaced before emoticons*, because `http://` contains `:/`, a sad-face emoticon;
the other order turns every URL into `httpEMO_NEG/...`.

**Stopword removal is disabled permanently.** Negation words appear in every standard
stopword list and negation flips sentiment.

---

## 4. Does stemming help? A measurement, not an assumption

Stemming and lemmatization merge `killed`/`kills`/`killing` into `kill`. On a small corpus
that should help. Here it risks erasing the phenomenon under study: *"killed the process"*
is routine description, and folding it into the emotional register of *kill* hides exactly
the confusion the project measures.

Rather than argue, we ran the same model (TF-IDF (1,2) + Logistic Regression at defaults)
three times, changing only the normalization mode, scored on **validation**:

| Mode | Vocabulary | Accuracy | Macro-F1 | neutral→negative |
|---|---|---|---|---|
| **none** | 12,723 | 0.7938 | **0.7863** | 10.4% |
| lemma | 12,186 | 0.7831 | 0.7754 | 11.2% |
| stem | 12,371 | 0.7908 | 0.7844 | **10.0%** |

**Selected: `none`.** The honest reading is that normalization barely matters here — the
gap between `none` and `stem` is 0.0019 macro-F1, about **2 of 650 validation documents**,
well inside noise. The defensible justification is therefore not that `none` scored highest
but that it preserves the inflections the project studies and is simplest.

Two secondary observations belong in the record. Lemmatization was worst on both metrics
despite being the more sophisticated technique. And the two metrics **disagree**: `none`
wins on macro-F1 while `stem` has the lowest neutral→negative rate. That disagreement is a
miniature of the project's whole thesis.

Implementation note: lemmatization is POS-aware. `WordNetLemmatizer` treats every word as a
noun unless told otherwise, so `lemmatize("stopped")` returns `stopped` unchanged while
`lemmatize("stopped", "v")` returns `stop`. Without POS tagging the entire comparison would
have been a no-op.

> `python -m tools.run_ablation` → `results/ablation/normalization_se.csv`

---

## 5. Feature engineering

### 5.1 TF-IDF

| Config | n-grams | Vocabulary | Sparsity | Features/doc |
|---|---|---|---|---|
| `tfidf11` | (1,1) | 3,449 | 0.9931 | 23.7 |
| `tfidf12` | (1,2) | 12,723 | 0.9968 | 40.7 |
| `tfidf13` | (1,3) | 17,902 | 0.9974 | 46.7 |

**Most distinctive terms per class.** Ranking by mean TF-IDF returns `i, the, is, to` for
every class, so terms are scored by weight inside the class minus weight outside it:

| Class | Top terms |
|---|---|
| negative | `EMO_NEG, sad, i, hate, horrible, afraid, not, terrible` |
| neutral | `?, URL, you, use, how, what, see` |
| positive | `!, excellent, thanks, great, EMO_POS, excellent !, thanks !` |

Neutral is dominated by **question words** — most neutral StackOverflow posts are questions.
Positive picks up the bigrams `excellent !` and `thanks !`, which a unigram model cannot
represent.

### 5.2 ⚠️ The compound phrases are not in the corpus

The proposal motivates n-grams with *fatal error*, *kill the process* and *null pointer
exception*. **None occurs even once** in the training split.

| Phrase | (1,1) | (1,2) | (1,3) | neg | neu | pos |
|---|---|---|---|---|---|---|
| fatal error | — | — | — | 0 | 0 | 0 |
| null pointer | — | — | — | 0 | 0 | 0 |
| kill the process | — | — | — | 0 | 0 | 0 |
| null pointer exception | — | — | — | 0 | 0 | 0 |
| **does not** | — | ✓ | ✓ | 46 | 52 | 26 |
| **not work** | — | ✓ | ✓ | 12 | 17 | 5 |
| **thank you** | — | ✓ | ✓ | 0 | 0 | 54 |
| does not work | — | — | ✓ | 5 | 11 | 5 |

Nine of fourteen tested phrases reached the (1,3) vocabulary; the five that did not are
exactly the canonical jargon examples. What n-grams actually capture in this corpus is
**negation** and **politeness**.

### 5.3 Tuning, and a rejected winner

`min_df` ∈ {1,2,5} × `max_features` ∈ {None, 20k, 50k}. The top validation score was
`tfidf13` with `min_df=1` at 0.7960 — using **130,078 features for 3,031 documents** with a
train/validation gap of 0.196. Its lead over the default was ~8 of 650 documents, while one
standard error on this validation set is ≈0.016 macro-F1.

We therefore stated a selection rule before inspecting results — *among settings within one
standard error of the best, take the smallest vocabulary* (the standard one-standard-error
rule) — which ships `min_df=5/5/2`: statistically tied, far smaller, and built from terms
that appear in more than one post.

### 5.4 Embeddings

| Embedding | Vocabulary | OOV tokens | Placeholders known |
|---|---|---|---|
| GloVe (pretrained, 100d) | 400,000 | 2.4% | none |
| Word2Vec (self-trained, 100d) | 3,752 | 3.9% | `CODE, URL, USER, EMO_POS, EMO_NEG` |

**The nearest-neighbour table — the clearest single piece of evidence in the project.**

| Word | Times in train | GloVe neighbours | Word2Vec neighbours |
|---|---|---|---|
| `kill` | 3 | killing, destroy, **shoot, attack, poison** | activities, clarify, song, exec |
| `crash` | 9 | accident, **plane, collision, airplane** | bus, modifying, press, gc |
| `error` | 65 | errors, mistake, incorrect, fault | occurs, **assertion, failed, throws** |
| `fatal` | **0** | deadly, deaths, **poisoning** | *out of vocabulary* |
| `abort` | **0** | aborted, takeoff, **fetuses** | *out of vocabulary* |

GloVe places `kill` beside *shoot* and *poison*, `abort` beside *fetuses*, `crash` beside
*airplane*. A general-English model reading a StackOverflow post genuinely sees violence and
aviation disasters. **This is the domain-shift claim demonstrated directly.**

Word2Vec's neighbours are noise, and the frequency column says why: `kill` appears 3 times,
`fatal`/`hang`/`abort` zero times. Word2Vec needs hundreds of occurrences per word. This is
the corpus-size limitation the proposal anticipated, now measured. The one probe with
enough data, `error` (65×), *does* yield sensible technical neighbours — which shows the
mechanism works when the data allows it.

Settings search: 100d CBOW 0.7043 ≈ 100d skip-gram 0.7042 > 50d (~0.68).

> `python -m src.features.features_tfidf` · `python -m src.features.features_embed`

---

## 6. Models

Fifteen models: five representations × three classifiers. Tuned by 5-fold cross-validation
**inside the training split**; the test split was read once, after all decisions were final.

Two constraints worth stating. `LinearSVC` has no `predict_proba`, so it is wrapped in
`CalibratedClassifierCV` — LIME, SHAP and the demo all require probabilities.
`MultinomialNB` is **impossible** on embeddings, which contain negative values, so the
generative model becomes `GaussianNB` there. That is a real consequence of choosing a dense
representation, not a technicality.

### 6.1 Why a linear model and not a transformer

A fine-tuned BERT would almost certainly beat 0.8275 macro-F1 here. Its absence is a
decision, not an oversight: the deliverable is an **audit** of *which words* the model
relies on, and that question is answerable exactly on a linear model and only
approximately on a transformer.

| Aspect | Linear SVM (what we have) | BERT (contextual transformer) |
|---|---|---|
| Explainability faithfulness | **Mathematically exact** — SHAP is closed-form for a linear model (φᵢ = wᵢ(xᵢ − E[xᵢ])). Zero approximation error. | **Approximation** — requires gradient paths or perturbation sampling (LIME/SHAP); the explanation is itself a fitted model. |
| Tokenisation | Whole words and n-grams; *fatal error* is one feature whose weight is read straight off the coefficient vector. | Subword pieces (`['kill', '##ing']`); uncommon jargon is split into fragments that must be re-aggregated before a word-level importance even exists. |
| Compute cost for explanations | SHAP for the **entire test set** in seconds, on CPU. | Hundreds of forward passes **per sentence** — hours without a dedicated GPU. |
| Masking artefacts | N/A — removing a feature is exact, not simulated. | Masking words creates out-of-distribution sentences the model never trained on, so the measured importance partly reflects the masking. |

Three consequences are specific to this project and decided the matter:

1. **The exactness is load-bearing.** Our headline explainability result is the Spearman
   agreement ρ = 0.872 between LIME and SHAP (§8.2). That number means something *only*
   because SHAP is exact on a linear model — it measures LIME's sampling noise against a
   known ground truth. On a transformer both sides are approximations, and their agreement
   could no longer distinguish "both describe the model" from "both share a sampling bias".
2. **Subword tokenisation attacks the research question directly.** The phenomenon is
   *word-level* lexical bias — whether *kill*, *fatal*, *abort* read as hostility. WordPiece
   splits exactly this low-frequency jargon into fragments, so the quantity being measured
   would first have to be rebuilt by a heuristic the linear model never needs.
3. **The stress-test ladder is a controlled comparison.** §9's result — false alarms falling
   from VADER 70% → GloVe 40% → Word2Vec 28% → TF-IDF 14% — works because each
   representation carries a *known, ordered* amount of general English. BERT is the next
   rung on that axis — a measurement this ladder sets up rather than one it is missing:
   adding it would extend the result, not invalidate it.

**In summary.** *Could* we use BERT? Yes — it is the ideal candidate for testing whether
contextual attention eliminates the vocabulary bias measured here. *Why didn't we?*
A linear model gave a transparent, exact baseline on which
vocabulary bias can be isolated mathematically, without the confounds of deep attention and
expensive sampling. That baseline is a precondition for interpreting what BERT would change.

---

## 7. Results

### 7.1 Test set

| Model | Representation | Kind | Test macro-F1 | Accuracy | neut→neg | neg recall |
|---|---|---|---|---|---|---|
| **tfidf13_svm** | TF-IDF | discriminative | **0.8275** | 0.8308 | 14.4% | 76.1% |
| tfidf13_lr | TF-IDF | discriminative | 0.8233 | 0.8262 | 14.8% | 76.7% |
| tfidf12_lr | TF-IDF | discriminative | 0.8108 | 0.8138 | 14.0% | 75.0% |
| tfidf12_svm | TF-IDF | discriminative | 0.8103 | 0.8154 | 12.8% | 69.3% |
| tfidf11_svm | TF-IDF | discriminative | 0.8081 | 0.8123 | 12.0% | 71.0% |
| tfidf11_lr | TF-IDF | discriminative | 0.8070 | 0.8092 | 13.6% | 75.0% |
| tfidf12_mnb | TF-IDF | generative | 0.7601 | 0.7662 | 12.0% | 62.5% |
| tfidf13_mnb | TF-IDF | generative | 0.7546 | 0.7600 | 13.2% | 61.9% |
| tfidf11_mnb | TF-IDF | generative | 0.7331 | 0.7400 | 8.4% | 56.2% |
| w2v_svm | Word2Vec | discriminative | 0.7059 | 0.7138 | 11.6% | 55.1% |
| glove_lr | GloVe | discriminative | 0.7054 | 0.7077 | 17.2% | 63.1% |
| w2v_lr | Word2Vec | discriminative | 0.7020 | 0.7046 | 20.4% | 64.8% |
| glove_svm | GloVe | discriminative | 0.6936 | 0.7015 | 12.4% | 52.8% |
| w2v_gnb | Word2Vec | generative | 0.5813 | 0.5769 | 49.6% | 89.2% |
| glove_gnb | GloVe | generative | 0.5668 | 0.5677 | 39.2% | 79.5% |
| *majority* | *baseline* | — | *0.1852* | *0.3846* | — | — |
| *VADER* | *baseline* | — | *0.7079* | *0.7154* | *25.2%* | — |

**Outcome #1 — which representation handles SE vocabulary best?**

| Representation | Best | Mean |
|---|---|---|
| **TF-IDF** | **0.8275** | 0.7928 |
| Word2Vec | 0.7059 | 0.6631 |
| GloVe | 0.7054 | 0.6553 |

TF-IDF wins decisively. Notably the n-gram range barely matters (`tfidf11_lr` 0.8070 vs
`tfidf13_lr` 0.8233), which follows from §5.2: the compound phrases that would justify wider
windows are not present.

**Outcome #2 — generative vs discriminative.** Discriminative **0.7694** mean against
**0.6792** generative; best 0.8275 vs 0.7601. McNemar between the best TF-IDF and best
embedding model gives *p* = 4.7e-9. The gap is consistent across all five representations.
GaussianNB suffers worst (0.57–0.58) because its assumption — independent, normally
distributed dimensions — fits embeddings poorly.

**A trap worth naming.** `tfidf11_mnb` has the lowest neutral→negative rate (8.4%), which
looks like the least biased model. It finds only 56% of genuinely negative posts against 76%
for the best model. Its low rate is reluctance to predict "negative" at all — a model that
never did so would score a perfect 0%. The two columns must always be read together.

### 7.2 Error analysis

For `tfidf13_svm`, 36 of 250 neutral test posts (14.4%) were predicted negative:

| Tag | Count |
|---|---|
| other | 17 |
| mixed or contrastive | 13 |
| **contains SE jargon** | **5** |
| short question | 1 |

Only 5 of 36 errors involve jargon. **On this corpus the model's errors are not primarily
caused by technical vocabulary** — consistent with the vocabulary being nearly absent.

---

## 8. Explainability

### 8.1 What the model actually uses

LIME over 120 gold-neutral test posts and exact SHAP over 250, aggregated per token, mean
weight toward *negative*:

| Method | Strongest negative-driving tokens |
|---|---|
| LIME | `afraid, me, extremely, i, m, am, !, was, very` |
| SHAP | `hate, painful, line, me, pain, afraid, will not, can not, still` |

Both are dominated by **first-person and affective language**, not technical vocabulary.
**Zero lexicon words occur three or more times** in the neutral test posts — the fourth
independent confirmation that the jargon is not in this corpus. The Domain Bias Score the
proposal specified therefore cannot be computed from Senti4SD, which is why §9 exists.

### 8.2 LIME and SHAP agree

Spearman rank correlation **0.872** across 80 shared tokens. Two methods with different
mathematics — local linear surrogate vs Shapley allocation — converge on the same account.
That is evidence both describe the model rather than their own sampling noise.

### 8.3 Stability

Across five random seeds on the 20 fixed posts, the mean Jaccard overlap of the top-5 tokens
is **0.907** (sd 0.164). High, but not 1.0: individual explanations should be read as
indicative, and aggregate tables are the safer evidence.

---

## 9. Stress test — the central result

Four analyses established that Senti4SD cannot answer the research question. A 60-sentence
set was written to do so: **40 harsh-but-neutral**, **20 genuine complaints** (10 with no
jargon), including **10 minimal pairs** using the same jargon word with opposite intent.

| Model | False alarm ↓ | Neg recall ↑ | Jargon | Plain | Pairs | Accuracy |
|---|---|---|---|---|---|---|
| glove_gnb | **2.5%** | 70% | 90% | 50% | **90%** | 88.3% |
| tfidf12_lr | 5.0% | 50% | 70% | 30% | 60% | 80.0% |
| tfidf12_svm | 5.0% | 45% | 70% | 20% | 60% | 78.3% |
| **tfidf13_svm** | **7.5%** | **75%** | 80% | 70% | 70% | 86.7% |
| tfidf11_lr | 7.5% | 65% | 70% | 60% | 60% | 83.3% |
| w2v_gnb | 10.0% | 80% | 80% | 80% | 70% | 86.7% |
| tfidf13_mnb | 30.0% | 70% | 90% | 50% | 50% | 68.3% |
| w2v_lr | 45.0% | 65% | 60% | 70% | 40% | 58.3% |
| glove_svm | 55.0% | 75% | 90% | 60% | 50% | 55.0% |
| glove_lr | 62.5% | 85% | 100% | 70% | 50% | 53.3% |
| **VADER** | **70.0%** | 75% | 90% | 60% | 40% | 45.0% |

### The headline

**VADER labels 70% of harsh-but-neutral SE sentences negative.** The best test model,
`tfidf13_svm`, labels **7.5%**, while still catching 75% of genuine complaints.

### Outcome #3, answered in one column

Mean false-alarm rate by representation:

| Representation | Mean false alarm |
|---|---|
| VADER (general lexicon) | **70%** |
| GloVe (general English) | **40%** |
| Word2Vec (this corpus) | **28%** |
| TF-IDF (this corpus) | **14%** |

**The ordering tracks exactly how much general English each representation carries.**
Pretrained GloVe inherits the lexical bias; representations learned on StackOverflow largely
avoid it. This is the project's answer to *"do models rely on domain-appropriate cues or on
general-language bias?"* — the bias is real, inherited from pretraining, and quantifiable.

**Caveat on the table's leader.** `glove_gnb` has the lowest false-alarm rate but the worst
test macro-F1 of any model (0.5668). A cautious model scores well on this metric for free.
The claim should be anchored on `tfidf13_svm`, which is best on real data.

**Jargon vs plain recall.** `tfidf13_svm` catches 80% of jargon-bearing complaints against
70% of plain ones (+10 points); `glove_gnb` shows +40 points. A large gap means the model
needs technical vocabulary to notice hostility at all.

---

## 10. The demo, and eight held-out probes

`python -m app.server` serves **JargonSense**, a Flask interface over the retained
`tfidf13_svm` pipeline. It shows the predicted label, the three class probabilities, and the
sentence shaded token-by-token by LIME weight.

One detail is worth stating because it is easy to get wrong: **the shading encodes
*evidence for* vs *evidence against the prediction*, not negative vs positive.** Red and
green already mean the two sentiment classes everywhere else on the page, so reusing them
for token weights would invert their meaning whenever the prediction is itself negative.
Tokens supporting the predicted class are drawn in the accent, tokens arguing against it in
slate. Each example also declares its gold label, so the page reports whether the model
*agreed* — the demo can fail in public.

### 10.1 Held-out probes

Eight of the sixteen built-in examples are **held-out probes** written for this report. They
appear in neither Senti4SD nor the stress set, and every jargon term in them occurs **zero
times** in the 3,031 training documents — *zombie*, *reap*, *poison*, *watchdog*, *orphan*,
*panic*, *abort*, *deadlock*, *scheduler*. They test the pipeline on vocabulary it
demonstrably never saw.

| Probe | Gold | Predicted | neg / neu / pos | |
|---|---|---|---|---|
| zombie | neutral | neutral | .04 / **.95** / .00 | ✓ |
| poison pill | neutral | neutral | .26 / **.73** / .01 | ✓ |
| watchdog | neutral | neutral | .14 / **.82** / .04 | ✓ |
| kernel panic | neutral | neutral | .12 / **.88** / .00 | ✓ |
| abort · calm | neutral | neutral | .05 / **.83** / .12 | ✓ |
| corruption | negative | negative | **.54** / .45 / .02 | ✓ |
| abort · angry | negative | *neutral* | .23 / .59 / .17 | ✗ |
| praise | positive | *neutral* | .16 / .81 / .03 | ✗ |

**All six harsh-but-neutral probes come back neutral**, several with high confidence, though
they are built entirely from words the model has no feature for. That is the mechanism §9
measures, restated: a TF-IDF model cannot inherit a hostile prior for a word it never saw.

The two failures fail in the **opposite direction to the one the literature warns about**.
Neither is a false alarm on jargon; both are *missed genuine sentiment*. "The service
aborted my transaction **again** and **lost two hours of work**" carries its negativity in
ordinary English, and "**brilliant** work on the lock ordering" carries its praise in a word
with 2 training occurrences. With the affective vocabulary too rare to have earned a strong
weight, the surrounding technical prose drags both to neutral. This is the same weakness the
test set records as 76.1% negative recall — and it is exactly the deficit a pretrained
contextual model would be expected to close (§6.1).

---

## 11. Limitations

1. **The corpus does not contain the phenomenon.** *fatal*, *hang* and *abort* occur zero
   times; *kill* three times. Four analyses confirmed it. The central claim rests on the
   stress set, not on Senti4SD.
2. **The stress set was written by one person** who already knew what the models were
   expected to get wrong. That is author bias. The minimal pairs are a partial defence —
   both halves share the jargon word, so no lexical shortcut is available — but blind
   cross-annotation by a second annotator was not performed.
3. **Word2Vec quality is limited by corpus size** (3,031 documents). Its neighbour lists are
   largely noise, as the proposal anticipated.
4. **LIME and SHAP are approximations.** Measured stability is 0.907 mean Jaccard, not 1.0.
5. **Single corpus, single domain.** Whether the failure mode generalises to other technical
   domains is untested here.
6. **No contextual models.** A fine-tuned transformer might substantially reduce this bias;
   we did not test it. §6.1 argues why — we traded accuracy for exact explainability — but
   it remains a limitation regardless of the justification.
7. **Labels carry annotation noise**, Fleiss' κ = 0.759 — good, not perfect. Some
   "errors" are disagreements the raters also had.

---

## 12. Conclusion

A tuned TF-IDF + linear SVM reaches 0.8275 test macro-F1 on Senti4SD, ahead of VADER by
0.12. Discriminative models beat the generative one decisively, and TF-IDF beats both
embedding families.

The project's more useful finding is methodological. The standard SE sentiment corpus does
not contain the harsh technical vocabulary that motivates SE sentiment research, so the
canonical claim cannot be tested on it — a fact that four independent analyses in this
project surfaced and that we have not seen stated elsewhere. Tested on purpose-built
sentences, the bias is unambiguous and its magnitude tracks how much general English the
representation carries.

---

## References

Calefato, F., Lanubile, F., Maiorano, F., & Novielli, N. (2018). Sentiment Polarity
Detection for Software Development. *Empirical Software Engineering*, 23(3), 1352–1382.

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
*NeurIPS 30*.

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word
Representation. *EMNLP*, 1532–1543.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the
Predictions of Any Classifier. *KDD*, 1135–1144.

---

## Appendix — reproducing every number

```bash
pip install -r requirements.txt
python -m src.acquisition.download
python -m src.extraction.build_se               # corpus, Fleiss' kappa
python -m src.preprocessing.splits --domain se  # frozen splits
python -m tools.run_baselines --domain se       # majority, VADER
python -m tools.run_ablation                    # section 4
python -m src.features.features_tfidf           # section 5.1-5.3
python -m src.features.features_embed           # section 5.4
python -m src.modeling.models_tfidf             # 9 models
python -m src.modeling.models_embed             # 6 models
python -m tools.run_final_evaluation            # section 7
python -m src.evaluation.explain_lime           # section 8
python -m src.evaluation.explain_shap           # section 8
python -m src.evaluation.stress_test            # section 9
python -m pytest -q                             # 90 tests
```

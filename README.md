# SE Sentiment Classifier

This repository contains one production pipeline for sentiment classification
of StackOverflow communication:

```
raw text -> SE cleaning -> TF-IDF (1,3) -> calibrated Linear SVM
```

The retained model is `tfidf13_svm`. It accepts raw text and returns one of
`negative`, `neutral`, or `positive`, with calibrated class probabilities.

The project targets this one corpus and this one pipeline. Alternative
representations and classifiers were evaluated during model selection; only
the retained pipeline is shipped here.

## Results

On the frozen 650-document Senti4SD test split:

| Metric | Score |
|---|---:|
| Accuracy | 0.8308 |
| Macro-F1 | 0.8275 |
| Negative recall | 0.7614 |
| Neutral predicted negative | 14.4% |

On the 60-sentence technical-vocabulary stress set it reaches 0.8667 accuracy,
7.5% false alarms on harsh-but-neutral sentences, and 75% negative recall.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train and evaluate

Run commands from the repository root:

```bash
python -m src.acquisition.download
python -m src.extraction.build_se
python -m src.preprocessing.splits --domain se
python -m src.modeling.models_tfidf
python -m tools.run_final_evaluation
python -m src.evaluation.stress_test
python -m src.evaluation.explain_lime
python -m src.evaluation.explain_shap
python -m pytest -q
```

The trained pipeline is written to `models/se/tfidf13_svm.joblib`.

## Web interface

```bash
python -m app.server
```

The server prints the local URL. The interface provides predictions,
probabilities, LIME token contributions, final test metrics, and stress-test
metrics for the retained pipeline.

## Layout

```text
src/acquisition/download.py        fetch the Senti4SD corpus
src/extraction/clean_text.py       SE text cleaning and placeholders
src/extraction/build_se.py         gold standard -> clean.csv
src/preprocessing/normalize.py     normalization and frozen splits
src/features/features_tfidf.py     TF-IDF (1,3)
src/modeling/models_tfidf.py       calibrated Linear SVM
src/evaluation/                    metrics, LIME, SHAP, stress test
tools/run_final_evaluation.py      frozen test-set evaluation
app/                               single-model web interface
tests/                             retained pipeline contracts
docs/                              proposal, project plan, and report archive
```

# New-token embedding initialisation: mean-of-context vs attention-weighted

Adding new tokens to a pretrained vocabulary requires input embeddings for them, and the common cheap choice is the average of context embeddings. The paper "Token distillation: Attention-aware input embeddings for new tokens" (2026) argues that attention-aware embeddings for new tokens can do better. This repository runs a small, self-contained test of that idea on a text classification task using classical PPMI-SVD word vectors.

**Question:** Does an attention-weighted initialization of new-token embeddings beat a standard mean-of-context initialization on downstream classification of 20 Newsgroups text when about 200 domain tokens are added?

**Dataset:** 20 Newsgroups (headers, footers and quotes removed), fetched via scikit-learn over HTTP. Four downstream categories are used; the other 16 categories form the background corpus for base embeddings. This is not the paper's own dataset.

**Baseline:** each new token is initialised as the unweighted mean of the base vectors of words in its co-occurrence window in the downstream training text.

**Improved:** the same context words, but each is weighted by a softmax over its PPMI score with the new token, then summed.

Each document is the mean of its token vectors; a logistic regression is trained on the training split (regularisation chosen by cross-validation on that split) and scored on the held-out test split. The main metric is macro-F1, with accuracy as a secondary metric.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> K Dobler, D Elliott, G de Melo (2026). *Token distillation: Attention-aware input embeddings for new tokens*. International Conference on. https://proceedings.iclr.cc/paper_files/paper/2026/hash/e3cd4c3748274cfde024873d4ae91600-Abstract-Conference.html

Paper author: Prof. Gerard de Melo, Universität Potsdam (gerard.demelo@hpi.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

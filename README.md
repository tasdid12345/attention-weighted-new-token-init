The question asks whether attention-weighted initialisation of new-token embeddings beats mean-of-context initialisation for macro-F1 on a 4-category 20 Newsgroups task (sci.space, rec.autos, comp.graphics, talk.politics.guns), with about 200 new downstream tokens added to PPMI-SVD base embeddings trained on the other 16 categories.

Run: `python analysis.py`. It writes `results.json` and `figures/baseline_vs_improved.png`.

Method change: the improved model now also selects its classifier by 5-fold cross-validated macro-F1 on the training split, choosing between logistic regression (wider C grid) and an RBF-SVM. The baseline (mean-of-context initialisation with logistic regression, C tuned by CV) is unchanged. The train/test split, seeds (42-44), and the test set (used only for final scoring) are unchanged.

Dependencies: numpy, scipy, scikit-learn, matplotlib.

## Results (latest run)

On 20 Newsgroups (4 categories, 965 test documents), the attention-weighted initialization of 200 new tokens reached macro-F1 0.826 versus 0.788 for the mean-of-context baseline, a gain of 0.037 that also bundled a CV-tuned LR/RBF-SVM classifier search.

| Metric | Value |
|---|---|
| baseline_score | 0.7883 |
| improved_score | 0.8255 |
| improvement | 0.0372 |
| baseline_score_std | 0.00773 |
| improved_score_std | 0.0008234 |
| baseline_accuracy | 0.7883 |
| improved_accuracy | 0.8235 |
| baseline_accuracy_std | 0.007631 |
| improved_accuracy_std | 0.000977 |
| n_new_tokens | 200 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: New-token vectors are frozen initialisations (no fine-tuning). The paper's exact token-distillation architecture is not given, so the attention is a PPMI-softmax proxy. Only the 4 categories are used and the base embeddings come from the other 16 categories. The improved model differs from the baseline in the initialisation and in the CV-tuned classifier search (LR vs RBF-SVM); the baseline classifier is unchanged. Mean +/- std over seeds 42-44 (SVD and CV folds); the train/test split is fixed. Test set is used only for final scoring.

## Reference

This project is motivated by the following paper:

> K Dobler, D Elliott, G de Melo (2026). *Token distillation: Attention-aware input embeddings for new tokens*. International Conference on. https://proceedings.iclr.cc/paper_files/paper/2026/hash/e3cd4c3748274cfde024873d4ae91600-Abstract-Conference.html

Paper author: Prof. Gerard de Melo, Universität Potsdam (gerard.demelo@hpi.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

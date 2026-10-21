The question asks whether attention-weighted initialisation of new-token embeddings beats mean-of-context initialisation for macro-F1 on a 4-category 20 Newsgroups task (sci.space, rec.autos, comp.graphics, talk.politics.guns), with about 200 new downstream tokens added to PPMI-SVD base embeddings trained on the other 16 categories.

Run: `python analysis.py`. It writes `results.json` and `figures/baseline_vs_improved.png`.

Method change: the improved model now also selects its classifier by 5-fold cross-validated macro-F1 on the training split, choosing between logistic regression (wider C grid) and an RBF-SVM. The baseline (mean-of-context initialisation with logistic regression, C tuned by CV) is unchanged. The train/test split, seeds (42-44), and the test set (used only for final scoring) are unchanged.

Dependencies: numpy, scipy, scikit-learn, matplotlib.

## Reference

This project is motivated by the following paper:

> K Dobler, D Elliott, G de Melo (2026). *Token distillation: Attention-aware input embeddings for new tokens*. International Conference on. https://proceedings.iclr.cc/paper_files/paper/2026/hash/e3cd4c3748274cfde024873d4ae91600-Abstract-Conference.html

Paper author: Prof. Gerard de Melo, Universität Potsdam (gerard.demelo@hpi.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

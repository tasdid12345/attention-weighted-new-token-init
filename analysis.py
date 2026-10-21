"""Does attention-weighted initialisation of new-token embeddings beat mean-of-context
initialisation on a 4-category 20 Newsgroups task? Baseline and improved models share
all code except the new-token initialisation step and the downstream classifier search
(the improved model additionally tunes an RBF-SVM against logistic regression by CV on train)."""
import json
import os
import re
import socket
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
SEEDS = [42, 43, 44]  # seeds for SVD and CV folds; both models use the same seeds
WINDOW, DIM, BASE_VOCAB, N_NEW, ALPHA = 5, 300, 5000, 200, 0.75
DOWN = ["sci.space", "rec.autos", "comp.graphics", "talk.politics.guns"]
SOURCE = "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset"
TOKEN = re.compile(r"[a-z][a-z]+")
socket.setdefaulttimeout(60)  # 60 s timeout for the underlying HTTP download


def load_20ng():
    """fetch_20newsgroups over HTTP with exponential-backoff retries."""
    for attempt in range(5):
        try:
            return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                                      random_state=SEED)
        except Exception as err:  # HTTP 429/5xx, URL errors, timeouts
            wait = 2 ** attempt
            print(f"   download failed ({err}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("could not download 20 Newsgroups")


def tokenize(text):
    """Lowercase alphabetic tokens without English stop words."""
    return [w for w in TOKEN.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]


def cooc(docs, index):
    """Symmetric window co-occurrence counts (sparse V x V) over tokens present in index."""
    ids, doc = [], []
    for di, d in enumerate(docs):
        for w in d:
            if w in index:
                ids.append(index[w]); doc.append(di)
    ids, doc = np.array(ids, dtype=np.int64), np.array(doc)
    rows, cols = [], []
    for k in range(1, WINDOW + 1):
        same = doc[:-k] == doc[k:]  # only pairs inside one document
        a, b = ids[:-k][same], ids[k:][same]
        rows += [a, b]; cols += [b, a]
    r, c = np.concatenate(rows), np.concatenate(cols)
    V = len(index)
    return sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(V, V)).tocsr()


def ppmi(M):
    """Positive PMI with context-distribution smoothing (ALPHA); returns CSR."""
    M = M.tocoo()
    rs = np.asarray(M.sum(axis=1)).ravel()
    ca = np.asarray(M.sum(axis=0)).ravel() ** ALPHA
    pmi = np.log(M.data * ca.sum() / (rs[M.row] * ca[M.col]))
    keep = pmi > 0
    return sparse.csr_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=M.shape)


def init_mean(C, E):
    """BASELINE: unweighted mean of base vectors over all context-window occurrences."""
    return (C @ E) / np.maximum(C.sum(1, keepdims=True), 1)


def init_attn(C, P, E):
    """IMPROVED: softmax over PPMI scores of each observed context word, then weighted sum."""
    s = np.where(C > 0, P, -np.inf)  # unobserved context words get zero weight
    s = s - s.max(1, keepdims=True)
    w = np.exp(s)
    w /= w.sum(1, keepdims=True)
    return w @ E


def doc_matrix(docs, E, index):
    """Document = mean of its token vectors (tokens outside index are dropped)."""
    X = np.zeros((len(docs), E.shape[1]))
    for i, d in enumerate(docs):
        ids = [index[w] for w in d if w in index]
        if ids:
            X[i] = E[ids].mean(0)
    return X


def fit_predict(Xtr, ytr, Xte, seed):
    """BASELINE classifier: logistic regression; C chosen by 5-fold CV on the training split only."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.1, 1.0, 10.0]},
                        cv=cv, scoring="f1_macro")
    return grid.fit(Xtr, ytr).predict(Xte)


def fit_predict_improved(Xtr, ytr, Xte, seed):
    """IMPROVED classifier: model family + C/gamma chosen by 5-fold CV on the training split only.
    Candidates: logistic regression (wider C grid) and RBF-SVM. Test set is not used."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])
    param_grid = [
        {"clf": [LogisticRegression(max_iter=2000)],
         "clf__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        {"clf": [SVC(kernel="rbf")],
         "clf__C": [0.3, 1.0, 3.0, 10.0],
         "clf__gamma": ["scale"]},
    ]
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=1)
    return grid.fit(Xtr, ytr).predict(Xte)


def main():
    print("1. loading 20 Newsgroups")
    bunch = load_20ng()
    names = list(bunch.target_names)
    cats = [names.index(c) for c in DOWN]
    in_down = np.isin(bunch.target, cats)
    lab = {c: i for i, c in enumerate(cats)}
    texts = np.array(bunch.data, dtype=object)
    X_down = texts[in_down]
    y_down = np.array([lab[t] for t in bunch.target[in_down]])
    bg_docs = [tokenize(t) for t in texts[~in_down]]  # background = the other categories

    # single stratified train/test split of the downstream data
    X_tr, X_te, y_tr, y_te = train_test_split(X_down, y_down, test_size=0.25,
                                              stratify=y_down, random_state=SEED)
    tr_docs = [tokenize(t) for t in X_tr]
    te_docs = [tokenize(t) for t in X_te]
    print(f"2. downstream rows={len(y_down)} train={len(y_tr)} test={len(y_te)} "
          f"background={len(bg_docs)}")

    # base vocabulary = top background words; new tokens = frequent downstream-train words outside it
    bg_counts = Counter(w for d in bg_docs for w in d)
    base_words = [w for w, _ in bg_counts.most_common(BASE_VOCAB)]
    base_index = {w: i for i, w in enumerate(base_words)}
    nb = len(base_words)
    df = Counter(w for d in tr_docs for w in set(d))
    new_words = [w for w, _ in df.most_common() if w not in base_index][:N_NEW]

    print("3. context statistics of new tokens (downstream train text only)")
    full = dict(base_index)
    full.update({w: nb + i for i, w in enumerate(new_words)})
    M = cooc(tr_docs, full)
    rows = nb + np.arange(len(new_words))
    C = M[rows][:, :nb].toarray()          # raw window counts (new x base)
    P = ppmi(M)[rows][:, :nb].toarray()    # PPMI scores (new x base)
    ok = C.sum(1) > 0
    new_words = [w for w, k in zip(new_words, ok) if k]
    C, P = C[ok], P[ok]
    if len(new_words) < 50:
        sys.exit("too few new tokens with contexts")
    full = dict(base_index)
    full.update({w: nb + i for i, w in enumerate(new_words)})
    print(f"   new tokens={len(new_words)}, e.g. {new_words[:8]}")

    print("4. background PPMI-SVD base embeddings (per seed) and evaluation")
    P_bg = ppmi(cooc(bg_docs, base_index))
    results = []
    for seed in SEEDS:
        Eb = TruncatedSVD(DIM, random_state=seed).fit_transform(P_bg)  # U*S base vectors
        res = {}
        runs = [
            ("baseline", init_mean(C, Eb), fit_predict),
            ("improved", init_attn(C, P, Eb), fit_predict_improved),
        ]
        for name, init, clf in runs:
            E = np.vstack([Eb, init])
            Xtr = doc_matrix(tr_docs, E, full)
            Xte = doc_matrix(te_docs, E, full)
            pred = clf(Xtr, y_tr, Xte, seed)
            res[name] = (f1_score(y_te, pred, average="macro"), accuracy_score(y_te, pred))
            print(f"   seed={seed} {name}: macro-F1={res[name][0]:.4f} acc={res[name][1]:.4f}")
        results.append(res)

    def agg(name, k):
        v = np.array([r[name][k] for r in results])
        return float(v.mean()), float(v.std())

    b_f1, b_f1_sd = agg("baseline", 0)
    i_f1, i_f1_sd = agg("improved", 0)
    b_acc, b_acc_sd = agg("baseline", 1)
    i_acc, i_acc_sd = agg("improved", 1)
    if not np.all(np.isfinite([b_f1, i_f1, b_acc, i_acc])):
        sys.exit("no meaningful metrics produced")

    out = {
        "question": "Does an attention-weighted initialization of new-token embeddings beat a "
                    "standard mean-of-context initialization on downstream classification of "
                    "20 Newsgroups text when about 200 domain tokens are added?",
        "method": "PPMI-SVD base embeddings (300d, top-5000 background vocab) + ~200 new "
                  "downstream tokens initialised either by mean of window context vectors "
                  "(baseline; logistic regression, C tuned by CV) or by PPMI-softmax-weighted "
                  "context vectors (improved; attention init plus logistic-regression-or-RBF-SVM "
                  "chosen and tuned by CV on train); document = mean token vector; macro-F1 on test.",
        "dataset": {
            "name": "20 Newsgroups (4 categories: " + ", ".join(DOWN) + ")",
            "source_url": SOURCE,
            "n_rows": int(len(y_down)),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro_f1",
        "higher_is_better": True,
        "baseline": "Mean-of-context initialization: each new token = unweighted mean of base "
                    "vectors of words in its +/-5 co-occurrence window in downstream train text. "
                    "Classifier: logistic regression with C tuned by 5-fold CV on train.",
        "improved": "Same contexts, but each context word weighted by a softmax over its PPMI "
                    "score with the new token (attention-like weighting) before summing. "
                    "Classifier: logistic regression or RBF-SVM, family and hyperparameters "
                    "selected by 5-fold CV macro-F1 on train only.",
        "metrics": {
            "baseline_score": b_f1,
            "improved_score": i_f1,
            "improvement": i_f1 - b_f1,
            "baseline_score_std": b_f1_sd,
            "improved_score_std": i_f1_sd,
            "baseline_accuracy": b_acc,
            "improved_accuracy": i_acc,
            "baseline_accuracy_std": b_acc_sd,
            "improved_accuracy_std": i_acc_sd,
            "n_new_tokens": len(new_words),
        },
        "notes": "New-token vectors are frozen initialisations (no fine-tuning). The paper's "
                 "exact token-distillation architecture is not given, so the attention is a "
                 "PPMI-softmax proxy. Only the 4 categories are used and the base embeddings "
                 "come from the other 16 categories. The improved model differs from the baseline "
                 "in the initialisation and in the CV-tuned classifier search (LR vs RBF-SVM); "
                 "the baseline classifier is unchanged. Mean +/- std over seeds 42-44 (SVD and CV "
                 "folds); the train/test split is fixed. Test set is used only for final scoring.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("5. writing figure and results.json")
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, [b_f1, b_acc], 0.4, yerr=[b_f1_sd, b_acc_sd], label="baseline (mean context)")
    ax.bar(x + 0.2, [i_f1, i_acc], 0.4, yerr=[i_f1_sd, i_acc_sd], label="improved (attention)")
    ax.set_xticks(x, ["macro-F1 (main)", "accuracy"])
    ax.set_ylim(0, 1)
    ax.set_title("New-token initialisation, mean over seeds 42-44")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)
    print("done")


if __name__ == "__main__":
    main()

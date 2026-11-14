"""Selective-QA methods. Each method maps an example to (answer, confidence); the system answers
when confidence >= tau and abstains otherwise.

Baselines
  B1  reader-prob    top span, conf = its softmax prob (null answer included in the softmax)
  B2  reader-margin  top span, conf = span logit - null logit
Verification (answer unchanged, only the confidence comes from NLI)
  V   nli:<verifier>|<hyp>|<chunk>      conf = P(entail) of the top span
Proposed: NLI re-ranking of the top-k candidates
  R   rerank:<verifier>|<hyp>|<chunk>   pick argmax_k  lam*reader_prob_k + (1-lam)*P(entail)_k
                                        (reader probs renormalised over the k spans),
                                        conf = same mix with the absolute prob; lam tuned on calib
  C   combo:<...>                       top span, conf = logistic regression on
                                        [reader prob, margin, P(entail), P(contra)] fit on calib
  RC  rerank + combo                    answer picked as in R, conf = logistic regression on
                                        [prob_k, margin_k, P(entail)_k, P(contra)_k, P(entail)_top1,
                                        null prob] of the picked span k, fit on calib
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import best_threshold, f1_score, is_correct


def nli_of(ex, key, k):
    """[ent, con] for candidate k under score key 'verifier|hyp|chunk'."""
    return ex["nli"][key][k]


def _row(ex, k, conf):
    cands = ex["candidates"]
    ans = cands[k]["text"] if cands else ""
    return {"answer": ans, "conf": float(conf),
            "correct": is_correct(ans, ex["answers"]),
            "f1": f1_score(ans, ex["answers"]) if ex["answers"] else 0.0}


def reader_prob(ex):
    return _row(ex, 0, ex["candidates"][0]["prob"] if ex["candidates"] else 0.0)


def reader_margin(ex):
    c = ex["candidates"]
    return _row(ex, 0, c[0]["logit"] - ex["null_logit"] if c else -1e9)


def nli_verify(ex, key):
    if not ex["candidates"]:
        return _row(ex, 0, 0.0)
    return _row(ex, 0, nli_of(ex, key, 0)[0])


def rerank_choice(ex, key, lam):
    """Index of the candidate R picks (spans only; reader probs renormalised over them)."""
    c = ex["candidates"]
    total = sum(x["prob"] for x in c) or 1.0
    return int(np.argmax([lam * c[k]["prob"] / total + (1 - lam) * nli_of(ex, key, k)[0]
                          for k in range(len(c))]))


def nli_rerank(ex, key, lam):
    """Selection uses reader probs renormalised over the spans (otherwise, when the null answer takes
    almost all the mass, every span prob is ~0 and the choice would be NLI-only); the confidence keeps
    the absolute prob so that a likely-unanswerable question still gets a low score."""
    c = ex["candidates"]
    if not c:
        return _row(ex, 0, 0.0)
    k = rerank_choice(ex, key, lam)
    return _row(ex, k, lam * c[k]["prob"] + (1 - lam) * nli_of(ex, key, k)[0])


def combo_features(ex, key):
    c = ex["candidates"]
    if not c:
        return [0.0, -10.0, 0.0, 0.0]
    ent, con = nli_of(ex, key, 0)
    return [c[0]["prob"], c[0]["logit"] - ex["null_logit"], ent, con]


class Combo:
    def __init__(self, key):
        self.key = key
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, calib):
        X = [combo_features(ex, self.key) for ex in calib]
        y = [reader_prob(ex)["correct"] for ex in calib]
        self.clf.fit(X, y)
        return self

    def __call__(self, ex):
        p = self.clf.predict_proba([combo_features(ex, self.key)])[0, 1]
        return _row(ex, 0, p)


def rerank_combo_features(ex, key, k):
    c = ex["candidates"]
    if not c:
        return [0.0, -10.0, 0.0, 0.0, 0.0, 1.0]
    ent, con = nli_of(ex, key, k)
    return [c[k]["prob"], c[k]["logit"] - ex["null_logit"], ent, con, nli_of(ex, key, 0)[0], ex["null_prob"]]


class RerankCombo:
    """R picks the answer, a logistic regression (with the reader's null signal) gives the confidence."""

    def __init__(self, key, lam):
        self.key, self.lam = key, lam
        self.clf = LogisticRegression(max_iter=1000)

    def _pick(self, ex):
        return rerank_choice(ex, self.key, self.lam) if ex["candidates"] else 0

    def fit(self, calib):
        ks = [self._pick(ex) for ex in calib]
        X = [rerank_combo_features(ex, self.key, k) for ex, k in zip(calib, ks)]
        y = [_row(ex, k, 0.0)["correct"] for ex, k in zip(calib, ks)]
        self.clf.fit(X, y)
        return self

    def __call__(self, ex):
        k = self._pick(ex)
        p = self.clf.predict_proba([rerank_combo_features(ex, self.key, k)])[0, 1]
        return _row(ex, k, p)


def calib_threshold(method, calib):
    """tau maximising SQuAD2-F1 of `method` on the calib split (the system answers when conf >= tau)."""
    out = [method(ex) for ex in calib]
    return best_threshold([o["conf"] for o in out], [o["f1"] for o in out], [ex["answers"] for ex in calib])

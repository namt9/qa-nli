"""SQuAD EM/F1 and selective-QA metrics (risk-coverage, AURC, AUROC, bootstrap CI)."""
import collections
import re
import string

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------- official SQuAD normalisation ----------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, golds):
    if not golds:
        return float(pred == "")
    return max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)


def f1_score(pred, golds):
    if not golds:
        return float(pred == "")
    best = 0.0
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        common = collections.Counter(p_toks) & collections.Counter(g_toks)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p_toks), same / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def is_correct(pred, golds, f1_threshold=0.5):
    """An answered prediction is correct if it overlaps a gold answer (F1 >= threshold).
    Answering an unanswerable question (no golds) is always wrong."""
    return bool(golds) and f1_score(pred, golds) >= f1_threshold


# ---------- selective QA ----------
def risk_coverage(conf, correct):
    """Sort by confidence (desc); return coverage and risk (error rate among answered) at each cut."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    n = np.arange(1, len(c) + 1)
    return n / len(c), 1 - np.cumsum(c) / n


def aurc(conf, correct):
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(conf, correct)
    return float(risk.mean())


def acc_at_coverage(conf, correct, cov):
    coverage, risk = risk_coverage(conf, correct)
    k = max(1, int(round(cov * len(coverage))))
    return float(1 - risk[k - 1])


def auroc(conf, correct):
    correct = np.asarray(correct)
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, conf))


def squad2_f1_with_threshold(conf, f1_if_answered, golds_list, tau):
    """SQuAD 2.0 style F1 when abstaining below tau (abstain on unanswerable = 1, on answerable = 0)."""
    total = 0.0
    for c, f1, golds in zip(conf, f1_if_answered, golds_list):
        total += f1 if c >= tau else float(not golds)
    return total / len(conf)


def best_threshold(conf, f1_if_answered, golds_list):
    """Pick tau maximising SQuAD2 F1 (use on the calib split only)."""
    cands = np.unique(np.concatenate([[-np.inf], np.asarray(conf, float)]))
    scores = [squad2_f1_with_threshold(conf, f1_if_answered, golds_list, t) for t in cands]
    return float(cands[int(np.argmax(scores))])


def bootstrap_ci(metric_fn, *arrays, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI of metric_fn(*resampled arrays). Arrays are resampled jointly (paired)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(*[a[idx] for a in arrays]))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(metric_fn, conf_a, correct_a, conf_b, correct_b, n_boot=1000, seed=0, alpha=0.05):
    """metric(A) - metric(B) on the same questions, with a paired bootstrap CI and a two-sided p-value
    (how often the resampled difference falls on the other side of 0). A and B may give different
    answers, so each keeps its own correctness array; only the resampled question ids are shared."""
    rng = np.random.default_rng(seed)
    ca, ya, cb, yb = (np.asarray(a, float) for a in (conf_a, correct_a, conf_b, correct_b))
    n = len(ca)
    diff = metric_fn(ca, ya) - metric_fn(cb, yb)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = metric_fn(ca[idx], ya[idx]) - metric_fn(cb[idx], yb[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return float(diff), float(lo), float(hi), float(p)

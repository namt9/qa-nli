"""Edge cases of SQuAD 2.0 questions, as boolean flags per question id
(shared by scripts/edge_cases.py and scripts/error_analysis.py)."""
import json
import os
from itertools import combinations

from .hypothesis import concat, template
from .metrics import f1_score
from .text import jaccard, negated, overlap, qtype, words
from .verifier import split_sentences

# flag, which questions it can apply to ("ans" / "unans" / "all"), name, how it is detected
EDGE_CASES = [
    ("minimal_pair", "unans", "Cặp gần giống, nhãn ngược",
     "cùng đoạn có một câu hỏi CÓ đáp án trùng ≥ 60% từ nội dung"),
    ("full_overlap", "unans", "Mọi từ của câu hỏi có trong đoạn", "overlap câu hỏi–đoạn văn = 1"),
    ("negation", "all", "Câu hỏi phủ định", "có not / never / no / n't …"),
    ("odd_question", "all", "Câu hỏi lỗi, thiếu từ để hỏi", "không có từ để hỏi hoặc dưới 5 từ"),
    ("template_fallback", "all", "Không đổi được thành câu khẳng định", "luật template phải dùng ghép thô"),
    ("long_context", "all", "Đoạn văn dài", "trên 250 từ"),
    ("ambiguous_gold", "ans", "Các đáp án gold không thống nhất", "≥ 2 đáp án gold, F1 giữa hai đáp án < 0.5"),
    ("long_answer", "ans", "Đáp án dài", "đáp án gold ngắn nhất ≥ 6 từ"),
    ("cross_sentence", "ans", "Đáp án vắt qua hai câu", "span gold không nằm trọn trong một câu"),
    ("repeated_answer", "ans", "Đáp án lặp lại trong đoạn", "chuỗi đáp án xuất hiện ≥ 2 lần"),
    ("numeric", "ans", "Đáp án là số / ngày", "đáp án gold có chữ số"),
]


def load_dev_spans(data_dir="data"):
    """id -> list of (answer_start, text) of the gold answers (empty for unanswerable questions)."""
    with open(os.path.join(data_dir, "squad_v2_dev.json"), encoding="utf-8") as f:
        raw = json.load(f)["data"]
    return {qa["id"]: [(a["answer_start"], a["text"]) for a in qa["answers"]]
            for art in raw for para in art["paragraphs"] for qa in para["qas"]}


def _sentence_spans(context):
    spans, pos = [], 0
    for s in split_sentences(context):
        at = context.find(s, pos)
        if at >= 0:
            spans.append((at, at + len(s)))
            pos = at + len(s)
    return spans


def edge_flags(rows, spans):
    """rows: id -> example (question, context, answers); spans: from load_dev_spans.
    Returns id -> set of edge-case flags."""
    answerable_by_ctx = {}
    for r in rows.values():
        if r["answers"]:
            answerable_by_ctx.setdefault(r["context"], []).append(r["question"])
    flags = {}
    for i, r in rows.items():
        q, ctx, f = r["question"], r["context"], set()
        if negated(q):
            f.add("negation")
        if qtype(q) == "other" or len(words(q)) < 5:
            f.add("odd_question")
        if template(q, "ANSWER") == concat(q, "ANSWER"):
            f.add("template_fallback")
        if len(words(ctx)) > 250:
            f.add("long_context")
        if not r["answers"]:
            if any(jaccard(q, q2) >= 0.6 for q2 in answerable_by_ctx.get(ctx, [])):
                f.add("minimal_pair")
            if overlap(q, ctx) >= 1.0:
                f.add("full_overlap")
        else:
            golds = r["answers"]
            if any(f1_score(a, [b]) < 0.5 for a, b in combinations(golds, 2)):
                f.add("ambiguous_gold")
            if min(len(words(a)) for a in golds) >= 6:
                f.add("long_answer")
            if any(ch.isdigit() for ch in golds[0]):
                f.add("numeric")
            if spans.get(i):
                start, text = spans[i][0]
                if ctx.count(text) >= 2:
                    f.add("repeated_answer")
                if not any(a <= start and start + len(text) <= b for a, b in _sentence_spans(ctx)):
                    f.add("cross_sentence")
        flags[i] = f
    return flags


def load_dev_plausible(data_dir="data"):
    """id -> (answer_start, text) of the first plausible answer of an unanswerable question."""
    with open(os.path.join(data_dir, "squad_v2_dev.json"), encoding="utf-8") as f:
        raw = json.load(f)["data"]
    return {qa["id"]: (qa["plausible_answers"][0]["answer_start"], qa["plausible_answers"][0]["text"])
            for art in raw for para in art["paragraphs"] for qa in para["qas"] if qa.get("plausible_answers")}


def sentence_at(context, start):
    """The sentence of `context` that contains character `start` (None if not found)."""
    for a, b in _sentence_spans(context):
        if a <= start < b:
            return context[a:b]
    return None

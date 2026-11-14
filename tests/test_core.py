import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify.hypothesis import template  # noqa: E402
from qaverify.metrics import aurc, best_threshold, f1_score, is_correct, risk_coverage  # noqa: E402
from qaverify.verifier import chunk_passage, split_sentences  # noqa: E402


def test_f1_and_correct():
    assert f1_score("the Eiffel Tower", ["Eiffel Tower"]) == 1.0
    assert is_correct("in 1945", ["1945"])
    assert not is_correct("1945", [])  # answering an unanswerable question is wrong
    assert f1_score("", []) == 1.0


def test_risk_coverage_and_aurc():
    cov, risk = risk_coverage([0.9, 0.8, 0.1], [1, 1, 0])
    assert np.allclose(cov, [1 / 3, 2 / 3, 1])
    assert np.allclose(risk, [0, 0, 1 / 3])
    # perfect ranking beats reversed ranking
    assert aurc([0.9, 0.8, 0.1], [1, 1, 0]) < aurc([0.1, 0.2, 0.9], [1, 1, 0])


def test_best_threshold_abstains_on_unanswerable():
    # high conf answerable correct, low conf unanswerable -> tau should abstain on the second
    tau = best_threshold([0.9, 0.2], [1.0, 0.0], [["x"], []])
    assert 0.2 < tau <= 0.9


def test_template():
    assert template("Who wrote Hamlet?", "Shakespeare") == "Shakespeare wrote Hamlet."
    assert template("What is the capital of France?", "Paris") == "Paris is the capital of France."
    assert template("When did the war end?", "1945") == "The war end in 1945."
    assert template("How many people live in Hanoi?", "8 million") == "8 million people live in Hanoi."
    assert template("In what year did Tesla die?", "1943") == "Tesla die in 1943."
    assert template("The war ended in what year?", "1945") == "The war ended in 1945."


def test_chunking():
    text = "Paris is big. It has 2 million people. The Seine flows through it. Dr. Who is not here."
    sents = split_sentences(text)
    assert sents[0] == "Paris is big." and len(sents) >= 4
    assert chunk_passage(text, "full") == [text]
    assert len(chunk_passage(text, "win3")) == len(sents) - 2


def test_rerank_uses_relative_reader_prob():
    from qaverify.methods import nli_rerank
    # null takes almost all the mass; reader still clearly prefers candidate 0 among the spans
    ex = {"answers": ["2 million"], "null_logit": 5.0,
          "candidates": [{"text": "2 million", "prob": 0.009, "logit": 1.0},
                         {"text": "75,000", "prob": 0.001, "logit": -1.0}],
          "nli": {"k": [[0.30, 0.0], [0.42, 0.0]]}}
    r = nli_rerank(ex, "k", lam=0.7)
    assert r["answer"] == "2 million" and r["correct"]
    assert r["conf"] < 0.2  # confidence stays low because the absolute prob is tiny


def test_paired_bootstrap_diff():
    from qaverify.metrics import paired_bootstrap_diff
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good, bad = y + rng.normal(0, 0.3, 300), rng.normal(0, 1, 300)
    d, lo, hi, p = paired_bootstrap_diff(aurc, good, y, bad, y, n_boot=300)
    assert d < 0 and hi < 0 and p < 0.01  # good ranking has clearly lower AURC
    d, lo, hi, p = paired_bootstrap_diff(aurc, good, y, good, y, n_boot=50)
    assert d == lo == hi == 0 and p == 1.0


def test_rerank_combo_answers_like_rerank():
    from qaverify.methods import RerankCombo, nli_rerank
    exs = []
    for i in range(20):
        right = i % 2  # NLI favours the correct candidate 1 on odd ids
        exs.append({"answers": ["b"], "null_logit": 0.0, "null_prob": 0.1 * (i % 3),
                    "candidates": [{"text": "a", "prob": 0.5, "logit": 1.0},
                                   {"text": "b", "prob": 0.4, "logit": 0.8}],
                    "nli": {"k": [[0.9 - 0.8 * right, 0.0], [0.1 + 0.8 * right, 0.0]]}})
    rc = RerankCombo("k", lam=0.3).fit(exs)
    for ex in exs:
        out = rc(ex)
        assert out["answer"] == nli_rerank(ex, "k", 0.3)["answer"] and 0 <= out["conf"] <= 1
    # confidence ranks the correct picks above the wrong ones
    assert rc(exs[1])["conf"] > rc(exs[0])["conf"]


def test_calib_threshold_matches_best_threshold():
    from qaverify.methods import calib_threshold, reader_prob
    calib = [{"answers": ["a"], "candidates": [{"text": "a", "prob": 0.9, "logit": 2.0}]},
             {"answers": [], "candidates": [{"text": "b", "prob": 0.2, "logit": 0.0}]}]
    tau = calib_threshold(reader_prob, calib)
    assert 0.2 < tau <= 0.9  # answers the confident correct one, abstains on the unanswerable one


def test_chunk_with_span():
    from qaverify.verifier import chunk_with_span
    text = "Paris is big. It has 2 million people. The Seine flows through it. Dr. Who is not here."
    start = text.index("2 million")
    assert chunk_with_span(text, start, "2 million", ("sent",)) == "It has 2 million people."
    win = chunk_with_span(text, start, "2 million", ("win3",))
    assert "2 million" in win and win.startswith("Paris is big.")
    # a span crossing a sentence boundary fits a window but no single sentence
    cross = text.index("people. The")
    assert chunk_with_span(text, cross, "people. The", ("sent",)) is None
    assert chunk_with_span(text, cross, "people. The", ("sent", "win3")) is not None


def test_text_helpers():
    from qaverify.text import negated, overlap, qtype, unanswerable_type
    assert negated("Why didn't the plan work?") and negated("What was never built?")
    assert not negated("What was built?")
    assert qtype("How many people live here?") == "how many"
    assert overlap("Who built the tower?", "The tower was built by Eiffel.") == 1.0
    sent = "In 1990 the largest city in Texas was Houston."
    assert unanswerable_type("What was not the largest city in Texas in 1990?", sent) == "NEG"
    assert unanswerable_type("What was the smallest city in Texas in 1990?", sent) == "ANT"
    assert unanswerable_type("What was the largest city in Ohio in 1990?", sent) == "ENT"
    assert unanswerable_type("What was the largest city in Texas?", sent) == "OTHER"


def test_edge_flags():
    from qaverify.edge import edge_flags
    ctx = "In 1990 the largest city in Texas was Houston. Houston grew fast. Houston is big."
    rows = {
        "a": {"question": "What was the largest city in Texas in 1990?", "context": ctx,
              "answers": ["Houston", "the city of Houston in the south"]},
        "u": {"question": "What was not the largest city in Texas in 1990?", "context": ctx, "answers": []},
        "u2": {"question": "Which was the largest city in Texas in 1990?", "context": ctx, "answers": []},
    }
    spans = {"a": [(ctx.index("Houston"), "Houston")], "u": [], "u2": []}
    f = edge_flags(rows, spans)
    assert "repeated_answer" in f["a"] and "ambiguous_gold" in f["a"] and "numeric" not in f["a"]
    assert "cross_sentence" not in f["a"]
    assert {"negation", "minimal_pair"} <= f["u"] and "full_overlap" not in f["u"]  # "not" is not in the passage
    assert {"minimal_pair", "full_overlap"} <= f["u2"] and "negation" not in f["u2"]

"""Turn a (question, answer) pair into an NLI hypothesis.

Three variants are compared in the ablation:
  concat   - "question answer" (Chen et al., 2021 show this is a surprisingly strong baseline)
  template - rule-based wh-substitution into a declarative sentence (falls back to concat)
  llm      - an instruction-tuned LLM rewrites (q, a) as one declarative sentence (optional, GPU)
"""
import re

AUX = r"(is|was|are|were)"
DO = r"(did|does|do)"
PREP = {"when": "in", "where": "in", "why": "because", "how": "by"}


def concat(q, a):
    return f"{q.strip()} {a.strip()}"


def _clean(q):
    q = q.strip().rstrip("?").strip()
    return re.sub(r"\s+", " ", q)


def template(q, a):
    """Best-effort declarative form. Grammar is not perfect (e.g. verbs stay in base form
    after removing 'did'), but NLI models are fairly robust to that."""
    s, a = _clean(q), a.strip()
    rules = [
        # "In what year did X happen" -> "X happen in A"
        (rf"^(in|on|at|from|during|by|since|for)\s+(what|which)\s+\w+\s+(?:{DO}|{AUX})\s+(.+)$",
         lambda m: f"{m.group(5)} {m.group(1).lower()} {a}"),
        # "How many people live in X" -> "A people live in X"
        (r"^how\s+(many|much)\s+(.+)$", lambda m: f"{a} {m.group(2)}"),
        # "Who was the first president" -> "A was the first president"
        (rf"^(who|what|which)\s+{AUX}\s+(.+)$", lambda m: f"{a} {m.group(2)} {m.group(3)}"),
        # "What year did the war end" / "What did Tesla invent" -> "the war end A"
        (rf"^(what|which)(?:\s+\w+)?\s+{DO}\s+(.+)$", lambda m: f"{m.group(3)} {a}"),
        # "When did the war end" -> "the war end in A"
        (rf"^(when|where|why|how)\s+{DO}\s+(.+)$",
         lambda m: f"{m.group(3)} {PREP[m.group(1).lower()]} {a}"),
        # "When was the treaty signed" -> "the treaty signed in A"
        (rf"^(when|where)\s+{AUX}\s+(.+)$",
         lambda m: f"{m.group(3)} {PREP[m.group(1).lower()]} {a}"),
        # Subject questions: "Who wrote Hamlet" -> "A wrote Hamlet"
        (r"^(who|what|which\s+\w+)\s+(?!(?:is|was|are|were|did|does|do|can|could|will|would)\b)(.+)$",
         lambda m: f"{a} {m.group(2)}"),
    ]
    for pattern, build in rules:
        m = re.match(pattern, s, flags=re.IGNORECASE)
        if m:
            return _finish(build(m))
    # Wh-word in the middle: "The war ended in what year" -> "The war ended in A"
    m = re.search(r"\b(what|which|who|whom|when|where)(\s+(year|month|day|century|country|city|type|kind))?\b",
                  s, flags=re.IGNORECASE)
    if m and m.start() > 0:
        return _finish(s[:m.start()] + a + s[m.end():])
    return concat(q, a)


def _finish(sent):
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent[0].upper() + sent[1:] + ("" if sent.endswith(".") else ".")


class LLMRewriter:
    """Optional: rewrite (q, a) with an instruction-tuned LLM. Needs a GPU to be practical."""

    PROMPT = ("Rewrite the question and its answer as a single declarative sentence. "
              "Output only the sentence.\n\n"
              "Question: When did the Second World War end?\nAnswer: 1945\n"
              "Sentence: The Second World War ended in 1945.\n\n"
              "Question: Who wrote Hamlet?\nAnswer: Shakespeare\n"
              "Sentence: Shakespeare wrote Hamlet.\n\n"
              "Question: {q}\nAnswer: {a}\nSentence:")

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device=None, max_new_tokens=48):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, pairs, batch_size=32):
        out = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            prompts = [self.tok.apply_chat_template(
                [{"role": "user", "content": self.PROMPT.format(q=q, a=a)}],
                tokenize=False, add_generation_prompt=True) for q, a in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.extend(t.strip().split("\n")[0] or concat(q, a) for t, (q, a) in zip(texts, chunk))
        return out

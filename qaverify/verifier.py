"""NLI verifier: P(entailment | premise chunk, hypothesis), aggregated over chunks of the passage."""
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .reader import get_device

# Short aliases used on the command line and in result tables
VERIFIERS = {
    "nli-xs": "cross-encoder/nli-deberta-v3-xsmall",      # ~70M  (22M backbone + embeddings)
    "nli-base": "cross-encoder/nli-deberta-v3-base",      # ~184M
    "nli-large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",  # ~435M
}
# Approximate total parameter counts, for the size plot
PARAMS = {"nli-xs": 70.8e6, "nli-base": 184e6, "nli-large": 435e6}

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_passage(text, mode):
    """full: whole passage (truncated by the model) | sent: one sentence | win3: 3-sentence sliding window."""
    if mode == "full":
        return [text]
    sents = split_sentences(text) or [text]
    if mode == "sent":
        return sents
    if mode.startswith("win"):
        w = int(mode[3:])
        if len(sents) <= w:
            return [" ".join(sents)]
        return [" ".join(sents[i:i + w]) for i in range(len(sents) - w + 1)]
    raise ValueError(mode)


def chunk_with_span(context, start, text, modes=("sent", "win3")):
    """First chunk (trying `modes` in order) that contains the span `text` starting at char `start`;
    None if the span crosses every chunk boundary. Used to build training pairs that look like inference."""
    for mode in modes:
        pos = 0
        for chunk in chunk_passage(context, mode):
            at = context.find(chunk, pos)
            if at < 0:  # a window joins sentences with one space; the passage may not
                continue
            if at <= start and start + len(text) <= at + len(chunk):
                return chunk
            if mode == "sent":
                pos = at + len(chunk)
    return None


class NLIVerifier:
    def __init__(self, model_name, device=None, max_length=512):
        model_name = VERIFIERS.get(model_name, model_name)
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_length = max_length
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.ent = next(i for i, l in labels.items() if l.startswith("entail"))
        self.con = next(i for i, l in labels.items() if l.startswith("contra"))

    @torch.no_grad()
    def score_pairs(self, premises, hypotheses, batch_size=32):
        """Return an (n, 2) array of [P(entail), P(contradiction)]."""
        n = len(premises)
        out = np.zeros((n, 2), dtype=np.float32)
        # Sort by length so each batch has similar padding
        order = sorted(range(n), key=lambda i: len(premises[i]) + len(hypotheses[i]))
        for b in range(0, n, batch_size):
            idx = order[b:b + batch_size]
            enc = self.tokenizer([premises[i] for i in idx], [hypotheses[i] for i in idx],
                                 truncation="only_first", max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            out[idx, 0] = probs[:, self.ent]
            out[idx, 1] = probs[:, self.con]
        return out

    def verify(self, items, chunk_modes=("full", "sent", "win3"), batch_size=32):
        """items: list of (passage, [hypotheses]).
        Returns, per item, {mode: [[ent, con] per hypothesis]} where ent is the max entailment over
        chunks and con is the contradiction prob of that same chunk."""
        premises, hyps, keys = [], [], []
        cache = {}
        for it, (passage, hlist) in enumerate(items):
            for mode in chunk_modes:
                for ci, chunk in enumerate(chunk_passage(passage, mode)):
                    for hi, h in enumerate(hlist):
                        pair = (chunk, h)
                        if pair not in cache:
                            cache[pair] = len(premises)
                            premises.append(chunk)
                            hyps.append(h)
                        keys.append((it, mode, hi, cache[pair]))
        scores = self.score_pairs(premises, hyps, batch_size)

        res = [{m: [[-1.0, 0.0] for _ in hl] for m in chunk_modes} for _, hl in items]
        for it, mode, hi, pid in keys:
            ent, con = scores[pid]
            if ent > res[it][mode][hi][0]:
                res[it][mode][hi] = [float(ent), float(con)]
        return res

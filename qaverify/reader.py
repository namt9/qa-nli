"""Extractive reader that returns the top-k candidate spans (not just the best one),
so that the verifier can re-rank them."""
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def get_device(device=None):
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class ExtractiveReader:
    def __init__(self, model_name="deepset/roberta-base-squad2", device=None,
                 max_length=384, stride=128, top_k=5, n_best=20, max_answer_len=30):
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_length, self.stride = max_length, stride
        self.top_k, self.n_best, self.max_answer_len = top_k, n_best, max_answer_len

    @torch.no_grad()
    def predict(self, examples, batch_size=16):
        """For each example return {"candidates": [{text, start, logit, prob}], "null_logit", "null_prob"}.
        prob is a softmax over the null answer + all kept spans, so it is comparable across examples."""
        out = []
        for i in range(0, len(examples), batch_size):
            out.extend(self._predict_batch(examples[i:i + batch_size]))
        return out

    def _predict_batch(self, batch):
        enc = self.tokenizer(
            [e["question"] for e in batch], [e["context"] for e in batch],
            truncation="only_second", max_length=self.max_length, stride=self.stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding=True,
            return_tensors="pt",
        )
        feat2ex = enc.pop("overflow_to_sample_mapping").tolist()
        offsets = enc.pop("offset_mapping").tolist()
        seq_ids = [enc.sequence_ids(f) for f in range(len(feat2ex))]
        inputs = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        res = self.model(**inputs)
        starts = res.start_logits.float().cpu().numpy()
        ends = res.end_logits.float().cpu().numpy()

        spans = [dict() for _ in batch]  # (start_char, end_char) -> best logit
        nulls = [np.inf for _ in batch]
        for f, ex_idx in enumerate(feat2ex):
            s_log, e_log, off, sid = starts[f], ends[f], offsets[f], seq_ids[f]
            # HF convention: null score = CLS logits; keep the minimum over features of one example
            nulls[ex_idx] = min(nulls[ex_idx], float(s_log[0] + e_log[0]))
            ctx_mask = np.array([s == 1 for s in sid])
            s_cand = [j for j in np.argsort(-s_log) if ctx_mask[j]][:self.n_best]
            e_cand = [j for j in np.argsort(-e_log) if ctx_mask[j]][:self.n_best]
            for s in s_cand:
                for e in e_cand:
                    if e < s or e - s + 1 > self.max_answer_len:
                        continue
                    key = (off[s][0], off[e][1])
                    score = float(s_log[s] + e_log[e])
                    if score > spans[ex_idx].get(key, -np.inf):
                        spans[ex_idx][key] = score

        results = []
        for ex, cand_map, null in zip(batch, spans, nulls):
            # Merge spans with identical surface text (keep best logit)
            by_text = {}
            for (a, b), score in cand_map.items():
                text = ex["context"][a:b].strip()
                if text and score > by_text.get(text, (-np.inf, 0))[0]:
                    by_text[text] = (score, a)
            ranked = sorted(by_text.items(), key=lambda kv: -kv[1][0])[:self.top_k]
            logits = np.array([null] + [s for _, (s, _) in ranked])
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            results.append({
                "candidates": [
                    {"text": t, "start": a, "logit": s, "prob": float(p)}
                    for (t, (s, a)), p in zip(ranked, probs[1:])
                ],
                "null_logit": float(null),
                "null_prob": float(probs[0]),
            })
        return results

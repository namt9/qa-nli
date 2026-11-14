"""Step 2b (member B): fine-tune an NLI verifier on QA->NLI pairs built from SQuAD 2.0 TRAIN
(never dev), then use it like any other verifier with run_verify.py.

  python scripts/finetune_verifier.py --out runs/models/nli-base-qa
  python scripts/finetune_verifier.py --out runs/models/nli-base-qa2 --n_swap 20000
  python scripts/finetune_verifier.py --out runs/models/nli-base-qa3 --n_pairs 10000
  python scripts/finetune_verifier.py --out runs/models/nli-large-qa --base nli-large --batch_size 8 --grad_accum 2
  python scripts/run_verify.py --run runs/squad11k --verifier runs/models/nli-base-qa --hyps template

Pairs: premise = the sentence or 3-sentence window holding the span (as at inference),
hypothesis = template(question, span).
  pos   answerable question + its gold answer                         -> entailment
  trap  unanswerable question + its plausible answer (the planted bait) -> neutral (not entailed)
  wrong answerable question + the gold answer of ANOTHER question on the same paragraph
        (a real span of the passage, but not the answer)               -> neutral (not entailed)
  pair  (--n_pairs) natural minimal pairs of SQuAD 2.0 itself: an unanswerable question and an answerable
        question of the same paragraph sharing >= 60% of their content words. Both go in, so every hard
        negative comes with its matching hard positive: twin + gold -> entailment, unanswerable + plausible
        answer -> neutral
  swap  (--n_swap) answerable question with one entity / number swapped for another of the same kind
        taken from the passage, + the original answer: every word is still in the passage, so it targets
        the lexical-overlap heuristic                                   -> neutral (not entailed)
The 3-way head of the start model is kept so P(entail) and P(contra) mean the same as before.
"""
import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify.data import load_squad_v2  # noqa: E402
from qaverify.hypothesis import template  # noqa: E402
from qaverify.text import STOP, jaccard  # noqa: E402
from qaverify.verifier import VERIFIERS, chunk_with_span  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--base", default="nli-base", help="alias or HF name of the starting NLI model")
p.add_argument("--out", default="runs/models/nli-base-qa")
p.add_argument("--data_dir", default="data")
p.add_argument("--n_per_kind", type=int, default=20000, help="pairs of each kind (pos / trap / wrong)")
p.add_argument("--n_swap", type=int, default=0, help="extra entity-swap negatives (0 = none)")
p.add_argument("--n_pairs", type=int, default=0, help="extra natural minimal pairs, 2 pairs of text each (0 = none)")
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--grad_accum", type=int, default=1, help="batches per optimizer step (large model on a T4: 8 x 2)")
p.add_argument("--max_length", type=int, default=256)
p.add_argument("--dev_frac", type=float, default=0.03, help="held-out share of the TRAIN pairs for a sanity check")
p.add_argument("--seed", type=int, default=13)
args = p.parse_args()


# a number, or a run of capitalised words ("Middle East", "Nikola Tesla")
_NOT_ENT = STOP | {"some", "many", "most", "several", "both", "each", "all", "other", "however", "although",
                   "while", "despite", "following", "since", "during", "after", "before"}
_ENT = re.compile(r"\b(?:\d[\d,.]*\d|\d|[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)*)\b")


def swap_entity(q, a, premise, rng):
    """Replace one number / capitalised phrase of q (not at its start, not in the answer) by another of the
    same kind taken from the premise, so the new question keeps its words in the premise; None if impossible."""
    spots = [m for m in _ENT.finditer(q) if m.start() > 0 and m.group() not in a]
    if not spots:
        return None
    m = rng.choice(spots)
    is_num = m.group()[0].isdigit()
    pool = {t for t in _ENT.findall(premise) if t[0].isdigit() == is_num and t not in q and t not in a
            and t.lower() not in _NOT_ENT and not (len(t) < 3 and not is_num)}
    if not pool:
        return None
    return q[:m.start()] + rng.choice(sorted(pool)) + q[m.end():]


def build_pairs(rng):
    load_squad_v2("train", args.data_dir)  # downloads train-v2.0 if missing
    with open(os.path.join(args.data_dir, "squad_v2_train.json"), encoding="utf-8") as f:
        raw = json.load(f)["data"]
    kinds = {"pos": [], "trap": [], "wrong": [], "swap": [], "pair_pos": [], "pair_neg": []}
    for art in raw:
        for para in art["paragraphs"]:
            ctx = para["context"]
            golds = [(qa["answers"][0]["answer_start"], qa["answers"][0]["text"])
                     for qa in para["qas"] if qa["answers"]]
            if args.n_pairs:
                answerable = [qa for qa in para["qas"] if qa["answers"]]
                for qa in para["qas"]:
                    if qa["answers"] or not qa.get("plausible_answers"):
                        continue
                    twin = max(answerable, key=lambda t: jaccard(qa["question"], t["question"]), default=None)
                    if twin is not None and jaccard(qa["question"], twin["question"]) >= 0.6:
                        pa, ga = qa["plausible_answers"][0], twin["answers"][0]
                        # the pair is kept together (same index) so both halves are sampled or neither
                        kinds["pair_pos"].append((twin["question"].strip(), ga["text"], ga["answer_start"], ctx))
                        kinds["pair_neg"].append((qa["question"].strip(), pa["text"], pa["answer_start"], ctx))
            for qa in para["qas"]:
                q = qa["question"].strip()
                if qa["answers"]:
                    s, a = qa["answers"][0]["answer_start"], qa["answers"][0]["text"]
                    kinds["pos"].append((q, a, s, ctx))
                    others = [(s2, a2) for s2, a2 in golds if a2.lower() not in a.lower() and a.lower() not in a2.lower()]
                    if others:
                        s2, a2 = rng.choice(others)
                        kinds["wrong"].append((q, a2, s2, ctx))
                    if args.n_swap:
                        kinds["swap"].append((q, a, s, ctx))  # swapped once its premise is known
                elif qa.get("plausible_answers"):
                    pa = qa["plausible_answers"][0]
                    kinds["trap"].append((q, pa["text"], pa["answer_start"], ctx))
    pairs = []
    want = {"pos": args.n_per_kind, "trap": args.n_per_kind, "wrong": args.n_per_kind, "swap": args.n_swap}
    for kind, label in (("pos", "entailment"), ("trap", "neutral"), ("wrong", "neutral"), ("swap", "neutral")):
        if not want[kind]:
            continue
        pool = kinds[kind]
        rng.shuffle(pool)
        got = 0
        for q, a, s, ctx in pool:
            # sentence or 3-sentence window at random, as both are used at inference
            chunk = chunk_with_span(ctx, s, a, rng.sample(["sent", "win3"], 2))
            if chunk is None:
                continue
            if kind == "swap":
                q = swap_entity(q, a, chunk, rng)
                if q is None:
                    continue
            pairs.append({"premise": chunk, "hypothesis": template(q, a), "label": label, "kind": kind})
            got += 1
            if got == want[kind]:
                break
        print(f"{kind}: {got} pairs (pool {len(pool)})")
    if args.n_pairs:
        # both halves of a minimal pair go in together, or neither
        order = list(range(len(kinds["pair_pos"])))
        rng.shuffle(order)
        got = 0
        for i in order:
            halves = []
            for (q, a, s, ctx), label, kind in ((kinds["pair_pos"][i], "entailment", "pair_pos"),
                                                (kinds["pair_neg"][i], "neutral", "pair_neg")):
                chunk = chunk_with_span(ctx, s, a, rng.sample(["sent", "win3"], 2))
                if chunk is None:
                    break
                halves.append({"premise": chunk, "hypothesis": template(q, a), "label": label, "kind": kind})
            if len(halves) == 2:
                pairs += halves
                got += 1
                if got == args.n_pairs:
                    break
        print(f"minimal pairs: {got} (pool {len(order)})")
    rng.shuffle(pairs)
    return pairs


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if os.path.exists(os.path.join(args.out, "config.json")):
        sys.exit(f"{args.out} already holds a model - delete it to train again")
    pairs = build_pairs(rng)
    n_dev = int(len(pairs) * args.dev_frac)
    dev, train = pairs[:n_dev], pairs[n_dev:]

    name = VERIFIERS.get(args.base, args.base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    # Train in fp32 (autocast adds fp16 compute): some checkpoints, e.g. the large NLI one, are stored in fp16 and
    # recent transformers load them as such, which GradScaler refuses ("Attempting to unscale FP16 gradients")
    model = AutoModelForSequenceClassification.from_pretrained(name).float().to(device)
    label2id = {l.lower(): i for i, l in model.config.id2label.items()}
    ent, neu = label2id["entailment"], label2id["neutral"]

    def batches(rows, shuffle):
        rows = list(rows)
        if shuffle:
            rng.shuffle(rows)
        for b in range(0, len(rows), args.batch_size):
            chunk = rows[b:b + args.batch_size]
            enc = tok([r["premise"] for r in chunk], [r["hypothesis"] for r in chunk], truncation="only_first",
                      max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            y = torch.tensor([ent if r["label"] == "entailment" else neu for r in chunk], device=device)
            yield chunk, enc, y

    @torch.no_grad()
    def evaluate():
        """Accuracy of 'P(entail) > 0.5' per kind on the held-out train pairs."""
        model.eval()
        hit = {}
        for chunk, enc, _ in batches(dev, False):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                pe = torch.softmax(model(**enc).logits.float(), -1)[:, ent].cpu().tolist()
            for r, v in zip(chunk, pe):
                hit.setdefault(r["kind"], []).append((v > 0.5) == (r["label"] == "entailment"))
        model.train()
        return {k: round(sum(v) / len(v), 3) for k, v in sorted(hit.items())}

    before = evaluate()
    print("before:", before)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size) // args.grad_accum
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    model.train()
    t0, step, micro = time.time(), 0, 0
    opt.zero_grad()
    for epoch in range(args.epochs):
        for _, enc, y in batches(train, True):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                eta = (time.time() - t0) / step * (steps - step)
                print(f"step {step}/{steps}  loss {loss.item():.3f}  eta {eta / 60:.1f} min", flush=True)
    print("after:", evaluate())

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_info.json"), "w") as f:
        json.dump({**vars(args), "n_train": len(train), "n_dev": len(dev), "dev_acc_before": before, "dev_acc": evaluate()}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()

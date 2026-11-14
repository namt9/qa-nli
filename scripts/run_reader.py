"""Step 1 (member A): run the extractive reader on a SQuAD 2.0 subset, keep the top-k candidates
and build NLI hypotheses for each of them.

  python scripts/run_reader.py --n 2000 --out runs/squad2k

If preds.jsonl already exists the reader is not run again; with --llm the missing 'llm'
hypotheses are added to the existing file.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify.data import load_dataset, read_jsonl, sample, split_of, write_jsonl  # noqa: E402
from qaverify.hypothesis import LLMRewriter, concat, template  # noqa: E402
from qaverify.reader import ExtractiveReader  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="squad_v2_dev", help="squad_v2_dev | addsent | addsent_orig | mrqa:<Name>")
p.add_argument("--n", type=int, default=2000, help="number of questions sampled (>= size of the set: all)")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--reader", default="deepset/roberta-base-squad2")
p.add_argument("--top_k", type=int, default=5)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--llm", default=None, help="e.g. Qwen/Qwen2.5-1.5B-Instruct to add 'llm' hypotheses (GPU)")
p.add_argument("--out", default="runs/squad2k")
p.add_argument("--data_dir", default="data")
args = p.parse_args()

out_path = os.path.join(args.out, "preds.jsonl")
if os.path.exists(out_path):
    rows = read_jsonl(out_path)
    print(f"found {out_path} ({len(rows)} questions), reader skipped")
    if not args.llm or all("llm" in r["hyp"] for r in rows):
        sys.exit(0)
else:
    examples = sample(load_dataset(args.dataset, args.data_dir), args.n, args.seed)
    print(f"{len(examples)} questions, {sum(e['is_impossible'] for e in examples)} unanswerable")

    t0 = time.time()
    reader = ExtractiveReader(args.reader, top_k=args.top_k)
    preds = reader.predict(examples, batch_size=args.batch_size)
    print(f"reader done in {time.time() - t0:.0f}s on {reader.device}")

    rows = []
    for ex, pr in zip(examples, preds):
        answers = [c["text"] for c in pr["candidates"]]
        rows.append({
            **ex, **pr,
            "split": split_of(ex["id"]),
            "hyp": {
                "concat": [concat(ex["question"], a) for a in answers],
                "template": [template(ex["question"], a) for a in answers],
            },
        })

if args.llm:
    rewriter = LLMRewriter(args.llm)
    flat = [(r["question"], c["text"]) for r in rows for c in r["candidates"]]
    out = iter(rewriter(flat))
    for r in rows:
        r["hyp"]["llm"] = [next(out) for _ in r["candidates"]]

# Write then rename, so a disconnect cannot leave a half-written preds.jsonl
write_jsonl(out_path + ".tmp", rows)
os.replace(out_path + ".tmp", out_path)
print("saved", out_path)

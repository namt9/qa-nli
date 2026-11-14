"""Step 2 (member B): score every candidate of preds.jsonl with an NLI verifier.
Writes runs/<run>/scores/<verifier>.jsonl so several verifiers can be run independently
(e.g. on two Colab sessions) and merged later by analyze.py.

  python scripts/run_verify.py --run runs/squad2k --verifier nli-xs
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify.data import read_jsonl  # noqa: E402
from qaverify.verifier import NLIVerifier  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad2k")
p.add_argument("--verifier", default="nli-xs", help="alias (nli-xs, nli-base, nli-large) or HF model name")
p.add_argument("--hyps", default="concat,template")
p.add_argument("--chunks", default="full,sent,win3")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--group", type=int, default=64, help="examples per scoring call (progress granularity)")
p.add_argument("--limit", type=int, default=None, help="only the first N examples (smoke test)")
args = p.parse_args()

rows = read_jsonl(os.path.join(args.run, "preds.jsonl"))[:args.limit]
hyps, chunks = args.hyps.split(","), args.chunks.split(",")
alias = args.verifier.split("/")[-1]
path = os.path.join(args.run, "scores", f"{alias}.jsonl")

# Resume: results are appended group by group, so a Colab disconnect loses at most one group.
# Done is tracked per (id, setting), so adding a hypothesis type later (e.g. llm) only scores that one.
done = {}
if os.path.exists(path):
    # A session killed mid-write (e.g. GPU quota) can leave a truncated last line: drop it and rewrite
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    good = []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        good.append(line if line.endswith("\n") else line + "\n")
        done.setdefault(r["id"], set()).update(r["nli"])
    if len(good) < len(lines):
        print(f"dropped {len(lines) - len(good)} broken line(s) in {path}")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.writelines(good)
        os.replace(path + ".tmp", path)
todo = []
for r in rows:
    missing = [h for h in hyps
               if any(f"{alias}|{h}|{m}" not in done.get(r["id"], ()) for m in chunks)]
    if missing:
        todo.append((r, missing))
print(f"{alias}: {len(rows) - len(todo)} already scored, {len(todo)} to go")
if not todo:
    sys.exit(0)
verifier = NLIVerifier(args.verifier)
print("device:", verifier.device)

os.makedirs(os.path.dirname(path), exist_ok=True)
t0, n_done = time.time(), 0
for g in range(0, len(todo), args.group):
    group = todo[g:g + args.group]
    # One item per (example, hypothesis type); hypotheses of the item = the k candidates
    items = [(r["context"], r["hyp"][h]) for r, missing in group for h in missing]
    res = iter(verifier.verify(items, chunks, args.batch_size))
    out = []
    for r, missing in group:
        nli = {}
        for h in missing:
            per_mode = next(res)
            for m in chunks:
                nli[f"{alias}|{h}|{m}"] = per_mode[m]
        out.append({"id": r["id"], "nli": nli})
    with open(path, "a", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n_done += len(group)
    eta = (time.time() - t0) / n_done * (len(todo) - n_done)
    print(f"{n_done}/{len(todo)}  {time.time() - t0:.0f}s  eta {eta / 60:.1f} min", flush=True)
print("saved", path)

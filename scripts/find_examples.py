"""Pick concrete examples for the slides (run after analyze.py, which writes settings.json).

  python scripts/find_examples.py --run runs/squad2k [--verifier nli-large]

Writes runs/<run>/results/examples.md with counts and a few test examples of each case:
  fixed     reader's top answer wrong, NLI re-ranking (R) picks a correct one
  broken    reader correct, R picks a wrong one
  caught    unanswerable question, reader confident (prob >= 0.5), NLI says not entailed (< 0.2)
  missed    wrong answer that NLI fully accepts (entail >= 0.8)
  false_alarm  correct answer that NLI rejects (entail < 0.2)
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import read_jsonl  # noqa: E402
from qaverify.metrics import is_correct  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad2k")
p.add_argument("--verifier", default=None, help="default: the largest verifier in settings.json")
p.add_argument("--k", type=int, default=3, help="examples per case")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

res_dir = os.path.join(args.run, "results")
settings = json.load(open(os.path.join(res_dir, "settings.json")))
order = ["nli-xs", "nli-base", "nli-large"]
verifier = args.verifier or max(settings, key=lambda v: order.index(v) if v in order else -1)
key, lam = settings[verifier]["key"], settings[verifier]["lam"]
hyp = key.split("|")[1]

rows = {r["id"]: r for r in read_jsonl(os.path.join(args.run, "preds.jsonl"))}
for path in glob.glob(os.path.join(args.run, "scores", "*.jsonl")):
    for s in read_jsonl(path):
        rows[s["id"]].setdefault("nli", {}).update(s["nli"])
test = [r for r in rows.values() if r["split"] == "test" and key in r.get("nli", {}) and r["candidates"]]

cases = {c: [] for c in ("fixed", "broken", "caught", "missed", "false_alarm")}
for ex in test:
    top = ex["candidates"][0]
    top_ok = is_correct(top["text"], ex["answers"])
    rr = M.nli_rerank(ex, key, lam)
    ent = ex["nli"][key][0][0]
    if not top_ok and rr["correct"]:
        cases["fixed"].append(ex)
    if top_ok and not rr["correct"]:
        cases["broken"].append(ex)
    if not ex["answers"] and top["prob"] >= 0.5 and ent < 0.2:
        cases["caught"].append(ex)
    if not top_ok and ent >= 0.8:
        cases["missed"].append(ex)
    if top_ok and ent < 0.2:
        cases["false_alarm"].append(ex)


def show(ex):
    ctx = ex["context"]
    lines = [f"**Q:** {ex['question']}  ",
             f"**Gold:** {ex['answers'] or '(unanswerable)'}  ",
             f"**Context:** {ctx[:600]}{'...' if len(ctx) > 600 else ''}", "",
             "| # | Candidate | P_reader | P_entail | P_contra | Hypothesis |", "|---|---|---|---|---|---|"]
    for k, c in enumerate(ex["candidates"]):
        ent, con = ex["nli"][key][k]
        h = ex["hyp"][hyp][k].replace("|", "/")
        lines.append(f"| {k + 1} | {c['text'].replace('|', '/')} | {c['prob']:.2f} | {ent:.2f} | {con:.2f} | {h} |")
    lines.append(f"\nNull prob: {ex['null_prob']:.2f}. R picks: **{M.nli_rerank(ex, key, lam)['answer']}**\n")
    return "\n".join(lines)


rng = random.Random(args.seed)
out = [f"# Examples ({verifier}, setting `{key}`, lambda={lam})\n",
       f"Test questions with candidates: {len(test)}\n",
       "| Case | Count |", "|---|---|"]
out += [f"| {c} | {len(v)} |" for c, v in cases.items()]
for c, v in cases.items():
    out.append(f"\n## {c} ({len(v)})\n")
    for ex in rng.sample(v, min(args.k, len(v))):
        out.append(show(ex))
path = os.path.join(res_dir, "examples.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\n".join(out[:4 + len(cases)]))
print("saved", path)

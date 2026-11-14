"""Step 6b (both): read the labelled sheets from make_annotation.py, report agreement and how the
reader / NLI / combo behave on each type of unanswerable question.

  python scripts/analyze_annotation.py --run runs/squad11k --set sheet    # random sample
  python scripts/analyze_annotation.py --run runs/squad11k --set check    # agreement set
  python scripts/analyze_annotation.py --run runs/squad11k --set errors   # questions C answers wrongly

Overlap rows: the label from resolved.csv (id,label) if present, else the label both agree on,
else member A's label (and the row is listed in <set>_disagreements.csv).
Writes runs/<run>/annotation/{<set>_annotation.md, <set>_disagreements.csv}.
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import load_run  # noqa: E402
from qaverify.metrics import f1_score  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad2k")
p.add_argument("--data_dir", default="data")
p.add_argument("--set", default="sheet", help="sheet prefix: sheet (random), check or errors")
p.add_argument("--verifier", default="nli-large", help="its setting is read from results/settings.json")
args = p.parse_args()
ann_dir = os.path.join(args.run, "annotation")
CODES = ["NEG", "ANT", "ENT", "MUT", "IMP", "OTH", "ANS"]


def read_sheet(who):
    with open(os.path.join(ann_dir, f"{args.set}_{who}.csv"), encoding="utf-8-sig") as f:
        return {r["id"]: r["label"].strip().upper() for r in csv.DictReader(f)}


a, b = read_sheet("A"), read_sheet("B")
bad = {i: l for s in (a, b) for i, l in s.items() if l and l not in CODES}
if bad:
    sys.exit(f"unknown labels (use {', '.join(CODES)}): {bad}")
both = [i for i in a if i in b and a[i] and b[i]]
resolved = {}
if os.path.exists(os.path.join(ann_dir, "resolved.csv")):
    with open(os.path.join(ann_dir, "resolved.csv"), encoding="utf-8-sig") as f:
        resolved = {r["id"]: r["label"].strip().upper() for r in csv.DictReader(f)}

labels = {}
for src in (b, a):  # A wins on overlap unless resolved
    labels.update({i: l for i, l in src.items() if l})
labels.update(resolved)
disagree = [i for i in both if a[i] != b[i]]
with open(os.path.join(ann_dir, f"{args.set}_disagreements.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label_A", "label_B", "resolved"])
    for i in disagree:
        w.writerow([i, a[i], b[i], resolved.get(i, "")])

# ---------- model behaviour on the labelled questions ----------
with open(os.path.join(args.data_dir, "squad_v2_dev.json"), encoding="utf-8") as f:
    plausible = {qa["id"]: list(dict.fromkeys(x["text"] for x in qa.get("plausible_answers", [])))
                 for art in json.load(f)["data"] for para in art["paragraphs"] for qa in para["qas"]}
rows = load_run(args.run)
with open(os.path.join(args.run, "results", "settings.json")) as f:
    key = json.load(f)[args.verifier]["key"]

calib = [r for r in rows.values() if r["split"] == "calib"]
combo = M.Combo(key).fit(calib)


systems = {"B2": M.reader_margin, "V": lambda ex: M.nli_verify(ex, key), "C": combo}
taus = {n: M.calib_threshold(m, calib) for n, m in systems.items()}


def answered(name, ex):
    return systems[name](ex)["conf"] >= taus[name]


def top1_is_plausible(ex):
    return bool(ex["candidates"]) and bool(plausible.get(ex["id"])) and \
        f1_score(ex["candidates"][0]["text"], plausible[ex["id"]]) >= 0.5


by = collections.defaultdict(list)
for i, l in labels.items():
    by[l].append(rows[i])
table = []
for c in CODES:
    g = by.get(c, [])
    if not g:
        continue
    ent = [M.nli_of(ex, key, 0)[0] for ex in g if ex["candidates"]]
    table.append([c, len(g), f"{100 * len(g) / len(labels):.0f}",
                  f"{100 * np.mean([top1_is_plausible(ex) for ex in g]):.0f}",
                  f"{np.mean(ent):.2f}" if ent else "-"] +
                 [f"{100 * np.mean([answered(n, ex) for ex in g]):.0f}" for n in systems])

title = {"sheet": "mẫu ngẫu nhiên", "check": "bộ kiểm tra độ đồng thuận",
         "errors": "câu C vẫn trả lời (lỗi của hệ thống)"}.get(args.set, args.set)
md = [f"# Gán nhãn câu không đáp án: {title} ({len(labels)} câu, NLI = `{key}`)\n"]
if both:
    md.append(f"Độ đồng thuận trên {len(both)} câu chung: {100 * (1 - len(disagree) / len(both)):.0f}% trùng nhãn, "
              f"Cohen's kappa = {cohen_kappa_score([a[i] for i in both], [b[i] for i in both]):.2f}. "
              f"Câu lệch: {len(disagree)} (xem {args.set}_disagreements.csv), đã thống nhất: "
              f"{sum(i in resolved for i in disagree)}.\n")
md += ["| Loại | n | % | Top-1 = plausible (%) | P_entail TB top-1 | B2 trả lời (%) | V trả lời (%) | C trả lời (%) |",
       "|---|---|---|---|---|---|---|---|"]
md += ["| " + " | ".join(str(x) for x in r) + " |" for r in table]
md.append("\n'... trả lời (%)': tỉ lệ câu trong loại mà hệ thống vẫn trả lời ở ngưỡng chọn trên calib "
          "(câu không đáp án nên trả lời là sai; thấp hơn là tốt hơn). Với n nhỏ, chỉ nên đọc xu hướng.\n")
with open(os.path.join(ann_dir, f"{args.set}_annotation.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

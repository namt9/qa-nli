"""Step 5 (both): analyse the dataset itself and how its properties drive reader / NLI behaviour.

  python scripts/data_analysis.py --run runs/squad2k

Part A (data only): full SQuAD 2.0 dev vs our sample, question types, lengths, answer types,
question-context overlap, negation, the 'plausible answers' of unanswerable questions.
Part B (data x models, test split): reader and NLI by question type, gold rank in top-k
(re-ranking ceiling), answer lengths, P(entail) by group, does the system fall for the plausible answer.

Outputs in runs/<run>/results/: data_analysis.md, data_qtype.png, data_overlap.png, data_entail.png,
data_topk.png
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify.data import read_jsonl  # noqa: E402
from qaverify.metrics import f1_score, is_correct  # noqa: E402
from qaverify.text import QTYPES, STOP, negated, overlap, qtype, words  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad2k")
p.add_argument("--data_dir", default="data")
p.add_argument("--key", default=None, help="NLI setting for part B (default: nli-large from settings.json)")
args = p.parse_args()
out_dir = os.path.join(args.run, "results")
os.makedirs(out_dir, exist_ok=True)

# Reference categorical palette (validated for <= 3 series), recessive ink for text and grid
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

MONTHS = "january february march april may june july august september october november december".split()


def answer_type(a):
    w = words(a)
    if any(m in w for m in MONTHS) or re.fullmatch(r"(c\. )?\d{3,4}s?", a.strip().lower()):
        return "date / year"
    if re.search(r"\d", a) or (w and w[0] in {"one", "two", "three", "four", "five", "six", "seven",
                                                 "eight", "nine", "ten", "hundred", "thousand"}):
        return "number / quantity"
    toks = a.split()
    if toks and all(t[:1].isupper() for t in toks if t.lower() not in STOP):
        return "proper noun"
    return "common phrase"


def load_raw_dev():
    """Full SQuAD 2.0 dev, keeping plausible_answers of unanswerable questions."""
    with open(os.path.join(args.data_dir, "squad_v2_dev.json"), encoding="utf-8") as f:
        raw = json.load(f)["data"]
    out = []
    for art in raw:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                out.append({"id": qa["id"], "title": art["title"], "question": qa["question"].strip(),
                            "context": para["context"],
                            "answers": list(dict.fromkeys(a["text"] for a in qa["answers"])),
                            "plausible": list(dict.fromkeys(a["text"] for a in qa.get("plausible_answers", [])))})
    return out


def pct(x):
    return f"{100 * np.mean(x):.1f}" if len(x) else "-"


def med_p90(x):
    return f"{np.median(x):.0f} / {np.percentile(x, 90):.0f}" if len(x) else "-"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# ---------- load ----------
dev = load_raw_dev()
plausible = {r["id"]: r["plausible"] for r in dev}
preds = read_jsonl(os.path.join(args.run, "preds.jsonl"))
for path in sorted(__import__("glob").glob(os.path.join(args.run, "scores", "*.jsonl"))):
    by_id = {r["id"]: r for r in preds}
    for s in read_jsonl(path):
        by_id[s["id"]].setdefault("nli", {}).update(s["nli"])
key = args.key
if key is None:
    settings_path = os.path.join(out_dir, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            key = json.load(f).get("nli-large", {}).get("key")
if key is None and preds and preds[0].get("nli"):
    key = sorted(preds[0]["nli"])[0]
test = [r for r in preds if r["split"] == "test"]
md = ["# Phân tích bộ dữ liệu SQuAD 2.0\n"]

# ---------- A1 overview ----------
groups = {"SQuAD 2.0 dev (đầy đủ)": dev, f"Mẫu {len(preds)} câu": preds,
          "calib": [r for r in preds if r["split"] == "calib"], "test": test}
rows = []
for name, g in groups.items():
    rows.append([name, len(g), pct([not r["answers"] for r in g]), len({r["title"] for r in g}),
                 len({r["context"] for r in g})])
md += ["## A1. Tổng quan\n", md_table(["Tập", "Số câu", "% không đáp án", "Số bài (title)", "Số đoạn văn"], rows)]

# ---------- A2 question types ----------
rows = []
share = {}
for t in QTYPES:
    full = [r for r in dev if qtype(r["question"]) == t]
    samp = [r for r in preds if qtype(r["question"]) == t]
    ans = sum(bool(r["answers"]) for r in dev)
    una = len(dev) - ans
    share[t] = (sum(bool(r["answers"]) for r in full) / ans, sum(not r["answers"] for r in full) / una)
    rows.append([t, len(full), pct([1] * len(full) + [0] * (len(dev) - len(full))),
                 pct([not r["answers"] for r in full]), len(samp)])
md += ["## A2. Loại câu hỏi (từ để hỏi)\n",
       md_table(["Loại", "Số câu (dev)", "% của dev", "% không đáp án trong loại", "Số câu (mẫu)"], rows)]

# ---------- A3 lengths ----------
rows = []
for name, cond in (("có đáp án", True), ("không đáp án", False)):
    g = [r for r in dev if bool(r["answers"]) == cond]
    gold = [len(words(a)) for r in g for a in (r["answers"] or r["plausible"])[:1]]
    rows.append([name, med_p90([len(words(r["question"])) for r in g]),
                 med_p90([len(words(r["context"])) for r in g]), med_p90(gold),
                 f"{np.mean([len(r['answers']) for r in g]):.2f}" if cond else "-"])
md += ["## A3. Độ dài (số từ, trung vị / p90, dev đầy đủ)\n",
       md_table(["Nhóm", "Câu hỏi", "Đoạn văn", "Đáp án (hoặc plausible)", "Số đáp án khác nhau"], rows),
       "\nVới câu không đáp án, cột đáp án dùng *plausible answer* do người gán nhãn ghi lại.\n"]

# ---------- A4 answer types ----------
rows = []
ans_dev = [r for r in dev if r["answers"]]
una_dev = [r for r in dev if not r["answers"] and r["plausible"]]
cats = ["date / year", "number / quantity", "proper noun", "common phrase"]
for c in cats:
    rows.append([c, pct([answer_type(r["answers"][0]) == c for r in ans_dev]),
                 pct([answer_type(r["plausible"][0]) == c for r in una_dev])])
md += ["## A4. Loại đáp án (heuristic, dev đầy đủ)\n",
       md_table(["Loại", "% đáp án thật (có đáp án)", "% plausible answer (không đáp án)"], rows)]

# ---------- A5 overlap, negation, plausible in context ----------
ov = {c: [overlap(r["question"], r["context"]) for r in dev if bool(r["answers"]) == c] for c in (True, False)}
neg = {c: [negated(r["question"]) for r in dev if bool(r["answers"]) == c] for c in (True, False)}
plaus_in_ctx = [any(a in r["context"] for a in r["plausible"]) for r in dev if not r["answers"]]
rows = [["có đáp án", f"{np.mean(ov[True]):.2f}", pct(np.array(ov[True]) >= 0.8), pct(neg[True]), "-"],
        ["không đáp án", f"{np.mean(ov[False]):.2f}", pct(np.array(ov[False]) >= 0.8), pct(neg[False]),
         pct(plaus_in_ctx)]]
md += ["## A5. Câu không đáp án được viết để \"trông như có đáp án\"\n",
       md_table(["Nhóm", "Overlap câu hỏi–đoạn văn (TB)", "% overlap ≥ 0.8", "% có từ phủ định",
                 "% plausible answer nằm trong đoạn văn"], rows),
       "\nOverlap = tỉ lệ từ nội dung của câu hỏi xuất hiện trong đoạn văn.\n"]

# ---------- B: test split x models ----------
md += [f"\n# B. Dữ liệu ảnh hưởng thế nào tới reader và NLI (test, {len(test)} câu, NLI = `{key}`)\n"]


def top1(r):
    return r["candidates"][0]["text"] if r["candidates"] else ""


def gold_rank(r):
    """1-based rank of the first correct candidate, 0 if none (answerable only)."""
    for i, c in enumerate(r["candidates"]):
        if is_correct(c["text"], r["answers"]):
            return i + 1
    return 0


def ent(r, k=0):
    return r["nli"][key][k][0] if key and r.get("nli") and r["candidates"] else float("nan")


def margin(r):
    return r["candidates"][0]["logit"] - r["null_logit"] if r["candidates"] else -1e9


def hits_plausible(r):
    return bool(plausible.get(r["id"])) and f1_score(top1(r), plausible[r["id"]]) >= 0.5


# B1 by question type
rows = []
for t in QTYPES:
    g = [r for r in test if qtype(r["question"]) == t]
    if not g:
        continue
    a = [r for r in g if r["answers"]]
    u = [r for r in g if not r["answers"]]
    rows.append([t, len(g), pct([not r["answers"] for r in g]),
                 pct([is_correct(top1(r), r["answers"]) for r in a]),
                 pct([gold_rank(r) > 0 for r in a]),
                 pct([margin(r) < 0 for r in u]),
                 f"{np.nanmean([ent(r) for r in u]):.2f}" if u else "-"])
md += ["## B1. Theo loại câu hỏi\n",
       md_table(["Loại", "n", "% không đáp án", "Top-1 đúng (có đáp án)", "Gold trong top-5 (có đáp án)",
                 "Reader tự từ chối (margin<0, không đáp án)", "P_entail TB top-1 (không đáp án)"], rows)]

# B2 gold rank -> re-ranking ceiling
ans_test = [r for r in test if r["answers"]]
ranks = np.array([gold_rank(r) for r in ans_test])
rows = [[f"top-{k}", pct((ranks >= 1) & (ranks <= k))] for k in range(1, 6)]
md += ["## B2. Vị trí ứng viên đúng trong top-5 (câu có đáp án): trần của re-ranking\n",
       md_table(["k", "% có ứng viên đúng trong top-k"], rows),
       f"\nTop-1 sai nhưng có ứng viên đúng ở hạng 2–5: {pct((ranks >= 2))}% câu có đáp án "
       f"({int((ranks >= 2).sum())} câu) — đây là số câu tối đa re-ranking có thể sửa.\n"]

# B3 lengths of gold vs candidates
gold_len = [min(len(words(a)) for a in r["answers"]) for r in ans_test]
top1_len = [len(words(top1(r))) for r in test if r["candidates"]]
all_len = [len(words(c["text"])) for r in test for c in r["candidates"]]
best_ent_len = [len(words(r["candidates"][int(np.argmax([ent(r, k) for k in range(len(r["candidates"]))]))]["text"]))
                for r in test if r["candidates"] and key and r.get("nli")]
md += ["## B3. Độ dài: đáp án gold so với ứng viên (số từ, trung vị / p90)\n",
       md_table(["", "Độ dài"], [["Gold ngắn nhất (có đáp án)", med_p90(gold_len)],
                                 ["Ứng viên top-1 của reader", med_p90(top1_len)],
                                 ["Mọi ứng viên top-5", med_p90(all_len)],
                                 ["Ứng viên có P_entail cao nhất", med_p90(best_ent_len)]])]
if key:
    corr = []
    for r in test:
        c = r["candidates"]
        lens = [len(words(x["text"])) for x in c]
        if len(set(lens)) > 1 and r.get("nli"):
            corr.append(np.corrcoef(lens, [ent(r, k) for k in range(len(c))])[0, 1])
    md += [f"\nTương quan (Pearson, trung bình theo câu) giữa độ dài ứng viên và P_entail: "
           f"{np.nanmean(corr):+.2f}.\n"]

# B4 entailment by group, plausible answers
una_test = [r for r in test if not r["answers"]]
grp = {"có đáp án, top-1 đúng": [r for r in ans_test if is_correct(top1(r), r["answers"])],
       "có đáp án, top-1 sai": [r for r in ans_test if not is_correct(top1(r), r["answers"])],
       "không đáp án": una_test}
rows = [[n, len(g), f"{np.nanmean([ent(r) for r in g]):.2f}", pct([ent(r) >= 0.5 for r in g])]
        for n, g in grp.items()]
md += ["## B4. NLI có phân biệt được các nhóm không?\n",
       md_table(["Nhóm", "n", "P_entail TB của top-1", "% P_entail ≥ 0.5"], rows)]
hp = [r for r in una_test if hits_plausible(r)]
nhp = [r for r in una_test if not hits_plausible(r)]
rows = [["top-1 trùng plausible answer", len(hp), f"{np.nanmean([ent(r) for r in hp]):.2f}",
         pct([margin(r) < 0 for r in hp])],
        ["top-1 khác plausible answer", len(nhp), f"{np.nanmean([ent(r) for r in nhp]):.2f}",
         pct([margin(r) < 0 for r in nhp])]]
md += ["\nCâu không đáp án, tách theo việc top-1 của reader có trùng *plausible answer* (F1 ≥ 0.5) hay không:\n\n",
       md_table(["Nhóm", "n", "P_entail TB", "Reader tự từ chối (margin<0)"], rows)]

with open(os.path.join(out_dir, "data_analysis.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

# ---------- figures ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed: skipping plots")
    sys.exit(0)

plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "legend.frameon": False})


def finish(fig, ax, name):
    ax.grid(axis="y" if name != "data_qtype.png" else "x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=180)
    plt.close(fig)


# Question types: share within answerable vs within unanswerable (dev)
order = [t for t in QTYPES if share[t][0] + share[t][1] > 0][::-1]
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(y + 0.2, [100 * share[t][0] for t in order], 0.38, color=BLUE, label="có đáp án")
ax.barh(y - 0.2, [100 * share[t][1] for t in order], 0.38, color=ORANGE, label="không đáp án")
ax.set_yticks(y, order)
ax.set_xlabel("% câu hỏi trong nhóm (SQuAD 2.0 dev)")
ax.legend(loc="lower right")
finish(fig, ax, "data_qtype.png")

# Overlap histogram
bins = np.linspace(0, 1, 21)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(ov[True], bins, color=BLUE, histtype="step", lw=2, density=True,
        label=f"có đáp án (TB {np.mean(ov[True]):.2f})")
ax.hist(ov[False], bins, color=ORANGE, histtype="step", lw=2, density=True,
        label=f"không đáp án (TB {np.mean(ov[False]):.2f})")
ax.set_xlabel("Overlap từ nội dung câu hỏi với đoạn văn")
ax.set_ylabel("Mật độ")
ax.legend(loc="upper left")
finish(fig, ax, "data_overlap.png")

# P(entail) of top-1 by group
if key:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (n, g), col in zip(grp.items(), (BLUE, AQUA, ORANGE)):
        v = [ent(r) for r in g if r["candidates"]]
        ax.hist(v, np.linspace(0, 1, 21), color=col, histtype="step", lw=2, density=True,
                label=f"{n} (n={len(g)})")
    ax.set_xlabel(f"P(entail) của ứng viên top-1 ({key})")
    ax.set_ylabel("Mật độ")
    ax.legend(loc="upper center")
    finish(fig, ax, "data_entail.png")

# Gold rank (cumulative) -> re-ranking ceiling
fig, ax = plt.subplots(figsize=(5, 3.4))
cum = [100 * np.mean((ranks >= 1) & (ranks <= k)) for k in range(1, 6)]
ax.plot(range(1, 6), cum, "o-", color=BLUE, lw=2, ms=8)
for k, v in zip(range(1, 6), cum):
    ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center", color=INK, fontsize=9)
ax.set_xticks(range(1, 6))
ax.set_xlabel("k (top-k ứng viên của reader)")
ax.set_ylabel("% câu có đáp án có ứng viên đúng")
ax.set_ylim(min(cum) - 5, 100)
finish(fig, ax, "data_topk.png")
print("figures saved in", out_dir)

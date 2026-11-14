"""Step 9 (both): edge cases of the dataset and how much each one hurts (test split, run after analyze.py).

  python scripts/edge_cases.py --run runs/squad11k [--verifier nli-large-qa]

For every edge case of qaverify/edge.py: how many test questions have it, and how often the reader alone
(B2) and the combo C make the right decision at their calib threshold (answer correctly an answerable
question, abstain on an unanswerable one), next to the rate over all questions of the same kind.

Writes runs/<run>/results/edge_cases.md, edge_cases.png and edge_cases_examples.md (a few concrete test
questions per edge case, preferring ones where the reader alone decides wrongly).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import load_run  # noqa: E402
from qaverify.edge import EDGE_CASES, edge_flags, load_dev_plausible, load_dev_spans, sentence_at  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad11k")
p.add_argument("--data_dir", default="data")
p.add_argument("--verifier", default=None, help="verifier of C (default: nli-large-qa if scored, else nli-large)")
args = p.parse_args()
res_dir = os.path.join(args.run, "results")
GRAY, BLUE, INK, MUTED, GRID = "#9aa0a8", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

rows = load_run(args.run)
with open(os.path.join(res_dir, "settings.json")) as f:
    settings = json.load(f)
verifier = args.verifier or ("nli-large-qa" if "nli-large-qa" in settings else "nli-large")
key = settings[verifier]["key"]
calib = [r for r in rows.values() if r["split"] == "calib"]
test = [r for r in rows.values() if r["split"] == "test"]
spans = load_dev_spans(args.data_dir)
flags = edge_flags(rows, spans)
plausible = load_dev_plausible(args.data_dir)

systems = {"B2": M.reader_margin, "C": M.Combo(key).fit(calib)}
right, answered = {}, {}
for n, m in systems.items():
    tau = M.calib_threshold(m, calib)
    out = [m(ex) for ex in test]
    answered[n] = np.array([o["conf"] >= tau for o in out])
    # right decision: answer an answerable question correctly, abstain on an unanswerable one
    right[n] = np.array([(o["conf"] >= tau and o["correct"]) if ex["answers"] else o["conf"] < tau
                         for o, ex in zip(out, test)], float)
has = np.array([bool(ex["answers"]) for ex in test])
top1_ok = np.array([bool(ex["answers"]) and M.reader_prob(ex)["correct"] for ex in test], float)
ent = np.array([M.nli_of(ex, key, 0)[0] if ex["candidates"] else 0.0 for ex in test])

base = {"ans": has, "unans": ~has}
md = [f"# Edge case của SQuAD 2.0 và mức ảnh hưởng (test {len(test)} câu, C dùng `{key}`)\n",
      "\"Đúng\" = trả lời đúng câu có đáp án, hoặc từ chối câu không đáp án, ở ngưỡng chọn trên calib.\n",
      "| Edge case | Áp dụng | Cách phát hiện | n | % nhóm | Top-1 reader đúng | P_entail TB | B2 đúng | C đúng | C − B2 |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for kind, name in (("ans", "Mọi câu có đáp án"), ("unans", "Mọi câu không đáp án")):
    m = base[kind]
    top1 = f"{100 * top1_ok[m].mean():.1f}" if kind == "ans" else "–"
    md.append(f"| **{name}** | {kind} | – | {int(m.sum())} | 100 | {top1} | {ent[m].mean():.2f} | "
              f"{100 * right['B2'][m].mean():.1f} | {100 * right['C'][m].mean():.1f} | "
              f"{100 * (right['C'][m].mean() - right['B2'][m].mean()):+.1f} |")
plot = []
for flag, kind, name, how in EDGE_CASES:
    m = np.array([flag in flags[ex["id"]] for ex in test])
    groups = ["ans", "unans"] if kind == "all" else [kind]
    for g in groups:
        mm = m & base[g]
        if mm.sum() < 20:
            continue
        label = name + (f" ({'có ĐA' if g == 'ans' else 'không ĐA'})" if kind == "all" else "")
        b2, c = 100 * right["B2"][mm].mean(), 100 * right["C"][mm].mean()
        md.append(f"| {label} | {g} | {how} | {int(mm.sum())} | {100 * mm.sum() / base[g].sum():.1f} | "
                  + (f"{100 * top1_ok[mm].mean():.1f}" if g == "ans" else "–")
                  + f" | {ent[mm].mean():.2f} | {b2:.1f} | {c:.1f} | {c - b2:+.1f} |")
        plot.append((label, g, b2, c, 100 * right["B2"][base[g]].mean(), 100 * right["C"][base[g]].mean()))
md.append("\n\"% nhóm\" tính trong nhóm câu có đáp án hoặc không đáp án tương ứng. Nhóm dưới 20 câu bị bỏ.\n")
with open(os.path.join(res_dir, "edge_cases.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

# ---------- examples ----------
def source(ex):
    """Sentence holding the gold answer (or, for an unanswerable question, the plausible answer)."""
    start = spans[ex["id"]][0][0] if spans.get(ex["id"]) else plausible.get(ex["id"], (None,))[0]
    return sentence_at(ex["context"], start) if start is not None else None


def decision(n, i):
    ex = test[i]
    if not answered[n][i]:
        return "từ chối" + (" ✓" if not ex["answers"] else " ✗")
    return f"trả lời “{ex['candidates'][0]['text']}”" + (" ✓" if right[n][i] else " ✗")


ex_md = [f"# Ví dụ cho từng edge case (test; C dùng `{key}`)\n",
         "Ưu tiên câu mà B2 (reader một mình) quyết định sai, câu hỏi và câu gốc ngắn. ✓ / ✗ = quyết định đúng / sai.\n"]
for flag, kind, name, how in EDGE_CASES:
    idx = [i for i, ex in enumerate(test) if flag in flags[ex["id"]] and ex["candidates"] and source(ex)]
    idx.sort(key=lambda i: (right["B2"][i], len(test[i]["question"]) + len(source(test[i]))))
    ex_md.append(f"## {name}\n_{how}_\n")
    for i in idx[:3]:
        ex = test[i]
        gold = " | ".join(ex["answers"][:3]) if ex["answers"] else "(không có đáp án; mồi: " + \
            plausible.get(ex["id"], (0, "?"))[1] + ")"
        ex_md.append(f"- **Q:** {ex['question']}  \n  **Câu gốc:** {source(ex)}  \n  **Gold:** {gold}  \n"
                     f"  **B2:** {decision('B2', i)} · **C:** {decision('C', i)} · P_entail = {ent[i]:.2f}\n")
with open(os.path.join(res_dir, "edge_cases_examples.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(ex_md))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})
# Error rate relative to the same-kind average: positive = this edge case makes the system fail more
plot.sort(key=lambda t: (t[4] - t[2]))
y = np.arange(len(plot))
fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(plot) + 1))
ax.barh(y + 0.2, [t[4] - t[2] for t in plot], 0.38, color=GRAY, label="B2 (chỉ reader)")
ax.barh(y - 0.2, [t[5] - t[3] for t in plot], 0.38, color=BLUE, label=f"C ({verifier})")
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(y, [t[0] for t in plot])
ax.set_xlabel("Tỉ lệ quyết định sai tăng thêm so với mức chung của nhóm (điểm %)")
ax.legend(loc="lower right")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(res_dir, "edge_cases.png"), dpi=180)
print("saved", os.path.join(res_dir, "edge_cases.png"))

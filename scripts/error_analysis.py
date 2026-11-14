"""Step 10 (both): error analysis after the results: where the remaining errors are and why
(test split, run after analyze.py).

  python scripts/error_analysis.py --run runs/squad11k [--verifier nli-large-qa]

  1. Decision errors of B2 (reader alone) and C (reader + NLI) at their calib threshold, by kind.
  2. What the remaining errors of C look like: edge cases over-represented among them (vs. all questions
     of the same kind), how unanswerable questions were made, suspected label noise.
  3. Why combining helps: do the reader and the NLI verifier make the same mistakes?
  4. The most confident remaining errors, as examples.

Writes runs/<run>/results/error_analysis.md and error_analysis.png.
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
from qaverify.metrics import f1_score  # noqa: E402
from qaverify.text import unanswerable_type  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad11k")
p.add_argument("--data_dir", default="data")
p.add_argument("--verifier", default=None, help="verifier of C (default: nli-large-qa if scored, else nli-large)")
p.add_argument("--n_examples", type=int, default=4)
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
flags = edge_flags(rows, load_dev_spans(args.data_dir))
plausible = load_dev_plausible(args.data_dir)
names = {f: n for f, _, n, _ in EDGE_CASES}

systems = {"B2": M.reader_margin, "C": M.Combo(key).fit(calib)}
dec = {}
for n, m in systems.items():
    tau = M.calib_threshold(m, calib)
    out = [m(ex) for ex in test]
    dec[n] = {"answered": np.array([o["conf"] >= tau for o in out]), "conf": np.array([o["conf"] for o in out]),
              "f1": np.array([o["f1"] for o in out]), "correct": np.array([o["correct"] for o in out])}
has = np.array([bool(ex["answers"]) for ex in test])
ent = np.array([M.nli_of(ex, key, 0)[0] if ex["candidates"] else 0.0 for ex in test])
rprob = np.array([ex["candidates"][0]["prob"] if ex["candidates"] else 0.0 for ex in test])


def buckets(d):
    a, c, f1 = d["answered"], d["correct"], d["f1"]
    return {"Trả lời câu KHÔNG có đáp án": a & ~has,
            "Trả lời sai: lệch biên span (0 < F1 < 0.5)": a & has & ~c & (f1 > 0),
            "Trả lời sai: sai hẳn (F1 = 0)": a & has & ~c & (f1 == 0),
            "Từ chối câu có đáp án mà top-1 đúng": ~a & has & c}


bk = {n: buckets(d) for n, d in dec.items()}
md = [f"# Phân tích lỗi sau khi có kết quả (test {len(test)} câu; C dùng `{key}`)\n",
      "## 1. Lỗi quyết định ở ngưỡng chọn trên calib\n",
      "| Loại lỗi | B2 | C | C − B2 |", "|---|---|---|---|"]
for name in bk["B2"]:
    b, c = int(bk["B2"][name].sum()), int(bk["C"][name].sum())
    md.append(f"| {name} | {b} | {c} | {c - b:+d} |")
tb, tc = sum(int(v.sum()) for v in bk["B2"].values()), sum(int(v.sum()) for v in bk["C"].values())
md.append(f"| **Tổng** | **{tb}** | **{tc}** | **{tc - tb:+d}** ({100 * (tc - tb) / tb:+.0f}%) |")
md.append("\nTừ chối câu có đáp án mà top-1 sai không tính là lỗi: đó là từ chối hợp lý.\n")

# ---------- 2. what the remaining errors are ----------
def enrichment(mask, pool):
    """Edge cases over-represented in `mask` compared with `pool` (same kind of questions)."""
    out = []
    for f, n in names.items():
        fm = np.array([f in flags[ex["id"]] for ex in test])
        if fm[pool].sum() < 20 or mask.sum() == 0:
            continue
        share_err, share_all = fm[mask].mean(), fm[pool].mean()
        out.append((n, 100 * share_err, 100 * share_all, share_err / share_all if share_all else np.nan))
    return sorted(out, key=lambda t: -t[3])


fa = bk["C"]["Trả lời câu KHÔNG có đáp án"]
wrong = bk["C"]["Trả lời sai: lệch biên span (0 < F1 < 0.5)"] | bk["C"]["Trả lời sai: sai hẳn (F1 = 0)"]
md += ["## 2. Lỗi còn lại của C tập trung ở đâu\n",
       f"### 2a. {int(fa.sum())} câu không đáp án mà C vẫn trả lời\n",
       "| Edge case | % trong các lỗi này | % trong mọi câu không ĐA | gấp |", "|---|---|---|---|"]
md += [f"| {n} | {a:.1f} | {b:.1f} | ×{r:.1f} |" for n, a, b, r in enrichment(fa, ~has)]
types = {}
for i, ex in enumerate(test):
    if not ex["answers"] and ex["id"] in plausible:
        s = sentence_at(ex["context"], plausible[ex["id"]][0])
        types[i] = unanswerable_type(ex["question"], s) if s else "OTHER"
body = []
for t in ("NEG", "ANT", "ENT", "OTHER"):
    m = np.array([types.get(i) == t for i in range(len(test))])
    body.append(f"| {t} | {100 * m[fa].mean():.1f} | {100 * m[~has].mean():.1f} |")
md += ["\nTheo loại câu không đáp án (luật, khớp nhãn tay 54%):\n", "| Loại | % trong các lỗi này | % trong mọi câu không ĐA |",
       "|---|---|---|"] + body
suspect = fa & (rprob >= 0.9) & (ent >= 0.95)
md.append(f"\nNghi lỗi nhãn (reader xác suất ≥ 0.9 **và** NLI P_entail ≥ 0.95): {int(suspect.sum())} / {int(fa.sum())} "
          f"câu ({100 * suspect.sum() / max(1, fa.sum()):.0f}%). Trong 100 câu gán tay, 3% thực ra có đáp án.\n")
md += [f"### 2b. {int(wrong.sum())} câu có đáp án mà C trả lời sai\n",
       "| Edge case | % trong các lỗi này | % trong mọi câu có ĐA | gấp |", "|---|---|---|---|"]
md += [f"| {n} | {a:.1f} | {b:.1f} | ×{r:.1f} |" for n, a, b, r in enrichment(wrong, has)]
part = bk["C"]["Trả lời sai: lệch biên span (0 < F1 < 0.5)"]
md.append(f"\nLệch biên span (có chồng lấn với gold nhưng F1 < 0.5): {int(part.sum())} / {int(wrong.sum())} "
          f"({100 * part.sum() / max(1, wrong.sum()):.0f}%) — phần lớn là chuyện tiêu chí chấm, không hẳn là hiểu sai.\n")

# ---------- 3. complementarity ----------
v_conf = ent
b_conf = dec["B2"]["conf"]
y = dec["B2"]["correct"].astype(bool)  # V and B2 answer with the same top span
from scipy.stats import spearmanr  # noqa: E402
rho = spearmanr(b_conf, v_conf).correlation
k = len(test) // 2
top_b = np.zeros(len(test), bool)
top_b[np.argsort(-b_conf)[:k]] = True
top_v = np.zeros(len(test), bool)
top_v[np.argsort(-v_conf)[:k]] = True
err_b, err_v = top_b & ~y, top_v & ~y
both = err_b & err_v
b_err = bk["B2"]
b_all = np.zeros(len(test), bool)
for v in b_err.values():
    b_all |= v
c_all = np.zeros(len(test), bool)
for v in bk["C"].values():
    c_all |= v
md += ["## 3. Vì sao kết hợp hiệu quả: reader và NLI sai ở những câu khác nhau\n",
       f"- Tương quan hạng (Spearman) giữa margin của reader và P_entail: **{rho:.2f}** — hai tín hiệu liên quan nhưng không trùng.",
       f"- Ở coverage 50% (mỗi bên trả lời {k} câu tự tin nhất, cùng câu trả lời top-1): reader sai {int(err_b.sum())} câu, "
       f"NLI sai {int(err_v.sum())} câu, **chỉ {int(both.sum())} câu sai chung** "
       f"({100 * both.sum() / max(1, err_b.sum()):.0f}% lỗi của reader).",
       f"- Ở ngưỡng đã chọn: C sửa {int((b_all & ~c_all).sum())} lỗi của B2, gây thêm {int((c_all & ~b_all).sum())} lỗi mới, "
       f"giữ nguyên {int((b_all & c_all).sum())} lỗi chung.\n"]

# ---------- 4. examples ----------
def describe(i):
    ex = test[i]
    gold = "; ".join(ex["answers"][:2]) if ex["answers"] else "(không có đáp án)"
    fl = ", ".join(names[f] for f in sorted(flags[ex["id"]]) if f in names) or "–"
    return (f"**Q:** {ex['question']}  \n**Trả lời:** {ex['candidates'][0]['text']} · **Gold:** {gold}  \n"
            f"C = {dec['C']['conf'][i]:.2f}, reader prob = {rprob[i]:.2f}, P_entail = {ent[i]:.2f} · edge case: {fl}\n")


md.append("## 4. Những lỗi tự tin nhất còn lại của C\n")
for title, mask in (("Trả lời câu không có đáp án", fa), ("Trả lời sai câu có đáp án", wrong)):
    md.append(f"### {title}\n")
    idx = [i for i in np.argsort(-dec["C"]["conf"]) if mask[i]][:args.n_examples]
    md += [describe(i) for i in idx]

with open(os.path.join(res_dir, "error_analysis.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})
labels = list(bk["B2"])
yy = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
bb = ax.barh(yy + 0.2, [bk["B2"][l].sum() for l in labels], 0.38, color=GRAY, label="B2 (chỉ reader)")
bc = ax.barh(yy - 0.2, [bk["C"][l].sum() for l in labels], 0.38, color=BLUE, label=f"C ({verifier})")
for bars in (bb, bc):
    for r in bars:
        ax.annotate(f"{int(r.get_width())}", (r.get_width(), r.get_y() + r.get_height() / 2), textcoords="offset points",
                    xytext=(4, 0), va="center", color=INK, fontsize=8)
ax.set_yticks(yy, labels)
ax.set_xlabel("Số câu (test)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(res_dir, "error_analysis.png"), dpi=180)
print("saved", os.path.join(res_dir, "error_analysis.png"))

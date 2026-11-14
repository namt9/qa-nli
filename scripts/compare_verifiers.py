"""Step 8 (both): compare verifiers head to head on the test split (run after analyze.py).

  python scripts/compare_verifiers.py --run runs/squad11k

  1. Paired bootstrap of AURC differences for chosen pairs of verifiers, for V, R and C
     (each verifier with its best calib setting from settings.json).
  2. The lexical-overlap check of deep_analysis.py with ONE setting (template|win3) for every verifier,
     so the overlap groups are the same questions: how often NLI accepts an unanswerable question when
     every hypothesis word is in the premise vs otherwise, and how often it accepts a correct answer.

Writes runs/<run>/results/compare_verifiers.md and compare_hans.png.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import load_run  # noqa: E402
from qaverify.metrics import aurc, is_correct, paired_bootstrap_diff  # noqa: E402
from qaverify.text import content_words, words  # noqa: E402
from qaverify.verifier import chunk_passage  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad11k")
p.add_argument("--verifiers", default="nli-base,nli-base-qa,nli-base-qa2,nli-base-qa3,nli-large,nli-large-qa")
p.add_argument("--pairs", default="nli-base-qa:nli-base,nli-large-qa:nli-large,nli-large-qa:nli-base-qa,"
                                  "nli-base-qa2:nli-base-qa,nli-base-qa:nli-large,nli-base-qa3:nli-base-qa,"
                                  "nli-base-qa3-s14:nli-base-qa-s14,nli-base-qa3-s15:nli-base-qa-s15",
               help="A:B pairs, reported as A - B")
p.add_argument("--setting", default="template|win3", help="hyp|chunk used for the overlap check")
p.add_argument("--n_boot", type=int, default=1000)
args = p.parse_args()
res_dir = os.path.join(args.run, "results")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

rows = load_run(args.run)
with open(os.path.join(res_dir, "settings.json")) as f:
    settings = json.load(f)
verifiers = [v for v in args.verifiers.split(",") if v in settings]
calib = [r for r in rows.values() if r["split"] == "calib"]
test = [r for r in rows.values() if r["split"] == "test"]


def arrays(method):
    out = [method(ex) for ex in test]
    return np.array([o["conf"] for o in out]), np.array([o["correct"] for o in out], float)


sys_out = {}
for v in verifiers:
    key, lam = settings[v]["key"], settings[v]["lam"]
    sys_out[v] = {"V": arrays(lambda ex, k=key: M.nli_verify(ex, k)),
                  "R": arrays(lambda ex, k=key, l=lam: M.nli_rerank(ex, k, l)),
                  "C": arrays(M.Combo(key).fit(calib))}

md = [f"# So sánh trực tiếp các verifier (test {len(test)} câu)\n",
      "## 1. Kiểm định cặp: ΔAURC ×100 (A − B, âm = A tốt hơn), mỗi verifier dùng setting tốt nhất trên calib\n",
      "| A − B | V | R | C |", "|---|---|---|---|"]
for pair in args.pairs.split(","):
    a, b = pair.split(":")
    if a not in sys_out or b not in sys_out:
        continue
    cells = []
    for m in ("V", "R", "C"):
        d, lo, hi, pv = paired_bootstrap_diff(aurc, *sys_out[a][m], *sys_out[b][m], n_boot=args.n_boot)
        p_txt = f"p<{1 / args.n_boot:.3f}" if pv == 0 else f"p={pv:.3f}"
        cells.append(f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] {p_txt}")
    md.append(f"| {a} − {b} | " + " | ".join(cells) + " |")

unans = [ex for ex in test if not ex["answers"] and ex["candidates"]]
good = [ex for ex in test if ex["answers"] and ex["candidates"] and is_correct(ex["candidates"][0]["text"], ex["answers"])]
hyp, chunk = args.setting.split("|")


def overlap(ex):
    hw = content_words(ex["hyp"][hyp][0])
    if not hw:
        return np.nan
    return max(sum(w in set(words(c)) for w in hw) / len(hw) for c in chunk_passage(ex["context"], chunk))


ov = np.array([overlap(ex) for ex in unans])
full = ov >= 1.0
hans = {}
for v in verifiers:
    k = f"{v}|{args.setting}"
    if k not in unans[0].get("nli", {}):
        continue
    e = np.array([M.nli_of(ex, k, 0)[0] for ex in unans])
    hans[v] = (100 * np.mean(e[full] >= 0.5), 100 * np.mean(e[~full] >= 0.5), int((e >= 0.8).sum()),
               100 * np.mean([M.nli_of(ex, k, 0)[0] >= 0.5 for ex in good]))
md += ["", f"## 2. Trùng lặp từ vựng với cùng setting `{args.setting}` cho mọi verifier\n",
       f"Câu không đáp án của test: {len(unans)}, trong đó {int(full.sum())} câu có overlap = 1 "
       "(mọi từ nội dung của giả thuyết có trong mảnh premise).\n",
       "| Verifier | chấp nhận câu không ĐA, overlap = 1 | chấp nhận câu không ĐA, overlap < 1 | chênh lệch | "
       "chấp nhận mạnh (P ≥ 0.8) | chấp nhận đáp án đúng |", "|---|---|---|---|---|---|"]
md += [f"| {v} | {a:.1f}% | {b:.1f}% | {a - b:+.1f} | {n} | {g:.1f}% |" for v, (a, b, n, g) in hans.items()]
md.append("\n\"Chấp nhận\" = P(entail) của ứng viên top-1 ≥ 0.5.\n")
with open(os.path.join(res_dir, "compare_verifiers.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False})
names = list(hans)
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(7, 3.8))
b1 = ax.bar(x - 0.2, [hans[v][0] for v in names], 0.38, color=ORANGE, label="mọi từ có trong premise (overlap = 1)")
b2 = ax.bar(x + 0.2, [hans[v][1] for v in names], 0.38, color=BLUE, label="overlap < 1")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}%", (r.get_x() + r.get_width() / 2, r.get_height()), textcoords="offset points",
                    xytext=(0, 3), ha="center", color=INK, fontsize=8)
ax.set_xticks(x, names)
ax.set_ylabel("% câu không đáp án NLI chấp nhận")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(res_dir, "compare_hans.png"), dpi=180)
print("saved", os.path.join(res_dir, "compare_hans.png"))

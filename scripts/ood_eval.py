"""Step 11 (both): out-of-domain evaluation. Everything is learned on the SOURCE run's calib split (setting,
lambda, combo C, threshold tau) and applied unchanged to every question of each TARGET run, as in selective
QA under domain shift (Kamath et al., 2020).

  python scripts/ood_eval.py --source runs/squad11k --targets runs/addsent,runs/nq2k,runs/newsqa2k

For each target and verifier: AURC of B2 / V / R / C, paired bootstrap of the difference with B2, and what the
source threshold does there (coverage, risk = error rate among answered questions).
Writes <source>/results/ood_eval.md and ood_eval.png.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import load_run  # noqa: E402
from qaverify.metrics import aurc, paired_bootstrap_diff  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--source", default="runs/squad11k")
p.add_argument("--targets", default="runs/addsent,runs/nq2k,runs/newsqa2k")
p.add_argument("--verifiers", default="nli-base,nli-base-qa,nli-large,nli-large-qa")
p.add_argument("--n_boot", type=int, default=1000)
args = p.parse_args()
res_dir = os.path.join(args.source, "results")
COLORS = {"B2": "#9aa0a8", "V nli-large-qa": "#eb6834", "R nli-large-qa": "#7fb0e6", "C nli-large-qa": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

src = load_run(args.source)
with open(os.path.join(res_dir, "settings.json")) as f:
    settings = json.load(f)
calib = [r for r in src.values() if r["split"] == "calib"]
src_test = [r for r in src.values() if r["split"] == "test"]

systems = {"B2": M.reader_margin}
for v in args.verifiers.split(","):
    if v not in settings:
        continue
    key, lam = settings[v]["key"], settings[v]["lam"]
    systems[f"V {v}"] = lambda ex, k=key: M.nli_verify(ex, k)
    systems[f"R {v}"] = lambda ex, k=key, l=lam: M.nli_rerank(ex, k, l)
    systems[f"C {v}"] = M.Combo(key).fit(calib)
taus = {n: M.calib_threshold(m, calib) for n, m in systems.items()}


def needed_key(name):
    v = name.split(" ", 1)[1] if " " in name else None
    return settings[v]["key"] if v else None


def evaluate(rows, label):
    usable = {n: m for n, m in systems.items() if needed_key(n) is None or needed_key(n) in rows[0].get("nli", {})}
    res = {}
    for n, m in usable.items():
        out = [m(ex) for ex in rows]
        conf = np.array([o["conf"] for o in out])
        y = np.array([o["correct"] for o in out], float)
        ans = conf >= taus[n]
        res[n] = {"conf": conf, "y": y, "aurc": aurc(conf, y), "cov": ans.mean(),
                  "risk": 1 - y[ans].mean() if ans.any() else float("nan")}
    lines = [f"### {label} ({len(rows)} câu, {sum(not r['answers'] for r in rows)} không đáp án; "
             f"top-1 reader đúng {100 * np.mean([M.reader_prob(r)['correct'] for r in rows if r['answers']]):.1f}%)\n",
             "| Phương pháp | AURC ↓ | Δ so với B2 [95% CI] | p | coverage ở τ | risk ở τ |", "|---|---|---|---|---|---|"]
    for n, r in res.items():
        if n == "B2":
            cell, pv = "–", "–"
        else:
            d, lo, hi, pval = paired_bootstrap_diff(aurc, r["conf"], r["y"], res["B2"]["conf"], res["B2"]["y"],
                                                    n_boot=args.n_boot)
            cell, pv = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]", (
                f"<{1 / args.n_boot:.3f}" if pval == 0 else f"{pval:.3f}")
        lines.append(f"| {n} | {100 * r['aurc']:.1f} | {cell} | {pv} | {100 * r['cov']:.1f}% | {100 * r['risk']:.1f}% |")
    return lines, {n: r["aurc"] for n, r in res.items()}


md = ["# Đánh giá ngoài miền: học trên calib của SQuAD 2.0, áp dụng nguyên xi\n",
      "AURC ×100 (thấp hơn là tốt hơn). τ, λ, setting và C đều chọn trên calib của nguồn; risk = tỉ lệ sai "
      "trong các câu được trả lời.\n"]
table = {}
lines, table["SQuAD 2.0 (trong miền)"] = evaluate(src_test, "SQuAD 2.0 test (trong miền)")
md += lines + [""]
for t in args.targets.split(","):
    if not os.path.exists(os.path.join(t, "preds.jsonl")):
        print("skip (no preds):", t)
        continue
    rows = list(load_run(t).values())
    name = os.path.basename(t.rstrip("/"))
    lines, table[name] = evaluate(rows, name)
    md += lines + [""]
with open(os.path.join(res_dir, "ood_eval.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})
sets = list(table)
show = [n for n in COLORS if all(n in table[s] for s in sets)]
x = np.arange(len(sets))
w = 0.8 / max(1, len(show))
fig, ax = plt.subplots(figsize=(8, 3.8))
for j, n in enumerate(show):
    bars = ax.bar(x + (j - (len(show) - 1) / 2) * w, [100 * table[s][n] for s in sets], w * 0.95, color=COLORS[n], label=n)
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), textcoords="offset points",
                    xytext=(0, 2), ha="center", fontsize=7, color=INK)
ax.set_xticks(x, sets)
ax.set_ylabel("AURC ×100 (thấp hơn là tốt hơn)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(res_dir, "ood_eval.png"), dpi=180)
print("saved", os.path.join(res_dir, "ood_eval.png"))

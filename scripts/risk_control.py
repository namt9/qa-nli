"""Step 12 (both): thresholds with a statistical guarantee on the selective risk (run after analyze.py).

  python scripts/risk_control.py --run runs/squad11k [--targets runs/addsent,runs/nq2k,runs/newsqa2k]

Goal: answer as many questions as possible while P(wrong | answered) <= alpha holds with probability >= 1 - delta.
  naive  largest coverage whose empirical risk on the threshold-calibration set is <= alpha
  LTT-Bonf   Learn-then-Test (Angelopoulos et al., 2021): on a grid of 100 coverage levels, test H0: risk > alpha
             with the exact binomial p-value of the errors among answered questions, reject at delta / 100
             (Bonferroni), keep the largest rejected coverage.
  LTT-FS     same tests in a fixed order, from low to high coverage, each at level delta, stopping at the first
             one not rejected (fixed-sequence testing: also controls the error, with more power).
The candidate thresholds are FIXED before looking at the data the tests use: quantiles of each method's
confidence on the calib split (100 coverage levels), which is disjoint from the test split the tests run on.
Confidence scores are fixed (C is fitted on the calib split). Validity is checked on 200 random halvings of the
test split: thresholds from one half, realised risk on the other; the share of halvings with risk > alpha must
stay <= delta for LTT. With --targets, LTT-Bonf thresholds set on the whole in-domain test split are applied unchanged
to out-of-domain runs.

Writes runs/<run>/results/risk_control.md and risk_control.png.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import load_run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad11k")
p.add_argument("--targets", default="")
p.add_argument("--alphas", default="0.02,0.05,0.10")
p.add_argument("--delta", type=float, default=0.1)
p.add_argument("--n_splits", type=int, default=200)
args = p.parse_args()
res_dir = os.path.join(args.run, "results")
GRAY, LIGHT, BLUE, INK, MUTED, GRID = "#9aa0a8", "#7fb0e6", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
alphas = [float(a) for a in args.alphas.split(",")]

rows = load_run(args.run)
with open(os.path.join(res_dir, "settings.json")) as f:
    settings = json.load(f)
calib = [r for r in rows.values() if r["split"] == "calib"]
test = [r for r in rows.values() if r["split"] == "test"]
systems = {"B2": M.reader_margin}
for v in ("nli-large", "nli-large-qa"):
    if v in settings:
        systems[f"C {v}"] = M.Combo(settings[v]["key"]).fit(calib)


def scores(method, exs):
    out = [method(ex) for ex in exs]
    return np.array([o["conf"] for o in out]), np.array([o["correct"] for o in out], bool)


GRID_COV = np.linspace(0.01, 1.0, 100)


def thresholds(conf, ok, alpha, delta, grid):
    """{rule: threshold} for a fixed descending grid of candidate thresholds; +inf means answer nothing."""
    out = {"naive": np.inf, "LTT-Bonf": np.inf, "LTT-FS": np.inf}
    sequence_open = True
    for lam in grid:  # strictest (highest) threshold first
        ans = conf >= lam
        n, k = int(ans.sum()), int((~ok[ans]).sum())
        if n == 0:
            continue
        pval = binom.cdf(k, n, alpha)  # exact p-value of H0: risk > alpha, given n answered
        if k / n <= alpha:
            out["naive"] = lam
        if pval <= delta / len(grid):
            out["LTT-Bonf"] = lam
        if sequence_open and pval <= delta:
            out["LTT-FS"] = lam
        else:
            sequence_open = False
    return out


def realised(conf, ok, tau):
    ans = conf >= tau
    return ans.mean(), (1 - ok[ans].mean()) if ans.any() else 0.0


md = [f"# Ngưỡng có bảo đảm rủi ro (δ = {args.delta}, {args.n_splits} lần chia đôi tập test)\n",
      "Mục tiêu: rủi ro = P(sai | được trả lời) ≤ α. \"Vi phạm\" = tỉ lệ lần chia mà rủi ro thực tế ở nửa còn lại > α; "
      f"hai cách LTT bảo đảm vi phạm ≤ δ = {args.delta}.\n",
      "| Phương pháp | α | Cách chọn ngưỡng | Coverage TB | Rủi ro TB | Vi phạm |", "|---|---|---|---|---|---|"]
rng = np.random.default_rng(0)
splits = [rng.permutation(len(test)) for _ in range(args.n_splits)]
test_scores = {n: scores(m, test) for n, m in systems.items()}
# candidate thresholds fixed in advance from the calib split (disjoint from the test halves used below)
grids = {n: np.quantile(scores(m, calib)[0], 1 - GRID_COV) for n, m in systems.items()}
plot = {}
for n, (conf, ok) in test_scores.items():
    for a in alphas:
        for rule in ("naive", "LTT-Bonf", "LTT-FS"):
            covs, risks = [], []
            for perm in splits:
                half = len(perm) // 2
                cal, ev = perm[:half], perm[half:]
                tau = thresholds(conf[cal], ok[cal], a, args.delta, grids[n])[rule]
                cv, rk = realised(conf[ev], ok[ev], tau)
                covs.append(cv)
                risks.append(rk)
            viol = np.mean(np.array(risks) > a)
            md.append(f"| {n} | {a:.2f} | {rule} | {100 * np.mean(covs):.1f}% | {100 * np.mean(risks):.2f}% | "
                      f"{100 * viol:.0f}% |")
            plot[(n, a, rule)] = (np.mean(covs), viol)

if args.targets:
    md += ["", "## Ngưỡng LTT-Bonf đặt trên SQuAD, áp dụng nguyên xi sang bộ ngoài miền\n",
           "| Phương pháp | α | Bộ dữ liệu | Coverage | Rủi ro thực tế | Vượt α? |", "|---|---|---|---|---|---|"]
    taus = {(n, a): thresholds(*test_scores[n], a, args.delta, grids[n])["LTT-Bonf"] for n in systems for a in alphas}
    for t in args.targets.split(","):
        if not os.path.exists(os.path.join(t, "preds.jsonl")):
            continue
        trows = list(load_run(t).values())
        for n, m in systems.items():
            key = settings[n.split(" ", 1)[1]]["key"] if " " in n else None
            if key and key not in trows[0].get("nli", {}):
                continue
            conf, ok = scores(m, trows)
            for a in alphas:
                cv, rk = realised(conf, ok, taus[(n, a)])
                md.append(f"| {n} | {a:.2f} | {os.path.basename(t)} | {100 * cv:.1f}% | {100 * rk:.1f}% | "
                          f"{'**có**' if rk > a else 'không'} |")
with open(os.path.join(res_dir, "risk_control.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})
names = list(systems)
x = np.arange(len(alphas))
w = 0.8 / len(names)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for j, (n, col) in enumerate(zip(names, (GRAY, LIGHT, BLUE))):
    bars = ax.bar(x + (j - (len(names) - 1) / 2) * w, [100 * plot[(n, a, "LTT-Bonf")][0] for a in alphas], w * 0.95,
                  color=col, label=n)
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()), textcoords="offset points",
                    xytext=(0, 2), ha="center", fontsize=8, color=INK)
ax.set_xticks(x, [f"rủi ro ≤ {int(100 * a)}%" for a in alphas])
ax.set_ylabel("% câu được trả lời, có bảo đảm (LTT-Bonf)")
ax.legend(loc="upper left")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(res_dir, "risk_control.png"), dpi=180)
print("saved", os.path.join(res_dir, "risk_control.png"))

"""Step 7 (both): deeper analyses for the report, on the test split (run after analyze.py).

  python scripts/deep_analysis.py --run runs/squad11k

  1. Why NLI is fooled: lexical overlap between hypothesis and premise vs P(entail) (the HANS heuristic,
     McCoy et al. 2019), on unanswerable questions.
  2. How unanswerable questions were made, on ALL of them: a rule-based type (NEG / ANT / ENT / OTHER)
     checked against the manual labels, then reader / NLI / combo behaviour per type.
  3. How solid the main result is: cluster bootstrap by paragraph and by article, wins over the 35
     articles, other correctness criteria (EM, F1 >= 0.8), size of the calib split.
  4. What the combo C uses: feature ablation and coefficients, calibration (ECE, reliability),
     where along the coverage axis NLI helps.

Writes runs/<run>/results/deep_analysis.md and deep_*.png (with a _<verifier> suffix for any verifier
other than nli-large, e.g. deep_analysis_nli-base-qa.md).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import load_run  # noqa: E402
from qaverify.metrics import aurc, exact_match, f1_score, risk_coverage  # noqa: E402
from qaverify.text import content_words, unanswerable_type, words  # noqa: E402
from qaverify.verifier import chunk_passage, split_sentences  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad11k")
p.add_argument("--data_dir", default="data")
p.add_argument("--verifier", default="nli-large")
p.add_argument("--n_boot", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
res_dir = os.path.join(args.run, "results")
suffix = "" if args.verifier == "nli-large" else f"_{args.verifier}"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9aa0a8"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

rows = load_run(args.run)
with open(os.path.join(res_dir, "settings.json")) as f:
    settings = json.load(f)
key, lam = settings[args.verifier]["key"], settings[args.verifier]["lam"]
chunk_mode = key.split("|")[2]
calib = [r for r in rows.values() if r["split"] == "calib"]
test = [r for r in rows.values() if r["split"] == "test"]
with open(os.path.join(args.data_dir, "squad_v2_dev.json"), encoding="utf-8") as f:
    plausible = {qa["id"]: [(a["answer_start"], a["text"]) for a in qa.get("plausible_answers", [])]
                 for art in json.load(f)["data"] for para in art["paragraphs"] for qa in para["qas"]}
has_ans = np.array([bool(r["answers"]) for r in test])
md = [f"# Phân tích sâu (test {len(test)} câu, NLI = `{key}`, λ = {lam:.1f})\n"]


def table(header, body):
    return "\n".join(["| " + " | ".join(header) + " |", "|---" * len(header) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in body]) + "\n"


def pct(x):
    return f"{100 * np.mean(x):.1f}" if len(x) else "-"


combo = M.Combo(key).fit(calib)
systems = {"B2": M.reader_margin, "V": lambda ex: M.nli_verify(ex, key),
           "R": lambda ex: M.nli_rerank(ex, key, lam), "C": combo}
outs = {n: [m(ex) for ex in test] for n, m in systems.items()}
conf = {n: np.array([o["conf"] for o in v]) for n, v in outs.items()}
corr = {n: np.array([o["correct"] for o in v], float) for n, v in outs.items()}
taus = {n: M.calib_threshold(m, calib) for n, m in systems.items()}
answered = {n: conf[n] >= taus[n] for n in systems}
ent_top = np.array([M.nli_of(ex, key, 0)[0] if ex["candidates"] else 0.0 for ex in test])

# ---------- 1. lexical overlap (HANS) ----------
def hyp_overlap(ex):
    """Max over premise chunks of the share of the top-1 hypothesis' content words found in the chunk."""
    if not ex["candidates"]:
        return np.nan
    hw = content_words(ex["hyp"][key.split("|")[1]][0])
    if not hw:
        return np.nan
    return max(sum(w in set(words(c)) for w in hw) / len(hw) for c in chunk_passage(ex["context"], chunk_mode))


ov = np.array([hyp_overlap(ex) for ex in test])
bins = [(0, 0.6, "< 0.6"), (0.6, 0.8, "0.6–0.8"), (0.8, 1.0, "0.8–1"), (1.0, 1.01, "= 1 (mọi từ có trong premise)")]
body = []
for lo, hi, name in bins:
    u = (~has_ans) & (ov >= lo) & (ov < hi)
    a = has_ans & (corr["B2"] == 1) & (ov >= lo) & (ov < hi)
    body.append([name, int(u.sum()), f"{ent_top[u].mean():.2f}" if u.any() else "-", pct(ent_top[u] >= 0.5),
                 pct(answered["C"][u]), int(a.sum()), pct(ent_top[a] >= 0.5)])
fooled = (~has_ans) & (ent_top >= 0.8)
md += ["## 1. NLI bị lừa bởi trùng lặp từ vựng?\n",
       "Overlap = tỉ lệ từ nội dung của giả thuyết (câu hỏi + ứng viên top-1) có trong mảnh premise khớp nhất.\n",
       table(["Overlap", "n (không ĐA)", "P_entail TB", "% P_entail ≥ 0.5", "% C trả lời",
              "n (có ĐA, đúng)", "% P_entail ≥ 0.5"], body),
       f"\nCâu không đáp án bị NLI chấp nhận mạnh (P_entail ≥ 0.8): {int(fooled.sum())}; "
       f"trong đó {pct(ov[fooled] >= 1.0)}% có overlap = 1, so với {pct(ov[(~has_ans) & ~fooled] >= 1.0)}% "
       f"ở các câu không đáp án còn lại.\n"]

# ---------- 2. types of unanswerable questions ----------
def source_sentence(ex):
    pa = plausible.get(ex["id"])
    if not pa:
        return None
    start, text = pa[0]
    pos = 0
    for s in split_sentences(ex["context"]):
        at = ex["context"].find(s, pos)
        if at >= 0 and at <= start < at + len(s):
            return s
        pos = max(pos, at + len(s)) if at >= 0 else pos
    return None


auto = {}
for ex in test:
    if not ex["answers"]:
        s = source_sentence(ex)
        auto[ex["id"]] = unanswerable_type(ex["question"], s) if s else "OTHER"

manual = {}
ann_dir = os.path.join(args.run, "annotation")
for who in "AB":
    path = os.path.join(ann_dir, f"sheet_{who}.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            manual.update({r["id"]: r["label"].strip().upper() for r in csv.DictReader(f) if r["label"].strip()})
GROUP = {"NEG": "NEG", "ANT": "ANT", "ENT": "ENT", "MUT": "OTHER", "IMP": "OTHER", "OTH": "OTHER"}
pairs = [(GROUP[l], auto[i]) for i, l in manual.items() if l in GROUP and i in auto]
md += ["## 2. Câu không đáp án được tạo thế nào: trên toàn bộ tập test\n",
       "Luật: so câu hỏi với câu chứa plausible answer. NEG = khác biệt phủ định; ANT = có từ trái nghĩa; "
       "ENT = câu hỏi có thực thể/số không có trong câu gốc; còn lại OTHER (MUT, IMP, OTH).\n"]
if pairs:
    cats = ["NEG", "ANT", "ENT", "OTHER"]
    body = [[m] + [sum(1 for x, y in pairs if x == m and y == a) for a in cats] for m in cats]
    acc = np.mean([x == y for x, y in pairs])
    md += [f"\nĐối chiếu với nhãn tay ({len(pairs)} câu, bỏ ANS): luật trùng nhãn tay ở {100 * acc:.0f}% số câu.\n\n",
           table(["Nhãn tay \\ luật"] + cats, body)]
ids = np.array([ex["id"] for ex in test])
body = []
for c in ["NEG", "ANT", "ENT", "OTHER"]:
    m = np.array([auto.get(i) == c for i in ids])
    body.append([c, int(m.sum()), pct(m[~has_ans]), f"{ent_top[m].mean():.2f}", pct(answered["B2"][m]),
                 pct(answered["V"][m]), pct(answered["C"][m]), pct(answered["R"][m])])
md += ["\nToàn bộ câu không đáp án của test, theo loại do luật gán (\"trả lời\" = sai, thấp hơn là tốt hơn):\n\n",
       table(["Loại", "n", "% câu không ĐA", "P_entail TB", "B2 trả lời %", "V trả lời %", "C trả lời %",
              "R trả lời %"], body)]

# ---------- 3. robustness ----------
rng = np.random.default_rng(args.seed)


def cluster_diff(a, b, groups):
    """C(a) - C(b) AURC with a bootstrap that resamples whole groups (paragraphs or articles)."""
    uniq = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(args.n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq))])
        vals.append(aurc(conf[a][idx], corr[a][idx]) - aurc(conf[b][idx], corr[b][idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


groups = {"câu hỏi (iid)": np.arange(len(test)),
          "đoạn văn": np.array([hash(ex["context"]) for ex in test]),
          "bài (35)": np.array([ex["title"] for ex in test])}
body = []
for a in ("C", "R", "V"):
    d = aurc(conf[a], corr[a]) - aurc(conf["B2"], corr["B2"])
    cells = [f"{a} − B2", f"{100 * d:+.2f}"]
    for g in groups.values():
        lo, hi = cluster_diff(a, "B2", g)
        cells.append(f"[{100 * lo:+.2f}, {100 * hi:+.2f}]")
    body.append(cells)
md += ["## 3. Kết luận có vững không?\n", "### 3a. Bootstrap theo cụm (câu hỏi cùng đoạn văn không độc lập)\n",
       table(["ΔAURC ×100", "giá trị"] + [f"95% CI, lấy mẫu theo {g}" for g in groups], body)]

titles = sorted({ex["title"] for ex in test})
per_art = []
for t in titles:
    m = np.array([ex["title"] == t for ex in test])
    per_art.append((t, int(m.sum()), aurc(conf["B2"][m], corr["B2"][m]), aurc(conf["C"][m], corr["C"][m]),
                    aurc(conf["R"][m], corr["R"][m])))
wins_c = sum(c < b for _, _, b, c, _ in per_art)
wins_r = sum(r < b for _, _, b, _, r in per_art)


def sign_p(k, n):
    """Two-sided binomial sign test p-value for k wins out of n."""
    from math import comb
    tail = sum(comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


md += ["\n### 3b. Nhất quán qua các bài\n",
       f"C có AURC thấp hơn B2 ở **{wins_c}/{len(titles)}** bài (sign test p = {sign_p(wins_c, len(titles)):.4f}); "
       f"R thấp hơn B2 ở {wins_r}/{len(titles)} bài (p = {sign_p(wins_r, len(titles)):.4f}).\n"]


def with_criterion(name, fn):
    """Re-run all systems with another definition of a correct answer (combo refit on calib with it)."""
    saved = M.is_correct
    M.is_correct = fn
    try:
        cm = M.Combo(key).fit(calib)
        sy = {"B2": M.reader_margin, "V": lambda ex: M.nli_verify(ex, key),
              "R": lambda ex: M.nli_rerank(ex, key, lam), "C": cm}
        res = {}
        for n, m in sy.items():
            o = [m(ex) for ex in test]
            res[n] = aurc([x["conf"] for x in o], [x["correct"] for x in o])
        return res
    finally:
        M.is_correct = saved


crit = {"EM": lambda p_, g: bool(g) and exact_match(p_, g) == 1.0,
        "F1 ≥ 0.5 (mặc định)": lambda p_, g: bool(g) and f1_score(p_, g) >= 0.5,
        "F1 ≥ 0.8": lambda p_, g: bool(g) and f1_score(p_, g) >= 0.8}
body = []
for name, fn in crit.items():
    r_ = with_criterion(name, fn)
    body.append([name] + [f"{100 * r_[n]:.1f}" for n in ("B2", "V", "R", "C")] +
                [f"{100 * (r_['C'] - r_['B2']):+.1f}", f"{100 * (r_['R'] - r_['B2']):+.1f}"])
md += ["\n### 3c. Đổi tiêu chí \"trả lời đúng\" (AURC ×100)\n",
       table(["Tiêu chí", "B2", "V", "R", "C", "C − B2", "R − B2"], body)]

X_cal = np.array([M.combo_features(ex, key) for ex in calib])
y_cal = np.array([M.reader_prob(ex)["correct"] for ex in calib], int)
X_te = np.array([M.combo_features(ex, key) for ex in test])
y_te = corr["B2"]  # combo answers with the top span, like B2
curve = []
for n in (50, 100, 200, 500, 1000, len(calib)):
    vals = []
    for s in range(5 if n < len(calib) else 1):
        idx = np.random.default_rng(s).choice(len(calib), n, replace=False)
        if len(set(y_cal[idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000).fit(X_cal[idx], y_cal[idx])
        vals.append(aurc(clf.predict_proba(X_te)[:, 1], y_te))
    curve.append((n, np.mean(vals), np.std(vals)))
md += ["\n### 3d. Cần bao nhiêu câu calib để học C?\n",
       table(["Số câu calib", "AURC C (TB ± sd, 5 lần)", "so với B2"],
             [[n, f"{100 * m:.2f} ± {100 * s:.2f}", f"{100 * (m - aurc(conf['B2'], corr['B2'])):+.2f}"]
              for n, m, s in curve])]

# ---------- 4. inside the combo ----------
FEATS = ["P_reader", "margin", "P_entail", "P_contra"]
mu, sd = X_cal.mean(0), X_cal.std(0) + 1e-9
full = LogisticRegression(max_iter=1000).fit((X_cal - mu) / sd, y_cal)
body = [["tất cả 4", f"{100 * aurc(full.predict_proba((X_te - mu) / sd)[:, 1], y_te):.2f}", "-"]]
for drop in range(4):
    keep = [i for i in range(4) if i != drop]
    clf = LogisticRegression(max_iter=1000).fit(X_cal[:, keep], y_cal)
    body.append([f"bỏ {FEATS[drop]}", f"{100 * aurc(clf.predict_proba(X_te[:, keep])[:, 1], y_te):.2f}",
                 f"{full.coef_[0][drop]:+.2f}"])
for name, keep in (("chỉ margin", [1]), ("margin + P_entail", [1, 2])):
    clf = LogisticRegression(max_iter=1000).fit(X_cal[:, keep], y_cal)
    body.append([name, f"{100 * aurc(clf.predict_proba(X_te[:, keep])[:, 1], y_te):.2f}", "-"])
md += ["## 4. Bên trong phương pháp C\n", "### 4a. Bỏ từng đặc trưng (AURC test ×100) và hệ số chuẩn hoá\n",
       table(["Đặc trưng", "AURC", "hệ số (chuẩn hoá) của đặc trưng bị bỏ"], body)]


def ece(p_, y, n_bins=10):
    p_, y = np.clip(np.asarray(p_, float), 0, 1), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    tot, rel = 0.0, []
    for i in range(n_bins):
        m = (p_ >= edges[i]) & ((p_ < edges[i + 1]) if i < n_bins - 1 else (p_ <= 1))
        if m.any():
            tot += m.mean() * abs(p_[m].mean() - y[m].mean())
            rel.append((p_[m].mean(), y[m].mean(), int(m.sum())))
    return tot, rel


b1 = [M.reader_prob(ex) for ex in test]
prob_like = {"B1 reader-prob": (np.array([o["conf"] for o in b1]), np.array([o["correct"] for o in b1], float)),
             "V P(entail)": (conf["V"], corr["V"]), "C combo": (conf["C"], corr["C"])}
eces = {n: ece(*v) for n, v in prob_like.items()}
md += ["\n### 4b. Hiệu chỉnh xác suất (ECE, 10 bin, thấp hơn là tốt hơn)\n",
       table(["Độ tự tin", "ECE ×100"], [[n, f"{100 * e[0]:.1f}"] for n, e in eces.items()])]

cov_grid = np.linspace(0.05, 1.0, 20)
def risk_at(name):
    cov, risk = risk_coverage(conf[name], corr[name])
    return np.interp(cov_grid, cov, risk)
dr = {n: risk_at(n) - risk_at("B2") for n in ("C", "R", "V")}
md += ["\n### 4c. NLI giúp ở vùng coverage nào (risk − risk của B2, điểm %)\n",
       table(["Coverage"] + [f"{c:.0%}" for c in cov_grid[3::4]],
             [[n] + [f"{100 * v:+.1f}" for v in d[3::4]] for n, d in dr.items()])]

with open(os.path.join(res_dir, f"deep_analysis{suffix}.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

# ---------- figures ----------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False})


def save(fig, ax, name):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, name.replace(".png", f"{suffix}.png")), dpi=180)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(6, 3.6))
xs = np.arange(len(bins))
u_rate = [100 * np.mean(ent_top[(~has_ans) & (ov >= lo) & (ov < hi)] >= 0.5) for lo, hi, _ in bins]
ax.bar(xs, u_rate, 0.6, color=ORANGE)
for x, v in zip(xs, u_rate):
    ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(0, 4), ha="center", color=INK, fontsize=9)
ax.set_xticks(xs, [b[2].split(" (")[0] for b in bins])
ax.set_xlabel("Overlap từ nội dung giả thuyết – premise")
ax.set_ylabel("% câu không đáp án NLI chấp nhận")
save(fig, ax, "deep_overlap_entail.png")

fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
for (n, e), col in zip(eces.items(), (GRAY, ORANGE, BLUE)):
    r_ = e[1]
    ax.plot([a for a, _, _ in r_], [b for _, b, _ in r_], "o-", color=col, lw=2, ms=6, label=f"{n} (ECE {100 * e[0]:.1f})")
ax.set_xlabel("Độ tự tin dự đoán")
ax.set_ylabel("Tỉ lệ đúng thực tế")
ax.legend(loc="upper left", fontsize=8)
save(fig, ax, "deep_reliability.png")

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.axhline(0, color=GRAY, lw=1)
for (n, d), col in zip(dr.items(), (BLUE, AQUA, ORANGE)):
    ax.plot(cov_grid, 100 * d, color=col, lw=2, label=f"{n} − B2")
ax.set_xlabel("Coverage (tỉ lệ câu được trả lời)")
ax.set_ylabel("Δ risk so với B2 (điểm %)")
ax.legend(loc="lower right")
save(fig, ax, "deep_delta_risk.png")

fig, ax = plt.subplots(figsize=(6, 3.6))
order = sorted(per_art, key=lambda t: t[2] - t[3])
ax.bar(np.arange(len(order)), [100 * (b - c) for _, _, b, c, _ in order], 0.7,
       color=[BLUE if b > c else ORANGE for _, _, b, c, _ in order])
ax.axhline(0, color=GRAY, lw=1)
ax.set_xticks([])
ax.set_xlabel(f"{len(order)} bài Wikipedia (sắp theo mức cải thiện)")
ax.set_ylabel("AURC B2 − AURC C (điểm)")
save(fig, ax, "deep_articles.png")
print("figures saved in", res_dir)

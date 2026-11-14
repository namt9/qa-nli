"""Step 3 (both): merge reader predictions with all verifier scores, tune on the calib split,
report on the test split.

  python scripts/analyze.py --run runs/squad2k

Outputs in runs/<run>/results/:
  main.md        B1, B2 and, per verifier, the best V / R / C / RC setting (chosen on calib)
  significance.md  paired bootstrap of test AURC differences vs B1 and B2
  breakdown.md   answerable vs unanswerable: detecting unanswerable questions, AURC on answerable only,
                 what the calib threshold does on each group
  ablation.md    test AURC of every verifier x hypothesis x chunking (V only)
  all.csv        every method, every metric
  risk_coverage.png
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import read_jsonl  # noqa: E402
from qaverify.metrics import (acc_at_coverage, aurc, auroc, best_threshold,  # noqa: E402
                              bootstrap_ci, paired_bootstrap_diff, risk_coverage,
                              squad2_f1_with_threshold)
from qaverify.verifier import PARAMS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad2k")
p.add_argument("--n_boot", type=int, default=1000)
args = p.parse_args()
out_dir = os.path.join(args.run, "results")
os.makedirs(out_dir, exist_ok=True)

# ---------- load & merge ----------
rows = {r["id"]: r for r in read_jsonl(os.path.join(args.run, "preds.jsonl"))}
scored_ids = None
for path in sorted(glob.glob(os.path.join(args.run, "scores", "*.jsonl"))):
    scores = read_jsonl(path)
    ids = {sc["id"] for sc in scores}
    if len(ids) < len(rows):
        # a verifier still being scored would otherwise shrink every table to the questions it has done
        print(f"skip {os.path.basename(path)}: {len(ids)}/{len(rows)} questions scored")
        continue
    for sc in scores:
        rows[sc["id"]].setdefault("nli", {}).update(sc["nli"])
    scored_ids = ids if scored_ids is None else scored_ids & ids
data = [r for r in rows.values() if scored_ids is None or r["id"] in scored_ids]
calib = [r for r in data if r["split"] == "calib"]
test = [r for r in data if r["split"] == "test"]
keys = sorted(data[0].get("nli", {}).keys()) if data else []
print(f"{len(data)} examples ({len(calib)} calib / {len(test)} test), {len(keys)} NLI settings")


def run(method, exs):
    out = [method(ex) for ex in exs]
    return (np.array([o["conf"] for o in out]), np.array([o["correct"] for o in out], float),
            np.array([o["f1"] for o in out]), [ex["answers"] for ex in exs])


def evaluate(name, method):
    cc, yc, fc, gc = run(method, calib)
    ct, yt, ft, gt = run(method, test)
    tau = best_threshold(cc, fc, gc)
    lo, hi = bootstrap_ci(aurc, ct, yt, n_boot=args.n_boot)
    return {
        "method": name,
        "AURC": aurc(ct, yt), "AURC_lo": lo, "AURC_hi": hi,
        "AUROC": auroc(ct, yt),
        "Acc@50": acc_at_coverage(ct, yt, 0.5), "Acc@80": acc_at_coverage(ct, yt, 0.8),
        "Acc@100": acc_at_coverage(ct, yt, 1.0),
        "SQuAD2-F1": squad2_f1_with_threshold(ct, ft, gt, tau),
        "calib_AURC": aurc(cc, yc),
        "_curve": (ct, yt), "_tau": tau,
    }


def tune_lambda(key):
    grid = np.linspace(0, 1, 11)
    scores = [aurc(*run(lambda ex: M.nli_rerank(ex, key, lam), calib)[:2]) for lam in grid]
    return float(grid[int(np.argmin(scores))])


results = [evaluate("B1 reader-prob", M.reader_prob), evaluate("B2 reader-margin", M.reader_margin)]
ablation = {}
for key in keys:
    r = evaluate(f"V {key}", lambda ex, k=key: M.nli_verify(ex, k))
    results.append(r)
    ablation[key] = r

main = results[:2]
settings, per_verifier = {}, {}
for verifier in sorted({k.split("|")[0] for k in keys}):
    vkeys = [k for k in keys if k.startswith(verifier + "|")]
    best = min(vkeys, key=lambda k: ablation[k]["calib_AURC"])
    main.append(ablation[best])
    lam = tune_lambda(best)
    rr = evaluate(f"R {best} lam={lam:.1f}", lambda ex, k=best, l=lam: M.nli_rerank(ex, k, l))
    combo = M.Combo(best).fit(calib)
    cr = evaluate(f"C {best}", combo)
    rc = evaluate(f"RC {best} lam={lam:.1f}", M.RerankCombo(best, lam).fit(calib))
    results += [rr, cr, rc]
    main += [rr, cr, rc]
    settings[verifier] = {"key": best, "lam": lam}
    per_verifier[verifier] = {"V": ablation[best], "R": rr, "C": cr, "RC": rc}
with open(os.path.join(out_dir, "settings.json"), "w") as f:
    json.dump(settings, f, indent=2)

# ---------- write ----------
cols = ["AURC", "AUROC", "Acc@50", "Acc@80", "Acc@100", "SQuAD2-F1"]


def table(rs, title):
    lines = [f"## {title}", "", "| Method | " + " | ".join(cols) + " | AURC 95% CI |",
             "|---" * (len(cols) + 2) + "|"]
    for r in rs:
        vals = " | ".join(f"{r[c] * 100:.1f}" for c in cols)
        name = r["method"].replace("|", "\\|")
        lines.append(f"| {name} | {vals} | [{r['AURC_lo'] * 100:.1f}, {r['AURC_hi'] * 100:.1f}] |")
    return "\n".join(lines) + "\n"


note = (f"\nTest split: {len(test)} questions ({sum(not r['answers'] for r in test)} unanswerable). "
        "All numbers x100. AURC lower is better; the rest higher is better. "
        "Settings, lambda and the SQuAD2-F1 threshold are chosen on the calib split.\n")
with open(os.path.join(out_dir, "main.md"), "w") as f:
    f.write(table(main, "Main results") + note)

# Paired bootstrap on the test split: negative delta = lower AURC than the baseline (better)
b1, b2 = main[0], main[1]
with open(os.path.join(out_dir, "significance.md"), "w") as f:
    f.write("## Paired bootstrap: test AURC difference x100 (negative = better than the baseline)\n\n"
            "| Method | vs B1 | p | vs B2 | p |\n|---|---|---|---|---|\n")
    for r in main[1:]:
        cells = []
        for base in (b1, b2):
            if base is r:
                cells += ["-", "-"]
                continue
            d, lo, hi, pv = paired_bootstrap_diff(aurc, *r["_curve"], *base["_curve"], n_boot=args.n_boot)
            p_txt = f"<{1 / args.n_boot:.3f}" if pv == 0 else f"{pv:.3f}"
            cells += [f"{d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]", p_txt]
        f.write(f"| {r['method'].replace('|', chr(92) + '|')} | " + " | ".join(cells) + " |\n")
    f.write(f"\n{args.n_boot} paired resamples of the {len(test)} test questions; 95% percentile CI; "
            "two-sided p = share of resamples on the other side of 0.\n")

# Answerable vs unanswerable on the test split
has_ans = np.array([bool(r["answers"]) for r in test])
with open(os.path.join(out_dir, "breakdown.md"), "w") as f:
    f.write("## Answerable vs unanswerable (test, x100)\n\n"
            "| Method | AUROC has-answer | Answerable: AURC | Answerable: AUROC | "
            "@tau: answered unans. (lower better) | @tau: answered ans. | @tau: acc on answered ans. |\n"
            "|---|---|---|---|---|---|---|\n")
    for r in main:
        ct, yt = r["_curve"]
        answered = ct >= r["_tau"]
        cells = [auroc(ct, has_ans), aurc(ct[has_ans], yt[has_ans]), auroc(ct[has_ans], yt[has_ans]),
                 answered[~has_ans].mean(), answered[has_ans].mean(),
                 yt[has_ans & answered].mean() if (has_ans & answered).any() else float("nan")]
        f.write(f"| {r['method'].replace('|', chr(92) + '|')} | " + " | ".join(f"{c * 100:.1f}" for c in cells) + " |\n")
    f.write(f"\n{has_ans.sum()} answerable / {(~has_ans).sum()} unanswerable test questions. "
            "AUROC has-answer: how well the confidence separates answerable from unanswerable questions. "
            "Answerable AURC/AUROC: ranking of right vs wrong spans when an answer exists. "
            "@tau: share of each group answered at the SQuAD2-F1 threshold chosen on calib.\n")

verifiers = sorted({k.split("|")[0] for k in keys})
hyps = sorted({k.split("|")[1] for k in keys})
chunks = sorted({k.split("|")[2] for k in keys})
with open(os.path.join(out_dir, "ablation.md"), "w") as f:
    f.write("## Ablation: test AURC x100 (lower is better)\n\n")
    f.write("| Verifier | Hypothesis | " + " | ".join(chunks) + " |\n" + "|---" * (len(chunks) + 2) + "|\n")
    for v in verifiers:
        for h in hyps:
            cells = [f"{ablation[f'{v}|{h}|{c}']['AURC'] * 100:.1f}" if f"{v}|{h}|{c}" in ablation else "-"
                     for c in chunks]
            f.write(f"| {v} | {h} | " + " | ".join(cells) + " |\n")

with open(os.path.join(out_dir, "all.csv"), "w") as f:
    f.write("method," + ",".join(cols + ["AURC_lo", "AURC_hi", "calib_AURC"]) + "\n")
    for r in results:
        f.write(f"\"{r['method']}\"," + ",".join(f"{r[c]:.4f}" for c in cols + ["AURC_lo", "AURC_hi", "calib_AURC"]) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in main:
        cov, risk = risk_coverage(*r["_curve"])
        ax.plot(cov, risk, label=f"{r['method']} ({r['AURC'] * 100:.1f})", lw=1.5)
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Risk (error rate among answered)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_coverage.png"), dpi=150)

    # AURC vs verifier size (only verifiers with a known parameter count)
    sized = [v for v in per_verifier if v in PARAMS]
    if sized:
        sized.sort(key=lambda v: PARAMS[v])
        x = [PARAMS[v] / 1e6 for v in sized]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.axhline(main[0]["AURC"] * 100, color="gray", ls="--", label="B1 reader-prob")
        for m in ("V", "R", "C", "RC"):
            ax.plot(x, [per_verifier[v][m]["AURC"] * 100 for v in sized], "o-", label=m)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v}\n{p:.0f}M" for v, p in zip(sized, x)])
        ax.minorticks_off()
        ax.set_ylabel("Test AURC x100 (lower is better)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "aurc_vs_size.png"), dpi=150)
except ImportError:
    print("matplotlib not installed: skipping plot")

print(open(os.path.join(out_dir, "main.md")).read())
print(open(os.path.join(out_dir, "ablation.md")).read())
print(open(os.path.join(out_dir, "significance.md")).read())
print(open(os.path.join(out_dir, "breakdown.md")).read())

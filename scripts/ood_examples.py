"""Step 11b (both): why NLI helps on AddSent. AddSent appends one distractor sentence at the end of each
paragraph; this script counts how many reader errors come from that last sentence, how the verifier scores
them, and lists the most confident reader errors that the verifier rejects.

  python scripts/ood_examples.py --run runs/addsent --out runs/squad11k/results/ood_examples.md
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import load_run  # noqa: E402
from qaverify.verifier import split_sentences  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/addsent")
p.add_argument("--key", default="nli-large-qa|template|full")
p.add_argument("--out", default="runs/squad11k/results/ood_examples.md")
p.add_argument("--n", type=int, default=6)
args = p.parse_args()

rows = list(load_run(args.run).values())
conf, ok, adv, pe = [], [], [], []
for r in rows:
    b = M.reader_margin(r)
    last = split_sentences(r["context"])[-1]
    conf.append(b["conf"])
    ok.append(b["correct"])
    adv.append(r["candidates"][0]["start"] >= r["context"].rfind(last))
    pe.append(r["nli"][args.key][0][0])
conf, ok, adv, pe = map(np.array, (conf, ok, adv, pe))
ok, adv = ok.astype(bool), adv.astype(bool)
wa = ~ok & adv

md = [f"# AddSent: lỗi do câu gây nhiễu ({os.path.basename(args.run)}, verifier `{args.key}`)\n",
      f"- {len(rows)} câu; reader top-1 sai {(~ok).sum()} câu, trong đó {wa.sum()} "
      f"({100 * wa.sum() / max(1, (~ok).sum()):.0f}%) lấy đáp án từ câu cuối (câu gây nhiễu).",
      f"- P(entail) trung bình: đáp án đúng {pe[ok].mean():.2f}, đáp án sai lấy từ câu nhiễu {pe[wa].mean():.2f}.",
      f"- Verifier bác bỏ (P < 0.5) {100 * (pe[wa] < 0.5).mean():.0f}% đáp án sai từ câu nhiễu, "
      f"chấp nhận {100 * (pe[ok] >= 0.5).mean():.0f}% đáp án đúng.",
      f"- {100 * (conf[wa] >= np.median(conf[ok])).mean():.0f}% đáp án sai từ câu nhiễu có margin reader "
      "≥ trung vị margin của đáp án đúng (reader tự tin như khi đúng).\n",
      "## Lỗi reader tự tin nhất mà verifier bác bỏ\n"]
for i in [i for i in np.argsort(-conf) if wa[i] and pe[i] < 0.05][:args.n]:
    r = rows[i]
    md += [f"**Q:** {r['question']}  ",
           f"**Gold:** {' / '.join(r['answers'][:2])} · **Reader:** {r['candidates'][0]['text']} "
           f"(margin {conf[i]:.1f}) · **P(entail)** {pe[i]:.3f}  ",
           f"**Câu gây nhiễu:** {split_sentences(r['context'])[-1]}  ",
           f"**Giả thuyết:** {r['hyp']['template'][0]}\n"]
with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("\n".join(md))

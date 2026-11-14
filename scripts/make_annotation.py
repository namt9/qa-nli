"""Step 6a (both): sample unanswerable test questions for manual annotation by type
(categories of Rajpurkar et al. 2018).

  python scripts/make_annotation.py --run runs/squad2k                   # random:  sheet_A/B.csv
  python scripts/make_annotation.py --run runs/squad11k --mode check     # check:   check_A/B.csv
  python scripts/make_annotation.py --run runs/squad11k --mode errors    # errors:  errors_A/B.csv

  random  --n random unanswerable questions, one sheet per member, --overlap of them in both sheets
  check   --n new questions given to BOTH members, to be labelled independently (Cohen's kappa)
  errors  --n unanswerable questions that combo C (--verifier) still answers at its calib threshold,
          split between the two members without overlap

Questions already in any sheet of the run are never sampled again. Sheets show only the question,
the passage and the plausible answer (no model output) so labels are not biased by the models.
Fill the 'label' column with one code from guide.md, then run scripts/analyze_annotation.py --set <prefix>.
"""
import argparse
import csv
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qaverify import methods as M  # noqa: E402
from qaverify.data import load_run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/squad2k")
p.add_argument("--data_dir", default="data")
p.add_argument("--mode", choices=["random", "check", "errors"], default="random")
p.add_argument("--n", type=int, default=None, help="questions (default: random 100, check 20, errors 60)")
p.add_argument("--overlap", type=int, default=20, help="random mode: questions labelled by both members")
p.add_argument("--verifier", default="nli-large", help="errors mode: setting read from results/settings.json")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
out_dir = os.path.join(args.run, "annotation")
prefix = {"random": "sheet", "check": "check", "errors": "errors"}[args.mode]
n = args.n or {"random": 100, "check": 20, "errors": 60}[args.mode]

CATEGORIES = [
    ("NEG", "Phủ định", "Câu hỏi thêm hoặc bỏ phủ định (not, never, no...) so với câu trong đoạn văn.",
     "Đoạn văn: 'X was built in 1900.' → 'When was X not built?'"),
    ("ANT", "Trái nghĩa", "Thay một từ bằng từ trái nghĩa (first/last, largest/smallest, increase/decrease...).",
     "Đoạn văn: 'the largest city' → 'What is the smallest city...?'"),
    ("ENT", "Hoán đổi thực thể", "Thay một thực thể, số hoặc ngày bằng một thực thể khác (thường cũng có trong đoạn).",
     "Đoạn văn: 'Tesla moved to Paris in 1882.' → 'When did Edison move to Paris?'"),
    ("MUT", "Loại trừ lẫn nhau", "Hỏi về một phương án loại trừ với điều đoạn văn nói, nhưng không phải phủ định hay từ trái nghĩa trực tiếp (ngày thường/cuối tuần, đất liền/đảo...).",
     "Đoạn văn: 'The museum is open on weekdays.' → 'On which weekend day is the museum open?'"),
    ("IMP", "Điều kiện bất khả", "Thêm một điều kiện mà đoạn văn không thỏa (thời gian, địa điểm, đối tượng không khớp).",
     "Đoạn văn: 'In 1990, the river flooded.' → 'What flooded in 1850?'"),
    ("OTH", "Khác (trung tính)", "Câu hỏi hợp lý nhưng đoạn văn không chứa thông tin trả lời, không thuộc các loại trên.",
     "Đoạn văn chỉ nói về dân số → 'What is the city's main export?'"),
    ("ANS", "Thực ra trả lời được", "Người gán nhãn thấy đoạn văn CÓ trả lời câu hỏi (lỗi nhãn của SQuAD 2.0). Ghi đáp án vào cột note.",
     ""),
]

with open(os.path.join(args.data_dir, "squad_v2_dev.json"), encoding="utf-8") as f:
    plausible = {qa["id"]: list(dict.fromkeys(a["text"] for a in qa.get("plausible_answers", [])))
                 for art in json.load(f)["data"] for para in art["paragraphs"] for qa in para["qas"]}

rows = load_run(args.run)
used = set()
for path in glob.glob(os.path.join(out_dir, "*_[AB].csv")):
    with open(path, encoding="utf-8-sig") as f:
        used |= {r["id"] for r in csv.DictReader(f)}
pool = [r for r in rows.values() if r["split"] == "test" and not r["answers"] and r["id"] not in used]
if args.mode == "errors":
    with open(os.path.join(args.run, "results", "settings.json")) as f:
        key = json.load(f)[args.verifier]["key"]
    calib = [r for r in rows.values() if r["split"] == "calib"]
    combo = M.Combo(key).fit(calib)
    tau = M.calib_threshold(combo, calib)
    pool = [r for r in pool if combo(r)["conf"] >= tau]
    print(f"{len(pool)} unlabelled unanswerable test questions answered by C ({key}) at tau={tau:.3f}")
pool.sort(key=lambda r: r["id"])
picked = random.Random(args.seed).sample(pool, min(n, len(pool)))

if args.mode == "random":
    half = (len(picked) + args.overlap) // 2
    sheets = {"A": picked[:half], "B": picked[len(picked) - half:]}  # the middle `overlap` rows are in both
elif args.mode == "check":
    sheets = {"A": picked, "B": picked}
else:
    sheets = {"A": picked[:len(picked) // 2], "B": picked[len(picked) // 2:]}

os.makedirs(out_dir, exist_ok=True)
for who in sheets:
    if os.path.exists(os.path.join(out_dir, f"{prefix}_{who}.csv")):
        sys.exit(f"{prefix}_{who}.csv already exists in {out_dir} (may hold labels) - delete it to regenerate")
for who, part in sheets.items():
    path = os.path.join(out_dir, f"{prefix}_{who}.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel shows Vietnamese/Unicode
        w = csv.writer(f)
        w.writerow(["no", "id", "question", "plausible_answer", "context", "label", "note"])
        for i, r in enumerate(part, 1):
            w.writerow([i, r["id"], r["question"], " | ".join(plausible.get(r["id"], [])), r["context"], "", ""])
    print("saved", path, len(part), "rows")

guide = ["# Hướng dẫn gán nhãn câu không có đáp án\n",
         "| Bộ | File | Cách làm |", "|---|---|---|",
         "| random | `sheet_A.csv`, `sheet_B.csv` | câu không đáp án ngẫu nhiên; mỗi người một file |",
         "| check | `check_A.csv`, `check_B.csv` | **cùng** các câu cho cả hai người; gán **riêng, không trao đổi** "
         "cho tới khi cả hai xong (để tính Cohen's kappa) |",
         "| errors | `errors_A.csv`, `errors_B.csv` | câu không đáp án mà hệ thống C vẫn trả lời (lỗi của hệ thống); "
         "mỗi người một nửa |\n",
         "**Cách làm:** đọc câu hỏi, tìm câu trong đoạn văn chứa *plausible answer* (Ctrl+F), so sánh câu hỏi với câu đó "
         "rồi ghi **một** mã vào cột `label`. Không xem kết quả của mô hình khi gán nhãn.\n",
         "**Thứ tự quyết định:** xét lần lượt `ANS` → `NEG` → `ANT` → `ENT` → `MUT` → `IMP` → `OTH`, dừng ở mã đầu tiên khớp. "
         "Phân vân thì ghi mã còn lại vào `note`.\n",
         "| Mã | Loại | Định nghĩa | Ví dụ minh họa |", "|---|---|---|---|"]
guide += [f"| `{c}` | {name} | {d} | {ex} |" for c, name, d, ex in CATEGORIES]
guide += ["\nCác loại theo Rajpurkar, Jia, Liang (2018), *Know What You Don't Know*, bảng phân loại câu không đáp án.",
          "Với câu lệch nhãn, hai người thống nhất và ghi vào `resolved.csv` (cột `id,label`), rồi chạy "
          "`python scripts/analyze_annotation.py --run " + args.run + " --set <sheet|check|errors>`.\n"]
with open(os.path.join(out_dir, "guide.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(guide))
print("saved", os.path.join(out_dir, "guide.md"))

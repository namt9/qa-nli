"""Load SQuAD 2.0 (and out-of-domain QA sets), sample a fixed subset, read/write jsonl, calib/test split."""
import gzip
import hashlib
import json
import os
import random
import re
import urllib.request

SQUAD_URLS = {
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
}


def load_squad_v2(split="dev", cache_dir="data"):
    """Return a flat list of {id, title, question, context, answers, is_impossible}."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"squad_v2_{split}.json")
    if not os.path.exists(path):
        urllib.request.urlretrieve(SQUAD_URLS[split], path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["data"]

    examples = []
    for article in raw:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                # Deduplicate gold answers, keep order
                answers = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                examples.append({
                    "id": qa["id"],
                    "title": article["title"],
                    "question": qa["question"].strip(),
                    "context": para["context"],
                    "answers": answers,
                    "is_impossible": qa.get("is_impossible", not answers),
                })
    return examples


# Out-of-domain sets (all questions answerable). AddSent: SQuAD 1.1 dev with an adversarial distractor sentence
# added to the passage (Jia & Liang, 2017); MRQA: the shared-task dev sets (Fisch et al., 2019).
ADDSENT_URL = "https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/"
MRQA_URL = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/{name}.jsonl.gz"


def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_addsent(cache_dir="data", adversarial=True):
    """AddSent questions in the flat format; adversarial=True keeps only the perturbed copies (ids with '-'),
    False only the 1000 original SQuAD 1.1 questions they were made from."""
    path = _download(ADDSENT_URL, os.path.join(cache_dir, "squad_addsent.json"))
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["data"]
    return [{"id": qa["id"], "title": art["title"], "question": qa["question"].strip(), "context": para["context"],
             "answers": list(dict.fromkeys(a["text"] for a in qa["answers"])), "is_impossible": False}
            for art in raw for para in art["paragraphs"] for qa in para["qas"] if ("-" in qa["id"]) == adversarial]


def load_mrqa(name, cache_dir="data"):
    """An MRQA dev set (e.g. NaturalQuestionsShort, NewsQA, HotpotQA) in the flat format. HTML-like tags such
    as <P> or <Td> are removed from the passage (answers never contain them)."""
    path = _download(MRQA_URL.format(name=name), os.path.join(cache_dir, f"mrqa_{name}.jsonl.gz"))
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        next(f)  # header line
        for line in f:
            para = json.loads(line)
            ctx = re.sub(r"\s+", " ", re.sub(r"</?[A-Za-z][^>]*>", " ", para["context"])).strip()
            for qa in para["qas"]:
                out.append({"id": qa["qid"], "title": name, "question": qa["question"].strip(), "context": ctx,
                            "answers": list(dict.fromkeys(qa["answers"])), "is_impossible": False})
    return out


def load_dataset(name, cache_dir="data"):
    """squad_v2_dev | addsent | addsent_orig | mrqa:<Name>"""
    if name == "squad_v2_dev":
        return load_squad_v2("dev", cache_dir)
    if name in ("addsent", "addsent_orig"):
        return load_addsent(cache_dir, adversarial=name == "addsent")
    if name.startswith("mrqa:"):
        return load_mrqa(name.split(":", 1)[1], cache_dir)
    raise ValueError(f"unknown dataset {name}")


def sample(examples, n, seed=42):
    if n is None or n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(examples, n)


def split_of(example_id, calib_frac=0.25):
    """Deterministic calib/test assignment by hashing the id (same split for everyone)."""
    h = int(hashlib.md5(example_id.encode()).hexdigest(), 16)
    return "calib" if (h % 10000) / 10000 < calib_frac else "test"


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(run):
    """preds.jsonl of a run with every scores/<verifier>.jsonl merged into r['nli'], keyed by id."""
    import glob
    rows = {r["id"]: r for r in read_jsonl(os.path.join(run, "preds.jsonl"))}
    for path in sorted(glob.glob(os.path.join(run, "scores", "*.jsonl"))):
        for s in read_jsonl(path):
            rows[s["id"]].setdefault("nli", {}).update(s["nli"])
    return rows

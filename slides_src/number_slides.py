"""Write a page number "n / N" into every slide (except the cover) following deck.json order.
Re-run after reordering/adding slides:  python slides_src/number_slides.py
"""
import json, re
from pathlib import Path

SRC = Path(__file__).resolve().parent / "project"
order = json.loads((SRC / "deck.json").read_text())["order"]
N = len(order)
TAG = re.compile(r'\n?<p data-pgnum[^>]*>.*?</p>')
for n, sid in enumerate(order, 1):
    f = SRC / "slides" / f"{sid}.html"
    s = TAG.sub("", f.read_text())
    if n > 1:
        dark = "background:#16202c" in s.split(">", 1)[0]
        color = "#8a94a3" if dark else "#9aa0a8"
        p = (f'<p data-pgnum style="position:absolute;right:56px;bottom:36px;font-size:22px;'
             f'font-weight:500;color:{color}">{n} / {N}</p>')
        s = s.replace("<aside>", p + "\n<aside>", 1) if "<aside>" in s else s.replace("</section>", p + "\n</section>", 1)
    f.write_text(s)
print("numbered", N - 1, "slides")

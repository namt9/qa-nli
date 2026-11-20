"""Build thuat_ngu.html (glossary page to keep open while presenting) from slides_src/glossary.json.
Usage: python slides_src/build_glossary.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "thuat_ngu.html"
groups = json.loads((HERE / "glossary.json").read_text())

cards = []
for title, rows in groups:
    trs = "".join(f"<tr><th>{t}</th><td>{m}</td></tr>" for t, m in rows)
    cards.append(f'<section class="card"><h2>{title}</h2><table>{trs}</table></section>')

page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thuật ngữ và ký hiệu</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700&display=swap">
<style>
:root{{--bg:#f6f5f1;--card:#fff;--ink:#1c2128;--mute:#5a6170;--line:#e2e0d8;--acc:#1f63b5;--hl:#fff3b0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1319;--card:#16202c;--ink:#f3f1ea;--mute:#a9b0bb;--line:#2a3646;--acc:#7fb3f0;--hl:#5a4a00}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.45 'Be Vietnam Pro',Arial,sans-serif;padding:24px 16px 48px}}
header{{max-width:1200px;margin:0 auto 20px;display:flex;flex-wrap:wrap;gap:12px 24px;align-items:end;justify-content:space-between}}
h1{{margin:0;font-size:28px}} header p{{margin:4px 0 0;color:var(--mute);font-size:15px}}
input{{font:inherit;padding:10px 14px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink);width:min(320px,100%)}}
main{{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-top:6px solid var(--acc);border-radius:14px;padding:16px 22px}}
h2{{margin:0 0 8px;font-size:19px;color:var(--acc)}}
table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;vertical-align:top;padding:8px 0;border-top:1px solid var(--line)}}
th{{font-weight:600;padding-right:16px;width:38%}}
tr:first-child th,tr:first-child td{{border-top:0}}
tr.hide{{display:none}} tr.hit{{background:var(--hl)}}
</style></head><body>
<header><div><h1>Thuật ngữ và ký hiệu</h1><p>Kiểm chứng câu trả lời QA bằng NLI · tra cứu khi báo cáo</p></div>
<input id="q" type="search" placeholder="Tìm: B2, AURC, premise…" autofocus></header>
<main>{''.join(cards)}</main>
<script>
const q=document.getElementById('q'),rows=[...document.querySelectorAll('tr')];
q.addEventListener('input',()=>{{const k=q.value.trim().toLowerCase();
 rows.forEach(r=>{{const m=!k||r.textContent.toLowerCase().includes(k);r.classList.toggle('hide',!m);r.classList.toggle('hit',!!k&&m);}});}});
</script></body></html>
"""
OUT.write_text(page)
print("->", OUT)

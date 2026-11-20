"""Build a standalone HTML deck (and optionally PDF) from slides_src/project.

Usage (from the repo root):
  python slides_src/build_html_deck.py            # -> slides_seminar.html
  python slides_src/build_html_deck.py --pdf      # also slides_seminar.pdf (needs google-chrome)
Images referenced as /_blob/<id> are inlined from the PNGs listed in slides_src/blobs.json.
"""
import base64, html, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = HERE / "project"
OUT = REPO / "slides_seminar.html"
BLOBS = {k: REPO / v for k, v in json.loads((HERE / "blobs.json").read_text()).items()}

deck = json.loads((SRC / "deck.json").read_text())
cache = {}


def blob(m):
    bid = m.group(1)
    if bid not in cache:
        cache[bid] = "data:image/png;base64," + base64.b64encode(BLOBS[bid].read_bytes()).decode()
    return f'src="{cache[bid]}"'


sections = []
for sid in deck["order"]:
    s = (SRC / "slides" / f"{sid}.html").read_text()
    s = re.sub(r'src="/_blob/([0-9a-f]{32})"', blob, s)
    assert "/_blob/" not in s, sid
    sections.append(s.strip())

title = html.escape(deck["title"])
font = deck["faces"]["be-vietnam-pro"]["href"]

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;background:#0e1319;overflow:hidden;font-family:'Be Vietnam Pro',Arial,sans-serif}
#stage{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform-origin:0 0}
#stage>section{position:absolute!important;inset:0;width:1920px;height:1080px;overflow:hidden}
#stage>section:not(.on){display:none!important}
h1{font-size:96px;font-weight:600} h2{font-size:64px;font-weight:600} h3{font-size:44px;font-weight:600}
p{font-size:32px}
ul,ol{padding-left:1.15em} li{padding-left:.15em}
table{border-collapse:collapse} th{font-weight:600}
img{display:block}
aside{display:none!important}
x-shape{display:block;flex-shrink:0}
x-shape[kind=arrow-right]{clip-path:polygon(0 30%,60% 30%,60% 0,100% 50%,60% 100%,60% 70%,0 70%)}
x-shape[kind=arrow-left]{clip-path:polygon(100% 30%,40% 30%,40% 0,0 50%,40% 100%,40% 70%,100% 70%)}
x-shape[kind=arrow-down]{clip-path:polygon(30% 0,70% 0,70% 60%,100% 60%,50% 100%,0 60%,30% 60%)}
x-shape[kind=arrow-up]{clip-path:polygon(30% 100%,70% 100%,70% 40%,100% 40%,50% 0,0 40%,30% 40%)}
x-shape[kind=ellipse]{border-radius:50%}
#bar{position:fixed;right:12px;bottom:10px;display:flex;gap:8px;align-items:center;font:14px Arial,sans-serif;color:#c9ccd2;opacity:.25;transition:opacity .2s;z-index:5}
#bar:hover{opacity:1}
#bar button{background:#22303f;color:#f3f1ea;border:0;border-radius:6px;padding:6px 10px;font:inherit;cursor:pointer}
#notes{position:fixed;left:0;right:0;bottom:0;max-height:38vh;overflow:auto;background:rgba(14,19,25,.94);color:#f3f1ea;
  font:18px/1.5 Arial,sans-serif;padding:16px 24px 48px;display:none;z-index:4;border-top:1px solid #33404f}
#notes.show{display:block}
#notes b{color:#7fb3f0}
#help{position:fixed;top:10px;left:50%;transform:translateX(-50%);background:rgba(14,19,25,.9);color:#f3f1ea;font:15px Arial,sans-serif;
  padding:8px 14px;border-radius:8px;z-index:6;transition:opacity 1s}
@media print{
  @page{size:1920px 1080px;margin:0}
  html,body{height:auto;overflow:visible;background:#fff}
  #stage{position:static;transform:none!important;width:auto;height:auto}
  #stage>section,#stage>section:not(.on){position:relative!important;display:flex!important;break-after:page;page-break-after:always}
  #bar,#notes,#help{display:none!important}
}
"""

JS = """
const S=[...document.querySelectorAll('#stage>section')], stage=document.getElementById('stage'),
 notes=document.getElementById('notes'), num=document.getElementById('num');
let i=Math.max(0,Math.min(S.length-1,(parseInt(location.hash.slice(1))||1)-1));
function fit(){const k=Math.min(innerWidth/1920,innerHeight/1080);stage.style.transform=`scale(${k}) translate(-50%,-50%)`;}
function show(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,j)=>s.classList.toggle('on',j===i));
 num.textContent=(i+1)+' / '+S.length;history.replaceState(null,'','#'+(i+1));
 const a=S[i].querySelector('aside');notes.innerHTML='<b>Slide '+(i+1)+'</b> — '+(a?a.innerHTML:'(không có ghi chú)');}
function fs(){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();}
addEventListener('resize',fit);
addEventListener('keydown',e=>{const k=e.key;
 if(['ArrowRight','ArrowDown','PageDown',' ','Enter'].includes(k)){show(i+1);e.preventDefault();}
 else if(['ArrowLeft','ArrowUp','PageUp','Backspace'].includes(k)){show(i-1);e.preventDefault();}
 else if(k==='Home')show(0); else if(k==='End')show(S.length-1);
 else if(k==='n'||k==='N')notes.classList.toggle('show'); else if(k==='f'||k==='F')fs();});
stage.addEventListener('click',e=>show(e.clientX>innerWidth/3?i+1:i-1));
document.getElementById('prev').onclick=()=>show(i-1);document.getElementById('next').onclick=()=>show(i+1);
document.getElementById('nbtn').onclick=()=>notes.classList.toggle('show');document.getElementById('fbtn').onclick=fs;
setTimeout(()=>document.getElementById('help').style.opacity=0,4000);
// shrink text when a slide overflows (like Slides' auto-fit, min 60%)
function shrink(s){
 s.classList.add('on');
 const top=s.getBoundingClientRect().top, sc=s.getBoundingClientRect().height/1080;
 const abs=[...s.children].filter(c=>getComputedStyle(c).position==='absolute'&&c.tagName!=='ASIDE'&&!c.hasAttribute('data-pgnum'));
 const flow=[...s.querySelectorAll('*')].filter(e=>e.tagName!=='ASIDE'&&!abs.some(a=>a===e||a.contains(e)));
 let limit=1080-72; abs.forEach(a=>{const t=(a.getBoundingClientRect().top-top)/sc; if(t>540) limit=Math.min(limit,t-16);});
 const bottom=()=>Math.max(...flow.map(e=>(e.getBoundingClientRect().bottom-top)/sc));
 const txt=flow.filter(e=>/font-size/.test(e.getAttribute('style')||'')&&!/^H[12]$/.test(e.tagName)&&e!==s.firstElementChild);
 txt.forEach(e=>e.dataset.fs=parseFloat(e.style.fontSize));
 let k=1; while(bottom()>limit&&k>0.6){k-=0.03;txt.forEach(e=>e.style.fontSize=(e.dataset.fs*k)+'px');}
 if(k<1)s.dataset.fit=k.toFixed(2);
 s.classList.remove('on');
}
const go=()=>{const t=stage.style.transform;stage.style.transform='none';S.forEach(shrink);stage.style.transform=t;fit();show(i);};
fit();show(i);
(document.fonts&&document.fonts.ready?document.fonts.ready:Promise.resolve()).then(go);
"""

page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="{font}">
<style>{CSS}</style></head>
<body>
<div id="stage">
{chr(10).join(sections)}
</div>
<div id="notes"></div>
<div id="help">← → hoặc Space: chuyển slide · F: toàn màn hình · N: ghi chú người nói · Ctrl+P: xuất PDF</div>
<div id="bar"><button id="prev">‹</button><span id="num"></span><button id="next">›</button><button id="nbtn">Ghi chú (N)</button><button id="fbtn">Toàn màn hình (F)</button></div>
<script>{JS}</script>
</body></html>
"""
OUT.write_text(page)
print(len(sections), "slides,", len(cache), "images,", OUT.stat().st_size // 1024, "KB ->", OUT)

if "--pdf" in sys.argv:
    pdf = OUT.with_suffix(".pdf")
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--virtual-time-budget=8000",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf}", OUT.as_uri()],
                   check=True, stderr=subprocess.DEVNULL)
    print("->", pdf)

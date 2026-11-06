r"""Build a local listening page to hand-check aligned curse words (calibrates the automatic checks).

For a sample of curse words from aligned.jsonl it writes two WAVs (the word with context, and only
the aligned word span) and review.html next to them. Open the page in a browser, mark each word,
then press "Download answers". Everything stays on this PC.

    .venv-lab\Scripts\python lab\detector\make_review.py --n 60 --per-video 4
"""
import argparse
import html
import json
import random
import sys
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from align_clips import SR  # noqa: E402
from fetch_yodas import DEST  # noqa: E402

CHOICES = [("ok", "Curse word, timing right"), ("late", "Curse word, timing off"),
           ("wrong", "Not said / different word"), ("unsure", "Can't tell")]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Curse word review</title>
<style>
body{font:15px system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}
.item{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:12px 0}
.item.done{border-color:#7a7}.meta{color:#777;font-size:13px}mark{background:#fd6;padding:0 2px}
audio{height:32px;vertical-align:middle}label{margin-right:14px;white-space:nowrap}
#bar{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd}
</style></head><body>
<div id="bar"><b>Curse word review</b>: <span id="count"></span>
<button onclick="dl()">Download answers</button>
<div class="meta">Play "word only": does that short span contain the whole curse word? "Timing off" = the word
is there but the span cuts part of it or is mostly something else. Answers are kept in this browser until you download them.</div></div>
ITEMS
<script>
const KEY='curse-review-v1';
let ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function upd(){document.querySelectorAll('.item').forEach(d=>{const v=ans[d.id];d.classList.toggle('done',!!v);
 d.querySelectorAll('input').forEach(i=>i.checked=(i.value===v))});
 document.getElementById('count').textContent=Object.keys(ans).length+' / '+document.querySelectorAll('.item').length+' marked';}
document.addEventListener('change',e=>{if(e.target.type==='radio'){ans[e.target.name]=e.target.value;
 try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(_){} upd();}});
function dl(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(ans,null,1)],{type:'application/json'}));
 a.download='review_answers.json';a.click();}
upd();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--per-video", type=int, default=4, help="at most this many words from one video")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (DEST / "aligned.jsonl").open(encoding="utf-8")]
    items = [(r, c) for r in rows for c in r["curses"]]
    random.Random(a.seed).shuffle(items)
    # Half Parakeet-heard, half not, so both checks get calibrated.
    picked, per = [], {}
    for want in (True, False):
        k = 0
        for r, c in items:
            if c["heard"] == want and per.get(r["src"], 0) < a.per_video and k < a.n // 2:
                picked.append((r, c))
                per[r["src"]] = per.get(r["src"], 0) + 1
                k += 1
    random.Random(a.seed + 1).shuffle(picked)
    write_page(picked, DEST / "review")
    print(f"{len(picked)} words from {len(per)} videos -> {DEST / 'review' / 'review.html'}")


def write_page(picked, out, key="curse-review-v1"):
    """picked: [(clip row, curse dict with "i" = index into row["words"])] -> WAVs, review.html,
    items.json in out. key: the browser storage key, so separate reviews don't mix answers."""
    out.mkdir(parents=True, exist_ok=True)
    blocks, meta = [], []
    for n, (r, c) in enumerate(picked):
        audio, _ = sf.read(DEST / r["clip"], dtype="float32")
        w = r["words"][c["i"]]
        item = f"{r['id']}#{c['i']}"
        sf.write(out / f"{n:03d}_ctx.wav", audio[int(max(0, w["start"] - 2.0) * SR):int((w["end"] + 1.0) * SR)], SR)
        sf.write(out / f"{n:03d}_word.wav", audio[int(w["start"] * SR):int(w["end"] * SR)], SR)
        line = " ".join(f"<mark>{html.escape(x['w'])}</mark>" if j == c["i"] else html.escape(x["w"])
                        for j, x in enumerate(r["words"]))
        radios = "".join(f'<label><input type="radio" name="{html.escape(item)}" value="{v}"> {t}</label>'
                         for v, t in CHOICES)
        blocks.append(f'<div class="item" id="{html.escape(item)}"><div class="meta">#{n} &middot; video {r["src"]}'
                      f' &middot; {w["end"] - w["start"]:.2f} s</div><p>{line}</p>'
                      f'context <audio controls preload="none" src="{n:03d}_ctx.wav"></audio> &nbsp; '
                      f'word only <audio controls preload="none" src="{n:03d}_word.wav"></audio>'
                      f'<p>{radios}</p></div>')
        meta.append({"n": n, "key": item, "id": r["id"], "src": r["src"], "w": c["w"], "group": c["group"],
                     "score": c["score"], "heard": c.get("heard"), "line_score": r["line_score"]})
    page = PAGE.replace("ITEMS", "\n".join(blocks)).replace("curse-review-v1", key)
    (out / "review.html").write_text(page, encoding="utf-8")
    (out / "items.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

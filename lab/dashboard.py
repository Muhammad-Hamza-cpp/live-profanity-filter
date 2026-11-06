r"""Live progress page for the real-speech work on this laptop and on the Lambda instance.

    .venv-lab\Scripts\python lab\dashboard.py          # opens http://localhost:8765

Every 15 s it reads the laptop's progress files (data/realspeech/yodas, *.log), nvidia-smi and the
running Python jobs, and asks Lambda for the same over SSH (host alias "lambda" in ~/.ssh/config).
The page refreshes itself. Ctrl+C stops it. Read-only: it never starts or stops anything.
"""
import http.server
import json
import re
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YODAS = ROOT / "data" / "realspeech" / "yodas"
PORT = 8765
LAMBDA_DIR = "/lambda/nfs/Audio/lpf"
TEST_ARCHIVES, TRAIN_ARCHIVES = 62, 262

status = {"laptop": {}, "lambda": {}, "updated": None}


def ids(path):
    """Unique "id"s in a jsonl file (a crashed or doubled run can repeat rows)."""
    if not path.exists():
        return set()
    out = set()
    for line in path.open(encoding="utf-8", errors="replace"):
        m = re.search(r'"id": "([^"]+)"', line)
        if m:
            out.add(m.group(1))
    return out


def tail(path, n=1):
    if not path.exists():
        return ""
    lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if l.strip() and not re.search(r"Loading weights|[Ww]arn|redirects", l)]
    return "\n".join(lines[-n:])


def run(cmd, timeout=20, stdin=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=stdin,
                           encoding="utf-8", errors="replace")
        return r.stdout
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"


def gpu_local():
    out = run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
               "--format=csv,noheader,nounits"])
    try:
        name, util, used, total = [x.strip() for x in out.strip().split(",")]
        return {"name": name, "util": int(util), "mem": int(used), "mem_total": int(total)}
    except ValueError:
        return {"name": "?", "error": out.strip()[:200]}


def jobs_local():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | "
          "ForEach-Object { $_.CommandLine }")
    out = run(["powershell", "-NoProfile", "-Command", ps])
    seen = []
    for line in out.splitlines():
        if "uv\\python" in line:  # the venv launcher starts a second process with the same arguments
            continue
        args = re.sub(r'^.*?pythonw?(\.exe)?"?\s+', "", line.strip()).replace("\\", "/")
        if re.match(r"[\w./-]+\.pyw?\b", args) and "dashboard.py" not in args and args not in seen:
            seen.append(args)
    return seen


def laptop():
    clips, aligned, whisper = ids(YODAS / "clips.jsonl"), ids(YODAS / "aligned.jsonl"), ids(YODAS / "whisper.jsonl")
    done = (YODAS / "shards_done.txt").read_text().split() if (YODAS / "shards_done.txt").exists() else []
    test = YODAS / "testset.jsonl"
    n_curse = sum(len(json.loads(l)["curses"]) for l in test.open(encoding="utf-8")) if test.exists() else 0
    stages = [
        {"name": "Fetch test archives", "done": len(done), "total": TEST_ARCHIVES, "unit": "archives"},
        {"name": "Align words (MMS)", "done": len(aligned & clips), "total": len(clips), "unit": "clips"},
        {"name": "Check words (Whisper large-v3)", "done": len(whisper & aligned & clips),
         "total": len(aligned & clips), "unit": "clips", "log": tail(ROOT / "whisper2.log")},
        {"name": "Test set built", "done": n_curse, "total": None, "unit": "curse words",
         "note": "built" if test.exists() and n_curse > 40 else "waiting for the Whisper check"},
    ]
    evals = []
    for p in sorted(ROOT.glob("eval_real*.log")) + sorted(ROOT.glob("baselines*.log")):
        evals.append({"name": p.stem, "log": tail(p, 6)})
    return {"stages": stages, "evals": evals, "gpu": gpu_local(), "jobs": jobs_local()}


REMOTE = r'''
import json, os, re, glob, subprocess
D = "LAMBDA_DIR"; Y = D + "/data/realspeech/yodas"
def ids(p):
    s = set()
    if os.path.exists(p):
        for l in open(p, encoding="utf-8", errors="replace"):
            m = re.search(r'"id": "([^"]+)"', l)
            if m: s.add(m.group(1))
    return s
def tail(p, n=1):
    if not os.path.exists(p): return ""
    ls = [l for l in open(p, encoding="utf-8", errors="replace").read().splitlines()
          if l.strip() and not re.search(r"Loading weights|[Ww]arn|\r", l)]
    return "\n".join(ls[-n:])
clips, al, wh = ids(Y + "/clips.jsonl"), ids(Y + "/aligned.jsonl"), ids(Y + "/whisper.jsonl")
done = open(Y + "/shards_done.txt").read().split() if os.path.exists(Y + "/shards_done.txt") else []
steps = {}
for jl in glob.glob(D + "/jobs_*.log"):
    for l in open(jl):
        m = re.search(r"start (\S+): .*--steps (\d+)", l)
        if m: steps[m.group(1)] = int(m.group(2))
runs = []
for d in sorted(glob.glob(D + "/lab/detector/runs/*/")):
    name = os.path.basename(d.rstrip("/")); out = d + "stdout.log"
    if not os.path.exists(out): continue
    txt = open(out, encoding="utf-8", errors="replace").read()
    st = re.findall(r"step (\d+) loss ([\d.]+) .*? (\d+)s", txt)
    val = re.findall(r"F1 ([\d.]+)", txt)
    failed = "Traceback" in txt
    runs.append({"name": name, "step": int(st[-1][0]) if st else 0, "loss": float(st[-1][1]) if st else None,
                 "secs": int(st[-1][2]) if st else 0, "steps": steps.get(name, 8000),
                 "val_f1": float(val[-1]) if val else None,
                 "state": "done" if os.path.exists(d + "done") else "failed" if failed else "running",
                 "error": tail(out, 2) if failed else ""})
g = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(",")
ps = subprocess.run(["pgrep", "-af", "python|\\.sh"], capture_output=True, text=True).stdout.splitlines()
jobs = sorted({re.sub(r"^\d+\s+(\S*/)?(python3?|bash)\s+", "", p) for p in ps   # a set: make_tts workers repeat
               if re.search(r"lab/detector|\w+\.sh", p) and "pgrep" not in p})
up = float(open("/proc/uptime").read().split()[0])
print(json.dumps({
  "fetch": {"done": len(done), "clips": len(clips), "running": any("fetch_yodas" in j for j in jobs)},
  "aligned": len(al & clips), "whisper": len(wh & al & clips),
  "label_logs": {k: tail(D + "/" + k + ".log") for k in ("label_train", "label_rest")},
  "label_phase": {k: (re.findall(r"clips to (align|check)", open(D + "/" + k + ".log", errors="replace").read()) or [""])[-1]
                  if os.path.exists(D + "/" + k + ".log") else "" for k in ("label_train", "label_rest")},
  "tts": tail(D + "/gen_tts.log"), "runs": runs, "jobs": jobs[:12],
  "gpu": {"name": g[0].strip(), "util": int(g[1]), "mem": int(g[2]), "mem_total": int(g[3])} if len(g) == 4 else {},
  "load": open("/proc/loadavg").read().split()[0], "uptime_h": round(up / 3600, 2),
  "trainset": os.path.exists(Y + "/trainset.jsonl"), "testset": os.path.exists(Y + "/testset.jsonl")}))
'''.replace("LAMBDA_DIR", LAMBDA_DIR)


def lambda_():
    out = run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "lambda", "python3", "-"],
              timeout=40, stdin=REMOTE)
    try:
        s = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": out.strip()[-400:] or "no answer from Lambda"}
    f = s["fetch"]
    job = "label_rest" if s["label_logs"]["label_rest"] else "label_train"
    log = s["label_logs"][job]
    # Rows can sit in a write buffer for a while, so the job's own "n/m" line is fresher: m clips
    # were left to check when it started, n of them are done.
    m = re.search(r"(\d+)/(\d+)\s*$", log)
    if m and s["label_phase"][job] == "check" and s["aligned"] >= int(m.group(2)):
        s["whisper"] = max(s["whisper"], s["aligned"] - int(m.group(2)) + int(m.group(1)))
    s["stages"] = [
        {"name": "Fetch training archives", "done": f["done"], "total": TRAIN_ARCHIVES, "unit": "archives",
         "note": "includes the 62 test archives, marked done"},
        {"name": "Align words (MMS)", "done": s["aligned"], "total": f["clips"], "unit": "clips"},
        {"name": "Check words (Whisper large-v3)", "done": s["whisper"], "total": s["aligned"], "unit": "clips",
         "log": log},
        {"name": "Train/validation split", "done": 1 if s["trainset"] else 0, "total": 1, "unit": "",
         "note": "built" if s["trainset"] else "after labelling"},
    ]
    return s


def collect():
    while True:
        try:
            status["laptop"] = laptop()
        except Exception as e:
            status["laptop"] = {"error": f"{type(e).__name__}: {e}"}
        try:
            status["lambda"] = lambda_()
        except Exception as e:
            status["lambda"] = {"error": f"{type(e).__name__}: {e}"}
        status["updated"] = time.strftime("%H:%M:%S")
        time.sleep(15)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Real-speech progress</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1d2228;--muted:#6b7480;--line:#e3e6ea;--bar:#e9ecef;--ok:#2f9e6b;--run:#3b74d9;--bad:#d9483b;--code:#f1f3f5}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1b1e23;--text:#e6e9ed;--muted:#9aa3ad;--line:#2b3037;--bar:#2a2f36;--ok:#3fbf85;--run:#5b8ff0;--bad:#f06b5f;--code:#23272d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:16px 20px;display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
h1{font-size:18px;margin:0}.muted{color:var(--muted)}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;padding:0 20px 20px}
@media (max-width:480px){main{grid-template-columns:1fr;padding:0 16px 16px}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.card h2{font-size:15px;margin:0 0 4px}.sub{font-size:12px;color:var(--muted);margin-bottom:12px}
.stage{margin:10px 0}.row{display:flex;justify-content:space-between;gap:8px;font-size:13px}
.bar{height:8px;background:var(--bar);border-radius:4px;overflow:hidden;margin-top:4px}
.fill{height:100%;background:var(--run);transition:width .4s}.fill.ok{background:var(--ok)}
pre{background:var(--code);border-radius:6px;padding:6px 8px;margin:4px 0 0;font:12px/1.4 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11.5px;color:#fff}
.pill.running{background:var(--run)}.pill.done{background:var(--ok)}.pill.failed{background:var(--bad)}
.kv{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;margin-bottom:6px}.err{color:var(--bad)}
h3{font-size:13px;margin:16px 0 4px}
</style></head><body>
<header><h1>Real-speech test set &amp; training</h1><span class="muted" id="upd">loading…</span></header>
<main><section class="card" id="laptop"></section><section class="card" id="lambda"></section></main>
<script>
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function stage(s){const pct=s.total?Math.min(100,100*s.done/s.total):(s.done?100:0);
 const txt=s.total?`${s.done.toLocaleString()} / ${s.total.toLocaleString()} ${s.unit} (${pct.toFixed(0)}%)`:`${s.done.toLocaleString()} ${s.unit}`;
 return `<div class="stage"><div class="row"><span>${esc(s.name)}</span><span class="muted">${txt}</span></div>
 <div class="bar"><div class="fill ${pct>=100?"ok":""}" style="width:${pct}%"></div></div>
 ${s.note?`<div class="muted" style="font-size:12px">${esc(s.note)}</div>`:""}${s.log?`<pre>${esc(s.log)}</pre>`:""}</div>`}
function gpu(g){return g&&g.name?`<span>GPU ${esc(g.name)}: <b>${g.util}%</b>, ${(g.mem/1024).toFixed(1)} / ${(g.mem_total/1024).toFixed(1)} GB</span>`:""}
function jobs(j){return j&&j.length?`<h3>Running jobs</h3><pre>${j.map(esc).join("\n")}</pre>`:`<h3>Running jobs</h3><div class="muted">none</div>`}
function laptop(d){if(d.error)return `<h2>Laptop</h2><div class="err">${esc(d.error)}</div>`;
 return `<h2>Laptop (RTX 4050)</h2><div class="sub">Test set, v3 and baselines on real speech</div>
 <div class="kv">${gpu(d.gpu)}</div>${(d.stages||[]).map(stage).join("")}
 ${(d.evals||[]).map(e=>`<h3>${esc(e.name)}</h3><pre>${esc(e.log)}</pre>`).join("")}${jobs(d.jobs)}`}
function eta(r){if(r.state!=="running"||!r.step)return"";const s=r.secs/r.step*(r.steps-r.step);return Math.round(s/60)+" min"}
function lam(d){if(d.error)return `<h2>Lambda</h2><div class="err">${esc(d.error)}</div>`;
 const runs=d.runs||[];
 return `<h2>Lambda (A10)</h2><div class="sub">Training pool, labelling and training runs</div>
 <div class="kv">${gpu(d.gpu)}<span>load ${esc(d.load)}</span><span>instance up ${d.uptime_h} h (billed while running)</span></div>
 ${(d.stages||[]).map(stage).join("")}
 <h3>Training runs (${runs.filter(r=>r.state==="done").length} done / ${runs.length})</h3>
 ${runs.length?`<table><tr><th>run</th><th>state</th><th>progress</th><th>loss</th><th>val F1</th><th>ETA</th></tr>
 ${runs.map(r=>`<tr><td>${esc(r.name)}</td><td><span class="pill ${r.state}">${r.state}</span></td>
 <td><div class="bar" style="width:90px;display:inline-block;vertical-align:middle"><div class="fill ${r.state==="done"?"ok":""}" style="width:${100*r.step/r.steps}%"></div></div>
 <span class="muted"> ${r.step}/${r.steps}</span></td><td class="num">${r.loss??""}</td><td class="num">${r.val_f1??""}</td><td>${eta(r)}</td></tr>
 ${r.error?`<tr><td colspan="6"><pre class="err">${esc(r.error)}</pre></td></tr>`:""}`).join("")}</table>`:`<div class="muted">none yet</div>`}
 ${d.tts?`<h3>Synthetic data</h3><pre>${esc(d.tts)}</pre>`:""}${jobs(d.jobs)}`}
async function tick(){try{const s=await (await fetch("/status")).json();
 document.getElementById("laptop").innerHTML=laptop(s.laptop||{});document.getElementById("lambda").innerHTML=lam(s.lambda||{});
 document.getElementById("upd").textContent=s.updated?"updated "+s.updated+" · refreshes every 15 s":"collecting…";}
 catch(e){document.getElementById("upd").textContent="dashboard server not reachable"}}
tick();setInterval(tick,15000);
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(status).encode() if self.path == "/status" else PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if self.path == "/status" else "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    threading.Thread(target=collect, daemon=True).start()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"dashboard at {url} (Ctrl+C to stop)", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

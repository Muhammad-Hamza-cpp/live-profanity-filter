r"""Find curse words in the transcripts of public real-speech datasets, without downloading audio.

Feasibility scan for the real-speech test set (docs/HANDOFF.md, next steps 1): which lexicon words
occur, how often, and in how many distinct sources. A source is an archive.org item for People's
Speech and a video for YODAS2 (its video ids are anonymised, so channels can't be told apart).
Only transcript bytes are fetched:
- People's Speech: the parquet files are read column by column with HTTP range requests.
- YODAS2: the text sits in separate JSON files. Only en000-en007 (manual subtitles) are scanned.
  Auto-captions write profanity as "[ __ ]".

    .venv-lab\Scripts\python lab\detector\scan_transcripts.py peoples --limit 10    # quick test
    .venv-lab\Scripts\python lab\detector\scan_transcripts.py peoples yodas         # full scan, resumes
    .venv-lab\Scripts\python lab\detector\scan_transcripts.py --summary             # tables only

Writes to data/realspeech/:
- hits_<ds>.jsonl: one line per utterance with a curse word or a starred-out word
- done_<ds>.jsonl: one line per finished file with its utterance count and hours, so reruns resume
- summary.json
"""
import argparse
import collections
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_url

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from lexicon import GROUPS, all_forms  # noqa: E402

OUT = ROOT / "data" / "realspeech"
PEOPLES = "MLCommons/peoples_speech"
YODAS = "espnet/yodas2"
# The test and validation splits are the same in every People's Speech config; scan them once.
PEOPLES_DIRS = ("clean/train-", "clean_sa/train-", "dirty/train-", "dirty_sa/train-",
                "test/test-", "validation/validation-")
YODAS_DIRS = [f"data/en00{i}/text/" for i in range(8)]

FORM2GROUP = {f: g for f, g in all_forms() if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", f)}
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")
STARRED = re.compile(r"\b[a-z]+\*+[a-z*]*|\b[a-z]?\*{2,}[a-z]*|\[ __ \]")

lock = threading.Lock()
http = fsspec.filesystem("http")


def match(text):
    """([(form, group)], [starred tokens]) found in one transcript line."""
    t = text.lower().replace("’", "'")
    hits = [(w, FORM2GROUP[w]) for w in TOKEN.findall(t) if w in FORM2GROUP]
    return hits, STARRED.findall(t)


def cdn_url(repo, path):
    """Signed CDN url of a dataset file. Range requests against it don't count against the Hub's
    per-request limits, which matters for thousands of small reads without a token."""
    for attempt in range(8):
        r = requests.head(hf_hub_url(repo, path, repo_type="dataset"), allow_redirects=False,
                          timeout=60)
        if r.status_code in (301, 302, 307):
            loc = r.headers["location"]
            # Small files are served from the Hub's own cache (a relative redirect), not the CDN.
            return "https://huggingface.co" + loc if loc.startswith("/") else loc
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 2 ** attempt))
            continue
        r.raise_for_status()
    raise RuntimeError(f"gave up resolving {path}")


def retry(fn, *args):
    for attempt in range(5):
        try:
            return fn(*args)
        except Exception as e:  # network hiccups on a 25-min scan are normal
            if attempt == 4:
                raise
            print(f"  retry {args[-1]}: {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))


def scan_peoples(path):
    with http.open(cdn_url(PEOPLES, path), "rb", block_size=2 ** 16, cache_type="bytes") as f:
        tab = pq.ParquetFile(f, pre_buffer=True).read(columns=["id", "text", "duration_ms"])
    rows, secs = [], 0.0
    for uid, text, ms in zip(*(tab.column(c).to_pylist() for c in ("id", "text", "duration_ms"))):
        secs += ms / 1000
        hits, starred = match(text or "")
        if hits or starred:
            rows.append({"id": uid, "src": uid.split("/")[0], "start": 0.0, "dur": ms / 1000,
                         "text": text, "hits": hits, "starred": starred})
    return rows, tab.num_rows, secs


def scan_yodas(path):
    r = requests.get(cdn_url(YODAS, path), timeout=120)
    r.raise_for_status()
    rows, n, secs = [], 0, 0.0
    for video in r.json():
        for uid, text in video["text"].items():
            # <video id, may contain '-'>-<index>-<start>-<end>, times in 10 ms units
            start, end = (int(x) / 100 for x in uid.rsplit("-", 3)[2:])
            n += 1
            secs += end - start
            hits, starred = match(text or "")
            if hits or starred:
                rows.append({"id": uid, "src": video["audio_id"], "start": start,
                             "dur": round(end - start, 2), "text": text, "hits": hits,
                             "starred": starred})
    return rows, n, secs


DATASETS = {
    "peoples": (PEOPLES, PEOPLES_DIRS, ".parquet", scan_peoples),
    "yodas": (YODAS, YODAS_DIRS, ".json", scan_yodas),
}


def list_files(repo, prefixes, ext):
    if repo == YODAS:  # 100k+ files: list only the folders we want
        api = HfApi()
        return sorted(f.path for p in prefixes
                      for f in api.list_repo_tree(repo, path_in_repo=p.rstrip("/"), repo_type="dataset")
                      if f.path.endswith(ext))
    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                  if f.endswith(ext) and f.startswith(prefixes))


def run(ds, workers, limit):
    repo, prefixes, ext, scan = DATASETS[ds]
    hits_file, done_file = OUT / f"hits_{ds}.jsonl", OUT / f"done_{ds}.jsonl"
    done = set()
    if done_file.exists():
        done = {json.loads(l)["file"] for l in done_file.read_text(encoding="utf-8").splitlines()}
    files = [f for f in list_files(repo, prefixes, ext) if f not in done]
    if limit:
        files = files[:limit]
    print(f"{ds}: {len(files)} files to scan ({len(done)} already done)", flush=True)
    t0, n_done, n_hits = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(retry, scan, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                rows, n, secs = fut.result()
            except Exception as e:
                print(f"  FAILED {f}: {e}", flush=True)
                continue
            with lock:
                with hits_file.open("a", encoding="utf-8") as h:
                    for r in rows:
                        h.write(json.dumps({"file": f, **r}, ensure_ascii=False) + "\n")
                with done_file.open("a", encoding="utf-8") as d:
                    d.write(json.dumps({"file": f, "utts": n, "hours": round(secs / 3600, 4)}) + "\n")
            n_done += 1
            n_hits += sum(len(r["hits"]) for r in rows)
            if n_done % 25 == 0 or n_done == len(files):
                el = time.time() - t0
                eta = el / n_done * (len(files) - n_done)
                print(f"  {ds} {n_done}/{len(files)} files, {n_hits} curse tokens, "
                      f"{el / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True)


def summary():
    out = {}
    for ds in DATASETS:
        hits_file, done_file = OUT / f"hits_{ds}.jsonl", OUT / f"done_{ds}.jsonl"
        if not done_file.exists():
            continue
        done = [json.loads(l) for l in done_file.read_text(encoding="utf-8").splitlines()]
        hours = sum(d["hours"] for d in done)
        tokens, utts, srcs, forms = (collections.Counter(), collections.Counter(),
                                     collections.defaultdict(set), collections.Counter())
        starred_utts = 0
        rows = [json.loads(l) for l in hits_file.read_text(encoding="utf-8").splitlines()] \
            if hits_file.exists() else []
        for r in rows:
            starred_utts += bool(r["starred"])
            for form, g in r["hits"]:
                tokens[g] += 1
                forms[form] += 1
            for g in {g for _, g in r["hits"]}:
                utts[g] += 1
                srcs[g].add(r["src"])
        cat = collections.Counter()
        for g, n in tokens.items():
            cat[GROUPS[g][0]] += n
        out[ds] = {
            "files": len(done), "utterances": sum(d["utts"] for d in done), "hours": round(hours, 1),
            "curse_tokens": sum(tokens.values()),
            "curse_utterances": sum(1 for r in rows if r["hits"]),
            "curse_sources": len({r["src"] for r in rows if r["hits"]}),
            "starred_utterances": starred_utts,
            "per_category": dict(cat.most_common()),
            "per_group": {g: {"tokens": tokens[g], "utterances": utts[g], "sources": len(srcs[g])}
                          for g, _ in tokens.most_common()},
            "top_forms": dict(forms.most_common(40)),
        }
        o = out[ds]
        print(f"\n## {ds}: {o['files']} files, {o['utterances']:,} utterances, {o['hours']:,} h")
        print(f"curse tokens {o['curse_tokens']:,} in {o['curse_utterances']:,} utterances from "
              f"{o['curse_sources']:,} sources ({o['curse_tokens'] / max(hours, 1e-9):.2f} per hour); "
              f"utterances with starred-out words: {starred_utts:,}")
        print("by category: " + ", ".join(f"{c} {n:,}" for c, n in cat.most_common()))
        print("| group | tokens | utterances | sources |\n|---|---|---|---|")
        for g, s in o["per_group"].items():
            print(f"| {g} | {s['tokens']:,} | {s['utterances']:,} | {s['sources']:,} |")
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", choices=list(DATASETS))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="scan only this many new files (testing)")
    ap.add_argument("--summary", action="store_true", help="only print tables from saved hits")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.summary:
        for ds in a.datasets:
            run(ds, a.workers, a.limit)
    summary()


if __name__ == "__main__":
    main()

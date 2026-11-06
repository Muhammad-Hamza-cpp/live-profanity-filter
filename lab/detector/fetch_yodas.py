r"""Cut real-speech clips around transcript curse-word hits out of YODAS2 audio (stage 1 of 2).

YODAS2 packs about 80 soundtracks (24 kHz WAV) into each ~1.4 GB .tar.gz. A gzip stream can't be
read from the middle, so each archive is streamed start to end, the soundtracks with hits are kept
in memory, and everything else is dropped as it passes: disk use is only the clips. Uses network
and CPU, no GPU. Stage 2 (align_clips.py) aligns the words and checks them with Parakeet.

One clip per transcript line with a curse word: 4 s before the line (the detector's window), the
line itself, and 1.5 s after, as 16 kHz FLAC. Hits come from scan_transcripts.py. A few clean
lines per video are cut the same way ("kind": "clean"), for clipped-clean-word and sound-alike
scoring on the same speakers.

    .venv-lab\Scripts\python lab\detector\fetch_yodas.py --top 1 --shard data/en003/audio/00000123.tar.gz
    .venv-lab\Scripts\python lab\detector\fetch_yodas.py --top 60          # densest archives first, resumes

Writes data/realspeech/yodas/clips/<video>/<line>.flac, clips.jsonl (one line per clip) and
shards_done.txt.
"""
import argparse
import collections
import io
import json
import sys
import tarfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio.functional as AF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lexicon import NEGATIVES  # noqa: E402
from scan_transcripts import OUT, TOKEN, YODAS, cdn_url, match  # noqa: E402

SR = 16000
PRE, POST = 4.0, 1.5      # seconds of context kept before / after the transcript line
MAX_LINE = 20.0           # longer lines are usually caption-timing junk
CLEAN_PER_VIDEO = 3       # clean lines cut from each video too, for the clean-speech side of the test
NEG = {w.lower() for w in NEGATIVES if " " not in w}
DEST = OUT / "yodas"
lock = threading.Lock()


def load_hits():
    """{archive path: {video id: [lines with a curse word]}}, starred and repeated lines dropped."""
    shards = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    for line in (OUT / "hits_yodas.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if not r["hits"] or r["starred"] or not 0.05 < r["dur"] <= MAX_LINE:
            continue
        key = (r["src"], " ".join(r["text"].lower().split()))
        if key in seen:  # rolling captions repeat a line in overlapping segments
            continue
        seen.add(key)
        archive = r["file"].replace("/text/", "/audio/").replace(".json", ".tar.gz")
        shards[archive][r["src"]].append(r)
    return shards


def clean_lines(archive, videos):
    """{video: up to CLEAN_PER_VIDEO lines with no curse word}, sound-alike lines ("duck",
    "count") first. Lines within 1 s of a curse or starred line are skipped, since rolling
    captions overlap and the curse could be in the audio."""
    text = archive.replace("/audio/", "/text/").replace(".tar.gz", ".json")
    data = requests.get(cdn_url(YODAS, text), timeout=120).json()
    out = {}
    for v in data:
        video = v["audio_id"]
        if video not in videos:
            continue
        lines, bad = [], []
        for uid, t in v["text"].items():
            start, end = (int(x) / 100 for x in uid.rsplit("-", 3)[2:])
            hits, starred = match(t or "")
            if hits or starred:
                bad.append((start - 1.0, end + 1.0))
            elif 1.0 <= end - start <= 15.0:
                lines.append({"id": uid, "src": video, "start": start, "dur": round(end - start, 2),
                              "text": t, "hits": []})
        seen, ok = set(), []
        for r in lines:
            key = " ".join(r["text"].lower().split())
            if key in seen or any(a < r["start"] + r["dur"] and r["start"] < b for a, b in bad):
                continue
            seen.add(key)
            ok.append(r)
        ok.sort(key=lambda r: (not set(TOKEN.findall(r["text"].lower())) & NEG, zlib.crc32(r["id"].encode())))
        out[video] = ok[:CLEAN_PER_VIDEO]
    return out


def cut(audio, lines, video, archive, out):
    """Write one clip per line; return the metadata rows."""
    rows = []
    for r in lines:
        a = max(0.0, r["start"] - PRE)
        b = min(len(audio) / SR, r["start"] + r["dur"] + POST)
        if b - a < 1.0:
            continue
        name = r["id"].rsplit("-", 3)[1]
        path = DEST / "clips" / video / f"{name}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio[int(a * SR):int(b * SR)], SR)
        rows.append({
            "id": r["id"], "src": video, "archive": archive,
            "clip": path.relative_to(DEST).as_posix(), "clip_start": round(a, 3),
            # where the transcript line sits inside the clip; only this span has word labels
            "line": [round(r["start"] - a, 3), round(r["start"] + r["dur"] - a, 3)],
            "text": r["text"], "hits": r["hits"], "kind": "curse" if r["hits"] else "clean",
        })
    return rows


def fetch(archive, videos):
    t0, got, rows, nbytes = time.time(), 0, [], 0
    clean = clean_lines(archive, videos)
    with requests.get(cdn_url(YODAS, archive), stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tf:
            for m in tf:
                nbytes += m.size
                video = Path(m.name).stem
                if video not in videos:
                    continue  # tarfile skips the member's bytes when the loop moves on
                data, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(1)
                if len(data) < sr:  # some soundtracks are empty
                    got += 1
                    continue
                audio = AF.resample(torch.from_numpy(data), sr, SR).numpy()
                rows += cut(np.clip(audio, -1, 1), videos[video] + clean.get(video, []), video, archive, DEST)
                got += 1
                if got == len(videos):
                    break  # no need to download the rest of the archive
    mb = nbytes / 1e6
    print(f"  {archive}: {got}/{len(videos)} soundtracks, {len(rows)} clips, {mb:.0f} MB read, "
          f"{time.time() - t0:.0f} s ({mb / (time.time() - t0):.1f} MB/s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="the N archives with the most videos with hits")
    ap.add_argument("--start", type=int, default=0,
                    help="skip this many top archives (the training pool starts after the test archives)")
    ap.add_argument("--shard", nargs="*", default=[], help="specific archives (audio paths)")
    ap.add_argument("--jobs", type=int, default=4, help="archives downloaded at the same time")
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    done_file = DEST / "shards_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    shards = load_hits()
    # Rank by how many different videos have hits, not by hit count: the densest archives are a
    # few videos full of swearing, and a test set needs many speakers.
    ranked = sorted(shards, key=lambda s: -len(shards[s]))
    todo = [s for s in dict.fromkeys(ranked[a.start:a.start + a.top] + a.shard) if s not in done]
    print(f"{len(todo)} archives to fetch ({len(done)} done), {a.jobs} at a time", flush=True)

    def one(archive):
        for attempt in range(3):
            try:
                rows = fetch(archive, shards[archive])
                break
            except (requests.RequestException, tarfile.TarError, OSError) as e:
                print(f"  retry {archive}: {type(e).__name__}: {e}", flush=True)
                time.sleep(10)
            except Exception as e:  # a bad file in one archive mustn't stop the whole run
                print(f"  FAILED {archive}: {type(e).__name__}: {e}", flush=True)
                return
        else:
            print(f"  FAILED {archive}", flush=True)
            return
        with lock:
            with (DEST / "clips.jsonl").open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with done_file.open("a") as f:
                f.write(archive + "\n")

    # One stream gets 4-9 MB/s from the CDN; a few in parallel fill the line better.
    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(one, todo))


if __name__ == "__main__":
    main()

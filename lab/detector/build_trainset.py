r"""Split the YODAS2 training pool (Lambda) into training clips and a validation set, by video.

Same automatic checks as build_testset.py (Whisper hears every curse word, timing agrees, English,
no unlabelled curse word). The pool's archives never overlap the test set's 62, so no video is in
both. 10% of the videos (fixed hash) are held out for validation. Validation is capped at 2 curse
clips + 2 clean clips per video, like the test set; training keeps every eligible clip.

    python lab/detector/build_trainset.py

Writes data/realspeech/yodas/trainset.jsonl (video order fixed by hash, so taking a prefix of
videos gives the "N real curse words" subsets) and valset.jsonl, in the testset.jsonl row format.
"""
import collections
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_testset import DEST, PER_VIDEO, eligible, labels, load  # noqa: E402

VAL_FRACTION = 0.10


def row(r):
    keep = ("id", "src", "clip", "line", "text", "kind", "curses", "words")
    return {k: r[k] for k in keep if k in r} | {"labels": labels(r), "line_score": r["line_score"]}


def main():
    al, wh = load()
    split = json.loads((DEST / "split_videos.json").read_text()) if (DEST / "split_videos.json").exists() else {}
    test_videos = set(split.get("test", []))
    by_video = collections.defaultdict(list)
    for r in al:
        if eligible(r, wh.get(r["id"])) and r["src"] not in test_videos:
            by_video[r["src"]].append(r)
    videos = sorted(by_video, key=lambda v: zlib.crc32(v.encode()))
    n_val = int(len(videos) * VAL_FRACTION)
    val_v, train_v = videos[:n_val], videos[n_val:]
    with (DEST / "valset.jsonl").open("w", encoding="utf-8") as f:
        for v in val_v:
            curse = sorted((r for r in by_video[v] if r["curses"]), key=lambda r: r["id"])[:PER_VIDEO]
            clean = sorted((r for r in by_video[v] if not r["curses"]), key=lambda r: r["id"])[:PER_VIDEO]
            for r in curse + clean:
                f.write(json.dumps(row(r), ensure_ascii=False) + "\n")
    n_curse = 0
    with (DEST / "trainset.jsonl").open("w", encoding="utf-8") as f:
        for v in train_v:
            for r in sorted(by_video[v], key=lambda r: r["id"]):
                f.write(json.dumps(row(r), ensure_ascii=False) + "\n")
                n_curse += len(r["curses"])
    vc = sum(len(r["curses"]) for v in val_v for r in sorted((r for r in by_video[v] if r["curses"]),
                                                             key=lambda r: r["id"])[:PER_VIDEO])
    print(f"eligible: {sum(len(x) for x in by_video.values())} clips from {len(videos)} videos")
    print(f"train: {len(train_v)} videos, {n_curse} curse words | validation: {len(val_v)} videos, {vc} curse words")


if __name__ == "__main__":
    main()

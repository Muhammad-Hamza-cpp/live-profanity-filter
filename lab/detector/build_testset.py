r"""Pick the real-speech test set from aligned + Whisper-checked YODAS2 clips, split by video.

1. Automatic checks, no listening.
   - Curse clips: Whisper detects English and hears every labelled curse word with word
     confidence >= 0.5. It hears no unlabelled curse word. Each curse word's aligner time agrees
     with Whisper's: Whisper runs ~0.15 s early, so the offset must be -0.15 +- 0.25 s.
   - Clean clips: English, and Whisper hears no curse word.
   - On a 37-word hand-audited pilot sample (2026-11-05), 80% of words passing these checks
     (before the confidence cut) were real, correctly timed curse words. 2 had their timing off,
     1 wasn't said, and 2 couldn't be judged.
2. Split. Videos are ordered by a fixed hash. Test videos are taken in that order, each giving at
   most 2 curse clips (rarest word groups first) and 2 clean clips, until the target number of
   curse words is reached. Every other video goes to the training pool, so no voice is in both.

    .venv-lab\Scripts\python lab\detector\build_testset.py --target 450
    .venv-lab\Scripts\python lab\detector\build_testset.py --target 450 --page    # optional listening page

Writes to data/realspeech/yodas/: testset.jsonl and split_videos.json (plus review_test/ with
--page). In each row "labels" is [[word, kind, start, end]] in clip seconds, with kind "curse" or
"clean". Only words inside the transcript line are labelled; "*" tokens (numbers) aren't.
"""
import argparse
import collections
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_yodas import DEST  # noqa: E402
from make_review import write_page  # noqa: E402

WHISPER_OFFSET, WHISPER_TOL = -0.15, 0.25
MIN_PROB = 0.5  # Whisper's own confidence in the curse word
PER_VIDEO = 2


def load():
    # One row per clip: two runs over the same clips append duplicates.
    al = list({r["id"]: r for r in map(json.loads, (DEST / "aligned.jsonl").open(encoding="utf-8"))}.values())
    wh = {}
    for l in (DEST / "whisper.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        wh[r["id"]] = r
    return al, wh


def eligible(row, w):
    if w is None or w["lang"] != "en" or w.get("extra"):
        return False
    for c, wc in zip(row["curses"], w["curses"]):
        if not wc["heard"] or wc["prob"] < MIN_PROB:
            return False
        if abs(wc["start"] - row["words"][c["i"]]["start"] - WHISPER_OFFSET) > WHISPER_TOL:
            return False
    return True


def labels(row):
    """[[word, kind, start, end]]. A clip with a curse word that fails the hand-check is dropped
    later, not relabelled."""
    curse_i = {c["i"] for c in row["curses"]}
    return [[w["w"], "curse" if i in curse_i else "clean", w["start"], w["end"]]
            for i, w in enumerate(row["words"]) if w["w"] != "*"]


def select(al, wh, target):
    by_video = collections.defaultdict(lambda: {"curse": [], "clean": []})
    for r in al:
        kind = "curse" if r["curses"] else "clean"
        if eligible(r, wh.get(r["id"])):
            by_video[r["src"]][kind].append(r)
    videos = sorted(by_video, key=lambda v: zlib.crc32(v.encode()))
    group_n = collections.Counter()
    test, n_curse, test_videos = [], 0, []
    for v in videos:
        if n_curse >= target:
            break
        cands = by_video[v]["curse"]
        if not cands:
            continue
        # Prefer clips whose word groups the test set has least of so far.
        cands = sorted(cands, key=lambda r: (min(group_n[c["group"]] for c in r["curses"]), r["id"]))
        for r in cands[:PER_VIDEO]:
            test.append(r)
            n_curse += len(r["curses"])
            group_n.update(c["group"] for c in r["curses"])
        test += sorted(by_video[v]["clean"], key=lambda r: r["id"])[:PER_VIDEO]
        test_videos.append(v)
    all_videos = {r["src"] for r in al}
    return test, test_videos, sorted(all_videos - set(test_videos)), group_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=450, help="curse words in the test set")
    ap.add_argument("--page", action="store_true", help="also write a listening page (optional)")
    a = ap.parse_args()
    al, wh = load()
    test, test_videos, train_videos, group_n = select(al, wh, a.target)
    with (DEST / "testset.jsonl").open("w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps({k: r[k] for k in ("id", "src", "clip", "line", "text", "kind", "curses", "words")
                                if k in r} | {"labels": labels(r), "line_score": r["line_score"]},
                               ensure_ascii=False) + "\n")
    (DEST / "split_videos.json").write_text(json.dumps({"test": test_videos, "train": train_videos}, indent=0))
    picked = [(r, c) for r in test for c in r["curses"]]
    if a.page:
        write_page(picked, DEST / "review_test", key="curse-review-test-v1")
    n_clean = sum(1 for r in test if not r["curses"])
    n_clean_words = sum(k == "clean" for r in test for _, k, *_ in labels(r))
    print(f"eligible clips: {sum(eligible(r, wh.get(r['id'])) for r in al)} of {len(al)}")
    print(f"test: {len(picked)} curse words in {len(test) - n_clean} clips + {n_clean} clean clips "
          f"({n_clean_words} clean words), {len(test_videos)} videos; training pool: {len(train_videos)} videos")
    print("by group: " + ", ".join(f"{g} {n}" for g, n in group_n.most_common()))


if __name__ == "__main__":
    main()

r"""Second check on aligned real-speech clips: Whisper large-v3 (GPU). Parakeet garbles casual
swearing ("motherfucker" -> "buffer"); Whisper keeps profanity, so it is the main check.

For each clip in aligned.jsonl, Whisper transcribes the transcript line +-1 s with language
detection (a translated subtitle shows up as a non-English language), no VAD, no prompt. A curse
word counts as heard when Whisper writes a form of its group; Whisper's own time for that word is
kept to cross-check the aligner's timing.

    .venv-lab\Scripts\python lab\detector\verify_whisper.py          # resumes

Writes data/realspeech/yodas/whisper.jsonl: {id, lang, lang_p, text, curses: [{i, heard, start, end, prob}],
extra: [[group, start, end, prob]] (curse words Whisper heard that the subtitle lacks)}.
int8_float16 keeps it at ~2 GB of GPU memory, so the app can still start alongside it.
"""
import collections
import json
import re
import sys
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from align_clips import SR, WORD  # noqa: E402
from fetch_yodas import DEST  # noqa: E402
from scan_transcripts import FORM2GROUP  # noqa: E402

MARGIN = 1.0


def check(model, row):
    audio, _ = sf.read(DEST / row["clip"], dtype="float32")
    a = max(0.0, row["line"][0] - MARGIN)
    b = min(len(audio) / SR, row["line"][1] + MARGIN)
    segs, info = model.transcribe(audio[int(a * SR):int(b * SR)], beam_size=5, word_timestamps=True,
                                  vad_filter=False, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in s.words or []:
            for t in WORD.findall(w.word.lower().replace("’", "'")):
                words.append((t, round(w.start + a, 3), round(w.end + a, 3), round(w.probability, 3)))
    # Match each transcript curse word to the nearest unused Whisper word of the same group.
    pool = collections.defaultdict(list)
    for t, s, e, p in words:
        if t in FORM2GROUP:
            pool[FORM2GROUP[t]].append([s, e, p])
    curses = []
    for c in row["curses"]:
        w = row["words"][c["i"]]
        cands = pool[c["group"]]
        if cands:
            best = min(cands, key=lambda x: abs(x[0] - w["start"]))
            cands.remove(best)
            curses.append({"i": c["i"], "heard": True, "start": best[0], "end": best[1], "prob": best[2]})
        else:
            curses.append({"i": c["i"], "heard": False})
    # Curse words Whisper hears that the subtitle doesn't have: a "clean" line with one of these
    # isn't clean, and a curse line may hold more curse words than labelled.
    extra = [[g, s, e, p] for g, ws in pool.items() for s, e, p in ws]
    return {"id": row["id"], "lang": info.language, "lang_p": round(info.language_probability, 3),
            "text": " ".join(t for t, *_ in words), "curses": curses, "extra": extra}


def main():
    import argparse
    import zlib
    from faster_whisper import WhisperModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n: only clips whose id hashes to i of n (parallel workers)")
    ap.add_argument("--skip-ids", help="file of ids another running job is already checking")
    a = ap.parse_args()
    i_sh, n_sh = map(int, a.shard.split("/"))
    out = DEST / "whisper.jsonl"
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    if a.skip_ids:
        done |= set(Path(a.skip_ids).read_text().split())
    rows = {r["id"]: r for r in map(json.loads, (DEST / "aligned.jsonl").open(encoding="utf-8"))}
    rows = [r for i, r in rows.items() if i not in done and zlib.crc32(i.encode()) % n_sh == i_sh]
    print(f"{len(rows)} clips to check", flush=True)
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            try:
                res = check(model, r)
            except Exception as e:
                print(f"  {r['id']}: {type(e).__name__}: {e}", flush=True)
                continue
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()  # the Lambda filesystem buffers ~MBs otherwise; a crash would lose them
            if i % 100 == 0:
                print(f"  {i}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()

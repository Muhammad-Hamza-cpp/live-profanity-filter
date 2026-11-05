r"""Check a recorded session ("Save audio for training") against what was actually said.

Whisper large-v3 transcribes the recording offline - full context, no time pressure - and every
mute the detector made is classified: a real curse word (caught), a clean word (false, in speech),
or no speech at all (false, non-speech). Curse words Whisper heard that weren't muted are misses.
Short clips of the false mutes and misses are written out so they can be checked by ear.

    .venv-lab\Scripts\python lab\detector\review_recording.py data\recordings\20261104-163523
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from censor import add_cuda_dlls, is_profane, load_words  # noqa: E402

SR = 16000
NEAR = 0.15  # a mute within this many seconds of a curse word counts as that word's mute


def load_session(d):
    parts = [json.loads(line) for line in open(d / "parts.jsonl")]
    mutes = [json.loads(line) for line in open(d / "mutes.jsonl")] if (d / "mutes.jsonl").exists() else []
    t0 = parts[0]["start"]
    end = max(p["start"] - t0 + sf.info(d / p["file"]).duration for p in parts)
    audio = np.zeros(int(end * SR) + SR, np.float32)
    for p in parts:
        a, sr = sf.read(d / p["file"], dtype="float32")
        i = int((p["start"] - t0) * SR)
        audio[i:i + len(a)] = a
    for m in mutes:
        m["start"] -= t0
        m["end"] -= t0
    return audio, mutes


def transcribe(audio, cache, win=10.0, hop=5.0):
    """Whisper large-v3 over short overlapping windows. On game audio, one long pass drops whole
    stretches of dialogue and the VAD filter drops speech under gunfire, so: no VAD, 10 s windows,
    each word taken from the window where it sits in the middle, and only segments Whisper is
    confident are speech (it invents text on pure noise)."""
    if cache.exists():
        return json.loads(cache.read_text())
    add_cuda_dlls()
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    words, n = [], len(audio) / SR
    for t in np.arange(0, max(n - win, 0) + hop, hop):
        a = audio[int(t * SR):int((t + win) * SR)]
        if np.sqrt(np.mean(a ** 2)) < 1e-4:  # digital silence
            continue
        lo = t + (win - hop) / 2 if t > 0 else 0.0  # this window owns words centred in [lo, hi)
        hi = t + (win + hop) / 2 if t + win < n else n
        segs, _ = model.transcribe(a, language="en", beam_size=5, word_timestamps=True,
                                   vad_filter=False, condition_on_previous_text=False)
        for s in segs:
            if s.no_speech_prob >= 0.6 or s.avg_logprob <= -1.0:
                continue
            for w in s.words or []:
                if lo <= t + (w.start + w.end) / 2 < hi:
                    words.append({"word": w.word.strip(), "start": t + w.start, "end": t + w.end,
                                  "p": w.probability})
    cache.write_text(json.dumps(words))
    return words


def main():
    d = Path(sys.argv[1])
    d = d if d.is_absolute() else ROOT / d
    audio, mutes = load_session(d)
    words = transcribe(audio, d / "whisper_large_v3_windows.json")
    patterns = load_words(ROOT / "words.txt")
    curse = [w for w in words if is_profane(w["word"], patterns)]
    print(f"{len(audio) / SR / 60:.1f} min, Whisper heard {len(words)} words, {len(curse)} of them curse words; "
          f"detector made {len(mutes)} mutes")

    def overlaps(m, w, slack=0.0):
        return m["start"] < w["end"] + slack and m["end"] > w["start"] - slack

    kinds = []
    for m in mutes:
        if any(overlaps(m, w, NEAR) for w in curse):
            kinds.append("caught")
        elif any(overlaps(m, w) for w in words):
            kinds.append("false, in speech")
        else:
            kinds.append("false, no speech")
        m["kind"] = kinds[-1]
    for k in ("caught", "false, in speech", "false, no speech"):
        sel = [m for m in mutes if m["kind"] == k]
        late = sum(m["late"] for m in sel)
        print(f"  {k:<17} {len(sel):4d} mutes ({len(sel) / max(len(mutes), 1):.0%}), {late} marked late")

    missed, late_words = [], []
    for w in curse:
        hit = [m for m in mutes if overlaps(m, w, NEAR)]
        if not hit:
            missed.append(w)
        elif all(m["late"] for m in hit):
            late_words.append(w)
    print(f"curse words Whisper heard: {len(curse)} - muted {len(curse) - len(missed)}, "
          f"of those only late {len(late_words)}, never muted {len(missed)}")

    out = d / "review"
    out.mkdir(exist_ok=True)

    def clip(name, s, e):
        a, b = max(0, int((s - 1.0) * SR)), min(len(audio), int((e + 1.0) * SR))
        sf.write(out / name, audio[a:b], SR)

    for m in mutes:
        m["heard_nearby"] = " ".join(w["word"] for w in words if overlaps(m, w, 0.5))
    for i, m in enumerate([m for m in mutes if m["kind"] != "caught"][:40]):
        clip(f"false_{i:03d}_{m['group']}_{m['start']:.0f}s.wav", m["start"], m["end"])
    for i, w in enumerate(missed[:40]):
        clip(f"missed_{i:03d}_{w['start']:.0f}s.wav", w["start"], w["end"])
    (out / "review.json").write_text(json.dumps({"mutes": mutes, "curse_words": curse, "missed": missed}, indent=1))

    print("\nfalse mutes in speech - the words nearby (Whisper):")
    for m in [m for m in mutes if m["kind"] == "false, in speech"][:25]:
        print(f"  {m['start']:7.1f}s  group {m['group']:<10} p={m['prob']:.2f}  \"{m.get('heard_nearby', '')}\"")
    print("\nmissed curse words (Whisper):")
    for w in missed[:25]:
        ctx = " ".join(x["word"] for x in words if abs(x["start"] - w["start"]) < 1.5)
        print(f"  {w['start']:7.1f}s  {w['word']:<12} (Whisper p={w['p']:.2f})  \"{ctx}\"")
    print(f"\nclips for listening: {out}")


if __name__ == "__main__":
    main()

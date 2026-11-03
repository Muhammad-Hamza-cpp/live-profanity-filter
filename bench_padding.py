"""How much silence to add around each detected word: leaks vs. clean words lost.

Runs the live-loop simulation once per model (with no padding), then for each padding setting
renders the censored audio and transcribes it, which is what you would actually hear.
Needs the clips made by bench.py.
"""
import difflib
import re

import numpy as np

from bench import SR, TEST, load16, reference_words, simulate
from censor import HERE, add_cuda_dlls, is_profane, load_words

PADS = [(0.12, 0.20), (0.08, 0.15), (0.05, 0.12), (0.03, 0.08), (0.0, 0.05)]


def render(audio, events, pad_before, pad_after, fade=0.005):
    t = np.arange(len(audio)) / SR
    gain = np.ones(len(audio), np.float32)
    for _, s, e in events:
        s, e = s - pad_before, e + pad_after
        lo, hi = max(0, int((s - fade) * SR)), min(len(audio), int((e + fade) * SR))
        seg = t[lo:hi]
        gain[lo:hi] = np.minimum(gain[lo:hi], np.clip(np.maximum(s - seg, seg - e) / fade, 0, 1))
    return audio * gain


def norm(words):
    return [re.sub(r"[^a-z']", "", w.lower()) for w in words]


def main():
    add_cuda_dlls()
    from faster_whisper import WhisperModel
    patterns = load_words(HERE / "words.txt")
    clips = {k: load16(TEST / f"bench_{k}.wav") for k in ("normal", "fast", "noisy")}
    judge = WhisperModel("small.en", device="cuda", compute_type="float16")
    refs = {k: [w for w, _, _ in reference_words(judge, a)] for k, a in clips.items()}

    for model_name in ("tiny.en", "small.en"):
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        events = {k: simulate(model, a, 0.4 if model_name == "small.en" else 0.3, 4.0, patterns, 0, 0)[0]
                  for k, a in clips.items()}
        for pb, pa in PADS:
            leaked, kept, total_clean, leaked_words = 0, 0, 0, []
            for k, audio in clips.items():
                heard = [w for w, _, _ in reference_words(judge, render(audio, events[k], pb, pa))]
                leaks = [w for w in heard if is_profane(w, patterns)]
                leaked += len(leaks)
                leaked_words += leaks
                ref_clean = [w for w in norm(refs[k]) if not is_profane(w, patterns)]
                heard_clean = [w for w in norm(heard) if not is_profane(w, patterns)]
                sm = difflib.SequenceMatcher(None, ref_clean, heard_clean, autojunk=False)
                kept += sum(b.size for b in sm.get_matching_blocks())
                total_clean += len(ref_clean)
            print(f"{model_name:<9} pad before {pb:.2f}s after {pa:.2f}s | curse words heard: {leaked} {leaked_words} "
                  f"| clean words kept: {kept}/{total_clean} ({100 * kept / total_clean:.1f}%)", flush=True)


if __name__ == "__main__":
    main()

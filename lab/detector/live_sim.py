r"""Run the app's real engine (censor.Censor + detect_forever, its own thread) in real time without
any audio devices: one thread feeds a WAV in 10 ms blocks like the input stream does, another
pulls output blocks like the speaker stream does. Then score the output by loudness like
selftest.py. Useful when VB-Cable isn't installed.

    .venv-lab\Scripts\python lab\detector\live_sim.py [delay]
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from censor import DETECTOR, Censor, add_cuda_dlls, is_profane, load_model, load_words  # noqa: E402
from selftest import TEST, load16, make_speech  # noqa: E402

SR, BLOCK = 48000, 480


def main():
    delay = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    TEST.mkdir(exist_ok=True)
    original = TEST / "original.wav"
    make_speech(original)
    a, sr = sf.read(original, dtype="float32")
    from scipy.signal import resample_poly
    a = resample_poly(a, SR, sr).astype(np.float32)
    a = np.concatenate([np.zeros(SR, np.float32), a, np.zeros(2 * SR, np.float32)])
    stereo = np.stack([a, a], 1)

    patterns = load_words(ROOT / "words.txt")
    model, _ = load_model(DETECTOR)
    c = Censor(model, patterns, SR, 2, delay, 0.06, 0.02, step=0.05, threshold=0.5, confirm=2, verbose=True)
    threading.Thread(target=c.detect_forever, daemon=True).start()
    out = []

    def speaker():
        t = time.perf_counter()
        while len(out) * BLOCK < len(stereo):
            t += BLOCK / SR
            out.append(c.render(BLOCK))
            time.sleep(max(0, t - time.perf_counter()))
    threading.Thread(target=speaker, daemon=True).start()
    t = time.perf_counter()
    for i in range(0, len(stereo), BLOCK):  # input stream: real-time pace
        c.push(stereo[i:i + BLOCK])
        t += BLOCK / SR
        time.sleep(max(0, t - time.perf_counter()))
    time.sleep(delay + 0.5)
    c.stop.set()
    rec = np.concatenate(out)[:, 0]
    censored = TEST / "censored_detector.wav"
    sf.write(censored, rec, SR)
    print(f"muted {c.muted_count} region(s), {c.late_count} late")

    # The output is the input shifted by the delay: line them up and compare loudness per word.
    orig, out16 = load16(original), resample_poly(rec, 16000, SR).astype(np.float32)
    lead = int((1.0 + delay) * 16000)
    out16 = out16[lead:lead + len(orig)]
    add_cuda_dlls()
    from faster_whisper import WhisperModel
    judge = WhisperModel("small.en", device="cuda", compute_type="float16")
    segs, _ = judge.transcribe(orig, language="en", beam_size=5, word_timestamps=True)
    words = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]

    def kept(s, e):
        i, j = int(s * 16000), int(e * 16000)
        return float(np.sqrt(np.mean(out16[i:j] ** 2)) / (np.sqrt(np.mean(orig[i:j] ** 2)) + 1e-9))
    bad = [(w, kept(s, e)) for w, s, e in words if is_profane(w, patterns)]
    clean = [(w, kept(s, e)) for w, s, e in words if not is_profane(w, patterns)]
    print("\nCurse words (loudness left, 0% = silent):")
    for w, k in bad:
        print(f"  {w:<12} {k:4.0%}  {'OK' if k < 0.2 else 'AUDIBLE'}")
    hurt = [(w, k) for w, k in clean if k < 0.7]
    print(f"Clean words left intact: {len(clean) - len(hurt)} of {len(clean)}"
          + (f"  (partly cut: {', '.join(f'{w} {k:.0%}' for w, k in hurt)})" if hurt else ""))
    print("PASS" if bad and all(k < 0.2 for _, k in bad) else "FAIL")


if __name__ == "__main__":
    main()

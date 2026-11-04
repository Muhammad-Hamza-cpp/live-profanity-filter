"""End-to-end test without using your speakers.

1. Windows text-to-speech makes test/original.wav with curse words in it.
2. censor.py runs live and records its output to test/censored.wav (nothing is played out loud).
3. The test WAV is played into the virtual cable in real time.
4. The recording is lined up with the original and each word's loudness is compared.
   (Loudness, not re-transcription: Whisper "hears" a silenced word anyway when context makes
   it obvious, e.g. "what the ___ is going on".)

    python selftest.py [delay] [extra censor.py options...]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from censor import DEFAULT_DELAY, HERE, add_cuda_dlls, is_profane, load_words

TEST = HERE / "test"
SR = 16000
TEXT = ("Hey everyone, welcome back to the channel. What the hell is going on with this game? "
        "This boss is so fucking hard, I swear. Oh shit, he hit me again. "
        "That was a damn good attempt though. Stop camping, you asshole. "
        "Alright, this level is clean, nice and friendly. Holy crap, we finally won. "
        "Thanks for watching, see you next time.")


def make_speech(path):
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{TEXT}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def load16(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_poly(audio, SR, sr).astype(np.float32)


def main():
    TEST.mkdir(exist_ok=True)
    original, censored = TEST / "original.wav", TEST / "censored.wav"
    make_speech(original)

    add_cuda_dlls()
    import sounddevice as sd
    import soundfile as sf
    audio, sr = sf.read(original, dtype="float32")
    length = len(audio) / sr
    print(f"Test speech: {length:.1f} s")

    delay = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DELAY)
    proc = subprocess.Popen(
        [sys.executable, "-u", str(HERE / "censor.py"), "--no-switch", "--no-speaker", "--delay", delay,
         "--record", str(censored), "--duration", str(length + float(delay) + 2), *sys.argv[2:]],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print("  censor> " + line.rstrip())
        if "READY" in line:
            break
    else:
        sys.exit("censor.py exited before it was ready")

    wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])
    cable_in = next(i for i, d in enumerate(sd.query_devices())
                    if d["hostapi"] == wasapi and d["max_output_channels"] > 0 and "CABLE Input" in d["name"])
    sd.play(audio, sr, device=cable_in, extra_settings=sd.WasapiSettings(auto_convert=True), blocking=True)

    for line in proc.stdout:
        print("  censor> " + line.rstrip())
    proc.wait()

    # Line the recording up with the original.
    from scipy.signal import correlate
    orig, rec = load16(original), load16(censored)
    offset = int(np.argmax(correlate(rec, orig, mode="valid", method="fft")))
    rec = rec[offset:offset + len(orig)]

    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(orig, language="en", beam_size=5, word_timestamps=True)
    words = [(w.word.strip(), w.start, w.end) for s in segments for w in s.words]
    patterns = load_words(HERE / "words.txt")

    def kept(s, e):
        """How much of the word's loudness survived (0 = fully silenced)."""
        a, b = int(s * SR), int(e * SR)
        return float(np.sqrt(np.mean(rec[a:b] ** 2)) / (np.sqrt(np.mean(orig[a:b] ** 2)) + 1e-9))

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

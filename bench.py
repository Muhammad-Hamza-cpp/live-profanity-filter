"""Latency/accuracy benchmark for the live filter.

Replays test speech through the same windowed transcription the live filter uses, timing every
call on the GPU, and works out for each curse word how much delay would have been needed to
silence all of it. Lower "safe delay" = less lag you hear.

    python bench.py                       # all models x steps
    python bench.py --models small.en --steps 0.3
"""
import argparse
import json
import subprocess
import time

import numpy as np

from censor import HERE, add_cuda_dlls, is_profane, load_words

TEST = HERE / "test"
SR = 16000
TEXT = ("Alright guys, let's get into it. Man, this damn update broke everything. "
        "I logged in and the whole map is different, what the fuck. Okay, okay, let me check the settings real quick. "
        "Bullshit, that guy was behind a wall! He is such a dickhead, honestly. "
        "Anyway, the new weapon is pretty cool, I like the sound of it. Oh crap, there are three of them. Run, run, run! "
        "Son of a bitch, I almost had him. That was a pretty good fight though, not gonna lie. "
        "Holy shit, look at that view. Let's grab some loot and heal up. "
        "What a motherfucker, he was camping the whole time. Stop hiding, you asshole. "
        "Hell yeah, we got the win! Thanks for hanging out today, and I will see you all tomorrow.")


def make_speech(path, voice, rate):
    txt = TEST / "bench_text.txt"
    txt.write_text(TEXT, encoding="utf-8")
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"try {{ $s.SelectVoice('{voice}') }} catch {{}}; $s.Rate = {rate}; "
          f"$s.SetOutputToWaveFile('{path}'); $s.Speak((Get-Content -Raw '{txt}')); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def load16(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return resample_poly(a, SR, sr).astype(np.float32)


def add_noise(speech, snr_db, seed=0):
    """Game-ish background: rumble noise plus changing synth chords."""
    rng = np.random.default_rng(seed)
    n = len(speech)
    noise = np.cumsum(rng.normal(0, 1, n)).astype(np.float32)
    noise -= np.convolve(noise, np.ones(400) / 400, mode="same")  # remove drift
    t = np.arange(n) / SR
    music = np.zeros(n, np.float32)
    for i in range(0, n, SR // 2):
        f = rng.choice([220, 247, 262, 294, 330, 392, 440])
        seg = slice(i, min(i + SR // 2, n))
        music[seg] = sum(np.sin(2 * np.pi * f * k * t[seg]) / k for k in (1, 1.5, 2))
    bg = noise / noise.std() + music / music.std()
    speech_rms = np.sqrt(np.mean(speech[np.abs(speech) > 0.01] ** 2))
    bg *= speech_rms / bg.std() / (10 ** (snr_db / 20))
    return (speech + bg).astype(np.float32)


def reference_words(model, audio):
    segments, _ = model.transcribe(audio, language="en", beam_size=5, word_timestamps=True)
    return [(w.word.strip(), w.start, w.end) for s in segments for w in s.words]


def simulate(model, audio, step, window, patterns, pad_before=0.12, pad_after=0.20):
    """Same loop as Censor.transcribe_forever, on a simulated clock. Returns mute events."""
    audio = np.concatenate([audio, np.zeros(SR * 2, np.float32)])
    events, times = [], []
    t = step
    while t * SR <= len(audio):
        end = int(t * SR)
        start = max(0, end - int(window * SR))
        chunk = audio[start:end]
        p = 0.0
        if np.sqrt(np.mean(chunk ** 2)) >= 1e-4:
            t0 = time.perf_counter()
            segments, _ = model.transcribe(chunk, language="en", beam_size=1, temperature=0.0,
                                           word_timestamps=True, condition_on_previous_text=False)
            words = [w for s in segments for w in (s.words or [])]
            p = time.perf_counter() - t0
            times.append(p)
            for w in words:
                if is_profane(w.word, patterns):
                    events.append((t + p, start / SR + w.start - pad_before, start / SR + w.end + pad_after))
        t = max(t + step, t + p)  # live loop can't start the next window before this one finishes
    return events, times


def score(ref, events, patterns):
    bad = [(w, s, e) for w, s, e in ref if is_profane(w, patterns)]
    clean = [(w, s, e) for w, s, e in ref if not is_profane(w, patterns)]
    needed, missed = [], []
    for w, s, e in bad:
        worst = 0.0
        # Check the core of the word (reference timestamps are themselves +-50 ms).
        for x in np.arange(s + 0.05, max(s + 0.06, e - 0.05), 0.01):
            covering = [dt for dt, ms, me in events if ms <= x <= me]
            if not covering:
                worst = float("inf")
                break
            worst = max(worst, min(covering) - x)
        (missed if worst == float("inf") else needed).append((w, worst))
    overmuted = 0
    for w, s, e in clean:
        grid = np.arange(s, e, 0.01)
        if len(grid) and np.mean([any(ms <= x <= me for _, ms, me in events) for x in grid]) > 0.5:
            overmuted += 1
    return bad, needed, missed, overmuted, len(clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny.en,base.en,small.en,large-v3-turbo")
    ap.add_argument("--steps", default="0.2,0.3,0.4")
    ap.add_argument("--window", type=float, default=4.0)
    args = ap.parse_args()

    add_cuda_dlls()
    from faster_whisper import WhisperModel
    TEST.mkdir(exist_ok=True)
    patterns = load_words(HERE / "words.txt")

    clips = {}
    for name, voice, rate in [("normal", "Microsoft David Desktop", 0), ("fast", "Microsoft Zira Desktop", 3)]:
        path = TEST / f"bench_{name}.wav"
        make_speech(path, voice, rate)
        clips[name] = load16(path)
    clips["noisy"] = add_noise(clips["normal"], snr_db=5)
    import soundfile as sf
    sf.write(TEST / "bench_noisy.wav", clips["noisy"], SR)

    ref_model = WhisperModel("small.en", device="cuda", compute_type="float16")
    refs = {k: reference_words(ref_model, clips["fast" if k == "fast" else "normal"]) for k in clips}
    del ref_model

    results = []
    for model_name in args.models.split(","):
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        noise = np.random.default_rng(0).normal(0, 0.1, SR * 4).astype(np.float32)
        list(model.transcribe(noise, language="en", beam_size=1, word_timestamps=True)[0])  # warm-up
        for step in map(float, args.steps.split(",")):
            row = {"model": model_name, "step": step, "clips": {}}
            all_times, all_needed, total_bad, total_missed, total_over, total_clean = [], [], 0, 0, 0, 0
            for clip, audio in clips.items():
                events, times = simulate(model, audio, step, args.window, patterns)
                bad, needed, missed, over, nclean = score(refs[clip], events, patterns)
                all_times += times
                all_needed += [n for _, n in needed]
                total_bad += len(bad)
                total_missed += len(missed)
                total_over += over
                total_clean += nclean
                row["clips"][clip] = {"missed": [w for w, _ in missed],
                                      "max_needed": round(max([n for _, n in needed], default=0), 2)}
            row.update(ms_per_call=round(1000 * float(np.mean(all_times)), 1),
                       gpu_busy=round(float(np.mean(all_times)) / step, 2),
                       curse_words=total_bad, missed=total_missed,
                       safe_delay=round(max(all_needed, default=0), 2),
                       p90_needed=round(float(np.percentile(all_needed, 90)) if all_needed else 0, 2),
                       clean_overmuted=f"{total_over}/{total_clean}")
            results.append(row)
            print(json.dumps(row), flush=True)
        del model

    print("\nmodel            step  ms/call  GPU busy  caught      safe delay  p90   clean words muted")
    for r in results:
        print(f"{r['model']:<16} {r['step']:<5} {r['ms_per_call']:<8} {r['gpu_busy']:<9} "
              f"{r['curse_words'] - r['missed']}/{r['curse_words']:<8} {r['safe_delay']:<11} "
              f"{r['p90_needed']:<5} {r['clean_overmuted']}")
    (TEST / "bench_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

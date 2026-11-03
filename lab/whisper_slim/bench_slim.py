"""Latency experiments on Whisper itself, scored exactly like bench.py.

Variants:
  - ctx=N : feed the encoder N mel frames (100 per second) instead of padding every window to 30 s.
  - pruned model dirs made by prune.py (layers removed from model.bin, no retraining).

    python bench_slim.py --profile                 # where the time goes in one 4 s call
    python bench_slim.py small.en:3000 small.en:450 tiny.en:450 --steps 0.2,0.3,0.4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from bench import SR, TEST, load16, reference_words, score, simulate  # noqa: E402
from censor import add_cuda_dlls, load_words  # noqa: E402

add_cuda_dlls()
import faster_whisper.transcribe as fwt  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

_orig_pad = fwt.pad_or_trim
ENC_FRAMES = [3000]


def short_pad(array, length=3000, *, axis=-1):
    """Pad to ENC_FRAMES instead of 3000 (never cutting real audio; even length for the stride-2 conv)."""
    n = max(ENC_FRAMES[0], array.shape[axis] + array.shape[axis] % 2)
    return _orig_pad(array, min(n, length), axis=axis)


fwt.pad_or_trim = short_pad


class GpuGuard:
    """Timings are only valid if nothing else is on the GPU (another bench may be running).
    wait() blocks until we're alone; clean() says whether anyone else showed up since."""

    def __init__(self):
        import os
        import subprocess
        import threading
        self.me, self.sp, self.dirty = os.getpid(), subprocess, False
        threading.Thread(target=self._poll, daemon=True).start()

    def others(self):
        out = self.sp.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split()
        return [p for p in out if p.isdigit() and int(p) != self.me]

    def _poll(self):
        while True:
            if self.others():
                self.dirty = True
            time.sleep(2)

    def wait(self):
        announced = False
        while self.others():
            if not announced:
                print("  (GPU busy with another process - waiting)", flush=True)
                announced = True
            time.sleep(5)
        self.dirty = False

    def clean(self):
        return not self.dirty and not self.others()


def load(spec):
    """'small.en:450' or 'path/to/pruned:450' -> (label, model, frames)."""
    name, _, frames = spec.partition(":")
    model = WhisperModel(name, device="cuda", compute_type="float16")
    noise = np.random.default_rng(0).normal(0, 0.1, SR * 4).astype(np.float32)
    ENC_FRAMES[0] = int(frames or 3000)
    list(model.transcribe(noise, language="en", beam_size=1, word_timestamps=True)[0])
    return f"{Path(name).name}:{ENC_FRAMES[0]}", model, ENC_FRAMES[0]


def profile(model, audio):
    """Split one live call into encoder / decoding / word alignment."""
    import ctranslate2
    out = {}
    feats = model.feature_extractor(audio)[:, :-1]
    for frames in (3000, 450):
        f = _orig_pad(feats, frames)[None]
        sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(f))
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.model.encode(sv, to_cpu=True)
            ts.append(time.perf_counter() - t0)
        out[f"encode {frames} frames"] = 1000 * float(np.median(ts))
    for frames in (3000, 450):
        ENC_FRAMES[0] = frames
        for wt in (False, True):
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                list(model.transcribe(audio, language="en", beam_size=1, temperature=0.0,
                                      word_timestamps=wt, condition_on_previous_text=False)[0])
                ts.append(time.perf_counter() - t0)
            out[f"transcribe {frames} frames, word_ts={wt}"] = 1000 * float(np.median(ts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--steps", default="0.2,0.3,0.4")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    args = ap.parse_args()

    patterns = load_words(ROOT / "words.txt")
    clips = {k: load16(TEST / f"bench_{k}.wav") for k in ("normal", "fast", "noisy")}
    guard = GpuGuard()

    if args.profile:
        for name in args.variants or ["tiny.en", "small.en"]:
            _, model, _ = load(name)
            chunk = clips["normal"][SR * 10: SR * 14]
            while True:
                guard.wait()
                res = profile(model, chunk)
                if guard.clean():
                    break
            for k, v in res.items():
                print(f"{name:<12}{k:<40}{v:7.1f} ms", flush=True)
            del model
        return

    # Reference transcript: same judge as bench.py (small.en, beam 5, full 30 s context).
    ENC_FRAMES[0] = 3000
    judge = WhisperModel("small.en", device="cuda", compute_type="float16")
    refs = {k: reference_words(judge, clips["fast" if k == "fast" else "normal"]) for k in clips}
    del judge

    for spec in args.variants:
        label, model, frames = load(spec)
        for step in map(float, args.steps.split(",")):
            while True:  # redo the run if another process used the GPU meanwhile (timings drive the simulated clock)
                guard.wait()
                all_times, all_needed, n_bad, n_missed, n_over, n_clean, missed_words = [], [], 0, 0, 0, 0, []
                worst = []
                for clip, audio in clips.items():
                    ENC_FRAMES[0] = frames
                    events, times = simulate(model, audio, step, 4.0, patterns)
                    bad, needed, missed, over, nclean = score(refs[clip], events, patterns)
                    all_times += times
                    all_needed += [n for _, n in needed]
                    worst += [(round(n, 2), f"{clip}:{w}") for w, n in needed]
                    n_bad, n_missed = n_bad + len(bad), n_missed + len(missed)
                    n_over, n_clean = n_over + over, n_clean + nclean
                    missed_words += [f"{clip}:{w}" for w, _ in missed]
                if guard.clean():
                    break
                print(f"  {label} step {step}: GPU was shared during the run, repeating", flush=True)
            row = dict(variant=label, step=step, ms_per_call=round(1000 * float(np.mean(all_times)), 1),
                       gpu_busy=round(float(np.mean(all_times)) / step, 2), caught=f"{n_bad - n_missed}/{n_bad}",
                       safe_delay=round(max(all_needed, default=0), 2),
                       p90_needed=round(float(np.percentile(all_needed, 90)) if all_needed else 0, 2),
                       clean_overmuted=f"{n_over}/{n_clean}", missed=missed_words,
                       worst=sorted(worst, reverse=True)[:3])
            print(json.dumps(row), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        del model


if __name__ == "__main__":
    main()

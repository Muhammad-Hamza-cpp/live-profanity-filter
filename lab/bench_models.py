"""Compare speech models for the live filter on the bench.py clips.

For each model/step it runs the live loop on a simulated clock (real GPU timings), then:
  - safe delay: smallest delay that silences every curse word completely
  - at that delay (+0.15 s margin): loudness left in each curse word, clean words cut
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench import SR, TEST, load16, reference_words, simulate  # noqa: E402
from censor import add_cuda_dlls, is_profane, load_words  # noqa: E402


class Parakeet:
    """Parakeet CTC behind the same .transcribe() shape as faster-whisper (words with times)."""

    def __init__(self, name="nvidia/parakeet-ctc-0.6b"):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForCTC.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
        self.blank = self.model.config.pad_token_id
        tok = self.proc.tokenizer
        self.pieces = tok.convert_ids_to_tokens(list(range(self.model.config.vocab_size)))

    def logits(self, audio):
        inputs = self.proc(audio, sampling_rate=SR, return_tensors="pt")
        inputs = {k: (v.cuda().bfloat16() if v.is_floating_point() else v.cuda()) for k, v in inputs.items()}
        with self.torch.inference_mode():
            return self.model(**inputs).logits[0].float().cpu().numpy()

    shift = 0.0  # timestamp correction, seconds
    gap = False  # mute from the previous word's last spike to the next word's first spike

    def transcribe(self, audio, **_):
        lg = self.logits(audio)
        frame = 0.08  # 10 ms mel hop x 8 subsampling
        ids = lg.argmax(-1)
        words, prev = [], -1
        for f, i in enumerate(ids):
            if i != self.blank and i != prev:
                piece = self.pieces[i]
                if piece.startswith("▁") or not words:
                    words.append([piece.lstrip("▁"), f, f])
                else:
                    words[-1][0] += piece
                    words[-1][2] = f
            prev = i
        words = [w for w in words if w[0]]
        if self.gap:
            # CTC spikes sit somewhere inside each word, so a word lies between its neighbours' spikes.
            n = len(lg)
            out = []
            for j, (w, s, e) in enumerate(words):
                lo = words[j - 1][2] + 1 if j else max(0, s - 5)
                hi = words[j + 1][1] if j + 1 < len(words) else min(n, e + 5)
                lo, hi = max(lo, s - 5), min(hi, e + 6)  # never more than 0.4 s around the spikes
                out.append(NS(word=w, start=lo * frame, end=hi * frame))
            return [NS(words=out)], None
        out = [NS(word=w, start=s * frame + self.shift, end=(e + 1) * frame + self.shift) for w, s, e in words]
        return [NS(words=out)], None


def evaluate(ref, events, delay, pb, pa, audio, patterns):
    """Render what you'd hear at this delay and measure each word's remaining loudness."""
    t = np.arange(len(audio)) / SR
    gain = np.ones(len(audio), np.float32)
    for dt, s, e in events:
        s, e = s - pb, e + pa
        lo, hi = max(0, int(s * SR)), min(len(audio), int(e * SR))
        if hi <= lo:
            continue
        # a sample at time x is played at x + delay; the mute only works if detected by then
        ok = t[lo:hi] + delay >= dt
        gain[lo:hi][ok] = 0
    heard = audio * gain

    def kept(s, e):
        a, b = int(s * SR), int(e * SR)
        return np.sqrt(np.mean(heard[a:b] ** 2)) / (np.sqrt(np.mean(audio[a:b] ** 2)) + 1e-9)
    bad = [kept(s, e) for w, s, e in ref if is_profane(w, patterns)]
    clean = [kept(s, e) for w, s, e in ref if not is_profane(w, patterns)]
    return bad, clean


def needed_delay(ref, events, patterns, pb, pa):
    worst = 0.0
    for w, s, e in ref:
        if not is_profane(w, patterns):
            continue
        for x in np.arange(s, e, 0.01):
            cov = [dt for dt, ms, me in events if ms - pb <= x <= me + pa]
            if not cov:
                return float("inf")
            worst = max(worst, min(cov) - x)
    return worst


def main():
    add_cuda_dlls()
    from faster_whisper import WhisperModel
    patterns = load_words(ROOT / "words.txt")
    clips = {k: load16(TEST / f"bench_{k}.wav") for k in ("normal", "fast", "noisy")}
    judge = WhisperModel("small.en", device="cuda", compute_type="float16")
    refs = {k: reference_words(judge, clips["fast" if k == "fast" else "normal"]) for k in clips}

    def parakeet(shift, gap=False):
        def make():
            if "pk" not in _cache:
                _cache["pk"] = Parakeet()
            _cache["pk"].shift, _cache["pk"].gap = shift, gap
            return _cache["pk"]
        return make
    _cache = {}
    candidates = [("whisper small.en", lambda: judge, [0.4], (0.06, 0.18))]
    candidates += [("parakeet-ctc-0.6b", parakeet(0.0), [0.1, 0.2, 0.3], (0.06, 0.18))]
    for pads in [(0.06, 0.18), (0.10, 0.22), (0.15, 0.25)]:
        candidates += [(f"pk +0.17 pad {pads[0]}/{pads[1]}", parakeet(0.17), [0.1, 0.2], pads)]
    for pads in [(0.0, 0.0), (0.04, 0.08)]:
        candidates += [(f"pk gap pad {pads[0]}/{pads[1]}", parakeet(0.0, gap=True), [0.1, 0.2], pads)]
    if len(sys.argv) > 1:
        candidates = [c for c in candidates if any(a in c[0] for a in sys.argv[1:])]

    print(f"{'model':<20}{'step':<6}{'ms/call':<9}{'GPU':<6}{'safe delay':<12}"
          f"{'at delay':<10}{'curse words silent':<20}{'clean words intact'}")
    for name, make, steps, (pb, pa) in candidates:
        model = make()
        # warm-up
        noise = np.random.default_rng(0).normal(0, 0.1, SR * 4).astype(np.float32)
        model.transcribe(noise, language="en", beam_size=1, word_timestamps=True)
        for step in steps:
            all_events, times = {}, []
            for k, a in clips.items():
                ev, tm = simulate(model, a, step, 4.0, patterns, 0, 0)
                all_events[k], times = ev, times + tm
            safe = max(needed_delay(refs[k], all_events[k], patterns, pb, pa) for k in clips)
            delay = round(safe + 0.15, 2)
            silent, total_bad, intact, total_clean = 0, 0, 0, 0
            for k, a in clips.items():
                bad, clean = evaluate(refs[k], all_events[k], delay, pb, pa, a, patterns)
                silent += sum(b < 0.2 for b in bad)
                total_bad += len(bad)
                intact += sum(c > 0.7 for c in clean)
                total_clean += len(clean)
            ms = 1000 * float(np.mean(times))
            print(f"{name:<20}{step:<6}{ms:<9.0f}{ms / 1000 / step:<6.0%}{safe:<12.2f}{delay:<10}"
                  f"{silent}/{total_bad:<18}{intact}/{total_clean} ({intact / total_clean:.0%})", flush=True)


if __name__ == "__main__":
    main()

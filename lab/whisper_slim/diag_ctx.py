"""Why is a short encoder context slower end-to-end? Look at what the decoder produces."""
import time
from bench_slim import ENC_FRAMES, GpuGuard, SR, TEST, WhisperModel, load16

guard = GpuGuard()
audio = load16(TEST / "bench_normal.wav")[SR * 10: SR * 14]
for name in ("tiny.en", "small.en"):
    m = WhisperModel(name, device="cuda", compute_type="float16")
    for frames, kw in [(3000, {}), (400, {}), (400, {"max_new_tokens": 40}),
                       (400, {"without_timestamps": True, "max_new_tokens": 40}),
                       (3000, {"without_timestamps": True})]:
        ENC_FRAMES[0] = frames
        list(m.transcribe(audio, language="en", beam_size=1, word_timestamps=True, **kw)[0])
        guard.wait()
        t0 = time.perf_counter()
        segs = list(m.transcribe(audio, language="en", beam_size=1, temperature=0.0, word_timestamps=True,
                                 condition_on_previous_text=False, **kw)[0])
        ms = 1000 * (time.perf_counter() - t0)
        toks = sum(len(s.tokens) for s in segs)
        text = "".join(s.text for s in segs).strip()
        print(f"{name:<9}{frames:<6}{str(kw):<52}{ms:6.0f} ms {toks:4} tok  {text[:90]}", flush=True)

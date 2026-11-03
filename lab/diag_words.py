"""Per curse word: when was it first detected, relative to when it was spoken? And timestamp error."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import TEST, load16, reference_words, simulate  # noqa: E402
from bench_models import Parakeet  # noqa: E402
from censor import add_cuda_dlls, is_profane, load_words  # noqa: E402

add_cuda_dlls()
from faster_whisper import WhisperModel  # noqa: E402

patterns = load_words(ROOT / "words.txt")
clips = {k: load16(TEST / f"bench_{k}.wav") for k in ("normal", "fast", "noisy")}
judge = WhisperModel("small.en", device="cuda", compute_type="float16")
refs = {k: reference_words(judge, clips["fast" if k == "fast" else "normal"]) for k in clips}
model = Parakeet()
model.shift = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
errs = []
print("clip    word          spoken        first detect  (after word end)  detected as start/end error")
for k, a in clips.items():
    events, _ = simulate(model, a, step, 4.0, patterns, 0, 0)
    for w, s, e in refs[k]:
        if not is_profane(w, patterns):
            continue
        near = [(dt, es, ee) for dt, es, ee in events if es < e + 0.3 and ee > s - 0.3]
        if not near:
            print(f"{k:<8}{w:<14}{s:5.2f}-{e:5.2f}   MISSED")
            continue
        dt, es, ee = min(near)
        errs.append((es - s, ee - e))
        print(f"{k:<8}{w:<14}{s:5.2f}-{e:5.2f}   {dt:6.2f}        {dt - e:+.2f}s            "
              f"{es - s:+.2f} / {ee - e:+.2f}")
errs = np.array(errs)
print(f"start error median {np.median(errs[:,0]):+.3f}  range [{errs[:,0].min():+.2f}, {errs[:,0].max():+.2f}] | end error median {np.median(errs[:,1]):+.3f}  range [{errs[:,1].min():+.2f}, {errs[:,1].max():+.2f}]")

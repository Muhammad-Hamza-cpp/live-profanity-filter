r"""Parakeet gap-fill with the encoder captured as a CUDA graph, so the check can run every few
tens of ms. Isolates how much of the delay is the step itself (no training involved).

    .venv-lab\Scripts\python lab\detector\bench_graph.py 0.03 0.05 0.1
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
from bench import SR, TEST, load16, reference_words, simulate  # noqa: E402
from bench_models import Parakeet, evaluate, needed_delay  # noqa: E402
from censor import add_cuda_dlls, load_words  # noqa: E402

WINDOW = 4.0


class GraphParakeet(Parakeet):
    """Every window is left-padded with silence to exactly WINDOW seconds, so one captured
    graph serves all calls. Word times are moved back by the padding."""

    def __init__(self):
        super().__init__()
        x = np.zeros(int(WINDOW * SR), np.float32)
        self.static = self._features(x)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self.model(**self.static)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.out = self.model(**self.static).logits

    def _features(self, audio):
        inputs = self.proc(audio, sampling_rate=SR, return_tensors="pt")
        return {k: (v.cuda().bfloat16() if v.is_floating_point() else v.cuda()) for k, v in inputs.items()}

    def logits(self, audio):
        for k, v in self._features(audio).items():
            self.static[k].copy_(v)
        self.graph.replay()
        return self.out[0].float().cpu().numpy()

    def transcribe(self, audio, **kw):
        pad = int(WINDOW * SR) - len(audio)
        if pad > 0:
            audio = np.concatenate([np.zeros(pad, np.float32), audio])
        segs, info = super().transcribe(audio, **kw)
        shift = pad / SR if pad > 0 else 0.0
        words = [NS(word=w.word, start=w.start - shift, end=w.end - shift)
                 for w in segs[0].words if w.end - shift > 0]
        return [NS(words=words)], info


def main():
    steps = [float(a) for a in sys.argv[1:]] or [0.03, 0.05, 0.1]
    add_cuda_dlls()
    from faster_whisper import WhisperModel
    patterns = load_words(HERE.parent.parent / "words.txt")
    clips = {k: load16(TEST / f"bench_{k}.wav") for k in ("normal", "fast", "noisy")}
    judge = WhisperModel("small.en", device="cuda", compute_type="float16")
    refs = {k: reference_words(judge, clips["fast" if k == "fast" else "normal"]) for k in clips}
    del judge
    model = GraphParakeet()
    model.gap = True
    pb, pa = 0.04, 0.08
    print(f"{'step':<6}{'ms/call':<9}{'GPU':<6}{'safe delay':<12}{'at delay':<10}"
          f"{'curse silent':<14}clean intact")
    for step in steps:
        events, times = {}, []
        for k, a in clips.items():
            events[k], tm = simulate(model, a, step, WINDOW, patterns, 0, 0)
            times += tm
        safe = max(needed_delay(refs[k], events[k], patterns, pb, pa) for k in clips)
        delay = round(safe + 0.15, 2)
        silent = total = intact = nclean = 0
        for k, a in clips.items():
            bad, clean = evaluate(refs[k], events[k], delay, pb, pa, a, patterns)
            silent += sum(b < 0.2 for b in bad)
            total += len(bad)
            intact += sum(c > 0.7 for c in clean)
            nclean += len(clean)
        ms = 1000 * float(np.mean(times))
        print(f"{step:<6}{ms:<9.1f}{ms / 1000 / step:<6.0%}{safe:<12.2f}{delay:<10}"
              f"{silent}/{total:<12}{intact}/{nclean} ({intact / nclean:.0%})", flush=True)


if __name__ == "__main__":
    main()

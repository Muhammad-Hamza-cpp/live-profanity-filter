r"""Transcribe-then-match baselines on the real-speech test set, scored exactly like the detector.

Each baseline runs the app's live loop on a simulated clock with real GPU timings (bench.simulate):
- whisper: Whisper small.en, step 0.4 s, 4 s window (the app's default engine)
- parakeet: Parakeet CTC 0.6B gap-fill as a CUDA graph, step 0.05 s
A word matching words.txt is muted with the app's pads (0.12 s before, 0.20 s after). Mute events
become a per-10 ms "first muted at" grid, and eval.score() then gives the same numbers as for the
detector: needed delay per curse word, curse words silenced, clean words intact, sound-alikes intact.

    .venv-lab\Scripts\python lab\detector\eval_baselines.py whisper parakeet
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))
sys.path.insert(0, str(HERE))
from eval import DELAYS, GRID, SR, real_streams, score  # noqa: E402

STEPS = {"whisper": 0.4, "parakeet": 0.05}
DELAYS_ALL = (*DELAYS, 0.6, 0.8, 1.2)  # 1.2 s is the app's Whisper delay


def load(name):
    from censor import add_cuda_dlls
    add_cuda_dlls()
    if name == "whisper":
        from faster_whisper import WhisperModel
        return WhisperModel("small.en", device="cuda", compute_type="float16")
    from bench_graph import GraphParakeet
    return GraphParakeet()


def to_grid(events, n):
    g = np.full(n, np.inf)
    for ready, ms, me in events:
        lo, hi = max(0, int(ms / GRID)), min(n, int(me / GRID) + 1)
        g[lo:hi] = np.minimum(g[lo:hi], ready)
    return g


def main():
    from bench import simulate
    from censor import load_words
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=list(STEPS))
    ap.add_argument("--real-file", default="testset.jsonl")
    a = ap.parse_args()
    patterns = load_words(ROOT / "words.txt")
    sets = real_streams(a.real_file)
    for name in a.models:
        model = load(name)
        need, agg, times, dur = [], {d: {} for d in DELAYS_ALL}, [], 0.0
        for audio, words in sets.values():
            events, tm = simulate(model, audio, STEPS[name], 4.0, patterns)
            times += tm
            r = score(audio, words, to_grid(events, int(len(audio) / SR / GRID) + 1), DELAYS_ALL)
            need += r["need"]
            dur += len(audio) / SR
            for d in DELAYS_ALL:
                for k, v in r[d].items():
                    agg[d][k] = agg[d].get(k, 0) + v
        need = np.array(need)
        fin = need[np.isfinite(need)]
        print(f"\n=== {name}: step {STEPS[name]} s, {1000 * np.mean(times):.0f} ms per call "
              f"(p99 {1000 * np.percentile(times, 99):.0f}) ===")
        print(f"[real] {len(need)} curse words, missed {int(np.sum(~np.isfinite(need)))} | needed delay: "
              f"median {np.median(fin):.2f}  p95 {np.percentile(fin, 95):.2f}  max {fin.max():.2f}")
        for d in DELAYS_ALL:
            x = agg[d]
            print(f"   at {d:.2f} s: curse silent {x['silent']}/{x['curse']}  "
                  f"clean intact {x['intact']}/{x['clean']} ({x['intact'] / max(x['clean'], 1):.1%})  "
                  f"sound-alikes intact {x['neg_intact']}/{x['neg']}  muted {x['muted_s'] / dur * 60:.1f} s/min")
        del model


if __name__ == "__main__":
    main()

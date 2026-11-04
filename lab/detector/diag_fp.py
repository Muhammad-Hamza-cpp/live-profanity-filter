r"""Where do the false mutes come from? For muted audio outside any curse word: how soon after it
was heard was it muted (edge guesses vs. late flicker), and which words get hit.

    .venv-lab\Scripts\python lab\detector\diag_fp.py runs\v1\head.pt --thr 0.7
"""
import argparse
import collections
from pathlib import Path

import numpy as np

from eval import GRID, HERE, _norm, first_mute, run, tts_streams
from detector import Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("head")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    sets = tts_streams()
    det = Detector(HERE / args.head)
    ages, hit_words, after = [], collections.Counter(), collections.Counter()
    fp_s = tot_s = 0.0
    for name, (audio, words) in sets.items():
        checks, _ = run(det, audio, args.step)
        n = int(len(audio) / 16000 / GRID) + 1
        g = first_mute(checks, n, args.thr, 0, 0)
        x = np.arange(n) * GRID
        curse = np.zeros(n, bool)
        for w, c, s, e in words:
            if c:
                curse[max(0, int((s - 0.1) / GRID)):int((e + 0.1) / GRID) + 1] = True
        fp = np.isfinite(g) & ~curse
        ages += list(g[fp] - x[fp])
        fp_s += fp.sum() * GRID
        tot_s += n * GRID
        for i, (w, c, s, e) in enumerate(words):
            a, b = int(s / GRID), max(int(s / GRID) + 1, int(e / GRID))
            if not c and fp[a:b].mean() > 0.3:
                hit_words[_norm(w)] += 1
                if i:
                    after[_norm(words[i - 1][0])] += 1
    ages = np.array(ages)
    print(f"false-muted (>0.1 s from any curse word): {fp_s / tot_s * 60:.1f} s/min")
    print("how soon after being heard each false-muted 10 ms was muted:")
    for lo, hi in [(-1, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 5)]:
        print(f"   {lo:>4}-{hi:<4} s: {np.mean((ages >= lo) & (ages < hi)):.0%}")
    print("most false-muted words:", hit_words.most_common(20))
    print("word before them:      ", after.most_common(15))


if __name__ == "__main__":
    main()

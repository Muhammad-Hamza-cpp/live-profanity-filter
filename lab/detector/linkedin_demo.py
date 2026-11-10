r"""Run detector v4 live on one sentence and save what it saw, for the LinkedIn explainer figure.

A held-out Piper voice (en_US-ryan-high, never used in training) says a game-style line. The
sentence is replayed through the same live loop as eval.py (a check every 50 ms, 13 ms per check,
2 checks in a row above 0.7, pads 0.06/0.02 s) and the result is saved to
data/linkedin_demo.npz for lab/detector/figures_linkedin.py.

    .venv-lab\Scripts\python lab\detector\linkedin_demo.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from detector import FRAME, SR, Detector  # noqa: E402
from eval import GRID, first_mute, run  # noqa: E402
from make_tts import load_voice, synth  # noqa: E402

VOICE = "en_US-ryan-high"
WORDS = [("Reload!", None), ("Oh", None), ("shit,", "shit"), ("they're", None), ("right", None),
         ("behind", None), ("us,", None), ("move!", None)]
DELAY, THR, PADS, CONFIRM, STEP, CHECK = 0.40, 0.7, (0.06, 0.02), 2, 0.05, 0.013
LEAD, TAIL = 0.6, 0.6  # silence around the sentence (s)


class FixedRng:
    """synth() draws speaking rate and expressiveness at random; use the middle of each range."""
    def uniform(self, lo, hi):
        return (lo + hi) / 2


def main():
    out = synth(load_voice(VOICE), WORDS, 0, FixedRng())
    if out is None:
        sys.exit("espeak's word split didn't line up with WORDS")
    speech, spans = out
    lead = int(LEAD * SR)
    audio = np.concatenate([np.zeros(lead, np.float32), speech, np.zeros(int(TAIL * SR), np.float32)])
    words = [(w.strip(",.!?"), g is not None, (s + lead) / SR, (e + lead) / SR) for w, g, s, e in spans]

    det = Detector(ROOT / "lab" / "detector" / "models" / "detector-v4.pt")
    checks, _ = run(det, audio, STEP, check_s=CHECK)
    n_grid = int(len(audio) / SR / GRID) + 1
    g = first_mute(checks, n_grid, THR, *PADS, confirm=CONFIRM)

    # What the model believed about each 80 ms frame at the last check before that frame had to play.
    n_frames = int(len(audio) / SR / FRAME)
    belief = np.zeros(n_frames)
    for ready, w0, p, _ in checks:
        pm = p.max(-1)
        for f in range(len(pm)):
            s = w0 + f * FRAME
            k = int(round(s / FRAME))
            if 0 <= k < n_frames and s < ready <= s + DELAY:
                belief[k] = pm[f]  # checks are in time order, so the latest one wins

    t = np.arange(n_grid) * GRID
    muted = g <= t + DELAY  # muted before it plays at this delay
    for w, c, s, e in words:
        sel = (t >= s) & (t < e)
        print(f"{w:>8} {'CURSE' if c else '':5} {s:5.2f}-{e:5.2f}s  muted {muted[sel].mean():5.0%}"
              f"  decided {np.min(g[sel]) - s:+.2f}s after onset")
    # The model's full output (frames x 41 groups) at the first check that heard all of the curse word.
    end_c = max(e for _, c, _, e in words if c)
    ready, w0, window_probs, _ = next(c for c in checks if c[0] >= end_c)
    np.savez(ROOT / "data" / "linkedin_demo.npz", audio=audio, muted=muted, belief=belief,
             window_probs=window_probs, window_start=w0, window_ready=ready,
             words=np.array([(w, int(c), s, e) for w, c, s, e in words], dtype=object),
             first_mute=g, delay=DELAY, thr=THR)
    print("saved data/linkedin_demo.npz")


if __name__ == "__main__":
    main()

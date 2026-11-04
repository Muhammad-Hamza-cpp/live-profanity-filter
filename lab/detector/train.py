r"""Train the detector head on the Piper data (encoder frozen).

Each example is a WINDOW-second slice of a stream of 2-3 utterances that ends "now"; labels say,
for every 80 ms frame in it, which curse group (if any) that frame is inside. Windows are often
cut to end partway through a curse word, because that is where the decision has to be made early.

    .venv-lab\Scripts\python lab\detector\train.py --name v1 --steps 6000
"""
import argparse
import json
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, fftconvolve, sosfilt

from detector import FRAME, GROUP_NAMES, SR, WINDOW, Head, encode, features, load_encoder

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "tts"
RUNS = HERE / "runs"
GIDX = {g: i for i, g in enumerate(GROUP_NAMES)}
N = int(WINDOW * SR)
MIN_OVERLAP = 0.02  # a frame counts as inside a word if it overlaps it by at least 20 ms


class Data:
    def __init__(self, split):
        audios, self.utts = [], []
        pos = 0
        for p in sorted(p for s in split.split(",") for p in (DATA / s).glob("shard_*.json")):
            z = np.load(p.with_suffix(".npz"))
            a, off = z["audio"], z["offsets"]
            for i, m in enumerate(json.loads(p.read_text())):
                words = [(GIDX[g] if g else -1, s, e) for _, g, s, e in m["words"]]
                self.utts.append((pos + off[i], off[i + 1] - off[i], words))
            audios.append(a)
            pos += len(a)
        self.audio = np.concatenate(audios)
        self.clean = [i for i, u in enumerate(self.utts) if all(g < 0 for g, _, _ in u[2])]
        self.curse = [i for i, u in enumerate(self.utts) if any(g >= 0 for g, _, _ in u[2])]
        print(f"{split}: {len(self.utts)} utterances, {len(self.audio) / SR / 3600:.1f} h, "
              f"{len(self.curse)} with curse words", flush=True)

    def get(self, i):
        s, n, words = self.utts[i]
        return self.audio[s:s + n].astype(np.float32) / 32767, words


# ---------- augmentation ----------

def colored_noise(n, rng):
    x = rng.normal(0, 1, n)
    kind = rng.integers(3)
    if kind == 1:   # pink-ish
        x = sosfilt(butter(1, 0.05, output="sos"), x) + 0.3 * x
    elif kind == 2:  # brown / rumble
        x = np.cumsum(x)
        x -= np.convolve(x, np.ones(400) / 400, mode="same")
    return x


def chords(n, rng):
    t = np.arange(n) / SR
    out = np.zeros(n)
    seg = int(rng.uniform(0.2, 0.8) * SR)
    for i in range(0, n, seg):
        f = rng.uniform(80, 600)
        sl = slice(i, min(i + seg, n))
        out[sl] = sum(np.sin(2 * np.pi * f * k * t[sl] + rng.uniform(0, 6)) / k for k in (1, 1.5, 2, 3))
    return out


def augment(x, data, rng):
    if rng.random() < 0.6:
        kind = rng.choice(["noise", "music", "babble"])
        if kind == "noise":
            bg = colored_noise(len(x), rng)
        elif kind == "music":
            bg = chords(len(x), rng) + 0.3 * colored_noise(len(x), rng)
        else:
            bg = np.zeros(len(x))
            for _ in range(rng.integers(1, 4)):
                b, _ = data.get(data.clean[rng.integers(len(data.clean))])
                o = rng.integers(-len(b) // 2, len(x))
                lo, hi = max(o, 0), min(o + len(b), len(x))
                if hi > lo:
                    bg[lo:hi] += b[lo - o:hi - o]
        rms = np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2)) if np.any(np.abs(x) > 0.01) else 0.1
        snr = rng.uniform(0, 25) if kind != "babble" else rng.uniform(6, 25)
        bg *= rms / (bg.std() + 1e-9) / 10 ** (snr / 20)
        x = x + bg
    if rng.random() < 0.25:  # room reverb
        rt60 = rng.uniform(0.15, 0.7)
        L = int(rt60 * SR)
        rir = rng.normal(0, 1, L) * np.exp(-6.9 * np.arange(L) / L)
        rir[0] = 1.0
        x = fftconvolve(x, rir / np.abs(rir).sum() * 3)[:len(x)]
    if rng.random() < 0.15:  # phone / cheap mic
        x = sosfilt(butter(4, [300, 3400], "bandpass", fs=SR, output="sos"), x)
    x = x / (np.abs(x).max() + 1e-9) * rng.uniform(0.1, 0.9)
    return x.astype(np.float32)


# ---------- examples ----------

def example(data, rng, aug=True):
    """One window and its (frames, groups) labels + loss mask."""
    ids = [data.curse[rng.integers(len(data.curse))] if rng.random() < 0.7 else rng.integers(len(data.utts))]
    ids = [rng.integers(len(data.utts))] * rng.integers(0, 2) + ids + [rng.integers(len(data.utts))]
    parts, words, pos = [], [], int(rng.uniform(0, 0.5) * SR)
    parts.append(np.zeros(pos, np.float32))
    for i in ids:
        a, ws = data.get(i)
        words += [(g, pos + s, pos + e) for g, s, e in ws]
        parts.append(a)
        gap = int(rng.uniform(0.0, 0.6) * SR)
        parts.append(np.zeros(gap, np.float32))
        pos += len(a) + gap
    x = np.concatenate(parts)
    curse = [(s, e) for g, s, e in words if g >= 0]
    if curse and rng.random() < 0.6:  # end the window around/inside a curse word
        s, e = curse[rng.integers(len(curse))]
        end = s + int(rng.uniform(-0.1, (e - s) / SR + 0.5) * SR)
    else:
        end = rng.integers(int(0.5 * SR), len(x) + 1)
    end = int(np.clip(end, int(0.3 * SR), len(x)))
    if aug:
        x = augment(x, data, rng)
    start = end - N
    w = x[max(start, 0):end]
    if start < 0:
        w = np.concatenate([np.zeros(-start, np.float32), w])
    return w, words, start, end


def labels(words, start, end, frames):
    y = np.zeros((frames, len(GROUP_NAMES)), np.float32)
    t0 = start + np.arange(frames) * FRAME * SR
    t1 = t0 + FRAME * SR
    mask = ((t0 < end) & (t1 > max(start, 0))).astype(np.float32)  # heard, and not left padding
    for g, s, e in words:
        if g < 0 or e <= start or s >= end:
            continue
        ov = np.minimum(t1, e) - np.maximum(t0, s)
        # Only what has been heard counts: a frame is a positive once 20 ms of the curse word in
        # it has been heard. Before that it is left out of the loss - neither "curse" (that taught
        # v1 to guess from context like "what the ...") nor "clean" (it may well be a curse word).
        heard = np.minimum(np.minimum(t1, e), end) - np.maximum(t0, s)
        y[heard >= MIN_OVERLAP * SR, g] = 1
        mask[(ov >= MIN_OVERLAP * SR) & (heard < MIN_OVERLAP * SR)] = 0
    return y, mask


def producer(data, q, seed, batch, aug):
    rng = np.random.default_rng(seed)
    while True:
        q.put([example(data, rng, aug) for _ in range(batch)])


def run_batch(fe, encoder, head, items, device="cuda"):
    feats, amask = features(fe, [w for w, _, _, _ in items], device)
    with torch.no_grad():
        states = encode(encoder, feats, amask)
    logits = head(states)
    frames = logits.shape[1]
    ys, ms = zip(*[labels(ws, s, e, frames) for _, ws, s, e in items])
    y = torch.from_numpy(np.stack(ys)).to(device)
    m = torch.from_numpy(np.stack(ms)).to(device)
    return logits, y, m


def frame_metrics(logits, y, m, thr=0.5):
    p = torch.sigmoid(logits).amax(-1)
    t = y.amax(-1)
    keep = m > 0
    pred, true = (p > thr)[keep], (t > 0)[keep]
    tp = (pred & true).sum().item()
    return tp, pred.sum().item(), true.sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--train", default="train", help="comma-separated data splits")
    args = ap.parse_args()

    torch.manual_seed(0)
    train, test = Data(args.train), Data("test")
    fe, encoder = load_encoder()
    head = Head().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    pos_weight = torch.full((len(GROUP_NAMES),), args.pos_weight, device="cuda")

    q = queue.Queue(maxsize=16)
    for i in range(args.threads):
        threading.Thread(target=producer, args=(train, q, i, args.batch, True), daemon=True).start()
    rng = np.random.default_rng(12345)
    val = [[example(test, rng, aug=rng.random() < 0.5) for _ in range(args.batch)] for _ in range(20)]

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")
    t0, best = time.time(), 0.0
    for step in range(1, args.steps + 1):
        head.train()
        logits, y, m = run_batch(fe, encoder, head, q.get())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight, reduction="none")
        loss = (loss.mean(-1) * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            msg = f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s q={q.qsize()}"
            print(msg, flush=True)
            log.write(msg + "\n")
        if step % 1000 == 0 or step == args.steps:
            head.eval()
            tp = npred = ntrue = 0
            vloss = 0.0
            with torch.no_grad():
                for items in val:
                    lg, yy, mm = run_batch(fe, encoder, head, items)
                    a, b, c = frame_metrics(lg, yy, mm)
                    tp, npred, ntrue = tp + a, npred + b, ntrue + c
            prec, rec = tp / max(npred, 1), tp / max(ntrue, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            msg = f"  val frames: precision {prec:.3f} recall {rec:.3f} F1 {f1:.3f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            ckpt = {"head": head.state_dict(), "groups": GROUP_NAMES, "step": step, "args": vars(args)}
            torch.save(ckpt, out / "last.pt")
            if f1 > best:
                best = f1
                torch.save(ckpt, out / "head.pt")


if __name__ == "__main__":
    main()

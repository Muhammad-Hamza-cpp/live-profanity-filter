r"""Train the detector head on the Piper data (encoder frozen), optionally mixed with real speech.

Each example is a WINDOW-second slice of a stream of 2-3 utterances that ends "now"; labels say,
for every 80 ms frame in it, which curse group (if any) that frame is inside. Windows are often
cut to end partway through a curse word, because that is where the decision has to be made early.

Real speech (--real N): YODAS2 clips from build_trainset.py, about N curse words' worth of whole
videos (-1 = all). A real window comes from one clip; only frames inside its transcript line
(+-0.5 s) are in the loss, because the context around it is unlabelled. When valset.jsonl exists,
the best checkpoint is picked on real validation speech, never on the test set.

    .venv-lab\Scripts\python lab\detector\train.py --name v1 --steps 6000
    python lab/detector/train.py --name real2000-s1 --train train,train_ctx --real 2000 --seed 1
Ablations: --label-unheard (v1's labels), --no-sfx, and --train train (no context negatives).
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
REAL = HERE.parent.parent / "data" / "realspeech" / "yodas"
LINE_MARGIN = 0.5   # words were aligned within the transcript line +- this; labels are complete there
LABEL_UNHEARD = False  # ablation: label a curse word's frames before they are heard (v1)


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


class RealData:
    """Real-speech clips (rows of trainset.jsonl / valset.jsonl). n_curse: take whole videos in
    file order until they hold this many curse words (-1 = all, 0 = none)."""

    def __init__(self, path, n_curse=-1):
        import soundfile as sf
        self.clips = []
        if n_curse == 0 or not path.exists():
            return
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        n, videos = 0, set()
        for r in rows:
            if n_curse > 0 and n >= n_curse and r["src"] not in videos:
                break
            videos.add(r["src"])
            a, _ = sf.read(REAL / r["clip"], dtype="float32")
            words = [(GIDX[c["group"]], int(r["words"][c["i"]]["start"] * SR), int(r["words"][c["i"]]["end"] * SR))
                     for c in r["curses"]]
            valid = (int((r["line"][0] - LINE_MARGIN) * SR), int((r["line"][1] + LINE_MARGIN) * SR))
            self.clips.append((a, words, valid))
            n += len(words)
        self.curse = [i for i, c in enumerate(self.clips) if c[1]]
        print(f"{path.name}: {len(self.clips)} clips from {len(videos)} videos, {n} curse words", flush=True)

    def example(self, rng, aug, tts):
        """Like example(), from one real clip; tts supplies babble for augmentation."""
        i = self.curse[rng.integers(len(self.curse))] if self.curse and rng.random() < 0.7 else rng.integers(len(self.clips))
        a, words, (vs, ve) = self.clips[i]
        if words and rng.random() < 0.6:
            _, s, e = words[rng.integers(len(words))]
            end = s + int(rng.uniform(-0.1, (e - s) / SR + 0.5) * SR)
        else:
            end = int(rng.uniform(max(vs, 0) + 0.3 * SR, ve + 0.5 * SR))
        end = int(np.clip(end, int(0.3 * SR), len(a)))
        x = augment(a, tts, rng) if aug else a
        start = end - N
        w = x[max(start, 0):end]
        if start < 0:
            w = np.concatenate([np.zeros(-start, np.float32), w])
        return w, words, start, end, (vs, ve)


class Sounds:
    """Non-speech sound effects (ESC-50: engines, fireworks, sirens, glass, thunder ...). Folds 1-4
    are for training; fold 5 is kept for eval.py. The detector has to learn these aren't speech."""

    def __init__(self, folds):
        import csv
        import soundfile as sf
        from scipy.signal import resample_poly
        root = HERE.parent.parent / "data" / "esc50_dl" / "ESC-50-master"
        self.clips = []
        if not root.exists():
            print("ESC-50 not found - training without sound effects", flush=True)
            return
        for r in csv.DictReader(open(root / "meta" / "esc50.csv")):
            if int(r["fold"]) in folds:
                a, sr = sf.read(root / "audio" / r["filename"], dtype="float32")
                a = resample_poly(a, SR, sr).astype(np.float32)
                nz = np.nonzero(np.abs(a) > 1e-3)[0]  # many clips are padded with digital silence
                if len(nz):
                    self.clips.append(a[nz[0]:nz[-1] + 1] / (np.abs(a).max() + 1e-9))
        print(f"sound effects: {len(self.clips)} clips from ESC-50 folds {folds}", flush=True)

    def stream(self, n, rng):
        """n samples of back-to-back random clips, sometimes with gaps."""
        out, pos = np.zeros(n, np.float32), 0
        while self.clips and pos < n:
            c = self.clips[rng.integers(len(self.clips))] * rng.uniform(0.2, 1.0)
            k = min(len(c), n - pos)
            out[pos:pos + k] = c[:k]
            pos += k + int(rng.uniform(0, 0.5) * SR) * (rng.random() < 0.5)
        return out


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
        kind = rng.choice(["noise", "music", "babble", "sfx", "sfx"] if data.sounds.clips else ["noise", "music", "babble"])
        if kind == "sfx":
            bg = data.sounds.stream(len(x), rng).astype(np.float64)
        elif kind == "noise":
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
        snr = {"babble": rng.uniform(6, 25), "sfx": rng.uniform(-5, 20)}.get(kind, rng.uniform(0, 25))
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
    """One window: (audio, curse words, start, end, labelled span or None)."""
    if data.real.clips and rng.random() < data.real_p:
        return data.real.example(rng, aug, data)
    if data.sounds.clips and rng.random() < 0.15:  # no speech at all: nothing may be muted
        x = data.sounds.stream(N + int(rng.uniform(0, 2) * SR), rng)
        if aug:
            x = augment(x, data, rng)
        return x[-N:], [], len(x) - N, len(x), None
    ids = [data.curse[rng.integers(len(data.curse))] if rng.random() < 0.7 else rng.integers(len(data.utts))]
    ids = [rng.integers(len(data.utts))] * rng.integers(0, 2) + ids + [rng.integers(len(data.utts))]
    parts, words, pos = [], [], int(rng.uniform(0, 0.5) * SR)
    parts.append(np.zeros(pos, np.float32))
    for i in ids:
        a, ws = data.get(i)
        words += [(g, pos + s, pos + e) for g, s, e in ws]
        parts.append(a)
        gap = int(rng.uniform(0.0, 0.6) * SR)
        if data.sounds.clips and rng.random() < 0.3:  # a sound effect between sentences
            gap = int(rng.uniform(0.2, 1.2) * SR)
            parts.append(data.sounds.stream(gap, rng) * rng.uniform(0.1, 0.8))
        else:
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
    return w, words, start, end, None


def labels(words, start, end, frames, valid=None):
    """valid: (start, end) sample span where the labels are complete (real clips); frames outside
    it are left out of the loss."""
    y = np.zeros((frames, len(GROUP_NAMES)), np.float32)
    t0 = start + np.arange(frames) * FRAME * SR
    t1 = t0 + FRAME * SR
    mask = ((t0 < end) & (t1 > max(start, 0))).astype(np.float32)  # heard, and not left padding
    if valid is not None:
        mask *= (t0 >= valid[0]) & (t1 <= valid[1])
    for g, s, e in words:
        if g < 0 or e <= start or s >= end:
            continue
        ov = np.minimum(t1, e) - np.maximum(t0, s)
        if LABEL_UNHEARD:  # ablation: v1's labels, the whole word even where not yet heard
            y[ov >= MIN_OVERLAP * SR, g] = 1
            continue
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
    feats, amask = features(fe, [w for w, *_ in items], device)
    with torch.no_grad():
        states = encode(encoder, feats, amask)
    logits = head(states)
    frames = logits.shape[1]
    ys, ms = zip(*[labels(ws, s, e, frames, v) for _, ws, s, e, v in items])
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", type=int, default=0, help="real curse words to train with (whole videos); -1 = all")
    ap.add_argument("--real-p", type=float, default=0.3, help="share of training windows from real clips")
    ap.add_argument("--label-unheard", action="store_true", help="ablation: v1's labels")
    ap.add_argument("--no-sfx", action="store_true", help="ablation: no sound-effect negatives")
    args = ap.parse_args()

    global LABEL_UNHEARD
    LABEL_UNHEARD = args.label_unheard
    torch.manual_seed(args.seed)
    train, test = Data(args.train), Data("test")
    train.sounds, test.sounds = Sounds(set() if args.no_sfx else {1, 2, 3, 4}), Sounds({5})
    train.real, train.real_p = RealData(REAL / "trainset.jsonl", args.real), args.real_p
    test.real, test.real_p = RealData(REAL / "valset.jsonl"), 1.0  # real validation, if there is one
    fe, encoder = load_encoder()
    head = Head().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    pos_weight = torch.full((len(GROUP_NAMES),), args.pos_weight, device="cuda")

    q = queue.Queue(maxsize=16)
    for i in range(args.threads):
        threading.Thread(target=producer, args=(train, q, args.seed * 100 + i, args.batch, True), daemon=True).start()
    rng = np.random.default_rng(12345)
    val = [[example(test, rng, aug=rng.random() < 0.5) for _ in range(args.batch)] for _ in range(20)]

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")
    t0, best, first = time.time(), 0.0, 1
    if (out / "last.pt").exists():  # resume after a crash (the data stream restarts, not resumes)
        ckpt = torch.load(out / "last.pt", map_location="cuda")
        if "opt" in ckpt:
            head.load_state_dict(ckpt["head"])
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            first, best = ckpt["step"] + 1, ckpt.get("best", 0.0)
            print(f"resumed {args.name} at step {ckpt['step']}", flush=True)
            log.write(f"resumed at step {ckpt['step']}\n")
    for step in range(first, args.steps + 1):
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
            log.flush()
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
            if f1 > best:
                best = f1
                torch.save({"head": head.state_dict(), "groups": GROUP_NAMES, "step": step, "args": vars(args)},
                           out / "head.pt")
            ckpt = {"head": head.state_dict(), "groups": GROUP_NAMES, "step": step, "args": vars(args),
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best}
            torch.save(ckpt, out / "last.tmp")
            (out / "last.tmp").replace(out / "last.pt")  # never leave a half-written checkpoint


if __name__ == "__main__":
    main()

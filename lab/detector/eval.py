r"""Replay audio through the detector on a simulated clock (real GPU timings) and score it.

For every curse word: the delay it needs to be silent before it plays. At fixed delays: curse
words silenced, clean words left intact, and sound-alike words ("duck", "count", ...) left intact.
Loudness-based, never re-transcribed (see docs/low-latency-research.md).

    .venv-lab\Scripts\python lab\detector\eval.py runs\v1\head.pt
    .venv-lab\Scripts\python lab\detector\eval.py runs\v1\head.pt --thr 0.3 0.5 0.7 --step 0.05
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
sys.path.insert(0, str(HERE))
from detector import FRAME, SR, WINDOW, Detector  # noqa: E402
from lexicon import NEGATIVES  # noqa: E402

GRID = 0.01  # seconds
DELAYS = (0.25, 0.30, 0.35, 0.40)
NEG = {w.lower() for w in NEGATIVES if " " not in w}


def run(det, audio, step, check_s=None):
    """Probabilities seen at every check: list of (time ready, window start, probs).
    check_s: use this fixed time per check instead of the measured one, so results don't depend
    on the GPU the evaluation happens to run on (13 ms = the RTX 4050 laptop)."""
    out, t, times = [], step, []
    while t * SR <= len(audio):
        end = int(t * SR)
        t0 = time.perf_counter()
        p, speech = det.score(audio[max(0, end - int(WINDOW * SR)):end])
        took = check_s if check_s is not None else time.perf_counter() - t0
        times.append(took)
        out.append((t + took, t - WINDOW, p, speech))
        t = max(t + step, t + took)
    return out, times


def run_batched(det, audio, step, check_s, batch=32):
    """Same checks as run() with a fixed check time, many windows per GPU call. With the check
    time fixed, check k happens at step + k * max(step, check_s) whatever the GPU, so the windows
    don't depend on each other. ~20x faster; for scoring many models."""
    import torch
    from detector import encode, features
    n, ts, t = int(WINDOW * SR), [], step
    while t * SR <= len(audio):  # the same float steps as run(): a 1-sample shift changes the frames
        ts.append(t)
        t = max(t + step, t + check_s)
    out = []
    for i in range(0, len(ts), batch):
        ws = []
        for t in ts[i:i + batch]:
            w = audio[max(0, int(t * SR) - n):int(t * SR)].astype(np.float32)
            ws.append(np.concatenate([np.zeros(n - len(w), np.float32), w]) if len(w) < n else w)
        f, m = features(det.fe, ws, det.device)
        with torch.inference_mode():
            states, last = encode(det.encoder, f, m, last=True)
            p = torch.sigmoid(det.head(states)).float().cpu().numpy()
            speech = (1 - torch.softmax(det.ctc(last).float(), -1)[..., det.blank]).cpu().numpy()
        out += [(t + check_s, t - WINDOW, pp, ss) for t, pp, ss in zip(ts[i:i + batch], p, speech)]
    return out, [check_s] * len(out)


def first_mute(checks, n_grid, thr, pad_before, pad_after, confirm=1, gate=None):
    """For each 10 ms grid point: earliest time a check decided to mute it (inf = never).
    confirm=K: only once K checks in a row put it above the threshold (filters one-check spikes).
    gate=(k, s): only frames with CTC speech probability > s within k frames count (speech gate)."""
    g = np.full(n_grid, np.inf)
    run_len = np.zeros(n_grid, np.int32)
    for ready, w0, p, speech in checks:
        above = np.zeros(n_grid, bool)
        ok = p.max(-1) > thr
        if gate:
            k, s_thr = gate
            near = np.convolve((speech > s_thr).astype(float), np.ones(2 * k + 1), mode="same") > 0
            ok &= near
        for f in np.nonzero(ok)[0]:
            s = w0 + f * FRAME
            if s >= ready:  # frame not heard yet
                continue
            lo = max(0, int((s - pad_before) / GRID))
            hi = min(n_grid, int((s + FRAME + pad_after) / GRID) + 1)
            above[lo:hi] = True
        seen = slice(max(0, int((w0 - pad_before) / GRID)), min(n_grid, int((ready + pad_after) / GRID) + 1))
        run_len[seen] = np.where(above[seen], run_len[seen] + 1, 0)
        hit = run_len >= confirm
        g[hit] = np.minimum(g[hit], ready)
    return g


def score(audio, words, g, delays=DELAYS):
    """words: [(word, is_curse, start_s, end_s)] -> per-curse needed delay, intact fractions."""
    x = np.arange(len(g)) * GRID
    need = []
    for w, curse, s, e in words:
        if curse:
            i = np.arange(int(s / GRID), max(int(s / GRID) + 1, int(e / GRID)))
            i = i[i < len(g)]
            need.append(float(np.max(g[i] - x[i])) if len(i) else np.inf)
    res = {"need": need}
    for d in delays:
        muted = g <= x + d
        gain = np.repeat(~muted, int(GRID * SR)).astype(np.float32)
        gain = np.pad(gain, (0, max(0, len(audio) - len(gain))), constant_values=1)[:len(audio)]
        heard = audio * gain

        def kept(s, e):
            a, b = int(s * SR), max(int(s * SR) + 1, int(e * SR))
            return np.sqrt(np.mean(heard[a:b] ** 2)) / (np.sqrt(np.mean(audio[a:b] ** 2)) + 1e-9)
        k = [(w, curse, kept(s, e)) for w, curse, s, e in words]
        res[d] = {
            "silent": sum(v < 0.2 for _, c, v in k if c), "curse": sum(c for _, c, _ in k),
            "intact": sum(v > 0.7 for _, c, v in k if not c), "clean": sum(not c for _, c, _ in k),
            "neg_intact": sum(v > 0.7 for w, c, v in k if not c and _norm(w) in NEG),
            "neg": sum(1 for w, c, _ in k if not c and _norm(w) in NEG),
            "muted_s": float(np.sum(muted)) * GRID,
        }
    return res


def _norm(w):
    return "".join(ch for ch in w.lower() if ch.isalpha() or ch == "'").strip("'")


def bench_clips():
    """The three bench.py clips; reference words from Whisper small.en like bench.py does."""
    from bench import TEST, load16, reference_words
    from censor import add_cuda_dlls, is_profane, load_words
    add_cuda_dlls()
    from faster_whisper import WhisperModel
    patterns = load_words(ROOT / "words.txt")
    clips = {k: load16(TEST / f"bench_{k}.wav") for k in ("normal", "fast", "noisy")}
    judge = WhisperModel("small.en", device="cuda", compute_type="float16")
    out = {}
    for k, a in clips.items():
        ref = reference_words(judge, clips["fast" if k == "fast" else "normal"])
        out[f"bench-{k}"] = (np.concatenate([a, np.zeros(SR, np.float32)]),
                             [(w, is_profane(w, patterns), s, e) for w, s, e in ref])
    del judge
    return out


def esc_streams(n_clips=300, seed=0):
    """ESC-50 fold 5 (held out from training): sirens, engines, fireworks ... joined into one stream."""
    import csv
    import soundfile as sf
    from scipy.signal import resample_poly
    root = ROOT / "data" / "esc50_dl" / "ESC-50-master"
    rows = [r for r in csv.DictReader(open(root / "meta" / "esc50.csv")) if r["fold"] == "5"]  # 1-4 train
    rng = np.random.default_rng(seed)
    parts = []
    for r in rng.choice(rows, n_clips, replace=False):
        a, sr = sf.read(root / "audio" / r["filename"], dtype="float32")
        a = resample_poly(a, SR, sr).astype(np.float32)
        parts += [a / (np.abs(a).max() + 1e-9) * 0.5, np.zeros(int(0.3 * SR), np.float32)]
    return {"esc-0": (np.concatenate(parts), [])}


def tts_streams(n_streams=4, per_stream=120, seed=0, split="test", prefix="tts"):
    """Held-out Piper voices/speakers, utterances joined with pauses into long streams."""
    data = sorted((ROOT / "data" / "tts" / split).glob("shard_*.json"))
    utts = []
    for p in data:
        z = np.load(p.with_suffix(".npz"))
        audio, off = z["audio"], z["offsets"]  # each z[...] access re-reads the whole array
        for i, m in enumerate(json.loads(p.read_text())):
            utts.append((audio[off[i]:off[i + 1]].copy(), m))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(utts))
    out = {}
    for s in range(n_streams):
        parts, words, pos = [], [], 0
        for j in order[s * per_stream:(s + 1) * per_stream]:
            a, m = utts[j]
            words += [(w, g is not None, (pos + a0) / SR, (pos + a1) / SR) for w, g, a0, a1 in m["words"]]
            gap = int(rng.uniform(0.2, 1.0) * SR)
            parts += [a.astype(np.float32) / 32767, np.zeros(gap, np.float32)]
            pos += len(a) + gap
        out[f"{prefix}-{s}"] = (np.concatenate(parts), words)
    return out


def real_streams(name="testset.jsonl", per_stream=100):
    """Real speech: YODAS2 (CC BY) clips from build_testset.py (or valset.jsonl from
    build_trainset.py, for tuning), joined with 0.5 s gaps. Each clip starts 4 s before its
    transcript line, so the detector's window holds real preceding speech. Only words inside the
    line are scored; the context around it has no labels."""
    import soundfile as sf
    path = ROOT / "data" / "realspeech" / "yodas" / name
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    gap = np.zeros(int(0.5 * SR), np.float32)
    out = {}
    for s in range(0, len(rows), per_stream):
        parts, words, pos = [], [], 0
        for r in rows[s:s + per_stream]:
            a, _ = sf.read(path.parent / r["clip"], dtype="float32")
            words += [(w, kind == "curse", pos / SR + st, pos / SR + e) for w, kind, st, e in r["labels"]]
            parts += [a, gap]
            pos += len(a) + len(gap)
        out[f"real-{s // per_stream}"] = (np.concatenate(parts), words)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("head")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--pad", type=float, nargs=2, default=[0.0, 0.0], metavar=("BEFORE", "AFTER"))
    ap.add_argument("--sets", default="bench,tts,ctx")
    ap.add_argument("--real-file", default="testset.jsonl", help="valset.jsonl to tune, testset.jsonl to report")
    ap.add_argument("--check-ms", type=float, help="fixed time per check (e.g. 13 = the laptop) instead of measured")
    ap.add_argument("--confirm", type=int, nargs="+", default=[1], help="checks in a row needed to mute")
    ap.add_argument("--gate", nargs="+", default=["none"],
                    help="speech gates to try: none, or K:S (CTC speech > S within K frames), e.g. 3:0.5")
    args = ap.parse_args()

    sets = {}
    if "bench" in args.sets:
        sets.update(bench_clips())
    if "tts" in args.sets:
        sets.update(tts_streams())
    if "esc" in args.sets:  # environmental sounds (ESC-50), no speech at all: anything muted is wrong
        sets.update(esc_streams())
    if "ctx" in args.sets:  # curse-word contexts with clean words in the slot: anything muted is wrong
        sets.update(tts_streams(n_streams=2, per_stream=200, split="test_ctx", prefix="ctx"))
    if "real" in args.sets:  # real speech (YODAS2); "muted" includes the unlabelled context around lines
        sets.update(real_streams(args.real_file))
    det = Detector(HERE / args.head if not Path(args.head).is_absolute() else args.head,
                   graph=not args.check_ms)  # the batched path doesn't use the CUDA graph
    checks = {}
    all_times = []
    for name, (audio, _) in sets.items():
        if args.check_ms:
            checks[name], tm = run_batched(det, audio, args.step, args.check_ms / 1000)
        else:
            checks[name], tm = run(det, audio, args.step)
        all_times += tm
    print(f"step {args.step} s | {1000 * np.mean(all_times):.1f} ms per check "
          f"(p99 {1000 * np.percentile(all_times, 99):.1f}) | pads {args.pad}")

    gates = [None if g == "none" else (int(g.split(":")[0]), float(g.split(":")[1])) for g in args.gate]
    for thr, confirm, gate in [(t, c, g) for g in gates for c in args.confirm for t in args.thr]:
        print(f"\n=== threshold {thr}, {confirm} check(s) in a row, speech gate {gate} ===")
        for group in ("bench", "tts", "ctx", "esc", "real"):
            names = [n for n in sets if n.startswith(group)]
            if not names:
                continue
            need, agg = [], {d: {} for d in DELAYS}
            for n in names:
                audio, words = sets[n]
                g = first_mute(checks[n], int(len(audio) / SR / GRID) + 1, thr, *args.pad, confirm, gate)
                r = score(audio, words, g)
                need += r["need"]
                for d in DELAYS:
                    for k, v in r[d].items():
                        agg[d][k] = agg[d].get(k, 0) + v
            need = np.array(need)
            dur = sum(len(sets[n][0]) for n in names) / SR
            fin = need[np.isfinite(need)]
            print(f"[{group}] {len(need)} curse words, missed {int(np.sum(~np.isfinite(need)))} | needed delay: "
                  f"median {np.median(fin):.2f}  p95 {np.percentile(fin, 95):.2f}  "
                  f"p99 {np.percentile(fin, 99):.2f}  max {fin.max():.2f}" if len(fin) else f"[{group}] nothing detected")
            for d in DELAYS:
                a = agg[d]
                print(f"   at {d:.2f} s: curse silent {a['silent']}/{a['curse']}  "
                      f"clean intact {a['intact']}/{a['clean']} ({a['intact'] / max(a['clean'], 1):.1%})  "
                      f"sound-alikes intact {a['neg_intact']}/{a['neg']}  "
                      f"muted {a['muted_s'] / dur * 60:.1f} s/min")


if __name__ == "__main__":
    main()

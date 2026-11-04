r"""Synthesize labelled training speech with Piper: many voices, exact word boundaries.

Piper's duration predictor says how many samples each phoneme gets, and espeak separates words
with a " " phoneme, so every word's start/end sample is known exactly - no forced aligner needed.

Output: data/tts/<split>/shard_NNNN.npz  (audio int16 @16 kHz, concatenated; offsets)
        data/tts/<split>/shard_NNNN.json (per utterance: voice, speaker, words [word, group, start, end])

    .venv-lab\Scripts\python lab\detector\make_tts.py --check            # boundary sanity check
    .venv-lab\Scripts\python lab\detector\make_tts.py --split train --utts 40000
    .venv-lab\Scripts\python lab\detector\make_tts.py --split test --utts 2000
"""
import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corpus  # noqa: E402

VOICES = HERE / "voices"
OUT = HERE.parent.parent / "data" / "tts"
SR = 16000
PUNCT = set(",.!?;:—…\"'()")
PER_SHARD = 250

# Unseen at training time: whole voices, plus the last speakers of each multi-speaker voice.
TEST_VOICES = {"en_US-ryan-high", "en_GB-alba-medium", "en_US-kristin-medium"}
TEST_SPEAKERS_FROM = {"en_US-libritts_r-medium": 850, "en_GB-vctk-medium": 100,
                      "en_US-l2arctic-medium": 20, "en_US-arctic-medium": 16}


def voice_list():
    out = []
    for p in sorted(VOICES.glob("*.onnx")):
        cfg = json.loads(p.with_suffix(".onnx.json").read_text(encoding="utf-8"))
        out.append((p.name[:-5], cfg.get("num_speakers", 1)))
    return out


def speakers(name, n, split):
    if split == "test":
        if name in TEST_VOICES:
            return list(range(n))
        return list(range(TEST_SPEAKERS_FROM[name], n)) if name in TEST_SPEAKERS_FROM else []
    if name in TEST_VOICES:
        return []
    return list(range(TEST_SPEAKERS_FROM.get(name, n)))


_voices = {}


def load_voice(name, threads=1):
    import onnxruntime
    from piper import PiperVoice
    if not hasattr(onnxruntime, "_DefaultSessionOptions"):
        # Piper builds its session with default options = one thread per core in every worker,
        # which oversubscribes the CPU many times over when workers run in parallel.
        onnxruntime._DefaultSessionOptions = onnxruntime.SessionOptions

        def options():
            o = onnxruntime._DefaultSessionOptions()
            o.intra_op_num_threads = o.inter_op_num_threads = threads
            return o
        onnxruntime.SessionOptions = options
    if name not in _voices:
        _voices.clear()  # one voice per worker at a time keeps memory flat
        _voices[name] = PiperVoice.load(VOICES / f"{name}.onnx", include_alignments=True)
    return _voices[name]


def synth(voice, words, speaker, rng):
    """Speak the words; return 16 kHz audio and (word, group, start, end) in samples, or None
    if espeak's word split doesn't line up with ours."""
    from piper import SynthesisConfig
    cfg = SynthesisConfig(speaker_id=speaker if voice.config.num_speakers > 1 else None,
                          length_scale=rng.uniform(0.75, 1.3), noise_scale=rng.uniform(0.45, 0.85),
                          noise_w_scale=rng.uniform(0.5, 1.0), normalize_audio=True)
    sr = voice.config.sample_rate
    audio, spans, phs, pos = [], [], [], 0
    for chunk in voice.synthesize(" ".join(w for w, _ in words), cfg, include_alignments=True):
        if chunk.phoneme_alignments is None:
            return None
        t, cur = pos, None
        for a in chunk.phoneme_alignments:
            ph = a.phoneme
            if ph == " ":
                if cur:
                    spans.append(cur[:2])
                    phs.append(cur[2])
                cur = None
            elif ph not in PUNCT and ph not in ("^", "$", "_"):
                cur = [t, t + a.num_samples, ph] if cur is None else [cur[0], t + a.num_samples, cur[2] + ph]
            t += a.num_samples
        if cur:
            spans.append(cur[:2])
            phs.append(cur[2])
        audio.append(chunk.audio_float_array)
        pos += len(chunk.audio_float_array)
        gap = int(rng.uniform(0.05, 0.35) * sr)  # pause between sentences
        audio.append(np.zeros(gap, np.float32))
        pos += gap
    if len(spans) != len(words):
        spans = merge_map(voice, words, spans, phs)
        if spans is None:
            return None
    a = np.concatenate(audio)
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    k = SR / sr
    return a, [(w, g, int(s * k), int(e * k)) for (w, g), (s, e) in zip(words, spans)]


STRESS = str.maketrans("", "", "ˈˌːʲ")


def merge_map(voice, words, spans, phs):
    """espeak runs unstressed function words together ("in the" -> one phoneme word), so there
    are fewer spans than words. Line them up by phoneme length, letting one span cover up to 3
    words; the words in a merged span share it. None if a curse word would share a span."""
    lens = []
    for w, _ in words:
        p = "".join(ph for s in voice.phonemize(w) for ph in s if ph not in PUNCT and ph != " ")
        lens.append(len(p.translate(STRESS)))
    glens = [len(p.translate(STRESS)) for p in phs]
    nw, ng = len(words), len(phs)
    INF = float("inf")
    cost = [[INF] * (ng + 1) for _ in range(nw + 1)]
    back = [[0] * (ng + 1) for _ in range(nw + 1)]
    cost[0][0] = 0
    for i in range(nw + 1):
        for j in range(ng):
            if cost[i][j] == INF:
                continue
            for k in (1, 2, 3):
                if i + k > nw:
                    break
                c = cost[i][j] + abs(glens[j] - sum(lens[i:i + k])) + (k - 1)
                if c < cost[i + k][j + 1]:
                    cost[i + k][j + 1], back[i + k][j + 1] = c, k
    if cost[nw][ng] == INF or cost[nw][ng] > 2 * ng:
        return None
    out, i, j = [None] * nw, nw, ng
    while j:
        k = back[i][j]
        if k > 1 and any(g for _, g in words[i - k:i]):
            return None
        for m in range(i - k, i):
            out[m] = spans[j - 1]
        i, j = i - k, j - 1
    return out


def make_shard(args):
    split, shard, seed, kind = args
    rng = random.Random(seed)
    pool = [(v, n) for v, n in voice_list() if speakers(v, n, split)]
    voice_name, n = rng.choices(pool, weights=[3 if n > 1 else 1 for _, n in pool])[0]
    spk = speakers(voice_name, n, split)
    voice = load_voice(voice_name)
    audios, metas, dropped = [], [], 0
    while len(metas) < PER_SHARD:
        words = corpus.make(rng, kind)
        speaker = rng.choice(spk)
        res = synth(voice, words, speaker, rng)
        if res is None:
            dropped += 1
            continue
        a, ws = res
        audios.append((a * 32767).astype(np.int16))
        metas.append({"voice": voice_name, "speaker": speaker, "words": ws})
    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(a) for a in audios])
    np.savez(d / f"shard_{shard:04d}.npz", audio=np.concatenate(audios), offsets=offsets)
    (d / f"shard_{shard:04d}.json").write_text(json.dumps(metas))
    return voice_name, dropped, offsets[-1] / SR


def check():
    """Compare Piper word boundaries with Whisper word timestamps on a few sentences."""
    sys.path.insert(0, str(HERE.parent.parent))
    from censor import add_cuda_dlls
    add_cuda_dlls()
    from faster_whisper import WhisperModel
    judge = WhisperModel("small.en", device="cuda", compute_type="float16")
    rng = random.Random(1)
    diffs, dropped = [], 0
    for name in ["en_US-libritts_r-medium", "en_GB-vctk-medium", "en_US-lessac-high", "en_US-l2arctic-medium"]:
        voice = load_voice(name)
        for _ in range(15):
            words = corpus.make(rng)
            res = synth(voice, words, rng.randrange(max(1, voice.config.num_speakers)), rng)
            if res is None:
                dropped += 1
                continue
            a, ws = res
            segs, _ = judge.transcribe(a, language="en", beam_size=5, word_timestamps=True)
            ref = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
            if len(ref) != len(ws):
                continue
            for (w, g, s, e), (rw, rs, re_) in zip(ws, ref):
                diffs.append((s / SR - rs, e / SR - re_))
        print(f"{name}: ok", flush=True)
    d = np.array(diffs)
    print(f"{len(d)} words compared, {dropped} sentences dropped (word split mismatch)")
    print(f"start: piper - whisper median {np.median(d[:, 0]) * 1000:+.0f} ms, "
          f"abs p90 {np.percentile(np.abs(d[:, 0]), 90) * 1000:.0f} ms")
    print(f"end:   piper - whisper median {np.median(d[:, 1]) * 1000:+.0f} ms, "
          f"abs p90 {np.percentile(np.abs(d[:, 1]), 90) * 1000:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--utts", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--kind", help="only this kind of sentence (curse, context, negative, clean)")
    args = ap.parse_args()
    if args.check:
        return check()
    shards = args.utts // PER_SHARD
    base = {"train": 0, "test": 10**6}.get(args.split, 2 * 10**6 + 10**5 * (sum(map(ord, args.split)) % 97))
    done = {int(p.stem[6:]) for p in (OUT / args.split).glob("shard_*.json")}
    todo = [(args.split, i, base + i, args.kind) for i in range(shards) if i not in done]
    print(f"{args.split}: {len(todo)} shards to make ({len(done)} already there)", flush=True)
    hours, drops = 0.0, 0
    with Pool(args.workers) as pool:
        for i, (voice, dropped, secs) in enumerate(pool.imap_unordered(make_shard, todo), 1):
            hours += secs / 3600
            drops += dropped
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} shards, {hours:.1f} h of speech, {drops} sentences dropped", flush=True)


if __name__ == "__main__":
    main()

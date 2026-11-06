r"""Align the transcript words of real-speech clips and check each curse word (stage 2 of 2, GPU).

For every clip from fetch_yodas.py:
- Word times: torchaudio's MMS forced aligner (20 ms frames, finer than Parakeet's 80 ms) aligns
  the transcript line to its span of the clip. A "*" token at each end absorbs speech the
  subtitle left out. Each word gets a score, the mean probability of its characters.
- Check: Parakeet CTC transcribes the same span greedily. A curse word counts as heard when a form
  from its group is in Parakeet's text.
A low line score usually means a translated subtitle, a mistimed caption or music. A curse word
whose score is low or that Parakeet didn't hear goes to the hand-check, not straight into the set.

    .venv-lab\Scripts\python lab\detector\align_clips.py            # resumes; prints a summary
    .venv-lab\Scripts\python lab\detector\align_clips.py --summary

Writes data/realspeech/yodas/aligned.jsonl: the clip row plus "words"
([{w, start, end, score, group}], times in clip seconds), "line_score", "asr" and "curses"
([{i, w, group, score, heard}]).
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detector import ENCODER, SR  # noqa: E402
from fetch_yodas import DEST  # noqa: E402
from lexicon import GROUPS  # noqa: E402
from scan_transcripts import FORM2GROUP  # noqa: E402

WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
MARGIN = 0.5  # seconds of audio either side of the caption's line, for loose caption timing


def words_of(text):
    """Transcript -> aligner words; anything with a digit becomes "*" (the aligner can't spell it)."""
    return ["*" if re.search(r"\d", w) else w
            for w in WORD.findall(text.lower().replace("’", "'"))]


class Aligner:
    def __init__(self, device="cuda"):
        from transformers import AutoModelForCTC, AutoProcessor
        self.device = device
        bundle = torchaudio.pipelines.MMS_FA
        self.mms = bundle.get_model(with_star=True).to(device).eval()
        self.tokenize, self.align = bundle.get_tokenizer(), bundle.get_aligner()
        self.proc = AutoProcessor.from_pretrained(ENCODER)
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.asr = AutoModelForCTC.from_pretrained(ENCODER, dtype=self.dtype).to(device).eval()

    @torch.inference_mode()
    def __call__(self, audio, text):
        words = words_of(text)
        x = torch.from_numpy(audio)[None].to(self.device)
        emission, _ = self.mms(x)
        spans = self.align(emission[0], self.tokenize(["*"] + words + ["*"]))[1:-1]
        sec = len(audio) / emission.shape[1] / SR
        out = []
        for w, sp in zip(words, spans):
            n = sum(len(s) for s in sp)
            out.append({"w": w, "start": round(sp[0].start * sec, 3), "end": round(sp[-1].end * sec, 3),
                        "score": round(sum(s.score * len(s) for s in sp) / n, 3),
                        "group": FORM2GROUP.get(w)})
        inputs = self.proc(audio, sampling_rate=SR, return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        ids = self.asr(**inputs).logits[0].argmax(-1).tolist()
        return out, self.proc.batch_decode([ids])[0]


def process(al, row):
    audio, sr = sf.read(DEST / row["clip"], dtype="float32")
    assert sr == SR
    a = max(0.0, row["line"][0] - MARGIN)
    b = min(len(audio) / SR, row["line"][1] + MARGIN)
    words, asr = al(audio[int(a * SR):int(b * SR)], row["text"])
    for w in words:
        w["start"], w["end"] = round(w["start"] + a, 3), round(w["end"] + a, 3)
    heard = collections.Counter(FORM2GROUP[t] for t in WORD.findall(asr.lower()) if t in FORM2GROUP)
    curses, used = [], collections.Counter()
    for i, w in enumerate(words):
        if w["group"]:
            used[w["group"]] += 1
            curses.append({"i": i, "w": w["w"], "group": w["group"], "score": w["score"],
                           "heard": used[w["group"]] <= heard[w["group"]]})
    real = [w["score"] for w in words if w["w"] != "*"]
    return {**row, "words": words, "line_score": round(float(np.mean(real)), 3) if real else 0.0,
            "asr": asr, "curses": curses}


def summary(rows, thr=0.5):
    cs = [(r, c) for r in rows for c in r["curses"]]
    if not cs:
        print("no curse words aligned yet")
        return
    n = len(cs)
    good = sum(c["score"] >= thr for _, c in cs)
    heard = sum(c["heard"] for _, c in cs)
    both = sum(c["score"] >= thr and c["heard"] for _, c in cs)
    print(f"{len(rows)} clips from {len({r['src'] for r in rows})} videos, {n} curse words")
    print(f"  aligned with score >= {thr}: {good} ({good / n:.0%})")
    print(f"  heard by Parakeet:           {heard} ({heard / n:.0%})")
    print(f"  both:                        {both} ({both / n:.0%})")
    print(f"  line score median {np.median([r['line_score'] for r in rows]):.2f}; "
          f"lines below 0.3: {sum(r['line_score'] < 0.3 for r in rows)}")
    per = collections.defaultdict(lambda: [0, 0])
    for _, c in cs:
        per[c["group"]][0] += 1
        per[c["group"]][1] += c["score"] >= thr and c["heard"]
    print("  by group (both / all): " + ", ".join(f"{g} {b}/{a}" for g, (a, b) in
                                                  sorted(per.items(), key=lambda kv: -kv[1][0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--device", default="cuda", help="cpu works too (~2 s a clip), e.g. while the app has the GPU")
    ap.add_argument("--threads", type=int, default=6, help="CPU threads when --device cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = DEST / "aligned.jsonl"
    rows = [json.loads(l) for l in out.open(encoding="utf-8")] if out.exists() else []
    if not a.summary:
        done = {r["id"] for r in rows}
        todo = [json.loads(l) for l in (DEST / "clips.jsonl").open(encoding="utf-8")]
        todo = [r for r in todo if r["id"] not in done]
        print(f"{len(todo)} clips to align", flush=True)
        al = Aligner(a.device)
        with out.open("a", encoding="utf-8") as f:
            for i, r in enumerate(todo, 1):
                try:
                    res = process(al, r)
                except Exception as e:  # one broken clip shouldn't stop the run
                    print(f"  {r['id']}: {type(e).__name__}: {e}", flush=True)
                    continue
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()  # the Lambda filesystem buffers ~MBs otherwise; a crash would lose them
                rows.append(res)
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
    summary(rows)


if __name__ == "__main__":
    main()

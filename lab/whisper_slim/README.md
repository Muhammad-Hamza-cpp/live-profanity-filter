# Whisper slimming experiments

Can Whisper itself be made faster (layers removed, weights edited, shorter encoder input) without losing accuracy? Short answer: not in a way that lowers the delay you hear.

| Variant | ms per check | Caught | Worst-case delay needed | Verdict |
|---|---|---|---|---|
| small.en (current) | 140 | 30/30 | 0.6-0.75 s | baseline |
| small.en, 2-6 decoder layers removed (no retraining) | - | - | - | broken: repeats text, drops curse words |
| small.en, 2 encoder layers removed | - | 26-30 | worse | encoder is only ~40 ms of a call, so ~7 ms gain at most |
| small.en, 4 s encoder input instead of 30 s | 650 | 29-30 | 2.2-3.0 s | encoder is 10x faster, but the decoder never stops; would need fine-tuning |
| distil-small.en (4 decoder layers, retrained) | 86 | 30/30 | 1.22 s | usual case like small.en, one word muted late |
| tiny.en | 50 | 30/30 | 0.48 s | still the lowest delay |

Compute is only 0.05-0.15 s of the 0.5-0.8 s delay. Most of the delay is Whisper waiting to hear audio after a word, so removing layers can't fix it. Worst-case figures vary by about ±0.15 s between runs (only 30 test words). Raw numbers: `results.jsonl`.

| File | What it is |
|---|---|
| `ct2_weights.py` | Reads/writes CTranslate2 `model.bin` (the file faster-whisper loads); prints per-layer sizes |
| `prune.py` | Copies a model with layers removed and alignment heads renumbered |
| `bench_slim.py` | `bench.py` scoring for these variants; waits for the GPU to be free and reruns overlapped measurements |
| `diag_ctx.py`, `probe_ctx.py`, `sanity.py`, `inspect_small.py` | One-off checks behind the table |

```
..\..\.venv\Scripts\python prune.py small.en --keep-decoder 0,1,3,5,6,7,8,9,10,11 --out models\small.en-dec10
..\..\.venv\Scripts\python bench_slim.py small.en:3000 distil-small.en:3000 models\small.en-dec10:3000 --steps 0.2
```

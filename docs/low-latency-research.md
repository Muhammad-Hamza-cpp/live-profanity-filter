# Low-latency research (in progress)

**Goal:** cut the audio delay as far as possible while still silencing every curse word, and keep collateral (clean words cut) low. The shipped app (`main`) uses Whisper `small.en` at a **1.2 s** delay, with `tiny.en` at 0.8 s in "Low lag" mode.

**Status:** off-the-shelf exploration is done. The best result is **Parakeet CTC 0.6B with "gap-fill" muting at about 0.65 s** (simulated). It is not in the app yet. The next steps are below.

---

## How we measure

All numbers come from the **bench clips** made by `bench.py` (written to `test/`, which is gitignored):

- **normal:** SAPI voice David, 10 curse words, about 57 s
- **fast:** SAPI voice Zira at rate +3
- **noisy:** normal mixed with rumble and synth chords at 5 dB SNR

`lab/bench_models.py` replays each clip through the live loop on a simulated clock, using real GPU timings. A window covers the last 4 s, and the next check runs at `max(t + step, t + processing_time)`. Reference word times come from Whisper `small.en` transcribing the full clip.

- **safe delay:** the smallest delay at which every sample of every curse word is muted before it plays. This is the worst case over all 30 words.
- **at delay:** safe delay + 0.15 s margin. At this delay the audio is rendered and each word's remaining loudness is measured:
  - **silent:** less than 20% of the original level left
  - **intact:** more than 70% left

> ⚠️ Do **not** judge leaks by re-transcribing the censored audio. Whisper fills in silenced words from context ("what the ___ is going on" → "hell"). Use loudness, as `selftest.py` and `evaluate()` do.

> ⚠️ This is synthetic TTS with only 30 curse words, so it's good for *comparing* approaches, but it is not proof of real-world accuracy. Another experiment (`lab/whisper_slim/`, not part of this branch) was using the GPU from 21:09 to 21:27 on 2026-11-03. That overlapped these runs, so ms/call figures from then may be inflated. Re-run on an idle GPU before final decisions.

## Results so far

| Approach | Step | ms/check | Safe delay | Run at | Curse silent | Clean intact |
|---|---|---|---|---|---|---|
| Whisper small.en (app default, `bench.py`, idle GPU) | 0.4 s | 148 | 0.76 s* | 1.2 s | 30/30 | ~95% |
| Whisper tiny.en + −0.14 s shift (app "Low lag") | 0.3 s | 54 | 0.49 s* | 0.8 s | 30/30 | ~96% |
| Parakeet CTC 0.6B, greedy word times | 0.1 s | 59 | 1.10 s | 1.25 s | 30/30 | 89% |
| Parakeet, +0.17 s shift, various pads | 0.1–0.2 s | 54 | ∞ (gaps) | – | 27–30/30 | 83–90% |
| **Parakeet, gap-fill, pad 0.04/0.08** | **0.1 s** | **54** | **0.49 s** | **0.64 s** | **30/30** | **88%** |
| Parakeet, gap-fill, pad 0.04/0.08 | 0.2 s | 53 | 1.17 s | 1.32 s | 30/30 | 88% |

\*`bench.py` measured only the core of each word (±50 ms) with Whisper's padding. `lab/bench_models.py` is stricter (the whole word), so the two sets of numbers aren't perfectly comparable.

### Key findings

1. **Whisper wastes most of its compute.** CTranslate2 Whisper pads every window to 30 s, so a 4 s check costs as much as a 30 s one: about 150 ms for small.en. Parakeet's cost scales with the input, about 50–60 ms per 4 s window. That lets us check every 0.1 s.
2. **Parakeet detects early.** It typically emits a curse word within ±0.1 s of the word's end, worst case +0.4 s (`lab/diag_words.py`). Whisper needs more trailing context.
3. **Parakeet's word boundaries are poor.** CTC spikes sit somewhere inside a word, so greedy word times were about 0.2 s early, with a range of −0.5 to +0.23 s. No constant shift fixes that.
   **Gap-fill** fixes it: mute from the previous word's last spike to the next word's first spike, capped at ±0.4 s around the word's own spikes. The word is always somewhere in that gap. The cost is more clean audio cut: 88% intact vs ~95% for Whisper.
4. **The physical floor is about 0.4 s.** A word can't be recognized before most of it is heard, and curse words last 0.2–0.45 s. Going lower means acting on the first syllable, which mutes false positives like "fun", "shift" and "duck".
5. **Step matters a lot for Parakeet.** 0.1 s gives a safe delay of 0.49 s and 0.2 s gives 1.17 s. The worst-case words are those detected only in a later window, so checking more often is cheap insurance at about 55% GPU.

### Gotchas (save yourself the debugging)

- **Parakeet + fp16 = NaN**, which decodes as `<unk>`. Use **bfloat16** or fp32. bf16 is accurate, and the speed is the same, since the cost is dominated by kernel-launch overhead rather than math. `torch.compile` / CUDA graphs could cut it further.
- **Parakeet frame = exactly 0.08 s** (10 ms mel hop × 8 subsampling). Don't use `len(audio) / n_frames`; it gives 78 ms and drifts by about 0.08 s across a 4 s window.
- `transformers` 5.x needs **librosa** for `ParakeetFeatureExtractor`. Feature extraction takes about 2 ms on CPU.
- **onnxruntime-gpu 1.31 expects CUDA 13** DLLs (`cublas64_13`, `cudart64_13`). Our NVIDIA pip wheels and torch are CUDA 12, so ORT silently falls back to CPU. To try `onnx-asr` with Parakeet TDT on GPU, install an older onnxruntime-gpu built for CUDA 12, or the CUDA 13 wheels.
- Installing `onnx-asr[gpu]` also pulled in CPU `onnxruntime`, which shadows the GPU build. Uninstall `onnxruntime` and force-reinstall `onnxruntime-gpu`.

## Reproduce

```powershell
# lab environment (separate from the app's .venv; ~6 GB)
python -m venv .venv-lab
.venv-lab\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv-lab\Scripts\python -m pip install -r lab\requirements-lab.txt

.venv\Scripts\python bench.py --models small.en --steps 0.4   # creates the test clips in test/
$env:TRANSFORMERS_VERBOSITY="error"
.venv-lab\Scripts\python lab\bench_models.py "pk gap"           # filter candidates by name substring
.venv-lab\Scripts\python lab\diag_words.py 0.1                  # per-word detection lag + timestamp error
```

| File | Purpose |
|---|---|
| `lab/bench_models.py` | Model comparison: a `Parakeet` adapter with the faster-whisper `.transcribe()` shape, gap-fill, and loudness-based evaluation |
| `lab/diag_words.py` | For each curse word: first detection time vs. when it was spoken, and start/end timestamp error |
| `lab/probe_parakeet.py`, `lab/probe2.py` | Speed per window length; precision (fp32/bf16/fp16) and feature vs. model time |
| `lab/requirements-lab.txt` | Lab packages (torch is installed separately from the CUDA index) |

---

## Next steps

### 1. Ship Parakeet as an "Ultra-low lag" mode (about 1 h)
- [ ] Move the `Parakeet` adapter (bf16, gap-fill, step 0.1) into `censor.py` behind the same interface as Whisper. The `Censor` loop already only needs words with start/end times.
- [ ] Dependencies: torch (CUDA) + transformers + librosa. That's large, about 4 GB, so consider making it an optional extra in `install.ps1`.
- [ ] Re-run `lab/bench_models.py` on an **idle GPU** to confirm 0.49 s, then `selftest.py` live at about 0.65 s.
- [ ] Add a preset to `app.pyw`, for example `"Ultra-low lag - Parakeet": model parakeet-ctc-0.6b, step 0.1, delay 0.65, pads 0.04/0.08`.
- [ ] Optional speedups: `torch.compile(mode="reduce-overhead")` with triton-windows, or CUDA graphs. Calls are launch-overhead bound (~50 ms regardless of length), so this could halve the time and allow a 0.05 s step.

### 2. Real-speech data (needed for any serious claim, and for training)
- [ ] **People's Speech** (`MLCommons/peoples_speech`, CC-BY-SA, about 30k hours of real archive.org audio). Stream only the transcript column (pyarrow + `HfFileSystem` column projection), find rows containing curse words, then fetch just those rows' audio.
- [ ] Get word boundaries with CTC forced alignment (`torchaudio.functional.forced_align` on Parakeet log-probs, or MMS_FA) and spot-check them by ear.
- [ ] Hold out a **real-speech test set**, a few hundred curse words plus clean speech, and report every result on it.
- [ ] More voices: Piper or Kokoro TTS with dozens of voices, plus noise and music augmentation. Optionally the user's own clips, kept local.

### 3. Train a frame-level "mute mask" detector (the real fix for boundaries and latency)
- [ ] Parakeet encoder (frozen at first) → small head → per-frame P(inside a curse word). The head can be causal or have limited right context.
- [ ] Train on forced-aligned real speech plus TTS. Loss: frame BCE, weighted against misses.
- [ ] Decode: mute where P > threshold, with hysteresis. Tune the threshold on the real test set for 0 leaks at the minimum collateral.
- [ ] Then try fine-tuning the top encoder layers, and **limited right context** (the cache-aware streaming FastConformer, NVIDIA Nemotron / Parakeet-unified, 80–160 ms lookahead) to approach the ~0.4 s floor.
- [ ] Fits on an RTX 3070 8 GB: the 0.6B encoder in bf16 runs inference with about 1.2 GB. Training only the head is easy, and partial fine-tuning needs gradient checkpointing.

### Other options considered
- **NVIDIA Nemotron 3 / Parakeet streaming (cache-aware FastConformer-RNNT, 80–1120 ms chunks):** the best streaming latency on paper, but needs NeMo, which is painful on Windows (consider WSL). Word timestamps for the streaming checkpoints are unconfirmed.
- **Parakeet TDT 0.6B v2 via `onnx-asr`:** better timestamps than CTC, likely. Blocked by the ORT/CUDA-13 issue above.
- **Pruned Whisper** (`lab/whisper_slim/`, a separate experiment, not in this branch): smaller decoder/encoder CT2 models. Whisper's 30 s padding still limits it.
- **Generic keyword spotters:** nothing open-source exists that is trained for profanity, so a custom detector (step 3) is the realistic version of this.

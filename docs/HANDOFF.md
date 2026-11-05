# Handoff: state of the mute-detector work

Last updated **2026-11-05**. Pick up from here. For background, [mute-detector.md](mute-detector.md) is the full research report and [related-work-and-datasets.md](related-work-and-datasets.md) covers literature and licences.

## 1. What the user wants
- **Goal:** worst-case audio delay **under 0.4 s**, with every curse word silenced. The app's default is 1.2 s (Whisper `small.en`).
- **Main concerns after playing Call of Duty MW4 with it:** **false positives** (muting game sound and clean dialogue) and **latency** (`LATE` mutes).
- **Also interested in:**
  - a possible **research paper** (arXiv first, then a workshop or Interspeech)
  - **LinkedIn** posts (figures exist; a draft post was given in chat)
  - training only on **copyright-safe data**

## 2. What exists (branch `research/mute-detector`, pushed up to `86f0eaf`)

| Piece | Where |
|---|---|
| Training vocabulary: 41 groups, 268 curse forms, 345 sound-alike negatives; consistency check vs `words.txt` | `lab/detector/lexicon.py`, `words.txt` |
| Sentence generator (curse / context / negative / clean) | `lab/detector/corpus.py` |
| Piper TTS data with exact word boundaries, parallel shards, train/test voice split, `merge_map` for espeak-merged words | `lab/detector/make_tts.py` |
| Model: frozen Parakeet CTC 0.6B encoder (blocks 12/16/20/24) + 0.93M head → P(frame in curse word) for each of 41 groups, every 80 ms; CUDA graph (~12 ms/check); `score()` also returns CTC speech probability | `lab/detector/detector.py` |
| Training: 4 s windows, heard-only labels, augmentation (noise/music/babble/ESC-50 effects/reverb/phone band), 15% sound-effects-only windows | `lab/detector/train.py` |
| Evaluation on a simulated live loop, scored by loudness; sets `bench` (SAPI voices), `tts` (held-out Piper voices), `ctx` (context negatives), `esc` (ESC-50 fold 5) | `lab/detector/eval.py` |
| False-mute diagnosis | `lab/detector/diag_fp.py` |
| App engine in real time without devices | `lab/detector/live_sim.py` |
| App integration: `Censor.detect_forever`, preset, timing log every 30 s, device rescan on Start, "Save audio for training" (FLAC parts + `mutes.jsonl`) | `censor.py`, `app.pyw`, `app-lab.bat` |
| Trained heads (3.7 MB each) | `lab/detector/models/detector-v2.pt`, **`detector-v3.pt` (what the app uses)** |
| LinkedIn/README figures (1200×1500) | `lab/detector/figures.py` → `docs/images/01..06-*.png` |

**App defaults** for the detector: v3, delay **0.40 s**, step 0.05 s, threshold 0.5, 2 checks in a row, pads 0.06/0.02 s, **speech gate off**.

## 3. Key results (details in mute-detector.md)
- **Baseline on this laptop:**
  - Whisper `small.en`: 0.77 s worst case.
  - Parakeet gap-fill: 0.66 s.
  - With a CUDA graph a Parakeet check costs ~12 ms instead of ~50 ms. A smaller step does **not** lower the worst case, which is set by when the transcript commits.
- **v1:** guessed from context: 69% of its false mutes fired within 0.1 s of the audio arriving. Cause: labels on frames not yet heard. **Fix (v2):** heard-only labels, plus 15k context-negative sentences.
- **v2:**
  - 235/235 held-out TTS curse words silenced at 0.35 s, with pads 0.06/0.02.
  - **First CoD session:** every real swear word blocked, but about 19 mutes a minute, mostly on game sound.
- **Speech gate** (CTC speech probability): cuts non-speech mutes by 85–100% but makes words late. Off by default.
- **v3:** ESC-50 sound effects as negatives.
  - Non-speech mutes: **1.1 → 0.1 s/min**.
  - Bench: 30/30 at 0.40 s.
  - VB-Cable self-test: borderline at 0.35 s (one word kept 22% loudness), clean at **0.40 s** (6/6, 0 late).

## 4. Open thread: review of the second game session (interrupted)
- **Recording:** `data/recordings/20261104-163523/`. About 32.5 min of real game audio, a **Call of Duty campaign with lots of dialogue and swearing, including some Spanish and French lines**. Then about 33 min of digital silence after the game stopped (parts ~65–130). It ran at 0.35 s; it's almost certainly v3 (the app started at 16:35, after v3 went in). **148 mutes**, 57% marked late.
- **The tool is `lab/detector/review_recording.py` (uncommitted).** It transcribes the recording with Whisper large-v3 and classifies every mute as caught / false in speech / false with no speech / missed, and writes clips to `review/`.
- **What's been learned about the reference transcript:**
  - Whisper **with VAD** drops most dialogue under game sound. **The first conclusion, "88% of mutes on non-speech", was wrong.**
  - **One long pass without VAD** drops whole stretches too (only 5 curse words found).
  - **Short clips without VAD** work. Re-transcribing the 130 "no speech" mute regions one by one found confident speech in 104 of them and real swearing in 22 ("get that skinny fuck out of the van", "those assholes are still fucking with us", "What the fuck did you do?").
- **Current code** transcribes in 10 s windows with a 5 s hop, no VAD, keeping only confident segments (`no_speech_prob < 0.6`, `avg_logprob > -1.0`) and taking each word from the window where it sits in the middle. The run was **interrupted before finishing**. The cache `whisper_large_v3_windows.json` doesn't exist yet; the older caches `whisper_large_v3.json` (VAD) and `whisper_large_v3_novad.json` are the unreliable ones.
- **Next:**
  1. Re-run `.venv-lab\Scripts\python lab\detector\review_recording.py data\recordings\20261104-163523`, about 5–10 min of GPU.
  2. Spot-check `review/*.wav` (the user can listen).
  3. Report caught / false / missed.
- **Expected story:** many v3 mutes are real swears in dialogue; the rest are clean dialogue (including non-English lines) and some pure game sound.
- **Privacy and licence:** this is copyrighted game audio and may include other players' voices. Keep it local and never publish or commit it.

## 5. Git state
Everything is committed and pushed to `origin/research/mute-detector`. The latest additions:
- the recorder skips silent parts
- `review_recording.py`
- the related-work doc
- this handoff and `CLAUDE.md`

Only gitignored data and logs are local-only (§9).

## 6. Licence and IP flags
- **ESC-50 is CC BY-NC.** v3 was trained on it, so **treat `detector-v3.pt` as research-only.** Retrain on **MUSAN (CC BY 4.0)** plus **FSD50K clips licensed CC0/CC BY** (filter per clip; includes gunshots and explosions).
- **Roblox patents** US 12,027,177 / 12,283,288 ("Artificial latency for moderating voice communication") cover delay-plus-mute at the system level. A paper should claim the detector and training method, not buffer-and-mute. Commercial use needs the claims reviewed by a professional.
- **Intel Bleep** (2021) is product prior art for a Windows voice-chat filter with category sliders.

## 7. Recommended next steps (in order)
1. **Finish the game-session review** (§4) and commit `review_recording.py` with the findings.
2. **Real-speech test set** (most important, for the paper and the product):
   - Search **YODAS2 / Emilia-YODAS (CC BY)** and **People's Speech (CC BY/BY-SA)** transcripts for lexicon words.
   - Fetch only the matching segments.
   - Force-align (CTC with Parakeet or MMS).
   - Hand-check 300–500 curse words.
   - Add clean spontaneous speech (AMI CC BY, Common Voice Spontaneous CC0).
3. **v4 on copyright-safe data:**
   - MUSAN plus CC0/CC BY FSD50K instead of ESC-50.
   - Non-English negatives (Multilingual LibriSpeech CC BY, VoxPopuli CC0), because game dialogue includes Spanish and French.
   - Some real positives (Google's TTS-KWS paper: a few thousand real clips close most of the synthetic-to-real gap).
   - Private game recordings as negatives and a private eval set, labelled by windowed Whisper large-v3.
4. **Onsets:** v3 at 0.35 s sometimes leaks the first ~60 ms. Try a start-offset regression in the head, or finer frames.
5. **Paper:**
   - baselines: Whisper + word list, Parakeet gap-fill, a streaming CTC keyword spotter, MuTox
   - curves of delay vs. recall vs. clean-word collateral
   - three seeds, and the ablation list in related-work-and-datasets.md §3
6. **App polish:** category toggles and a threshold slider in the UI.

## 8. Commands
```powershell
$env:TRANSFORMERS_VERBOSITY="error"
.venv-lab\Scripts\python lab\detector\lexicon.py                                   # vocab vs words.txt
.venv-lab\Scripts\python lab\detector\make_tts.py --split train --utts 40000     # ~30 min, 12 CPU workers
.venv-lab\Scripts\python lab\detector\train.py --name v4 --steps 8000 --train train,train_ctx   # ~20 min
.venv-lab\Scripts\python lab\detector\eval.py runs\v4\head.pt --thr 0.5 --confirm 2 --pad 0.06 0.02 --sets bench,tts,ctx,esc
.venv-lab\Scripts\python selftest.py 0.4 --model detector                        # through VB-Cable, nothing played aloud
.venv-lab\Scripts\python lab\detector\review_recording.py data\recordings\<session>
.venv-lab\Scripts\python lab\detector\figures.py
```
To ship a new head, copy `runs\vN\head.pt` to `lab\detector\models\detector-vN.pt` and point `DETECTOR_HEAD` in `censor.py` at it. Then re-run the selftest at 0.35 and 0.40 s.

## 9. Data on disk (gitignored, about 7 GB)
- `data/tts/{train, train_ctx, test, test_ctx}`: 40k + 15k + 2k + 1k Piper utterances; `.npz` audio plus `.json` word labels.
- `data/esc50_dl/ESC-50-master/`
- `data/recordings/20261104-163523/`
- `lab/detector/voices/`: 29 Piper voices.
- `lab/detector/runs/v1..v3/`: checkpoints and training logs. Eval logs are in the repo root (`eval_*.log`, `train_*.log`).
- The bench clips are in `test/` (made by `bench.py`).

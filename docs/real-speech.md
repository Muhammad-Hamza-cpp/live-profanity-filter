# Real speech: test set, training on real data, v4

Last updated **2026-11-06**. This continues [mute-detector.md](mute-detector.md), which covers v1–v3. Until now every number came from synthetic voices (Piper TTS) or a few SAPI clips. This report covers three steps:
1. a test set of **real, public speech**
2. training with real speech mixed in, on a cloud GPU (Lambda A10), 21 runs
3. **v4**, the model the app now uses

**Decision behind it (2026-11-05):** a paper must rest on data others can get. So no game audio is used for training or for any reported number. Everything below is public YouTube speech from [YODAS2](https://huggingface.co/datasets/espnet/yodas2) (CC BY 3.0) plus the existing Piper data.

## Summary

All on 450 real curse words from 306 YouTube videos, at **0.40 s of delay**, 2 checks in a row, pads 0.06/0.02 s. Each threshold was chosen on a separate validation set, never on the test set.

| System | Curse words silenced | Clean words intact | Sound-alikes intact | Muted per minute |
|---|---|---|---|---|
| Whisper `small.en` (the app's default engine) | 15.3% | 98.9% | 99.2% | 0.6 s |
| Parakeet CTC gap-fill | 75.8% | 97.0% | 97.0% | 1.4 s |
| v3, synthetic only (threshold 0.5, as shipped) | 93.8% | 82.0% | 78.8% | 9.7 s |
| v3, synthetic only (threshold 0.7) | 90.4% | 90.8% | 86.9% | 5.4 s |
| **Synthetic + all real (3 seeds, threshold 0.7)** | **93.0 ± 0.1%** | **95.0 ± 0.2%** | **92.0 ± 0.6%** | **3.5 s** |
| **v4** = seed 3 of the row above | 93.1% | 94.9% | 91.4% | 3.5 s |

- **The low delay holds on real speech.** At 0.40 s the detector silences 93% of real curse words. To silence as many, Whisper needs 1.2 s (404/450 = 90%) and Parakeet about 0.8 s (392/450 = 87%).
- **Real training speech fixes most false mutes.** The synthetic-only model damages about 1 in 5 clean words on real speech. With real speech mixed in it damages 1 in 20 and mutes 3.5 s a minute instead of 9.7 s. A near-perfect detector still clips about 4% of clean words, the ones next to a curse word: Whisper at 1.2 s keeps 96.0% intact.
- **A little real data does most of it.** 500 real curse words already close most of the gap. More data mainly makes the model surer, so it can use a higher threshold.
- **The ablations change nothing measurable.** Labelling unheard frames, dropping context negatives and dropping sound effects each stay within seed noise on real speech. Section 5 covers the one exception, sound effects on non-speech.

## 1. Where the real speech comes from

**Transcript scan, no audio** ([`scan_transcripts.py`](../lab/detector/scan_transcripts.py)). This scanned the transcripts of two datasets for every curse form in the lexicon and fetched only transcript bytes.
- **YODAS2.** Only `en000`–`en007` were scanned, because those are manual subtitles; auto-captions write profanity as `[ __ ]`. This covers 37,094 hours, about 88.5k lines with a curse word, in about 23k videos. Starred-out words ("f***") are recorded too, and clips near them are skipped, because the audio is probably bleeped.
- **People's Speech** (CC BY / BY-SA). 30,808 hours, 28,133 matching utterances. **Not used yet:** its source IDs don't parse into usable speaker groups, and that's still open.

**Audio** ([`fetch_yodas.py`](../lab/detector/fetch_yodas.py)). YODAS2 packs about 80 soundtracks into each 1.4 GB `.tar.gz`. A gzip stream can't be read from the middle, so each archive is streamed start to end and only the soundtracks with hits are kept.
- **Curse clips:** one per transcript line with a curse word. Each clip is 4 s before the line (the detector's window), the line, and 1.5 s after, as 16 kHz FLAC.
- **Clean clips:** up to 3 clean lines per video are cut the same way. Lines with sound-alike words go first, and lines within 1 s of a curse or starred line are skipped.
- **Archive order:** archives are ranked by how many *distinct videos* have hits. Ranking by raw hit count let one video supply 310 of 349 words in the first archive.

## 2. Labels, with automatic checks only

Nobody listens to the clips. Every label comes from two models that must agree.
1. **Timing:** [`align_clips.py`](../lab/detector/align_clips.py) runs torchaudio's MMS forced aligner, at 20 ms frames, on the subtitle line ± 0.5 s. A `*` token at each end absorbs speech the subtitle left out. On TTS, where the truth is known, 92% of word starts land within 40 ms.
2. **Was it said:** [`verify_whisper.py`](../lab/detector/verify_whisper.py) runs Whisper large-v3 on the line ± 1 s, with language detection, no VAD and no prompt. Parakeet turned out to be a poor check: it garbles casual swearing ("motherfucker" → "buffer").

A clip is used only if it passes all of these (`eligible()` in [`build_testset.py`](../lab/detector/build_testset.py)):
- Whisper detects **English**, which drops translated subtitles.
- Whisper hears **every** labelled curse word with word confidence ≥ 0.5, and **no** unlabelled one.
- The aligner's start time and Whisper's agree. Whisper runs about 0.12–0.15 s early, so the offset must be −0.15 ± 0.25 s.

**How good are these labels?** On a pilot sample of 37 words that the user listened to (2026-11-05), 80% of words passing the checks were real, correctly timed curse words. With the confidence cut it was 87%. So some of the "missed" words below are label errors, and the recall numbers are a slight underestimate. After the pilot the user asked for automatic checks only; [`make_review.py`](../lab/detector/make_review.py) still builds a listening page if one is ever wanted.

## 3. The splits

Everything is split **by video**, so no recording is in two splits. YODAS2 anonymises video IDs, so two videos from the same channel can't be matched. A speaker may therefore appear in both training and test through different videos. Treat this as a limitation.

| Split | Built by | Videos | Clips | Curse words | Notes |
|---|---|---|---|---|---|
| **Test** | `build_testset.py --target 450` on the 62 densest archives (laptop) | 306 | 434 curse + 588 clean | **450** (13,172 clean words, 1,103 sound-alikes) | at most 2 curse + 2 clean clips per video; rarest word groups first; 81.8 min |
| **Training pool** | `fetch_yodas.py --start 62 --top 200` then `build_trainset.py` (Lambda) | 1,540 | 3,311 curse + 4,400 clean | 3,477 | archives ranked 63–262, so no overlap with test; 10.1 h |
| **Validation** | 10% of the pool's videos (fixed hash) | 171 | 174 curse + 336 clean | 182 | capped like the test set |

Test set by category: strong 215, mild 210, British 17, slur 8. The pool had 8,500 eligible clips from 1,711 videos before the split.

## 4. Training with real speech

[`train.py`](../lab/detector/train.py) adds real clips to the v3 recipe:
- `--real N` takes whole videos in the pool's fixed order until they hold N curse words; −1 takes all.
- `--real-p 0.3` sets the share of training windows that come from real clips.
- A real window comes from one clip. **Only frames within the transcript line ± 0.5 s count in the loss**, because the audio around the line is unlabelled.
- Labels are heard-only, as in v2: a frame is positive only once the word has started being heard.
- Everything else is v3's recipe: the 40k + 15k Piper utterances (regenerated on Lambda with the same commands), ESC-50 sound effects, and the same noise and reverb augmentation.

Each run is 8,000 steps at batch 32, about 30 minutes on an A10, two runs at a time. `last.pt` (the final checkpoint) is what gets reported, so no checkpoint is picked by looking at the test set.

**Runs** ([`lab/lambda/jobs.txt`](../lab/lambda/jobs.txt)), each with seeds 1, 2 and 3:

| Group | What changes |
|---|---|
| `c0` | synthetic only (v3's recipe retrained, so seed noise is measured) |
| `c500`, `c2000`, `call` | + 500 / 2,000 / all 3,477 real curse words |
| `abl_unheard` | `call` + also label frames not heard yet (v1's labels) |
| `abl_noctx` | `call` without the 15k context-negative sentences |
| `abl_nosfx` | `call` without ESC-50 sound effects |

**Scoring** ([`eval.py`](../lab/detector/eval.py)) is the app's live loop on a simulated clock: a check every 50 ms, 2 in a row above threshold, pads 0.06/0.02 s, judged by loudness.
- `--check-ms 13` fixes each check's cost at the laptop's measured 13 ms, so results don't depend on which GPU ran them. v3 scores the same on Lambda as on the laptop (422 vs 421 of 450).
- A batched path runs about 10× faster; its decisions are 99.9% identical to running one check at a time.

**Threshold rule** ([`summarize_runs.py`](../lab/detector/summarize_runs.py)):
- Every run is scored on validation at thresholds 0.3–0.7.
- The rule maximises *silenced % + clean intact %* at 0.40 s, and ties go to the higher threshold.
- The test set is then scored once, at that threshold.

## 5. Results

### Every run (test set, 0.40 s, threshold chosen on validation)

| Group | Seeds | Threshold | Silenced % | Clean intact % | Sound-alikes intact % | Muted s/min | Non-speech (ESC-50) muted s/min |
|---|---|---|---|---|---|---|---|
| c0 (synthetic only) | 3 | 0.7 | 89.5 ± 0.6 | 91.8 ± 0.3 | 88.4 ± 0.4 | 4.9 ± 0.1 | 0.0 |
| c500 | 3 | 0.6 | 92.3 ± 1.0 | 93.7 ± 0.5 | 90.8 ± 1.6 | 4.5 ± 0.2 | 0.1 |
| c2000 | 3 | 0.7 | 92.2 ± 0.0 | 95.3 ± 0.1 | 92.7 ± 0.5 | 3.3 ± 0.0 | 0.0 |
| **call** | 3 | 0.7 | **93.0 ± 0.1** | **95.0 ± 0.2** | 92.0 ± 0.6 | 3.5 ± 0.1 | 0.0 |
| abl_unheard | 3 | 0.7 | 92.5 ± 0.3 | 95.1 ± 0.3 | 92.3 ± 0.4 | 3.5 ± 0.1 | 0.0 |
| abl_noctx | 3 | 0.7 | 92.8 ± 0.3 | 95.2 ± 0.1 | 92.6 ± 0.5 | 3.5 ± 0.1 | 0.0 |
| abl_nosfx | 3 | 0.7 | 92.5 ± 0.6 | 95.1 ± 0.2 | 92.2 ± 0.5 | 3.6 ± 0.1 | **0.2** |
| v3 (as shipped) | 1 | 0.7 | 90.4 | 90.8 | 86.9 | 5.4 | 0.0 |

The full per-run table is printed by `summarize_runs.py`.

### At the fixed threshold 0.5 (the app's old default)

| Group | Silenced % | Clean intact % | Muted s/min |
|---|---|---|---|
| v3 | 93.8 | 82.0 | 9.7 |
| c0 | 93.5 | 83.4 | 9.0 |
| c500 | 93.5 | 91.1 | 5.9 |
| c2000 | 95.3 | 90.4 | 6.0 |
| call | 95.6 | 90.2 | 6.2 |

### Baselines: delay needed (test set)

Transcribe-then-match baselines, run through the same live loop and scoring ([`eval_baselines.py`](../lab/detector/eval_baselines.py)). Each muted word gets the Whisper app's pads, 0.12 s before and 0.20 s after.

| Delay | Whisper `small.en` (step 0.4 s, 192 ms/call) silenced | intact | Parakeet gap-fill (step 0.05 s, 13 ms/call) silenced | intact |
|---|---|---|---|---|
| 0.25 s | 10/450 | 99.5% | 228/450 | 98.6% |
| 0.40 s | 69/450 | 98.9% | 341/450 | 97.0% |
| 0.60 s | 231/450 | 98.1% | 377/450 | 95.0% |
| 0.80 s | 360/450 | 97.0% | 392/450 | 94.4% |
| 1.20 s | 404/450 | 96.0% | 404/450 | 94.0% |

Median delay a curse word needs: **0.13–0.16 s** for the detector, 0.28 s for Parakeet and 0.62 s for Whisper.

### What it means
1. **The synthetic-to-real gap is in precision, not recall.** v3 already catches 94% of real curse words at 0.40 s. What fails on real speech is the clean side, with about 18% of clean words clipped. Real training speech fixes most of that.
2. **The learning curve flattens early.** At the fixed threshold 0.5, clean intact goes 83.4% (c0) → 91.1% (500 real words) → about 90% with more. With validation thresholds, 2,000 words or more let the model run at 0.7 and reach about 95% intact, near the 96% floor.3. **The design choices from v1–v3 don't show up on real speech.**
   - **Heard-only labels:** v1's guessing problem was mostly a templated-sentence artefact. Real speech doesn't have that context leak, so the ablation is flat.
   - **Context negatives:** real clean speech does their job.
   - **Sound effects:** they matter only for non-speech. Without them, mutes on ESC-50 rise from about 0.0 to 0.2 s/min. That's small, but it is game-sound behaviour, and ESC-50 is a weak stand-in for game audio.
4. **Thresholds ended at the top of the grid.** 0.7 was the highest threshold searched, so 0.8–0.9 may be better for the `call` and `c2000` models. Searching higher on validation is cheap and should be done before the paper's numbers are final.

## 6. v4 in the app

- **Which model:** `call_s3`, the trained run with the best validation score (191.3). Its final weights were copied to [`lab/detector/models/detector-v4.pt`](../lab/detector/models/) (3.7 MB, sha256 `05c636a3…`).
- **App settings:** `censor.py` points `DETECTOR_HEAD` at v4, with `DETECTOR_THRESHOLD = 0.7`. That threshold is the app preset's value and the CLI default. Delay 0.40 s.
- **VB-Cable self-test** (`selftest.py 0.4 --model detector`): 6/6 curse words at ≤ 6% loudness, 0 late, PASS; 52 of 55 clean words intact. At 0.35 s, "shit" kept 24% loudness (FAIL). This is the same onset limit as v3, so **0.40 s stays the floor**.
- **Not measured yet:** game audio. That needs a play session; it's not a reported result.

**Licence:** v4 was trained with ESC-50 (CC BY-NC) and Piper voices whose licences aren't audited yet. Like v3, it's **research-only**. The `abl_nosfx` runs, trained without ESC-50, are almost as good on real speech, so a version without the non-commercial licence is cheap to make.

## 7. Infrastructure (Lambda)

- **Hardware:** one `gpu_1x_a10` instance (us-west-1), 2026-11-05 to 2026-11-06 04:14, then terminated automatically.
- **Storage:** everything lived on the persistent filesystem **`Audio`** at `/lambda/nfs/Audio/lpf`, which **still exists**: clips, labels, runs and model caches (`HF_HOME`, `TORCH_HOME`). Only `~/venv` lived on the instance disk; [`setup_env.sh`](../lab/lambda/setup_env.sh) rebuilds it.

Scripts are in [`lab/lambda/`](../lab/lambda/), in pipeline order:

| Script | Does |
|---|---|
| `lpf_env.sh` | CUDA library paths, model caches on NFS, `cd` to the project. Every script sources it. |
| `setup_env.sh` | uv venv with Python 3.12, torch 2.11 cu128, transformers 5.22.1, faster-whisper 1.2.1, piper-tts 1.8 |
| `setup_data.sh` | 29 Piper voices (`voices.txt`), ESC-50, Parakeet / MMS / Whisper weights; checks Whisper runs on the GPU |
| `gen_tts.sh` | regenerates the 4 Piper splits (same commands as mute-detector.md §7) |
| `label_train.sh`, `label_rest.sh`, `label_fast.sh` | align, then Whisper-check, the training pool. `label_fast.sh` ran 3 Whisper workers in parallel on clips the first pass hadn't reached. |
| `build_split.sh` | `build_trainset.py` once labelling is done |
| `run_jobs.sh`, `run_queue.sh` | training workers. `run_queue.sh` takes jobs from `jobs.txt` with atomic lock directories and skips `done`/`failed` runs. Real-data jobs wait for `trainset.jsonl`. |
| `eval_queue.sh` | scores each finished run on validation, then test; also scores v3 as `v3_shipped` |
| `resume_all.sh` | after a crash or on a new instance, restarts whatever isn't finished; safe to run any time |

`train.py` writes `last.pt` atomically (via `last.tmp`) with optimizer and scheduler state, so a killed run resumes.

**On the laptop** ([`lab/`](../lab/)):
- [`dashboard.py`](../lab/dashboard.py): a progress page for both machines at `localhost:8765`.
- [`backup_lambda.sh`](../lab/backup_lambda.sh): copies the Lambda results (not the clips) to `data/lambda_backup/` every 30 min.
- [`finish_and_shutdown.sh`](../lab/finish_and_shutdown.sh): waits until both machines finish, pulls and checks every run's files, writes `results_overnight.txt`, terminates the instance through the Lambda API, then shuts the laptop down.
  - The API key is read from `~/.lambda_api_key`, outside the repo.
  - The script only terminates if the instance ID still maps to the expected IP.

## 8. Reproduce

```powershell
# Laptop: scan, test set, baselines
$env:TRANSFORMERS_VERBOSITY="error"
.venv-lab\Scripts\python lab\detector\scan_transcripts.py yodas --workers 16          # transcripts only, resumes
.venv-lab\Scripts\python lab\detector\fetch_yodas.py --top 60 --jobs 4                # densest archives, for the test set
.venv-lab\Scripts\python lab\detector\align_clips.py
.venv-lab\Scripts\python lab\detector\verify_whisper.py
.venv-lab\Scripts\python lab\detector\build_testset.py --target 450
.venv-lab\Scripts\python lab\detector\eval.py models\detector-v3.pt --sets real --thr 0.3 0.5 0.7 --confirm 2 --pad 0.06 0.02
.venv-lab\Scripts\python lab\detector\eval_baselines.py whisper parakeet
```

```bash
# Lambda (persistent filesystem mounted at /lambda/nfs/Audio): copy the repo to /lambda/nfs/Audio/lpf,
# copy data/realspeech/{hits_yodas.jsonl, yodas/split_videos.json, yodas/testset.jsonl + clips} there,
# then install lab/lambda/*.sh at the project root, ln -s /lambda/nfs/Audio/lpf/lpf_env.sh ~/lpf_env.sh
bash setup_env.sh && bash setup_data.sh && bash gen_tts.sh
bash resume_all.sh        # fetch -> label -> split -> 2 training workers -> eval queue; resumable
```

```powershell
# Laptop again: pull results and summarize
.venv-lab\Scripts\python lab\detector\summarize_runs.py data\lambda_backup\files\lab\detector\runs
```

Gitignored local data: `data/realspeech/` (scan hits, test clips, labels) and `data/lambda_backup/` (all 22 runs' checkpoints and logs, labels and splits). The clips are YouTube audio under CC BY 3.0. To release the test set, publish the clip IDs and times plus a script to fetch them, with attribution, not the audio itself.

## 9. Open items
1. **Search thresholds above 0.7 on validation**, then re-score test once.
2. **Figures:** delay vs. silenced vs. clean intact curves for the detector and both baselines, a learning curve over N real words, and results per category.
3. **People's Speech** as a second test source (fix the source-ID parsing), so the claim isn't tied to one dataset.
4. **A version that can be used commercially:** retrain `call` without ESC-50 (≈ `abl_nosfx`) using MUSAN / CC0 FSD50K, and audit the Piper voice licences.
5. **Onsets:** 0.35 s still leaks the start of some words. Try a start-offset regression in the head, or finer frames.
6. **A game session with v4** (private, qualitative only).
7. **Paper draft:** method, real-speech test set, baselines, learning curve, ablations, latency.

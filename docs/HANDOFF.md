# Handoff: state of the mute-detector work

Last updated **2026-11-06**. Pick up from here. Other docs:
- [real-speech.md](real-speech.md): the latest work (real-speech test set, training on real speech, v4).
- [mute-detector.md](mute-detector.md): v1–v3 on synthetic speech.
- [related-work-and-datasets.md](related-work-and-datasets.md): literature and licences.

## 1. What the user wants
- **Goal:** worst-case audio delay **under 0.4 s**, with every curse word silenced. The app's default is 1.2 s (Whisper `small.en`).
- **Main concerns after playing Call of Duty MW4 with it:** **false positives** (muting game sound and clean dialogue) and **latency** (`LATE` mutes).
- **Also interested in:**
  - a **research paper** (arXiv first, then a workshop or Interspeech)
  - **LinkedIn** posts (figures exist in `docs/images/`)
  - training only on **copyright-safe data**
- **Decision (2026-11-05): the paper uses public data only.** Game recordings are **not** used for training or for reported results, because nobody else can get them and the results couldn't be reproduced. A game session can appear at most as a private, qualitative case study.
- **Decision (2026-11-05): automatic checks only.** The user does not want to listen to clips or hand-check labels. Use model agreement (aligner + Whisper) instead.

## 2. Current state (2026-11-06)
- **App:** the "Ultra-low lag" preset uses **v4** (`lab/detector/models/detector-v4.pt`) at threshold **0.7** (`DETECTOR_THRESHOLD` in `censor.py`) and a delay of **0.40 s**.
  - VB-Cable self-test: PASS at 0.40 s (6/6, 0 late); FAIL at 0.35 s ("shit" kept 24% loudness).
  - The user is about to test it in a game.
- **Real speech** (details and every number in [real-speech.md](real-speech.md)), at 0.40 s:
  - **Test set:** 450 curse words + 13,172 clean words from 306 YouTube videos (YODAS2, CC BY 3.0), labelled by MMS alignment + Whisper large-v3 agreement.
  - **Synthetic + real training** (3 seeds): **93.0%** of curse words silenced, **95.0%** of clean words intact, 3.5 s muted per minute.
  - **v3:** 93.8% silenced but only 82.0% intact, 9.7 s/min.
  - **Whisper `small.en` at 0.40 s:** 15.3% silenced. It needs 1.2 s to reach 90%.
  - **Parakeet gap-fill at 0.40 s:** 75.8% silenced.
  - **Learning curve:** synthetic only → +500 → +2,000 → +all 3,477 real curse words. Most of the gain comes from the first 500.
  - **Ablations** (unheard labels, no context negatives, no sound effects): flat within seed noise on real speech.
- **Lambda:** the instance (gpu_1x_a10, 150.230.34.10) was **terminated automatically at 04:14 on 2026-11-06** after the results were pulled.
  - The persistent filesystem **`Audio`** (`/lambda/nfs/Audio/lpf`) still holds all clips, labels, runs and model caches. It costs storage money; delete it in the Lambda dashboard once nothing more is needed from it.
  - Everything important is also on the laptop in `data/lambda_backup/` (gitignored).

## 3. What exists (branch `research/mute-detector`)

| Piece | Where |
|---|---|
| Training vocabulary: 41 groups, 268 curse forms, 345 sound-alike negatives | `lab/detector/lexicon.py`, `words.txt` |
| Synthetic speech: sentence generator + Piper TTS with exact word boundaries | `lab/detector/corpus.py`, `make_tts.py` |
| Model: frozen Parakeet CTC 0.6B encoder + 0.93M head → per-80 ms-frame curse probability for 41 groups; CUDA graph (~12 ms/check) | `lab/detector/detector.py` |
| Training (synthetic + optional real clips, ablation flags, crash-safe resume) | `lab/detector/train.py` |
| Live-loop evaluation scored by loudness; sets `bench`, `tts`, `ctx`, `esc`, **`real`**; fixed check time (`--check-ms`) and a batched path | `lab/detector/eval.py` |
| Real-speech pipeline: transcript scan → clip fetch → MMS align → Whisper check → splits by video | `scan_transcripts.py`, `fetch_yodas.py`, `align_clips.py`, `verify_whisper.py`, `build_testset.py`, `build_trainset.py` (all in `lab/detector/`) |
| Baselines on real speech (Whisper `small.en`, Parakeet gap-fill) | `lab/detector/eval_baselines.py` |
| Per-run / per-group results with thresholds picked on validation | `lab/detector/summarize_runs.py` |
| Lambda scripts: env, data, TTS, labelling, split, job queue, eval queue, resume | `lab/lambda/` |
| Laptop tools: two-machine dashboard, 30-min Lambda backup, finish watcher (pull → summary → terminate → shutdown) | `lab/dashboard.py`, `lab/backup_lambda.sh`, `lab/finish_and_shutdown.sh` |
| App integration | `censor.py` (`detect_forever`), `app.pyw`, `app-lab.bat` |
| Trained heads (3.7 MB each) | `lab/detector/models/detector-v2.pt`, `detector-v3.pt`, **`detector-v4.pt` (what the app uses)** |
| Figures (v1–v3 era) | `lab/detector/figures.py` → `docs/images/01..06-*.png` |
| Game-session review tool (deprioritized) | `lab/detector/review_recording.py` |

## 4. Next steps (in order)
1. **The user plays a game with v4** and reports false mutes and leaks. This is qualitative only and stays private.
2. **Thresholds above 0.7:** most runs picked 0.7, the top of the grid searched. Re-run the validation eval at 0.7–0.9 on the `data/lambda_backup` checkpoints (laptop GPU), then re-score the test set once.
3. **Paper figures:**
   - delay vs. silenced vs. clean-intact curves (detector, Whisper, Parakeet)
   - learning curve over N real words
   - results per category (strong / mild / British / slur)
4. **Second test source:** People's Speech (fix the source-ID parsing in `scan_transcripts.py`). This keeps the claim from resting on one dataset.
5. **A model that can be used commercially:** retrain without ESC-50 (CC BY-NC) using MUSAN / CC0 FSD50K, and audit the 29 Piper voices' licences. `abl_nosfx` shows dropping ESC-50 costs almost nothing on real speech.
6. **Onsets:** 0.35 s still leaks the start of some words. Try a start-offset regression in the head, or finer frames.
7. **Paper draft (LaTeX):** method, test set construction, baselines, learning curve, ablations, latency on a laptop GPU, limitations. The limitations include automatic labels with about 87% precision on a pilot, one dataset, and anonymised video IDs, so a speaker could be in both training and test.
8. **App polish:** category toggles and a threshold slider. Avoid the add/delete/on-off word-list dialog claimed in US 12,300,222 (§6).

## 5. Old thread: review of the second game session (deprioritized)
- **Recording:** `data/recordings/20261104-163523/`, about 32.5 min of CoD campaign audio with v3 at 0.35 s, 148 mutes.
- **Tool:** `lab/detector/review_recording.py` re-transcribes it with Whisper large-v3 in 10 s windows (5 s hop, no VAD) and classifies each mute. It was interrupted before finishing.
- **Status:** not needed for the paper (§1 decision). Only worth finishing if the user wants a private sanity check of game behaviour.
- **Privacy:** copyrighted game audio, possibly other players' voices. Keep it local; never publish or commit it.

## 6. Licence, IP and security flags
- **Licences:**
  - **ESC-50 is CC BY-NC.** v3 and v4 were trained with it, so treat them as **research-only**.
  - **YODAS2 is CC BY 3.0:** attribute it. Release the test set as clip IDs and times plus a fetch script, not the audio.
- **Patents** (claims read 2026-11-05):
  - **Roblox** US 12,027,177 / 12,283,288 require a delay that depends on a toxicity score, inserted at pauses. The app's fixed delay plus muting falls outside them. A paper should claim the detector and the training method.
  - **US 12,300,222** (Numbers Fourteen Eight) claims a word list plus a specific add/delete/on-off UI. Avoid building that exact dialog.
  - **ClearPlay** US 9,066,046 expired on 2026-03-09.
  - **Intel Bleep** is product prior art only; no Intel patent was found.
- **Security:**
  - The Lambda API key (`~/.lambda_api_key`, outside the repo) and the `project-1` SSH key (`~/.ssh/lambda_project1`) were both pasted into a chat. **Delete both in the Lambda dashboard**, and use the `omen16` key (`~/.ssh/lambda_ed25519`) next time.
  - The Lambda workspace is shared ("Rodi's Account"). Never act on instances other than your own.

## 7. Commands
```powershell
$env:TRANSFORMERS_VERBOSITY="error"
.venv-lab\Scripts\python lab\detector\lexicon.py                                   # vocab vs words.txt
.venv-lab\Scripts\python lab\detector\eval.py models\detector-v4.pt --sets real --thr 0.7 --confirm 2 --pad 0.06 0.02 --check-ms 13
.venv-lab\Scripts\python lab\detector\eval_baselines.py whisper parakeet
.venv-lab\Scripts\python lab\detector\summarize_runs.py data\lambda_backup\files\lab\detector\runs
.venv-lab\Scripts\python selftest.py 0.4 --model detector                        # through VB-Cable, nothing played aloud
```
The real-speech pipeline and the Lambda run are in [real-speech.md §8](real-speech.md#8-reproduce). Synthetic data and v1–v3 training are in [mute-detector.md §7](mute-detector.md#7-reproduce).

**To ship a new head:**
1. Strip the run's `last.pt` to `{head, groups, step, args}` and save it as `lab/detector/models/detector-vN.pt`.
2. Point `DETECTOR_HEAD` (and `DETECTOR_THRESHOLD`) in `censor.py` at it.
3. Re-run the self-test at 0.35 and 0.40 s.

## 8. Data on disk (gitignored)
- `data/tts/{train, train_ctx, test, test_ctx}`: Piper utterances.
- `data/esc50_dl/ESC-50-master/`
- `data/realspeech/`: transcript-scan hits (`hits_yodas.jsonl`, `hits_peoples.jsonl`) and `yodas/`, which holds:
  - the test clips (`clips/`)
  - `clips.jsonl`, `aligned.jsonl`, `whisper.jsonl`
  - `testset.jsonl`, `split_videos.json`
- `data/lambda_backup/files/`: everything pulled from Lambda except the training clips:
  - the 22 runs with `last.pt`, `head.pt`, training and eval logs
  - `trainset.jsonl`, `valset.jsonl`, labels
  - the pipeline logs
- `data/recordings/20261104-163523/`: private game audio.
- `lab/detector/voices/`: 29 Piper voices.
- `lab/detector/runs/`: local v1–v3 runs.
- `results_overnight.txt`, `results_summary.md`: overnight result summaries (repo root, untracked).

## 9. Git
- Everything above is committed on `research/mute-detector` and pushed.
- `main` is the shipped Whisper app; don't change it unless asked.
- The GitHub repo `Muhammad-Hamza-cpp/live-profanity-filter` was made **private** on 2026-11-06.

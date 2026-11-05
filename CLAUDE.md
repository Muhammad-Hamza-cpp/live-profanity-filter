# CLAUDE.md

Windows desktop app that mutes curse words in all system audio. Apps play into VB-Cable; this program holds the audio back, detects curse words and mutes them, then plays the result to the real speakers.

- **Read [docs/HANDOFF.md](docs/HANDOFF.md) first.** It has the current state, results, open threads and next steps.
- Full research write-up: [docs/mute-detector.md](docs/mute-detector.md). Literature and dataset licences: [docs/related-work-and-datasets.md](docs/related-work-and-datasets.md).

## Branches
- `main`: the shipped app (Whisper `small.en`, 1.2 s delay). **Don't change it** unless asked.
- `research/mute-detector`: **the active branch**. A trained frame-level detector on a frozen Parakeet encoder, in the app as "Ultra-low lag" (v3, 0.40 s).
- `research/low-latency`, `research/whisper-slim`: earlier research. Read-only.

## Environment (Windows 11, RTX 4050 Laptop 6 GB, i5-14500HX, 32 GB)
- **Lab venv:** `.venv-lab`, with torch 2.11 cu128, transformers 5.22.1, faster-whisper, piper-tts 1.8, matplotlib. `.venv` (the app's own) doesn't exist on this machine; everything runs from `.venv-lab`.
- Run the app with `app-lab.bat`. The CLI is `.venv-lab\Scripts\python censor.py --model detector ...`.
- Set `TRANSFORMERS_VERBOSITY=error` to keep logs readable.
- VB-Cable is installed. Its devices are "CABLE Input" and "CABLE In 16ch"; the app uses CABLE Input and records from "CABLE Output". The user listens on **Headphone (Realtek(R) Audio)**.
- `selftest.py <delay> --model detector` checks the real audio path through VB-Cable without using the speakers.
- Gitignored local data: `data/` (TTS shards, ESC-50, recordings), `lab/detector/voices/` (Piper voices), `lab/detector/runs/`, `*.log`.

## Rules
- **Commits:** no `Co-Authored-By` or "Generated with" lines. Make logical commits, one per change. Commit or push only when the user asks.
- This clone has its own local git identity and credential helper for the user's personal GitHub account. Plain `git push` works; don't change the git config.
- **Never kill all `python.exe`/`pythonw.exe` processes.** Target processes by command line. The user may have the app open.
- **Judge muting by loudness, never by re-transcribing the output.** Whisper fills a silenced word back in from context.
- Before long GPU jobs, check nothing else is using the GPU (training, eval, the app). Concurrent jobs distort the timing numbers.

## Tooling gotchas
- **A `\n` inside a string literal sent through a Bash or PowerShell heredoc comes out as a real newline** and breaks Python string literals (this broke `censor.py` once). For code containing `"\n"`, use the Edit/Write tools, not shell scripts.
- PowerShell 5.1 wraps native stderr as `NativeCommandError`, and HF progress bars land there. That is noise, not a failure. Check the exit code and the log.
- The app runs under `pythonw`, which has no console, so exceptions in worker threads are silent. Check `data/recordings/*/mutes.jsonl` and the app log.
- Long `eval.py` runs take 10–25 min. Run them in the background and watch the log file.

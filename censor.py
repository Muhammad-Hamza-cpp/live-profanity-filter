"""Live profanity muter for Windows system audio.

Audio path while running:
    apps -> "CABLE Input" (made the Windows default) -> this program -> your real speakers

Everything is held back by --delay seconds. Meanwhile Whisper transcribes the audio on
the GPU, and any word from words.txt is silenced before it reaches the speakers.
Ctrl+C (or closing the window) puts your normal speakers back as the default device.
"""
import argparse
import ctypes
import os
import re
import site
import subprocess
import sys
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HELPER = HERE / "audio_default.ps1"
# Experimental trained detector (docs/mute-detector.md): needs the lab environment (torch).
DETECTOR = "detector"
DETECTOR_DIR = HERE / "lab" / "detector"
DETECTOR_HEAD = DETECTOR_DIR / "models" / "detector-v3.pt"
# Tuned with bench.py / bench_padding.py: small.en never needed more than 0.76 s, so 1.2 s leaves margin.
DEFAULT_MODEL = "small.en"
DEFAULT_DELAY = 1.2   # seconds
DEFAULT_STEP = 0.4    # seconds between transcriptions
# Measured bias of each model's word timestamps (seconds; negative = move earlier).
TIMESTAMP_SHIFT = {"tiny.en": -0.14}
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def add_cuda_dlls():
    # The pip-installed NVIDIA libraries ship DLLs that Windows won't find on its own.
    for sp in site.getsitepackages():
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            p = os.path.join(sp, sub)
            if os.path.isdir(p):
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]


# ---------- Windows default playback device ----------

def ps_helper(*args):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HELPER), *args],
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return out.stdout


def playback_devices():
    devs = []
    for line in ps_helper("list").splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3:
            devs.append({"id": parts[0], "name": parts[1], "default": parts[2] == "1"})
    return devs


# ---------- word list ----------

def load_words(path):
    patterns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            patterns.append(re.compile("^" + re.escape(line).replace(r"\*", ".*") + "$"))
    return patterns


def is_profane(word, patterns):
    w = word.strip().lower()
    if re.search(r"[a-z]\*", w):  # already bleeped by the transcriber, e.g. "f***"
        return True
    w = re.sub(r"[^a-z']", "", w).strip("'")
    return bool(w) and any(p.match(w) for p in patterns)


def mask(word):
    """'fucking,' -> 'f******' so logs don't show the words either."""
    w = re.sub(r"[^A-Za-z*]", "", word)
    return w[:1] + "*" * (len(w) - 1)


# ---------- engine ----------

class Censor:
    def __init__(self, model, patterns, sr, channels, delay, pad_before=0.06, pad_after=0.18,
                 window=4.0, step=0.4, verbose=False, log=log, on_mute=None, shift=0.0,
                 threshold=0.5, confirm=2, gate=None):
        self.model, self.patterns = model, patterns
        self.threshold, self.confirm, self.gate = threshold, confirm, gate
        self.events = []  # detector mutes: (start, end, group, prob, late), for recordings
        self.log, self.on_mute = log, on_mute
        self.sr, self.ch = sr, channels
        self.delay = int(delay * sr)
        self.pad_before, self.pad_after = pad_before - shift, pad_after + shift
        self.window, self.step = int(window * sr), int(step * sr)
        self.verbose = verbose
        self.fade = int(0.005 * sr)  # 5 ms ramps so mutes don't click
        g = gcd(16000, sr)
        self.up, self.down = 16000 // g, sr // g

        self.size = sr * 60  # 60 s ring buffer
        self.buf = np.zeros((self.size, channels), np.float32)
        self.write_pos = 0              # absolute sample index of next input sample
        self.play_pos = -self.delay     # absolute sample index of next output sample
        self.mutes = []                 # [start, end] absolute sample ranges to silence
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.muted_count = 0
        self.late_count = 0

    # input side
    def push(self, frames):
        n = len(frames)
        idx = np.arange(self.write_pos, self.write_pos + n) % self.size
        self.buf[idx] = frames[:, :self.ch]
        self.write_pos += n

    def _read(self, start, end):
        out = np.zeros((end - start, self.ch), np.float32)
        lo = max(start, 0, self.write_pos - self.size)
        if lo < end:
            out[lo - start:] = self.buf[np.arange(lo, end) % self.size]
        return out

    # output side
    def render(self, n):
        lag = self.write_pos - self.play_pos
        slack = int(0.3 * self.sr)
        if lag > self.delay + slack:      # output clock slower than input: skip ahead
            self.play_pos = self.write_pos - self.delay
        elif lag < self.delay - slack:    # output clock faster: hold until buffer refills
            return np.zeros((n, self.ch), np.float32)
        start = self.play_pos
        self.play_pos += n
        return self._read(start, start + n) * self._gain(start, n)[:, None]

    def _gain(self, start, n):
        g = np.ones(n, np.float32)
        with self.lock:
            regions = [tuple(m) for m in self.mutes if m[1] + self.fade > start]
        if regions:
            i = np.arange(start, start + n)
            for s, e in regions:
                if s - self.fade < start + n:
                    g = np.minimum(g, np.clip(np.maximum(s - i, i - e) / self.fade, 0, 1))
        return g

    def add_mute(self, s, e):
        with self.lock:
            # Keep already-played ranges for one window length: overlapping windows hear
            # the same word again, and that must not count as a new (late) word.
            self.mutes = [m for m in self.mutes if m[1] > self.write_pos - self.window - self.sr]
            for m in self.mutes:
                if s <= m[1] and e >= m[0]:
                    m[0], m[1] = min(m[0], s), max(m[1], e)
                    return False
            self.mutes.append([s, e])
            return True

    # transcription
    def transcribe_forever(self):
        from scipy.signal import resample_poly
        done = 0
        while not self.stop.is_set():
            end = self.write_pos
            if end - done < self.step:
                time.sleep(0.01)
                continue
            done = end
            start = max(0, end - self.window)
            audio = self._read(start, end).mean(axis=1)
            if np.sqrt(np.mean(audio ** 2)) < 1e-4:  # silence
                continue
            audio16 = resample_poly(audio, self.up, self.down).astype(np.float32)
            t0 = time.perf_counter()
            segments, _ = self.model.transcribe(
                audio16, language="en", beam_size=1, temperature=0.0,
                word_timestamps=True, condition_on_previous_text=False)
            words = [w for seg in segments for w in (seg.words or [])]
            took = time.perf_counter() - t0
            if self.verbose:
                self.log(f"({took * 1000:.0f} ms) " + "".join(w.word for w in words))
            for w in words:
                if not is_profane(w.word, self.patterns):
                    continue
                s = start + int((w.start - self.pad_before) * self.sr)
                e = start + int((w.end + self.pad_after) * self.sr)
                if self.add_mute(s, e):
                    self.muted_count += 1
                    margin = (s - self.play_pos) / self.sr
                    if margin < 0:
                        self.late_count += 1
                        self.log(f"muted '{mask(w.word)}' LATE by {-margin:.2f}s - increase the delay")
                    else:
                        self.log(f"muted '{mask(w.word)}'" + (f" ({margin:.2f}s to spare)" if self.verbose else ""))
                    if self.on_mute:
                        self.on_mute()

    def detect_forever(self):
        """Trained detector instead of a transcript: every step, score each 80 ms frame of the
        last 4 s, and mute audio that stays above the threshold for `confirm` checks in a row."""
        from scipy.signal import resample_poly
        from detector import FRAME, GROUP_NAMES, WINDOW
        groups = [i for i, g in enumerate(GROUP_NAMES) if g in enabled_groups(self.patterns)]
        win, frame, grid = int(WINDOW * self.sr), int(FRAME * self.sr), int(0.01 * self.sr)
        runs = {}  # 10 ms grid index -> checks in a row above the threshold
        done, times, last_report = 0, [], time.time()
        while not self.stop.is_set():
            end = self.write_pos
            if end - done < self.step:
                time.sleep(0.002)
                continue
            done = end
            audio = self._read(end - win, end).mean(axis=1)
            if np.sqrt(np.mean(audio[-self.step:] ** 2)) < 1e-4:  # silence
                runs = {}
                continue
            t0 = time.perf_counter()
            p, speech = self.model.score(resample_poly(audio, self.up, self.down).astype(np.float32))
            p = p[:, groups]
            took = time.perf_counter() - t0
            times.append(took)
            if time.time() - last_report > 30:  # is a game starving the GPU? (LATE mutes)
                t = np.array(times) * 1000
                self.log(f"detector checks: median {np.median(t):.0f} ms, p95 {np.percentile(t, 95):.0f} ms, "
                         f"{np.mean(t > self.step / self.sr * 1000):.0%} slower than the {self.step / self.sr * 1000:.0f} ms step")
                times, last_report = [], time.time()
            ok = p.max(-1) > self.threshold
            if self.gate:  # only where Parakeet's own CTC layer hears speech: not gunfire, music ...
                k, s_thr = self.gate
                ok &= np.convolve((speech > s_thr).astype(float), np.ones(2 * k + 1), mode="same") > 0
            above = {}
            for f in np.nonzero(ok)[0]:
                s = end - win + f * frame
                if s >= end:  # not heard yet
                    continue
                for i in range(s // grid, (s + frame) // grid):
                    above[i] = (groups[int(p[f].argmax())], float(p[f].max()))
            lo = (end - win) // grid
            runs = {i: runs.get(i, 0) + 1 for i in above if i >= lo}
            hit = sorted(i for i, n in runs.items() if n >= self.confirm)
            # contiguous runs of grid points -> mute ranges
            spans, start = [], None
            for a, b in zip(hit, hit[1:] + [None]):
                start = a if start is None else start
                if b != a + 1:
                    spans.append((start, a, above[start]))
                    start = None
            for a, b, (g, prob) in spans:
                s = a * grid - int(self.pad_before * self.sr)
                e = (b + 1) * grid + int(self.pad_after * self.sr)
                if e < self.play_pos - self.sr // 2:
                    continue  # a flicker on audio that played long ago, not a late word
                if self.add_mute(s, e):
                    self.muted_count += 1
                    self.events.append((s, e, GROUP_NAMES[g], prob, s < self.play_pos))
                    margin = (s - self.play_pos) / self.sr
                    name = mask(GROUP_NAMES[g])
                    if margin < 0:
                        self.late_count += 1
                        self.log(f"muted '{name}' LATE by {-margin:.2f}s - increase the delay")
                    else:
                        self.log(f"muted '{name}'" + (f" ({margin:.2f}s to spare)" if self.verbose else ""))
                    if self.on_mute:
                        self.on_mute()


def enabled_groups(patterns):
    """Detector groups whose words are still in words.txt (comment a word out to allow it)."""
    from lexicon import GROUPS
    return {g for g, (_, forms) in GROUPS.items()
            if any(is_profane(f, patterns) for f in forms if " " not in f)}


# ---------- session (shared by the app and the command line) ----------

_models = {}


def load_model(name, log=log):
    """Load (once) and warm up a Whisper model (or the detector). Returns (model, 'GPU' or 'CPU')."""
    if name == DETECTOR and name not in _models:
        sys.path.insert(0, str(DETECTOR_DIR))
        try:
            from detector import Detector
        except ImportError as e:
            raise RuntimeError(f"The detector needs the lab environment (.venv-lab, see app-lab.bat): {e}")
        if not DETECTOR_HEAD.exists():
            raise RuntimeError(f"Trained detector not found: {DETECTOR_HEAD}")
        log("Loading the trained detector (Parakeet encoder + head) ...")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        _models[name] = (Detector(DETECTOR_HEAD), "GPU")  # captures a CUDA graph = warmed up
    if name not in _models:
        add_cuda_dlls()
        from faster_whisper import WhisperModel
        log(f"Loading speech model '{name}' (the first time, it is downloaded) ...")
        try:
            model, where = WhisperModel(name, device="cuda", compute_type="float16"), "GPU"
        except Exception as e:  # no usable CUDA: fall back to CPU
            log(f"GPU not available ({e}); using CPU")
            model, where = WhisperModel(name, device="cpu", compute_type="int8"), "CPU"
        # Warm up the exact code path used live, so the first real words aren't slow.
        import scipy.signal  # noqa: F401  (slow first import)
        noise = np.random.default_rng(0).normal(0, 0.1, 16000 * 4).astype(np.float32)
        list(model.transcribe(noise, language="en", beam_size=1, word_timestamps=True)[0])
        _models[name] = (model, where)
    return _models[name]


def real_speakers(devs=None):
    return [d for d in (devs or playback_devices()) if "VB-Audio" not in d["name"]]


def find_sd_device(sd, wasapi, name_part, kind):
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == wasapi and d[f"max_{kind}_channels"] > 0 and name_part.lower() in d["name"].lower():
            return i
    return None


class Session:
    """One run of the filter: owns the audio streams and the Windows default-device switch."""

    def __init__(self, model=DEFAULT_MODEL, delay=DEFAULT_DELAY, step=DEFAULT_STEP, words=HERE / "words.txt",
                 output=None, switch_default=True, play=True, record=False, verbose=False,
                 log=log, on_mute=None, pad_before=0.06, pad_after=0.18, threshold=0.5, confirm=2,
                 gate=None, save_dir=None):
        self.model_name, self.delay, self.step, self.words = model, delay, step, words
        self.pads = pad_before, pad_after
        self.threshold, self.confirm, self.gate, self.save_dir = threshold, confirm, gate, save_dir
        self._saver = None
        self.output, self.switch_default, self.play, self.record = output, switch_default, play, record
        self.verbose, self.log, self.on_mute = verbose, log, on_mute
        self.censor, self.streams, self.recorded = None, [], []
        self.speaker, self._switched, self._stopped = None, False, False
        self._lock = threading.Lock()

    def start(self):
        add_cuda_dlls()
        import sounddevice as sd
        # PortAudio lists devices once per process. Rescan, so a device added or switched since
        # (VB-Cable installed, headphones plugged in) isn't opened through a stale entry.
        sd._terminate()
        sd._initialize()

        devs = playback_devices()
        cable = next((d for d in devs if d["name"].startswith("CABLE Input")), None)
        if cable is None:
            raise RuntimeError("VB-Audio 'CABLE Input' not found - is VB-Cable installed?")
        real = real_speakers(devs)
        if self.output:
            self.speaker = next((d for d in real if self.output.lower() in d["name"].lower()), None)
        else:
            self.speaker = next((d for d in real if d["default"]), None) or (real[0] if real else None)
        if self.speaker is None:
            raise RuntimeError("No speaker found. Devices: " + ", ".join(d["name"] for d in devs))

        wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])
        in_idx = find_sd_device(sd, wasapi, "CABLE Output", "input")
        out_idx = find_sd_device(sd, wasapi, self.speaker["name"], "output")
        if in_idx is None or (out_idx is None and self.play):
            raise RuntimeError(f"Could not open the audio devices (speaker: {self.speaker['name']}).")
        sr = int(sd.query_devices(in_idx)["default_samplerate"])
        channels = 2

        model, where = load_model(self.model_name, self.log)
        self.censor = Censor(model, load_words(self.words), sr, channels, self.delay, *self.pads, step=self.step,
                             verbose=self.verbose, log=self.log, on_mute=self.on_mute,
                             shift=TIMESTAMP_SHIFT.get(self.model_name, 0.0),
                             threshold=self.threshold, confirm=self.confirm, gate=self.gate)

        def on_input(indata, frames, t, status):
            self.censor.push(indata)

        def on_output(outdata, frames, t, status):
            block = self.censor.render(frames)
            outdata[:] = block
            if self.record:
                self.recorded.append(block.copy())

        settings = sd.WasapiSettings(auto_convert=True)
        self.streams = [sd.InputStream(device=in_idx, samplerate=sr, channels=channels, dtype="float32",
                                       callback=on_input, extra_settings=settings)]
        if self.play:
            self.streams.append(sd.OutputStream(device=out_idx, samplerate=sr, channels=channels, dtype="float32",
                                                callback=on_output, extra_settings=settings))
        try:
            for s in self.streams:
                s.start()
        except sd.PortAudioError as e:
            raise RuntimeError(f"Could not start audio on {self.speaker['name']} ({e}). If headphones were "
                               "just plugged in or out, or another app is using the device exclusively, "
                               "try Start again or pick the device under Speakers.") from e
        loop = self.censor.detect_forever if self.model_name == DETECTOR else self.censor.transcribe_forever
        threading.Thread(target=loop, daemon=True).start()
        if self.save_dir:
            self._saver = threading.Thread(target=self._save_loop, daemon=True)
            self._saver.start()
        if self.record and not self.play:  # no output device to pace us: render on our own clock
            threading.Thread(target=self._clock, args=(sr,), daemon=True).start()
        if self.switch_default:
            with self._lock:
                if self._stopped:  # stop() was called while we were starting up
                    raise RuntimeError("Cancelled")
                ps_helper("set", cable["id"])
                self._switched = True
        self.log(f"Filtering ON  | model {self.model_name} on {where} | delay {self.delay}s | "
                 f"output -> {self.speaker['name'] if self.play else '(none)'}")

    def _save_loop(self):
        """Keep what came in (16 kHz mono FLAC, a file every 30 s) and every mute decision, to train
        the detector on real game audio later. Everything stays in save_dir on this PC."""
        import json
        import soundfile as sf
        from scipy.signal import resample_poly
        c = self.censor
        out = Path(self.save_dir) / time.strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        (out / "session.json").write_text(json.dumps({
            "model": self.model_name, "delay": self.delay, "step": self.step, "threshold": self.threshold,
            "confirm": self.confirm, "gate": self.gate, "pads": self.pads, "sample_rate": 16000}, indent=2))
        self.log(f"Saving the audio for training to {out}")
        saved, part, n_events = 0, 0, 0
        while True:
            stopping = c.stop.wait(30)
            end = c.write_pos
            start = max(saved, end - c.size + c.sr)  # never older than the ring buffer still holds
            if end - start > c.sr // 10:  # skip a near-empty last part
                a = resample_poly(c._read(start, end).mean(axis=1), c.up, c.down).astype(np.float32)
                if np.abs(a).max() > 1e-4:  # nothing playing (game closed, menus): don't keep silence
                    sf.write(out / f"{part:04d}.flac", a, 16000)
                    with open(out / "parts.jsonl", "a") as f:
                        f.write(json.dumps({"file": f"{part:04d}.flac", "start": start / c.sr}) + "\n")
                    part += 1
                saved = end
            new, n_events = c.events[n_events:], len(c.events)
            if new:
                with open(out / "mutes.jsonl", "a") as f:
                    for s, e, g, prob, late in new:
                        f.write(json.dumps({"start": round(s / c.sr, 3), "end": round(e / c.sr, 3),
                                            "group": g, "prob": round(prob, 3), "late": bool(late)}) + "\n")
            if stopping:
                return

    def _clock(self, sr):
        last = time.perf_counter()
        while not self.censor.stop.is_set():
            n = int((time.perf_counter() - last) * sr)
            if n:
                self.recorded.append(self.censor.render(n))
                last += n / sr
            time.sleep(0.01)

    def stop(self):
        """Safe to call more than once, and from any thread."""
        with self._lock:
            self._stopped = True
            if self._switched:
                self._switched = False
                ps_helper("set", self.speaker["id"])
                self.log(f"Default output restored to {self.speaker['name']}")
        if self.censor and not self.censor.stop.is_set():
            self.censor.stop.set()
            self.log(f"Stopped. Muted {self.censor.muted_count} word(s), {self.censor.late_count} late.")
        streams, self.streams = self.streams, []
        for s in streams:
            s.stop()
            s.close()
        if self._saver:
            self._saver.join(timeout=10)  # last part of the recording
            self._saver = None

    def save_recording(self, path):
        import soundfile as sf
        if self.recorded:
            sf.write(path, np.concatenate(self.recorded), self.censor.sr)
            self.log(f"Saved {path}")


# ---------- command line ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, help=f"seconds audio is held back (default {DEFAULT_DELAY}; detector 0.4)")
    ap.add_argument("--step", type=float, help=f"seconds between checks (default {DEFAULT_STEP}; detector 0.05)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Whisper model (tiny.en, base.en, small.en, large-v3-turbo) or 'detector' (experimental)")
    ap.add_argument("--threshold", type=float, default=0.5, help="detector only: how sure before muting (0-1)")
    ap.add_argument("--gate", default="off",
                    help="detector only: mute only where speech is heard within K frames (K:S, or 'off')")
    ap.add_argument("--save", metavar="DIR", help="also keep the incoming audio + mute decisions here (training data)")
    ap.add_argument("--pads", type=float, nargs=2, metavar=("BEFORE", "AFTER"),
                    help="seconds of silence added around each word (default 0.06 0.18; detector 0.06 0.02)")
    ap.add_argument("--words", default=str(HERE / "words.txt"))
    ap.add_argument("--output", help="part of your speaker/headphone name (default: your current default device)")
    ap.add_argument("--no-switch", action="store_true", help="don't change the Windows default device")
    ap.add_argument("--no-speaker", action="store_true", help="don't play anything (use with --record)")
    ap.add_argument("--record", help="also save the censored output to this WAV file")
    ap.add_argument("--duration", type=float, help="stop after this many seconds")
    ap.add_argument("--verbose", action="store_true", help="print everything that is transcribed")
    args = ap.parse_args()

    det = args.model == DETECTOR
    args.delay = args.delay or (0.4 if det else DEFAULT_DELAY)
    args.step = args.step or (0.05 if det else DEFAULT_STEP)
    pads = args.pads or ((0.06, 0.02) if det else (0.06, 0.18))
    session = Session(model=args.model, delay=args.delay, step=args.step, words=args.words, output=args.output,
                      switch_default=not args.no_switch, play=not args.no_speaker, record=bool(args.record),
                      verbose=args.verbose, pad_before=pads[0], pad_after=pads[1], threshold=args.threshold,
                      gate=None if args.gate == "off" else (int(args.gate.split(":")[0]), float(args.gate.split(":")[1])),
                      save_dir=args.save)

    # Also restore the speakers if the console window is closed instead of Ctrl+C.
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
    def on_console_event(evt):
        if evt in (2, 5, 6):  # close, logoff, shutdown
            session.stop()
        return False
    ctypes.windll.kernel32.SetConsoleCtrlHandler(on_console_event, True)

    try:
        session.start()
        log("READY - press Ctrl+C to stop")
        t_end = time.time() + args.duration if args.duration else None
        while not t_end or time.time() < t_end:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        log(f"ERROR: {e}")
    finally:
        session.stop()
        if args.record:
            session.save_recording(args.record)


if __name__ == "__main__":
    main()

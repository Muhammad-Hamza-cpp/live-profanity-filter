"""Audio Censor - small desktop app around censor.Session (Start/Stop, settings, activity log)."""
import importlib.util
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from censor import DETECTOR, DETECTOR_HEAD, HERE, Session, ps_helper, playback_devices, real_speakers

SETTINGS = HERE / "settings.json"

# From bench.py / bench_padding.py. Delay = worst case measured + safety margin;
# pads = least silence around a word that let nothing leak through.
PRESETS = {
    "Balanced - videos & games (recommended)":
        {"model": "small.en", "step": 0.4, "delay": 1.2, "pad_before": 0.06, "pad_after": 0.18},
    "Low lag - light on GPU, for games":
        {"model": "tiny.en", "step": 0.3, "delay": 0.8, "pad_before": 0.08, "pad_after": 0.18},
    "Most accurate - heavy GPU, videos only":
        {"model": "large-v3-turbo", "step": 0.4, "delay": 1.5, "pad_before": 0.06, "pad_after": 0.18},
}
# Trained frame detector (docs/mute-detector.md). Only offered when it can run: lab environment + weights.
if DETECTOR_HEAD.exists() and importlib.util.find_spec("torch"):
    PRESETS["Ultra-low lag - trained detector (experimental)"] = {
        "model": DETECTOR, "step": 0.05, "delay": 0.35, "pad_before": 0.06, "pad_after": 0.02, "threshold": 0.5}


class App:
    def __init__(self, root):
        self.root = root
        self.session = None
        self.busy = False
        self.muted = 0
        self.events = queue.Queue()  # (kind, payload) from worker threads
        self.settings = self.load_settings()

        root.title("Audio Censor")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        style = ttk.Style()
        style.configure("Big.TButton", font=("Segoe UI", 12, "bold"), padding=(18, 8))
        style.configure("Status.TLabel", font=("Segoe UI", 12, "bold"))

        frm = ttk.Frame(root, padding=14)
        frm.grid(sticky="nsew")

        self.status = ttk.Label(frm, text="●  Filter is OFF", style="Status.TLabel", foreground="#888")
        self.status.grid(row=0, column=0, columnspan=2, sticky="w")
        self.button = ttk.Button(frm, text="Start", style="Big.TButton", command=self.toggle)
        self.button.grid(row=0, column=2, sticky="e")

        ttk.Separator(frm).grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Label(frm, text="Speakers").grid(row=2, column=0, sticky="w", pady=3, padx=(0, 10))
        self.speaker = ttk.Combobox(frm, state="readonly", width=38)
        self.speaker.grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)

        ttk.Label(frm, text="Mode").grid(row=3, column=0, sticky="w", pady=3, padx=(0, 10))
        self.preset = ttk.Combobox(frm, state="readonly", values=list(PRESETS), width=38)
        self.preset.grid(row=3, column=1, columnspan=2, sticky="ew", pady=3)
        self.preset.bind("<<ComboboxSelected>>", self.on_preset)

        ttk.Label(frm, text="Delay").grid(row=4, column=0, sticky="w", pady=3, padx=(0, 10))
        self.delay = tk.DoubleVar(value=self.settings.get("delay", next(iter(PRESETS.values()))["delay"]))
        self.delay_scale = ttk.Scale(frm, from_=0.3, to=4.0, variable=self.delay,
                                     command=lambda _: self.delay_label.config(text=f"{self.delay.get():.2f} s"))
        self.delay_scale.grid(row=4, column=1, sticky="ew", pady=3)
        self.delay_label = ttk.Label(frm, text=f"{self.delay.get():.2f} s", width=6)
        self.delay_label.grid(row=4, column=2, sticky="w", padx=(6, 0))

        row = ttk.Frame(frm)
        row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self.count_label = ttk.Label(row, text="Muted this session: 0")
        self.count_label.pack(side="left")
        ttk.Button(row, text="Edit word list", command=lambda: os.startfile(HERE / "words.txt")).pack(side="right")

        self.log_box = tk.Text(frm, height=12, width=58, state="disabled", font=("Consolas", 9),
                               relief="solid", borderwidth=1, wrap="word")
        self.log_box.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        ttk.Label(frm, text="Speech recognition runs on this PC - nothing is sent online.",
                  foreground="#888").grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))
        frm.columnconfigure(1, weight=1)

        saved = self.settings.get("preset")
        self.preset.set(saved if saved in PRESETS else next(iter(PRESETS)))
        self.refresh_speakers()
        self.root.after(100, self.pump)

    # ---- settings / devices ----
    def load_settings(self):
        try:
            return json.loads(SETTINGS.read_text())
        except Exception:
            return {}

    def save_settings(self):
        SETTINGS.write_text(json.dumps({"speaker": self.speaker.get(), "preset": self.preset.get(),
                                        "delay": round(self.delay.get(), 2)}, indent=2))

    def refresh_speakers(self):
        devs = playback_devices()
        speakers = real_speakers(devs)
        names = [d["name"] for d in speakers]
        self.speaker["values"] = names
        saved = self.settings.get("speaker")
        default = next((d["name"] for d in speakers if d["default"]), None)
        self.speaker.set(saved if saved in names else default or (names[0] if names else ""))
        # Left on the cable by a crash last time? Put real speakers back so there's sound.
        if any(d["default"] and "VB-Audio" in d["name"] for d in devs) and speakers:
            target = next(d for d in speakers if d["name"] == self.speaker.get())
            ps_helper("set", target["id"])
            self.add_log(f"Sound was left on the virtual cable - restored {target['name']}")

    def on_preset(self, _=None):
        self.delay.set(PRESETS[self.preset.get()]["delay"])
        self.delay_label.config(text=f"{self.delay.get():.2f} s")

    # ---- start / stop ----
    def toggle(self):
        if self.busy:
            return
        self.busy = True
        self.button.config(state="disabled")
        if self.session:
            self.set_status("Stopping...", "#c80")
            threading.Thread(target=self._stop_worker, daemon=True).start()
        else:
            self.save_settings()
            p = PRESETS[self.preset.get()]
            self.muted = 0
            self.count_label.config(text="Muted this session: 0")
            self.session = Session(model=p["model"], step=p["step"], delay=round(self.delay.get(), 2),
                                   pad_before=p["pad_before"], pad_after=p["pad_after"],
                                   threshold=p.get("threshold", 0.5),
                                   output=self.speaker.get() or None,
                                   log=lambda m: self.events.put(("log", m)),
                                   on_mute=lambda: self.events.put(("mute", None)))
            self.set_status("Starting...", "#c80")
            for w in (self.speaker, self.preset, self.delay_scale):
                w.config(state="disabled")
            threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self):
        session = self.session
        try:
            session.start()
            self.events.put(("started", None))
        except Exception as e:
            session.stop()
            self.events.put(("failed", str(e)))

    def _stop_worker(self):
        self.session.stop()
        self.events.put(("stopped", None))

    def pump(self):
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "log":
                self.add_log(payload)
            elif kind == "mute":
                self.muted += 1
                self.count_label.config(text=f"Muted this session: {self.muted}")
            elif kind == "started":
                self.busy = False
                self.set_status("●  Filter is ON", "#1a7f37")
                self.button.config(text="Stop", state="normal")
            elif kind in ("stopped", "failed"):
                self.session, self.busy = None, False
                self.set_status("●  Filter is OFF", "#888")
                self.button.config(text="Start", state="normal")
                self.speaker.config(state="readonly")
                self.preset.config(state="readonly")
                self.delay_scale.config(state="normal")
                if kind == "failed":
                    self.add_log(f"ERROR: {payload}")
                    messagebox.showerror("Audio Censor", payload)
        self.root.after(100, self.pump)

    def set_status(self, text, color):
        self.status.config(text=text, foreground=color)

    def add_log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", time.strftime("%H:%M:%S  ") + msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def on_close(self):
        if self.session:
            self.session.stop()  # always give the speakers back before exiting
        self.root.destroy()


if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()

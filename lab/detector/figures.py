r"""Render the summary figures in docs/images/ (1200x1500 px, 4:5 - sized for a LinkedIn post).

Every number comes from docs/mute-detector.md (measured on the RTX 4050 laptop).

    .venv-lab\Scripts\python -m pip install matplotlib
    .venv-lab\Scripts\python lab\detector\figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "images"

# Light chart chrome from the dataviz reference palette; one accent (blue) + de-emphasis gray.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, ACCENT, GRAY = "#e1e0d9", "#c3c2b7", "#2a78d6", "#b4b3ad"
ACCENT_WASH, CARD = "#e8f1fc", "#f3f2ee"
W, H, DPI = 6.0, 7.5, 200
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})


def canvas(title, subtitle):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    fig.text(0.07, 0.945, title, fontsize=17, fontweight="semibold", va="top", wrap=True)
    sub_y = 0.945 - 0.042 * (title.count("\n") + 1) - 0.022  # right under the title, however long
    fig.text(0.07, sub_y, subtitle, fontsize=10.5, color=INK2, va="top", linespacing=1.4)
    return fig


def footer(fig, text="Live profanity filter · research/mute-detector · measured on an RTX 4050 Laptop GPU"):
    fig.text(0.07, 0.03, text, fontsize=7.5, color=MUTED)


def box(ax, x, y, w, h, title, body="", accent=False, title_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018",
                                facecolor=ACCENT_WASH if accent else CARD,
                                edgecolor=ACCENT if accent else "none", linewidth=1.2))
    ax.text(x + 0.03, y + h - 0.022, title, fontsize=title_size, fontweight="semibold", va="top")
    if body:
        ax.text(x + 0.03, y + h - 0.022 - 0.038, body, fontsize=8.6, color=INK2, va="top", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 color=MUTED, linewidth=1.2))
    if label:
        ax.text((x0 + x1) / 2 + 0.02, (y0 + y1) / 2, label, fontsize=8, color=INK2, va="center")


def diagram_axes(fig, top=0.80, bottom=0.07):
    ax = fig.add_axes([0.05, bottom, 0.90, top - bottom])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def hbar(ax, labels, values, unit, highlight, fmt="{:.2f}"):
    """Horizontal bars, one highlighted; value at every tip (relief for the light gray)."""
    y = list(range(len(values)))[::-1]
    colors = [ACCENT if i == highlight else GRAY for i in range(len(values))]
    ax.barh(y, values, height=0.32, color=colors, zorder=2)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.02, yi, fmt.format(v) + unit, va="center", fontsize=11,
                fontweight="semibold", color=INK)
    ax.set_yticks(y, labels, fontsize=9.5)
    ax.set_xlim(0, max(values) * 1.25)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.set_facecolor(SURFACE)


# ---------- 1. summary card ----------

def summary():
    fig = canvas("Muting swear words in live game\naudio with 0.4 s of delay",
                 "A small detector trained on top of a frozen speech model,\n"
                 "running locally on a gaming laptop. No transcript, no cloud.")
    tiles = [
        ("Audio delay", "0.40 s", "was 1.2 s with Whisper in the same app"),
        ("Curse words caught", "30 / 30", "test voices never seen in training"),
        ("False mutes on non-speech", "−91%", "v3 vs v2, held-out sound effects"),
        ("GPU time per check", "12 ms", "a check every 50 ms (CUDA graph)"),
        ("Trained parameters", "0.93 M", "on a frozen 600 M Parakeet encoder"),
        ("Training speech", "31.6 h", "synthetic, ~1,070 voices, exact word timings"),
    ]
    ax = diagram_axes(fig, top=0.765, bottom=0.075)
    for i, (label, value, note) in enumerate(tiles):
        col, row = i % 2, i // 2
        x, y, w, h = 0.02 + col * 0.49, 0.69 - row * 0.335, 0.47, 0.30
        accent = i == 0
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor=ACCENT_WASH if accent else CARD,
                                    edgecolor=ACCENT if accent else "none", linewidth=1.2))
        ax.text(x + 0.035, y + h - 0.04, label, fontsize=9.5, color=INK2, va="top")
        ax.text(x + 0.035, y + h * 0.50, value, fontsize=26, fontweight="semibold", va="center")
        ax.text(x + 0.035, y + 0.045, note, fontsize=7.6, color=INK2, va="bottom", wrap=True)
    footer(fig, "Lab and synthetic-speech results; first live Call of Duty test: every real swear word was blocked.")
    return fig


# ---------- 2. how it works ----------

def how_it_works():
    fig = canvas("How it works", "Audio is held back 0.40 s. Every 50 ms the detector scores the last\n"
                 "4 s and mutes any part that is a curse word - before it plays.")
    ax = diagram_axes(fig)
    x, w = 0.06, 0.88
    steps = [
        ("Game, video, voice chat", "Windows plays everything into a virtual cable (VB-Cable)", False),
        ("Delay buffer: 0.40 s", "Audio waits here while the detector decides", False),
        ("Parakeet CTC 0.6B encoder  ·  frozen", "Last 4 s of audio → 24 Conformer layers, ~600 M parameters\n"
         "Encoder + head: 12 ms per check on the laptop GPU (one CUDA graph)", False),
        ("Detector head  ·  trained, 0.93 M parameters", "For every 80 ms frame: P(inside a curse word)\nfor each of 41 word groups (switch groups on/off)", True),
        ("Mute decision", "Mute where 2 checks in a row agree · 5 ms fades", False),
        ("Your headphones", "Clean audio, 0.40 s late", False),
    ]
    gap, top = 0.040, 0.98
    for i, (t, b, acc) in enumerate(steps):
        h = 0.085 + 0.032 * (b.count("\n") + 1)  # sized to its text
        y = top - h
        box(ax, x, y, w, h, t, b, accent=acc)
        if i:
            arrow(ax, 0.5, top + gap - 0.004, 0.5, top + 0.004)
        top = y - gap
    footer(fig)
    return fig


# ---------- 3. latency ----------

def latency():
    fig = canvas("3× less delay than the Whisper default",
                 "Audio delay you hear in each mode of the same app.\nAll modes silence every curse word in their tests.")
    ax = fig.add_axes([0.36, 0.30, 0.56, 0.48])
    hbar(ax, ["Whisper small.en\n(app default)", "Whisper tiny.en\n(Low lag mode)", "Trained detector\n(this work)"],
         [1.2, 0.8, 0.4], " s", highlight=2, fmt="{:.2f}")
    ax.set_xlabel("seconds of delay", fontsize=8.5, color=MUTED)
    fig.text(0.07, 0.22, "Why not lower? A word can't be called before its deciding sound is\n"
             "heard: \"ass\" vs \"class\", \"hell\" vs \"hello\". The detector calls a typical\n"
             "curse word 0.14 s after it starts (median); 99% within 0.35 s.",
             fontsize=8.8, color=INK2, va="top", linespacing=1.45)
    footer(fig)
    return fig


# ---------- 4. training data ----------

def training_data():
    fig = canvas("Training data, built from scratch",
                 "No dataset labels curse words frame by frame - so it was\n"
                 "synthesized, with word boundaries known to the sample.")
    ax = diagram_axes(fig)
    steps = [
        ("Lexicon", "41 word groups · 268 curse forms\n345 sound-alikes that must stay audible (duck, count, class …)"),
        ("Sentence generator", "Curse words in natural slots · the same slots with clean words\n(\"what the heck\") · sound-alikes · everyday sentences"),
        ("Piper text-to-speech", "29 voices, ~1,070 speakers, accents incl. L2-ARCTIC & VCTK\nPhoneme timings → word starts within 6 ms of the true onset"),
        ("31.6 h of labelled speech + 1,600 sound effects", "Noise, music, babble, reverb, phone band, ESC-50 effects\nWindows cut to end mid-word, where the early call is made"),
        ("Train the head  ·  20 min on the laptop", "Encoder frozen · loss only on audio already heard\nTest: held-out voices, unseen TTS engine, held-out sound effects"),
    ]
    h, gap = 0.142, 0.045
    for i, (t, b) in enumerate(steps):
        y = 0.98 - (i + 1) * h - i * gap
        box(ax, 0.06, y, 0.88, h, t, b, accent=i == 4)
        if i:
            arrow(ax, 0.5, y + h + gap - 0.004, 0.5, y + h + 0.004)
    footer(fig)
    return fig


# ---------- 5. what went wrong, and the fixes ----------

def iterations():
    fig = canvas("Three versions, two big mistakes",
                 "Each version was measured on a simulated live loop before\nthe next change. What broke, and what fixed it:")
    ax = diagram_axes(fig, top=0.80, bottom=0.08)
    rows = [
        ("v1", "Guessed from context", "Labels marked unheard audio as \"curse\", so the model\n"
         "learned \"what the …\" means a swear is coming. 69% of its\nfalse mutes fired within 0.1 s - guessing, not listening.",
         "Fix: learn only from audio already heard + 15,000\nsentences with clean words in the same slots"),
        ("v2", "Tripped by game sounds", "First Call of Duty session: every real swear word was\n"
         "blocked, but gunfire and music triggered mutes too -\nit had never heard non-speech sound.",
         "Fix: 1,600 real sound-effect clips as negatives\n(engines, fireworks, glass, thunder …)"),
        ("v3", "Current", "Non-speech false mutes: 1.1 → 0.1 s per minute (−91%)\n"
         "Curse words caught (held-out bench, 0.40 s): 30 / 30\nSame latency: median 0.14 s to call a word",
         "Next: train on recorded game audio, labelled\noffline by Whisper large-v3"),
    ]
    h, gap = 0.29, 0.035
    for i, (tag, head, body, fix) in enumerate(rows):
        y = 0.98 - (i + 1) * h - i * gap
        acc = i == 2
        ax.add_patch(FancyBboxPatch((0.04, y), 0.92, h, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor=ACCENT_WASH if acc else CARD,
                                    edgecolor=ACCENT if acc else "none", linewidth=1.2))
        ax.text(0.075, y + h - 0.03, tag, fontsize=18, fontweight="semibold", va="top",
                color=ACCENT if acc else INK)
        ax.text(0.20, y + h - 0.035, head, fontsize=11, fontweight="semibold", va="top")
        ax.text(0.20, y + h - 0.085, body, fontsize=8.4, color=INK2, va="top", linespacing=1.35)
        ax.text(0.20, y + 0.025, fix, fontsize=8.4, color=INK, va="bottom", linespacing=1.35,
                fontweight="semibold" if acc else "normal")
    footer(fig)
    return fig


# ---------- 6. CUDA graph ----------

def cuda_graph():
    fig = canvas("4× faster checks with one CUDA graph",
                 "Time for one 4 s window through the 600 M-parameter encoder\non the RTX 4050 laptop (bf16).")
    ax = fig.add_axes([0.36, 0.40, 0.56, 0.36])
    hbar(ax, ["Eager PyTorch", "CUDA graph"], [47.1, 11.7], " ms", highlight=1, fmt="{:.0f}")
    ax.set_xlabel("milliseconds per check", fontsize=8.5, color=MUTED)
    fig.text(0.07, 0.31, "Eager mode spends most of its time launching hundreds of small GPU\n"
             "kernels. Capturing the whole encoder + detector once as a CUDA graph\n"
             "and replaying it makes a check cost ~12 ms at any window length\n"
             "from 1 to 4 s - so the detector can check every 50 ms.",
             fontsize=8.8, color=INK2, va="top", linespacing=1.45)
    footer(fig)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in [("01-summary", summary), ("02-how-it-works", how_it_works), ("03-latency", latency),
                       ("04-training-data", training_data), ("05-iterations", iterations),
                       ("06-cuda-graph", cuda_graph)]:
        fig = make()
        fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    main()

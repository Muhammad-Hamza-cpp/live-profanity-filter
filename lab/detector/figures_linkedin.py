r"""Render the two LinkedIn images (1200x1500 px, 4:5, dark) into docs/images/linkedin/.

    07-how-it-works.png   the live loop on one real sentence: audio in, model scores, audio out
    08-architecture.png   detector v4: frozen Parakeet encoder + trained head, live rule, training data

Both use data/linkedin_demo.npz (run lab/detector/linkedin_demo.py first): detector v4's real
output on a held-out voice. Headline numbers are from docs/real-speech.md.

    .venv-lab\Scripts\python lab\detector\figures_linkedin.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from lexicon import GROUPS  # noqa: E402

OUT = ROOT / "docs" / "images" / "linkedin"
SR, FRAME = 16000, 0.08

BG, CARD, CARD2, LINE = "#0d1117", "#161b24", "#1e2531", "#2b3341"
INK, INK2, MUTED = "#f3f4f6", "#a9b1be", "#6c7585"
ACCENT, ACCENT_WASH = "#4c9bff", "#13233a"
CURSE, CURSE_WASH = "#ff6b5b", "#3b1f1f"
W, H, DPI = 6.0, 7.5, 200
R = H / W  # the canvas axes run 0..1 across and 0..R up, so units are equal both ways
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "text.color": INK})
SHOWN = ["fuck", "shit", "bitch", "ass", "damn", "hell", "crap", "bloody"]  # heatmap rows (no slurs)


def mask(w):
    """sh*t, f*ck, bl**dy: star the vowels after the first letter."""
    return w[0] + "".join("*" if c in "aeiou" else c for c in w[1:])


def canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, R)
    ax.axis("off")
    return fig, ax


def sub_axes(fig, x, y, w, h, xlim=None):
    """An axes placed in canvas units (y and h in 0..R)."""
    a = fig.add_axes([x, y / R, w, h / R], facecolor=BG)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xticks([])
    a.set_yticks([])
    if xlim:
        a.set_xlim(*xlim)
    return a


def header(ax, title, subtitle):
    ax.text(0.06, 1.195, title, fontsize=19, fontweight="bold", va="top", linespacing=1.15)
    ax.text(0.06, 1.195 - 0.052 * (title.count("\n") + 1) - 0.018, subtitle, fontsize=9.5,
            color=INK2, va="top", linespacing=1.4)


def card(ax, x, y, w, h, face=CARD, edge=None, r=0.016):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=face, edgecolor=edge or "none", linewidth=1.1))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10, color=color,
                                 linewidth=lw, shrinkA=0, shrinkB=0))


def badge(ax, x, y, n, color=ACCENT):
    ax.add_patch(Circle((x, y), 0.019, facecolor=color, edgecolor="none"))
    ax.text(x, y - 0.001, str(n), fontsize=9, fontweight="bold", ha="center", va="center", color=BG)


def footer(ax, text):
    ax.text(0.06, 0.022, text, fontsize=6.8, color=MUTED, va="bottom", linespacing=1.4)


def envelope(audio, t0=0.0, hop=0.01, ref=None):
    n = int(hop * SR)
    a = np.abs(audio[: len(audio) // n * n]).reshape(-1, n).max(1)
    return t0 + np.arange(len(a)) * hop, a / (ref or max(1e-6, np.abs(audio).max()))


def load_words(d):
    return [(str(w), bool(int(c)), float(s), float(e)) for w, c, s, e in d["words"]]


# ---------- 07: how it works, on real model output ----------

def how_it_works(d):
    audio, muted, belief, thr, delay = d["audio"], d["muted"], d["belief"], float(d["thr"]), float(d["delay"])
    words = load_words(d)
    cw = next(w for w in words if w[1])
    decided = float(np.min(d["first_mute"][int(cw[2] / 0.01):int(cw[3] / 0.01)])) - cw[2]
    xlim = (0.45, words[-1][3] + delay + 0.12)
    X, WD = 0.08, 0.86  # panel left edge and width

    fig, ax = canvas()
    header(ax, "A live profanity filter\nfor Call of Duty",
           "How the AI mutes a swear word before it reaches your headphones.\n"
           "Real output, on a voice it never heard in training.")

    def panel_title(y, n, text, color=ACCENT):
        badge(ax, 0.085, y, n, color)
        ax.text(0.12, y, text, fontsize=10.5, fontweight="semibold", va="center")

    def word_labels(a, shift=0.0, struck=False):
        for w, c, s, e in words:
            a.text((s + e) / 2 + shift, -1.3, mask(w.lower()) if c else w, ha="center", va="top",
                   fontsize=7.8, color=CURSE if c else INK2, fontweight="semibold" if c else "normal")

    ref = np.abs(audio).max()

    # 1. audio in
    panel_title(0.985, 1, "Sound comes in from the game, video or chat")
    a1 = sub_axes(fig, X, 0.79, WD, 0.165, xlim)
    t, env = envelope(audio, ref=ref)
    a1.axvspan(cw[2], cw[3], ymin=0.2, color=CURSE_WASH, zorder=0)
    a1.fill_between(t, -env, env, color=INK2, linewidth=0)
    sel = (t >= cw[2] - 0.005) & (t <= cw[3])
    a1.fill_between(t[sel], -env[sel], env[sel], color=CURSE, linewidth=0)
    a1.set_ylim(-1.75, 1.1)
    word_labels(a1)

    # 2. the model's belief per 80 ms frame, at the last check before that frame had to play
    panel_title(0.75, 2, "The AI rates every 80 ms: \"is this a swear word?\"")
    a2 = sub_axes(fig, X, 0.575, WD, 0.145, xlim)
    ft = np.arange(len(belief)) * FRAME
    keep = (ft >= xlim[0]) & (ft <= words[-1][3] + 0.05)
    a2.bar(ft[keep] + FRAME / 2, belief[keep], width=FRAME * 0.78,
           color=[CURSE if b > thr else ACCENT for b in belief[keep]], linewidth=0)
    a2.axhline(thr, color=INK2, lw=0.8, ls=(0, (3, 3)))
    a2.text(xlim[0] + 0.01, thr + 0.04, f"mute above {thr:.1f}", ha="left", va="bottom", fontsize=7.2, color=INK2)
    a2.axhline(0, color=LINE, lw=0.8)
    a2.set_ylim(0, 1.08)
    a2.annotate(f"sure {decided:.2f} s into the word\n(it had {delay:.2f} s to decide)",
                xy=(cw[3] + 0.05, 0.93), xytext=(cw[3] + 0.32, 0.93), fontsize=7.8, color=INK, va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8))

    # 3. audio out, 0.40 s later, word silenced
    panel_title(0.525, 3, f"You hear it {delay:.2f} s later, with the word gone", CURSE)
    a3 = sub_axes(fig, X, 0.315, WD, 0.18, xlim)
    out = audio.copy()
    m = np.repeat(muted, int(0.01 * SR))[: len(out)]
    out[: len(m)][m] = 0
    t3, env3 = envelope(out, t0=delay, ref=ref)
    a3.axvspan(cw[2] + delay, cw[3] + delay, ymin=0.2, color=CURSE_WASH, zorder=0)
    a3.fill_between(t3, -env3, env3, color=INK2, linewidth=0)
    a3.plot([cw[2] + delay, cw[3] + delay], [0, 0], color=CURSE, lw=1.4)
    a3.text((cw[2] + cw[3]) / 2 + delay, 0.55, "silence", ha="center", fontsize=7.8, color=CURSE,
            fontweight="semibold")
    # where the word was, and the 0.40 s shift
    a3.axvline(cw[2], ymin=0.25, ymax=0.95, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    a3.annotate("", xy=(cw[2] + delay, 0.95), xytext=(cw[2], 0.95),
                arrowprops=dict(arrowstyle="-|>", color=CURSE, lw=1.0, mutation_scale=8))
    a3.text(cw[2] + delay / 2, 1.0, f"+{delay:.2f} s", ha="center", va="bottom", fontsize=7.4, color=CURSE)
    a3.set_ylim(-1.75, 1.3)
    word_labels(a3, shift=delay)

    # stat tiles
    tiles = [("93%", "of real swear words\nsilenced"), ("95%", "of clean words\nleft untouched"),
             ("0.40 s", "of delay; Whisper\nneeds 1.2 s for 90%")]
    for i, (v, lab) in enumerate(tiles):
        x = 0.06 + i * 0.3
        card(ax, x, 0.09, 0.28, 0.15, face=ACCENT_WASH if i == 0 else CARD, edge=ACCENT if i == 0 else None)
        ax.text(x + 0.025, 0.198, v, fontsize=20, fontweight="bold", va="center")
        ax.text(x + 0.025, 0.135, lab, fontsize=7.8, color=INK2, va="center", linespacing=1.3)
    footer(ax, "Tiles: 450 swear words + 13,172 clean words from 306 public YouTube videos (YODAS2), 0.40 s delay.\n"
               "Example: held-out Piper voice; one take, typical of 10 (sh*t 100% silenced in every take).")
    return fig


# ---------- 08: architecture ----------

def architecture(d):
    fig, ax = canvas()
    header(ax, "Inside the AI profanity\nfilter for Call of Duty",
           "A big pretrained speech model hears the game audio; a tiny model I\n"
           "trained spots the swear words in it. No transcript, 12 ms per check.")

    # input
    card(ax, 0.06, 0.925, 0.88, 0.055, face=CARD2)
    ax.text(0.09, 0.9525, "Game audio in: the last 4 s", fontsize=10, fontweight="semibold", va="center")
    ax.text(0.91, 0.9525, "16 kHz → log-mel features, on the GPU", fontsize=8.5, color=INK2, va="center", ha="right")
    arrow(ax, 0.265, 0.925, 0.265, 0.905)

    # frozen encoder with 24 blocks
    card(ax, 0.06, 0.5, 0.41, 0.405)
    ax.text(0.085, 0.88, "① Hears the speech", fontsize=10.5, fontweight="semibold", va="top")
    ax.text(0.085, 0.848, "Parakeet CTC 0.6B encoder (NVIDIA)\npretrained · frozen · ~600 M params",
            fontsize=7.6, color=INK2, va="top", linespacing=1.3)
    top, step, taps = 0.79, 0.0108, (12, 16, 20, 24)
    ys = {}
    for k in range(1, 25):
        y = top - (k - 0.5) * step
        ys[k] = y
        tap = k in taps
        ax.add_patch(FancyBboxPatch((0.09, y - 0.0037), 0.29, 0.0074, boxstyle="round,pad=0,rounding_size=0.0035",
                                    facecolor=ACCENT if tap else "#3a4352", edgecolor="none"))
        if tap:
            ax.text(0.392, y, str(k), fontsize=7, color=ACCENT, va="center", fontweight="semibold")
    ax.text(0.085, 0.512, "24 layers; 4 of them feed the swear spotter", fontsize=7, color=INK2, va="bottom")

    # trained head
    card(ax, 0.53, 0.5, 0.41, 0.405, face=ACCENT_WASH, edge=ACCENT)
    ax.text(0.555, 0.88, "② Spots the swear words", fontsize=10.5, fontweight="semibold", va="top")
    ax.text(0.555, 0.848, "the part I trained\n0.93 M params · 3.7 MB file", fontsize=7.6, color=ACCENT, va="top",
            linespacing=1.3)
    steps = [("Learned layer mix", "softmax weights over blocks 12-24"),
             ("LayerNorm → Linear", "1024 → 256 · GELU"),
             ("2 × Conv1d, kernel 5", "GELU · residual · ±0.32 s context"),
             ("Linear → sigmoid", "256 → 41 word groups")]
    bh, gap, y0 = 0.056, 0.016, 0.786
    for i, (a, b) in enumerate(steps):
        y = y0 - i * (bh + gap) - bh
        card(ax, 0.555, y, 0.36, bh, face=CARD, r=0.01)
        ax.text(0.575, y + bh - 0.012, a, fontsize=8.4, fontweight="semibold", va="top")
        ax.text(0.575, y + 0.01, b, fontsize=7.2, color=INK2, va="bottom")
        if i:
            arrow(ax, 0.735, y + bh + gap, 0.735, y + bh, color=ACCENT)
        else:
            mix_y = y + bh / 2
    for k in taps:
        arrow(ax, 0.41, ys[k], 0.553, mix_y, color=ACCENT, lw=0.9)
    head_bottom = y0 - 3 * (bh + gap) - bh

    # output heatmap: the model's real output for the window ending just after the curse word
    p = d["window_probs"]  # (frames, groups)
    w0, ready = float(d["window_start"]), float(d["window_ready"])
    names = list(GROUPS)
    rows = np.stack([p[:, names.index(g)] for g in SHOWN])
    t_lo = 0.45
    f_lo = int(np.ceil((t_lo - w0) / FRAME))
    rows = rows[:, f_lo:]
    t_edges = w0 + (f_lo + np.arange(rows.shape[1] + 1)) * FRAME
    arrow(ax, 0.735, head_bottom, 0.735, 0.435, color=ACCENT)
    ax.text(0.06, 0.465, "③ A swear-word score for every 80 ms", fontsize=10.5, fontweight="semibold", va="bottom")
    ax.text(0.06, 0.442, "one row per word type; red = \"mute this\"", fontsize=8, color=INK2, va="bottom")
    hx, hy, hw, hh = 0.17, 0.3, 0.77, 0.125
    cmap = LinearSegmentedColormap.from_list("c", [CARD2, "#6b2f2b", CURSE])
    ha = sub_axes(fig, hx, hy, hw, hh)
    ha.pcolormesh(t_edges, np.arange(len(SHOWN) + 1), rows, cmap=cmap, vmin=0, vmax=1,
                  edgecolors=BG, linewidth=0.8)
    ha.set_ylim(len(SHOWN), 0)
    ha.set_xlim(t_edges[0], t_edges[-1])
    for i, g in enumerate(SHOWN):
        ax.text(hx - 0.012, hy + hh * (1 - (i + 0.5) / len(SHOWN)), mask(g), ha="right", va="center",
                fontsize=7, color=CURSE if g == "shit" else INK2, fontweight="semibold" if g == "shit" else "normal")
    ax.text(hx - 0.012, hy - 0.008, "+33 more", ha="right", va="top", fontsize=6.6, color=MUTED)
    for w, c, s, e in load_words(d):
        if s >= t_edges[0] and e <= t_edges[-1]:
            xx = hx + hw * ((s + e) / 2 - t_edges[0]) / (t_edges[-1] - t_edges[0])
            ax.text(xx, hy - 0.008, mask(w.lower()) if c else w, ha="center", va="top", fontsize=7.2,
                    color=CURSE if c else INK2, fontweight="semibold" if c else "normal")
    ax.text(hx + hw, hy - 0.008, "now", ha="right", va="top", fontsize=7.2, color=MUTED)

    # live rule
    ax.text(0.06, 0.238, "Live loop", fontsize=9.5, fontweight="semibold", va="bottom")
    chips = [("every 50 ms", "score the last 4 s"), ("12 ms on the GPU", "one CUDA graph"),
             ("mute if P > 0.7", "in 2 checks in a row"), ("play 0.40 s late", "with 5 ms fades")]
    cwid, cg = 0.2, 0.027
    for i, (a, b) in enumerate(chips):
        x = 0.06 + i * (cwid + cg)
        card(ax, x, 0.16, cwid, 0.07, face=CARD, r=0.01)
        ax.text(x + 0.016, 0.213, a, fontsize=8.2, fontweight="semibold", va="center")
        ax.text(x + 0.016, 0.18, b, fontsize=7, color=INK2, va="center")
        if i:
            arrow(ax, x - cg + 0.004, 0.195, x - 0.004, 0.195)

    # training data
    ax.text(0.06, 0.128, "Trained on · 8,000 steps, ~30 min on one A10 GPU", fontsize=9.5,
            fontweight="semibold", va="bottom")
    cols = [("31.6 h synthetic", "Piper TTS, ~1,070 voices;\nexact word timings, no aligner"),
            ("~10 h real speech", "public YouTube (YODAS2); labels\nwhere 2 models agree"),
            ("Hard negatives", "sound effects, sound-alikes\n(\"duck\", \"shift\"), noise")]
    for i, (a, b) in enumerate(cols):
        x = 0.06 + i * 0.3
        card(ax, x, 0.045, 0.28, 0.075, face=CARD, r=0.01)
        ax.text(x + 0.018, 0.1, a, fontsize=8.4, fontweight="semibold", va="center")
        ax.text(x + 0.018, 0.066, b, fontsize=6.8, color=INK2, va="center", linespacing=1.3)
    ax.text(0.06, 0.012, "Heatmap: the model's real output, as the word finishes, on a voice not used in training.",
            fontsize=6.8, color=MUTED, va="bottom")
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "data" / "linkedin_demo.npz", allow_pickle=True)
    for name, fn in (("07-how-it-works", how_it_works), ("08-architecture", architecture)):
        fig = fn(d)
        fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor=BG)
        plt.close(fig)
        print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    main()

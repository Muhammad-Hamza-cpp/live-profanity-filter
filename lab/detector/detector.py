"""Frame-level curse detector: frozen Parakeet encoder + a small head that outputs, for every
80 ms frame and every word group, the probability that the frame is inside that curse word.

No transcript is involved. The head sees the encoder's hidden states for a window that ends
"now", so it learns to call a word from whatever part of it has been heard - which is what
lets the delay drop below the length of the word plus the time a transcriber waits for context.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lexicon import GROUPS  # noqa: E402

ENCODER = "nvidia/parakeet-ctc-0.6b"
SR = 16000
FRAME = 0.08          # Parakeet output frame: 10 ms hop x 8 subsampling
WINDOW = 4.0          # seconds of audio per check (fixed, so one CUDA graph serves every call)
GROUP_NAMES = list(GROUPS)
LAYERS = (12, 16, 20, 24)  # outputs of these encoder blocks (1-24) are mixed into the head


class Head(nn.Module):
    def __init__(self, dim=1024, hidden=256, groups=len(GROUP_NAMES), layers=len(LAYERS)):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(layers))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, hidden)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
        self.out = nn.Linear(hidden, groups)

    def forward(self, states):
        """states: (layers, batch, frames, dim) -> logits (batch, frames, groups)"""
        w = torch.softmax(self.mix, 0)
        x = (w[:, None, None, None] * states.float()).sum(0)
        x = torch.nn.functional.gelu(self.proj(self.norm(x)))
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.out(x)


def load_encoder(device="cuda", ctc=False):
    """Feature extractor + frozen encoder (+ Parakeet's own CTC output layer if ctc=True)."""
    from transformers import AutoModelForCTC, AutoProcessor
    proc = AutoProcessor.from_pretrained(ENCODER)
    model = AutoModelForCTC.from_pretrained(ENCODER, dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if ctc:
        return proc.feature_extractor, model.encoder, model.ctc_head, model.config.pad_token_id
    return proc.feature_extractor, model.encoder


def features(fe, audio, device="cuda"):
    """audio: list of 1-D float32 arrays, all the same length -> encoder inputs on the GPU.

    Same steps as ParakeetFeatureExtractor.__call__ (which pads through numpy and so can't take
    GPU tensors), without the padding: every window here has the same length."""
    x = torch.from_numpy(np.stack(audio)).to(device, torch.float32)
    if fe.preemphasis is not None:
        x = torch.cat([x[:, :1], x[:, 1:] - fe.preemphasis * x[:, :-1]], dim=1)
    f = fe._torch_extract_fbank_features(x, device)
    n = (x.shape[1] + fe.n_fft // 2 * 2 - fe.n_fft) // fe.hop_length
    mask = (torch.arange(f.shape[1], device=device) < n)[None].expand(len(audio), -1)
    m = mask.unsqueeze(-1)
    mean = (f * m).sum(1, keepdim=True) / n
    std = torch.sqrt((((f * m) - mean) ** 2 * m).sum(1, keepdim=True) / (n - 1))
    f = (f - mean) / (std + 1e-5) * m
    return f.to(torch.bfloat16), mask


def encode(encoder, feats, mask, last=False):
    out = encoder(input_features=feats, attention_mask=mask, output_hidden_states=True)
    hs = out.hidden_states
    off = len(hs) - 24  # 1 if the block input is included at index 0, else 0
    states = torch.stack([hs[i - 1 + off] for i in LAYERS])
    return (states, out.last_hidden_state) if last else states


class Detector:
    """Streaming use: call probs(window) with the last WINDOW seconds of 16 kHz audio (left-padded
    with zeros if shorter); returns (frames, groups) probabilities for that window. score(window)
    also returns, per frame, the probability that Parakeet's own CTC layer hears speech there
    (not blank) - the speech gate. Encoder, head and CTC layer run as one CUDA graph (~12 ms on
    an RTX 4050)."""

    def __init__(self, head_path, device="cuda", graph=True):
        self.device = device
        self.fe, self.encoder, self.ctc, self.blank = load_encoder(device, ctc=True)
        ckpt = torch.load(head_path, map_location=device)
        self.head = Head().to(device).eval()
        self.head.load_state_dict(ckpt["head"])
        self.n = int(WINDOW * SR)
        f, m = features(self.fe, [np.zeros(self.n, np.float32)], device)
        self.static_f, self.static_m = f.clone(), m.clone()
        self.graph = None
        if graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(3):
                    self._run()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.inference_mode():
                self.out = self._run()

    def _run(self):
        states, last = encode(self.encoder, self.static_f, self.static_m, last=True)
        p = torch.sigmoid(self.head(states))[0]
        speech = 1 - torch.softmax(self.ctc(last).float(), -1)[0, :, self.blank]
        return torch.cat([p, speech[:, None]], 1)

    def score(self, audio):
        """-> (probs (frames, groups), speech (frames,))"""
        if len(audio) < self.n:
            audio = np.concatenate([np.zeros(self.n - len(audio), np.float32), audio])
        f, m = features(self.fe, [audio[-self.n:].astype(np.float32)], self.device)
        self.static_f.copy_(f)
        self.static_m.copy_(m)
        if self.graph is None:
            with torch.inference_mode():
                out = self._run().cpu().numpy()
        else:
            self.graph.replay()
            out = self.out.cpu().numpy()
        return out[:, :-1], out[:, -1]

    def probs(self, audio):
        return self.score(audio)[0]

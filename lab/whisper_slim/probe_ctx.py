"""Does CTranslate2's Whisper encoder accept fewer than 3000 mel frames? (CPU, tiny.en)"""
import sys
import numpy as np
import ctranslate2
from faster_whisper import WhisperModel

m = WhisperModel("tiny.en", device="cpu", compute_type="int8")
fe = m.feature_extractor
audio = np.random.default_rng(0).normal(0, 0.1, 16000 * 4).astype(np.float32)
feats = fe(audio)[:, :400]
for n in (3000, 400):
    f = np.zeros((80, n), np.float32)
    f[:, :min(n, feats.shape[1])] = feats[:, :n]
    try:
        out = m.model.encode(ctranslate2.StorageView.from_array(f[None]), to_cpu=True)
        print(n, "->", np.asarray(out).shape)
    except Exception as e:
        print(n, "ERROR:", e)

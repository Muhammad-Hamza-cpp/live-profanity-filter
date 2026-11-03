"""Quick look at Parakeet CTC in transformers: frame rate, vocab, speed, greedy output."""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForCTC, AutoProcessor

name = sys.argv[1] if len(sys.argv) > 1 else "nvidia/parakeet-ctc-0.6b"
root = Path(__file__).resolve().parent.parent
audio, sr = sf.read(root / "test" / "bench_normal.wav", dtype="float32")
audio = resample_poly(audio, 16000, sr).astype(np.float32) if sr != 16000 else audio

proc = AutoProcessor.from_pretrained(name)
model = AutoModelForCTC.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
print("feature extractor:", {k: v for k, v in proc.feature_extractor.to_dict().items() if not isinstance(v, (list, dict))})
print("vocab size:", model.config.vocab_size, "| blank/pad id:", getattr(model.config, "pad_token_id", None))


def run(chunk):
    inputs = proc(chunk, sampling_rate=16000, return_tensors="pt")
    inputs = {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in inputs.items()}
    with torch.inference_mode():
        return model(**inputs).logits[0].float().cpu()


for secs in (2, 3, 4, 8):
    chunk = audio[: 16000 * secs]
    run(chunk)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        logits = run(chunk)
    torch.cuda.synchronize()
    print(f"{secs}s audio -> {logits.shape[0]} frames ({secs / logits.shape[0] * 1000:.0f} ms/frame), "
          f"{(time.perf_counter() - t0) / 10 * 1000:.1f} ms per call")

logits = run(audio[: 16000 * 8])
ids = logits.argmax(-1).tolist()
print("greedy:", proc.batch_decode([ids])[0])
tok = proc.tokenizer
print("sample tokens:", [tok.convert_ids_to_tokens(i) for i in sorted(set(ids))][:30])
print("GPU memory:", torch.cuda.max_memory_allocated() // 2**20, "MB")

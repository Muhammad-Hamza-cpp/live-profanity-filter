import time, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import AutoModelForCTC, AutoProcessor
name = "nvidia/parakeet-ctc-0.6b"
root = Path(__file__).resolve().parent.parent
audio, sr = sf.read(root / "test" / "bench_normal.wav", dtype="float32")
audio = resample_poly(audio, 16000, sr).astype(np.float32)
proc = AutoProcessor.from_pretrained(name)
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    model = AutoModelForCTC.from_pretrained(name, torch_dtype=dtype).cuda().eval()
    chunk = audio[:16000 * 4]
    t0 = time.perf_counter()
    for _ in range(10):
        inputs = proc(chunk, sampling_rate=16000, return_tensors="pt")
    tf = (time.perf_counter() - t0) / 10
    inputs = {k: (v.cuda().to(dtype) if v.is_floating_point() else v.cuda()) for k, v in inputs.items()}
    with torch.inference_mode():
        model(**inputs); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10):
            logits = model(**inputs).logits[0]
        torch.cuda.synchronize()
    tm = (time.perf_counter() - t0) / 10
    ids = logits.float().argmax(-1).tolist()
    print(f"{str(dtype):<15} features {tf*1000:.1f} ms | model {tm*1000:.1f} ms | nan={torch.isnan(logits).any().item()} | "
          + proc.batch_decode([ids])[0][:120], flush=True)
    del model; torch.cuda.empty_cache()

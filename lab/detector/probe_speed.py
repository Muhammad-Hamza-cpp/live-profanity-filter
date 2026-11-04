r"""How fast is the Parakeet encoder per check, by window length, eager vs CUDA graph?

The check interval (step) can't be shorter than one call, and the worst-case delay grows with the
step, so this decides how low the delay can go on this GPU.

    .venv-lab\Scripts\python lab\detector\probe_speed.py
"""
import time

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

NAME = "nvidia/parakeet-ctc-0.6b"
SR = 16000


def timed(fn, n=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    proc = AutoProcessor.from_pretrained(NAME)
    model = AutoModelForCTC.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
    print(f"GPU {torch.cuda.get_device_name()} | weights {torch.cuda.memory_allocated() / 2**30:.2f} GB")
    audio = np.random.default_rng(0).normal(0, 0.1, SR * 4).astype(np.float32)
    print(f"{'window':<8}{'features (CPU)':<16}{'eager':<10}{'CUDA graph':<12}frames")
    for secs in (1.0, 1.5, 2.0, 3.0, 4.0):
        chunk = audio[:int(secs * SR)]
        t0 = time.perf_counter()
        for _ in range(10):
            inputs = proc(chunk, sampling_rate=SR, return_tensors="pt")
        feat_ms = (time.perf_counter() - t0) / 10 * 1000
        inputs = {k: (v.cuda().bfloat16() if v.is_floating_point() else v.cuda()) for k, v in inputs.items()}
        with torch.inference_mode():
            eager_ms = timed(lambda: model(**inputs).logits)
            frames = model(**inputs).logits.shape[1]
        graph_ms = float("nan")
        try:
            static = {k: v.clone() for k, v in inputs.items()}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(3):
                    model(**static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.inference_mode():
                out = model(**static).logits
            graph_ms = timed(g.replay)
            with torch.inference_mode():
                ref = model(**static).logits
            g.replay()
            assert torch.allclose(out.float(), ref.float(), atol=1e-2), "graph output differs"
        except Exception as e:
            print(f"  CUDA graph failed: {type(e).__name__}: {str(e)[:150]}")
        print(f"{secs:<8}{feat_ms:<16.1f}{eager_ms:<10.1f}{graph_ms:<12.1f}{frames}", flush=True)


if __name__ == "__main__":
    main()

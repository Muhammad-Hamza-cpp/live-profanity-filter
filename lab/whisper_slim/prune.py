"""Make a copy of a faster-whisper model with some decoder/encoder layers removed (no retraining).

    python prune.py small.en --keep-decoder 0,1,2,3,4,5,6,7,8,9,10,11 --keep-encoder 0,...,11 --out DIR
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import ct2_weights

HUB = Path.home() / ".cache/huggingface/hub"


def model_dir(name):
    repo = {"large-v3-turbo": "mobiuslabsgmbh--faster-whisper-large-v3-turbo"}.get(name, f"Systran--faster-whisper-{name}")
    return next((HUB / f"models--{repo}" / "snapshots").iterdir())


def prune(src, out, keep_decoder=None, keep_encoder=None):
    header, variables, aliases = ct2_weights.load(src / "model.bin")
    keep = {"decoder": keep_decoder, "encoder": keep_encoder}
    new = {}
    for name, v in variables.items():
        m = re.match(r"(encoder|decoder)/layer_(\d+)/(.*)", name)
        if m and keep[m[1]] is not None:
            part, i, rest = m[1], int(m[2]), m[3]
            if i not in keep[part]:
                continue
            name = f"{part}/layer_{keep[part].index(i)}/{rest}"
        new[name] = v
    out.mkdir(parents=True, exist_ok=True)
    ct2_weights.save(out / "model.bin", header, new, aliases)
    for f in src.iterdir():
        if f.name != "model.bin":
            shutil.copy(f, out / f.name)
    # Word timestamps come from cross-attention "alignment heads"; renumber them, drop those that are gone.
    cfg = json.loads((out / "config.json").read_text())
    if keep_decoder is not None:
        heads = [[keep_decoder.index(l), h] for l, h in cfg["alignment_heads"] if l in keep_decoder]
        if not heads:  # fall back to every head of the last half, like CTranslate2's default
            n = len(keep_decoder)
            heads = [[l, h] for l in range(n // 2, n) for h in range(new["decoder/num_heads"].item())]
        cfg["alignment_heads"] = heads
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--keep-decoder")
    ap.add_argument("--keep-encoder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parse = lambda s: [int(x) for x in s.split(",")] if s else None
    print(prune(model_dir(a.model), Path(a.out), parse(a.keep_decoder), parse(a.keep_encoder)))


if __name__ == "__main__":
    main()

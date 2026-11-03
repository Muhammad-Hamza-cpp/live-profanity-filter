"""Read / write CTranslate2 model.bin files (the format faster-whisper loads), so layers can be
inspected, removed or edited directly.

    python ct2_weights.py <model dir>      # print a per-block parameter summary
"""
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DTYPES = (np.float32, np.int8, np.int16, np.int32, np.float16, None)  # None = bfloat16


def _read_str(f):
    (n,) = struct.unpack("H", f.read(2))
    return f.read(n)[:-1].decode("utf-8")


def _write_str(f, s):
    f.write(struct.pack("H", len(s) + 1) + s.encode("utf-8") + b"\0")


def load(path):
    """Returns (header, {name: ndarray}, [(alias, target)])."""
    with open(path, "rb") as f:
        version, = struct.unpack("I", f.read(4))
        spec = _read_str(f)
        revision, count = struct.unpack("II", f.read(8))
        variables = {}
        for _ in range(count):
            name = _read_str(f)
            (rank,) = struct.unpack("B", f.read(1))
            shape = struct.unpack(f"{rank}I", f.read(4 * rank))
            dtype_id, nbytes = struct.unpack("<BI", f.read(5))
            raw = f.read(nbytes)
            dt = DTYPES[dtype_id]
            variables[name] = np.frombuffer(raw, dt).reshape(shape) if dt else (raw, shape, dtype_id)
        (n_alias,) = struct.unpack("I", f.read(4))
        aliases = [(_read_str(f), _read_str(f)) for _ in range(n_alias)]
    return (version, spec, revision), variables, aliases


def save(path, header, variables, aliases):
    version, spec, revision = header
    with open(path, "wb") as f:
        f.write(struct.pack("I", version))
        _write_str(f, spec)
        f.write(struct.pack("II", revision, len(variables)))
        for name, v in variables.items():
            _write_str(f, name)
            if isinstance(v, tuple):  # bfloat16 kept as raw bytes
                raw, shape, dtype_id = v
            else:
                raw, shape, dtype_id = v.tobytes(), v.shape, DTYPES.index(v.dtype.type)
            f.write(struct.pack("B", len(shape)) + struct.pack(f"{len(shape)}I", *shape))
            f.write(struct.pack("<BI", dtype_id, len(raw)) + raw)
        f.write(struct.pack("I", len(aliases)))
        for a, t in aliases:
            _write_str(f, a)
            _write_str(f, t)


def summary(model_dir):
    header, variables, aliases = load(Path(model_dir) / "model.bin")
    print(f"spec {header[1]} rev {header[2]}, {len(variables)} tensors, {len(aliases)} aliases")
    groups = defaultdict(int)
    for name, v in variables.items():
        n = int(np.prod(v.shape if not isinstance(v, tuple) else v[1]))
        groups[re.sub(r"(layer_\d+)/.*", r"\1", name)] += n
    total = sum(groups.values())
    for g, n in groups.items():
        print(f"  {g:<45}{n / 1e6:8.2f} M  {100 * n / total:5.1f}%")
    print(f"  {'TOTAL':<45}{total / 1e6:8.2f} M")
    return header, variables, aliases


if __name__ == "__main__":
    summary(sys.argv[1])

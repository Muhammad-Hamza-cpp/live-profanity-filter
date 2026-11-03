import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import ct2_weights as c
snap = next((Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small.en/snapshots").iterdir())
h, v, a = c.summary(snap)
print("aliases", a)
for k, x in v.items():
    if "layer_" not in k:
        small = getattr(x, "size", 99) < 40
        print(f"  {k:<45}{str(getattr(x, 'shape', '')):<14}{str(getattr(x, 'dtype', '')):<9}{x.tolist() if small else ''}")
print("decoder layer 0:")
for k, x in v.items():
    if k.startswith("decoder/layer_0/"):
        print(f"  {k:<55}{x.shape} {x.dtype}")

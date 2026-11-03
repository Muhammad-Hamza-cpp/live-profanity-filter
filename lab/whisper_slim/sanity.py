import filecmp, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import ct2_weights
from prune import model_dir
from bench import TEST, load16
from faster_whisper import WhisperModel

src = model_dir("tiny.en") / "model.bin"
out = Path(tempfile.gettempdir()) / "rt_model.bin"
ct2_weights.save(out, *ct2_weights.load(src))
print("round-trip identical:", filecmp.cmp(src, out, shallow=False))
out.unlink()

audio = load16(TEST / "bench_normal.wav")[16000 * 10: 16000 * 18]
for m in ["small.en", "models/small.en-dec10", "models/small.en-dec8", "models/small.en-dec6", "models/small.en-enc10"]:
    model = WhisperModel(m, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio, language="en", beam_size=1, word_timestamps=True)
    print(f"{m:<24}", "".join(s.text for s in segs).strip()[:160])

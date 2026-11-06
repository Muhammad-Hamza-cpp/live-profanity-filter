#!/bin/bash
# Lambda: Piper voices, ESC-50, and model downloads; checks Whisper runs on the GPU.
set -e
cd /lambda/nfs/Audio/lpf
mkdir -p lab/detector/voices data/esc50_dl
while read -r v; do
  lang=${v%%-*}; rest=${v#*-}; name=${rest%-*}; q=${rest##*-}; fam=${lang%%_*}
  for ext in onnx onnx.json; do
    f=lab/detector/voices/$v.$ext
    [ -f "$f" ] || curl -sfL -o "$f" "https://huggingface.co/rhasspy/piper-voices/resolve/main/$fam/$lang/$name/$q/$v.$ext"
  done
done < voices.txt
echo "voices: $(ls lab/detector/voices/*.onnx | wc -l)"
if [ ! -d data/esc50_dl/ESC-50-master ]; then
  curl -sfL -o data/esc50_dl/esc.zip https://github.com/karoldvl/ESC-50/archive/master.zip
  (cd data/esc50_dl && python3 -m zipfile -e esc.zip . && rm esc.zip)
fi
echo "esc50 clips: $(ls data/esc50_dl/ESC-50-master/audio | wc -l)"
source "$HOME/lpf_env.sh"
"$HOME/venv/bin/python" - <<'PY'
import numpy as np, torchaudio
from transformers import AutoModelForCTC, AutoProcessor
AutoProcessor.from_pretrained("nvidia/parakeet-ctc-0.6b")
AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-0.6b")
torchaudio.pipelines.MMS_FA.get_model()
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
segs, _ = m.transcribe(np.zeros(16000, np.float32))
list(segs)
print("models ok, whisper runs on the GPU")
PY
echo DATA_DONE

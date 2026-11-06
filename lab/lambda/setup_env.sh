#!/bin/bash
# Lambda: build the Python environment in ~/venv (instance disk; rebuilt on a new instance by
# resume_all.sh). Pinned to the versions the overnight runs used.
set -e
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv $HOME/venv --python 3.12
uv pip install --python $HOME/venv/bin/python torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $HOME/venv/bin/python transformers==5.22.1 accelerate==1.15.0 librosa==1.0.0 faster-whisper==1.2.1 \
  piper-tts==1.8.0 onnx==1.23.2 soundfile scipy numpy pyarrow==26.0.1 fsspec==2026.7.0 aiohttp requests \
  huggingface_hub==2.4.0 matplotlib nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
$HOME/venv/bin/python -c "import torch, torchaudio, transformers, faster_whisper; print(\"OK torch\", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo SETUP_DONE

# Source before running lab scripts on Lambda. Lives on the persistent filesystem; model caches
# live there too, so a new instance needn't download them again.
NV=$("$HOME/venv/bin/python" -c "import nvidia.cublas, os; print(os.path.dirname(os.path.dirname(nvidia.cublas.__path__[0])))" 2>/dev/null)
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH}"
export HF_HOME=/lambda/nfs/Audio/cache/hf TORCH_HOME=/lambda/nfs/Audio/cache/torch
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8
export PATH="$HOME/venv/bin:$PATH"
cd /lambda/nfs/Audio/lpf

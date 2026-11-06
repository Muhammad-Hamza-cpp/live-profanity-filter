#!/bin/bash
# Lambda: second labelling pass, for clips fetched after label_train.sh started.
source "$HOME/lpf_env.sh"
while pgrep -f "fetch_yodas.py|label_train.sh" > /dev/null; do sleep 30; done
set -e
python lab/detector/align_clips.py --device cuda
python lab/detector/verify_whisper.py
echo LABEL_DONE

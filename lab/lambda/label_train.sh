#!/bin/bash
# Lambda: once the training-pool fetch is done, align its clips (MMS) and check them with Whisper.
source "$HOME/lpf_env.sh"
while pgrep -f fetch_yodas.py > /dev/null; do sleep 30; done
until grep -q DATA_DONE setup_data.log; do sleep 30; done
set -e
python lab/detector/align_clips.py --device cuda
python lab/detector/verify_whisper.py
echo LABEL_DONE

#!/bin/bash
# Lambda: regenerate the Piper training/test speech (same commands as docs/mute-detector.md section 7).
source "$HOME/lpf_env.sh"
until grep -q DATA_DONE setup_data.log; do sleep 30; done
set -e
python lab/detector/make_tts.py --split train --utts 40000 --workers 24
python lab/detector/make_tts.py --split train_ctx --kind context --utts 15000 --workers 24
python lab/detector/make_tts.py --split test --utts 2000 --workers 24
python lab/detector/make_tts.py --split test_ctx --kind context --utts 1000 --workers 24
echo TTS_DONE

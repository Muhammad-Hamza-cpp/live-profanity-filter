#!/bin/bash
# Lambda: align the clips fetched after the first labelling pass started, then check them with 3
# Whisper workers in parallel, skipping the clips the first pass is still checking (pass1_ids.txt).
source /lambda/nfs/Audio/lpf/lpf_env.sh
set -e
python lab/detector/align_clips.py --device cuda
for i in 0 1 2; do python lab/detector/verify_whisper.py --shard $i/3 --skip-ids data/realspeech/yodas/pass1_ids.txt & done
wait
echo FAST_DONE

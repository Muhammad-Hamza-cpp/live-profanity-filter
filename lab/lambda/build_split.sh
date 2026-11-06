#!/bin/bash
# Lambda: once every training clip is labelled, build the train/validation split (once).
source /lambda/nfs/Audio/lpf/lpf_env.sh
[ -f data/realspeech/yodas/trainset.jsonl ] && { echo "split already built"; exit 0; }
until grep -q LABEL_DONE label_rest.log 2>/dev/null; do sleep 60; done
python lab/detector/build_trainset.py && echo SPLIT_DONE

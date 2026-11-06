#!/bin/bash
# Lambda: score every finished training run (and v3 as shipped) once the validation set exists.
# Final checkpoint (last.pt), check time fixed at the laptop's 13 ms, batched. Writes
# runs/<name>/eval_val.log (thresholds, tuned here) and eval_test.log (reported). Resumable.
source /lambda/nfs/Audio/lpf/lpf_env.sh
V3=lab/detector/runs/v3_shipped
mkdir -p $V3 && [ -f $V3/last.pt ] || { cp lab/detector/models/detector-v3.pt $V3/last.pt && touch $V3/done; }
ARGS="--confirm 2 --pad 0.06 0.02 --check-ms 13"
while true; do
  for d in lab/detector/runs/*/; do
    n=$(basename "$d")
    [ -f "$d/done" ] && [ -f data/realspeech/yodas/valset.jsonl ] && [ ! -f "$d/eval_test.log" ] \
      && [ ! -f "$d/eval_failed" ] || continue
    mkdir "$d/.evallock" 2>/dev/null || continue
    echo "$(date +%H:%M) eval $n"
    (cd lab/detector && python eval.py "runs/$n/last.pt" --sets real --real-file valset.jsonl \
        --thr 0.3 0.4 0.5 0.6 0.7 $ARGS) > "$d/eval_val.tmp" 2>&1 && mv "$d/eval_val.tmp" "$d/eval_val.log" \
    && (cd lab/detector && python eval.py "runs/$n/last.pt" --sets real,tts,ctx,esc \
        --thr 0.3 0.4 0.5 0.6 0.7 $ARGS) > "$d/eval_test.tmp" 2>&1 && mv "$d/eval_test.tmp" "$d/eval_test.log" \
    && echo "$(date +%H:%M) eval done $n" || { echo "$(date +%H:%M) eval FAILED $n"; touch "$d/eval_failed"; }
    rmdir "$d/.evallock"
  done
  sleep 120
done

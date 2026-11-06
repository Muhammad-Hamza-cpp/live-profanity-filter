#!/bin/bash
# Lambda: one training worker. Takes the next job in jobs.txt ("<name> <train.py args>") that
# isn't done and that no other worker holds (runs/<name>/.lock), runs it, repeats. Jobs with
# --real other than 0 wait until data/realspeech/yodas/trainset.jsonl exists. Lines can be added to
# jobs.txt at any time. A crashed run resumes from its last checkpoint (train.py does that).
# Start N workers for N runs at a time:   bash run_queue.sh 1 & bash run_queue.sh 2 &
source /lambda/nfs/Audio/lpf/lpf_env.sh
W=$1
while pgrep -f "run_jobs.sh" > /dev/null; do sleep 60; done   # the first queue (c0 runs) finishes first
while true; do
  picked=""
  while read -r name args; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    d=lab/detector/runs/$name
    [ -f "$d/done" ] || [ -f "$d/failed" ] && continue   # delete "failed" to retry a run
    if [[ "$args" =~ --real\ (-?[0-9]+) && "${BASH_REMATCH[1]}" != "0" && ! -f data/realspeech/yodas/trainset.jsonl ]]; then
      continue
    fi
    mkdir -p "$d"
    mkdir "$d/.lock" 2>/dev/null || continue   # atomic: another worker has it
    picked="$name"; break
  done < jobs.txt
  if [ -z "$picked" ]; then sleep 60; continue; fi
  echo "$(date +%H:%M) worker $W start $picked: $args"
  (cd lab/detector && python train.py --name "$picked" $args) >> "lab/detector/runs/$picked/stdout.log" 2>&1 \
    && touch "lab/detector/runs/$picked/done" && echo "$(date +%H:%M) worker $W done $picked" \
    || { echo "$(date +%H:%M) worker $W FAILED $picked"; touch "lab/detector/runs/$picked/failed"; }
  rmdir "lab/detector/runs/$picked/.lock"
done

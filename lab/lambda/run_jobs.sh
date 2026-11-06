#!/bin/bash
# Lambda: run training jobs from a file, PARALLEL at a time. Each line: "<name> <train.py args>".
# A job whose runs/<name>/done exists is skipped, so a job file can be rerun after a stop.
#   bash run_jobs.sh jobs_curve0.txt 2
source "$HOME/lpf_env.sh"
until grep -q TTS_DONE gen_tts.log; do sleep 30; done
JOBS=$1; PARALLEL=${2:-2}
run() {
  name=$1; shift
  d=lab/detector/runs/$name
  [ -f "$d/done" ] && { echo "skip $name"; return; }
  mkdir -p "$d"
  echo "$(date +%H:%M) start $name: $*"
  (cd lab/detector && python train.py --name "$name" "$@") > "$d/stdout.log" 2>&1 \
    && touch "$d/done" && echo "$(date +%H:%M) done $name" || echo "$(date +%H:%M) FAILED $name"
}
export -f run
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | xargs -P "$PARALLEL" -L 1 bash -c 'run "$@"' _
echo "ALL_DONE $JOBS"

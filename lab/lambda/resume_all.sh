#!/bin/bash
# Lambda: after a crash, reboot or a new instance, restart whatever isn't finished. Safe to run any
# time: each step skips finished work and nothing that is still running gets a second copy.
#   bash /lambda/nfs/Audio/lpf/resume_all.sh
D=/lambda/nfs/Audio/lpf
cd $D
[ -x "$HOME/venv/bin/python" ] || bash setup_env.sh >> setup_env.log 2>&1   # new instance: rebuild the env
source $D/lpf_env.sh
start() {  # start <pattern> <command> <log>: only if nothing matching <pattern> runs
  pgrep -f "$1" > /dev/null && { echo "running: $1"; return; }
  echo "starting: $2"
  (setsid nohup bash -c "$2" >> "$3" 2>&1 < /dev/null &)
}
if ! pgrep -f "run_queue.sh|run_jobs.sh|eval_queue.sh" > /dev/null; then   # no worker alive: their locks are stale
  find lab/detector/runs -maxdepth 2 \( -name .lock -o -name .evallock \) -type d -exec rmdir {} + 2>/dev/null
fi
start fetch_yodas.py "python lab/detector/fetch_yodas.py --start 62 --top 200 --jobs 12" fetch_train.log
start "label_train.sh|label_rest.sh|align_clips.py|verify_whisper.py" "bash label_rest.sh" label_rest.log
start build_split.sh "bash build_split.sh" build_split.log
start "run_queue.sh 1" "bash run_queue.sh 1" queue.log
start "run_queue.sh 2" "bash run_queue.sh 2" queue.log
start eval_queue.sh "bash eval_queue.sh" eval_queue.log

#!/bin/bash
# Wait until the laptop's evaluations (v3 + baselines) and every Lambda job (labelling, split,
# all runs in jobs.txt + v3_shipped, trained and scored) are finished. Then pull Lambda's results
# to data/lambda_backup/, check each run's files arrived, write results_overnight.txt, terminate
# the Lambda instance (its persistent filesystem "Audio" is kept), and shut this laptop down with
# a 2-minute warning ("shutdown /a" in a terminal cancels it).
# A failed run doesn't block: it is listed in the summary. If anything stays unfinished or the
# results can't be pulled, nothing is terminated and the laptop simply stays on.
#   bash lab/finish_and_shutdown.sh
cd "$(dirname "$0")/.."
DEST=data/lambda_backup
INSTANCE=1319a7ed710d439184165858548d7cb6  # the gpu_1x_a10 in us-west-1
INSTANCE_IP=150.230.34.10
API=https://cloud.lambda.ai/api/v1
log() { echo "$(date '+%H:%M') $*"; }

terminate_lambda() {
  # API key in ~/.lambda_api_key (kept outside the repo). Only terminates if the instance id still
  # belongs to our IP, so a reused id or a teammate's instance is never touched.
  local key info
  key=$(cat ~/.lambda_api_key 2>/dev/null) || { log "no API key - not terminating"; return 1; }
  info=$(curl -s -u "$key:" "$API/instances/$INSTANCE")
  echo "$info" | grep -q "\"ip\": *\"$INSTANCE_IP\"" || { log "instance $INSTANCE isn't $INSTANCE_IP any more - not terminating"; return 1; }
  curl -s -u "$key:" -H "Content-Type: application/json" -X POST \
    -d "{\"instance_ids\": [\"$INSTANCE\"]}" "$API/instance-operations/terminate" | grep -q "terminated_instances" \
    || { log "terminate request was refused"; return 1; }
  log "Lambda instance $INSTANCE ($INSTANCE_IP) terminated; the Audio filesystem keeps all data"
}

laptop_finished() {
  [ -f baselines_parakeet.log ] && grep -q -E "=== parakeet|Traceback" baselines_parakeet.log || return 1
  ! powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |
      Where-Object { \$_.CommandLine -match 'eval.py|eval_baselines.py' } | Select-Object -First 1" | grep -q .
}

lambda_status() {
  ssh -o ConnectTimeout=15 -o BatchMode=yes lambda python3 - <<'EOF'
import os, json
D = "/lambda/nfs/Audio/lpf"; R = D + "/lab/detector/runs"
names = [l.split()[0] for l in open(D + "/jobs.txt") if l.strip() and not l.startswith("#")] + ["v3_shipped"]
st = {}
for n in names:
    d = f"{R}/{n}"
    ex = lambda f: os.path.exists(f"{d}/{f}")
    st[n] = ("train_failed" if ex("failed") else "training" if not ex("done") else
             "ok" if ex("eval_test.log") else "eval_failed" if ex("eval_failed") else "evaluating")
lr = D + "/label_rest.log"
label = os.path.exists(lr) and "LABEL_DONE" in open(lr, errors="replace").read()
split = os.path.exists(D + "/data/realspeech/yodas/trainset.jsonl")
busy = [n for n, s in st.items() if s in ("training", "evaluating")]
print(json.dumps({"finished": label and split and not busy, "label": label, "split": split,
                  "busy": len(busy), "states": st}))
EOF
}

pull() {
  ssh -n -o ConnectTimeout=15 -o BatchMode=yes lambda \
    'cd /lambda/nfs/Audio/lpf && tar -czf - --ignore-failed-read data/realspeech/yodas/*.jsonl data/realspeech/yodas/*.txt data/realspeech/yodas/*.json lab/detector/runs *.log jobs*.txt 2>/dev/null' \
    > "$DEST/final.tgz.part" && [ -s "$DEST/final.tgz.part" ] || return 1
  mv "$DEST/final.tgz.part" "$DEST/final.tgz"
  rm -rf "$DEST/files" && mkdir -p "$DEST/files" && tar -xzf "$DEST/final.tgz" -C "$DEST/files"
}

mkdir -p "$DEST"
log "watching: laptop evaluations and Lambda jobs (checks every 10 min)"
while true; do
  L=no; laptop_finished && L=yes
  S=$(lambda_status 2>/dev/null | tail -1)
  F=$(echo "$S" | grep -o '"finished": [a-z]*' | cut -d' ' -f2)
  log "laptop finished: $L | lambda: $(echo "$S" | grep -o '"label": [a-z]*, "split": [a-z]*, "busy": [0-9]*' || echo unreachable)"
  [ "$L" = yes ] && [ "$F" = true ] && break
  sleep 600
done

log "everything finished - pulling Lambda results"
for try in 1 2 3; do pull && break; log "pull failed, retrying in 2 min"; sleep 120; done
[ -f "$DEST/final.tgz" ] || { log "could not pull results - NOT shutting down"; exit 1; }

missing=0
for n in $(echo "$S" | grep -o '"[a-z0-9_]*": "ok"' | cut -d'"' -f2); do
  for f in last.pt eval_val.log eval_test.log; do
    [ -s "$DEST/files/lab/detector/runs/$n/$f" ] || { log "missing locally: $n/$f"; missing=1; }
  done
done
[ $missing = 0 ] || { log "some results did not arrive - NOT shutting down"; exit 1; }

{
  echo "Overnight results ($(date '+%Y-%m-%d %H:%M')). Real test set, threshold 0.5, 2 checks, pads 0.06/0.02, 13 ms checks."
  echo "Lambda run states: $(echo "$S" | grep -o '"states": {.*}' )"
  for d in "$DEST"/files/lab/detector/runs/*/; do
    [ -f "$d/eval_test.log" ] || continue
    echo; echo "== $(basename "$d")"
    awk '/=== threshold 0.5,/{on=1} /=== threshold 0.6,/{on=0} on && /\[real\]|at 0.3|at 0.4/' "$d/eval_test.log"
  done
  for f in eval_real_v3.log baselines_whisper.log baselines_parakeet.log; do
    echo; echo "== laptop: $f"; grep -v -i -E "warn|loading weights|redirects" "$f" | tail -12
  done
} > results_overnight.txt
log "summary written to results_overnight.txt"
if terminate_lambda; then
  echo -e "\nLambda instance $INSTANCE terminated at $(date '+%H:%M'); data is on the Audio filesystem." >> results_overnight.txt
else
  echo -e "\nWARNING: the Lambda instance was NOT terminated and is still billing - terminate it in the dashboard." >> results_overnight.txt
fi
log "shutting down in 2 minutes (run 'shutdown /a' to cancel)"
powershell -NoProfile -Command "shutdown /s /t 120 /c 'Laptop and Lambda jobs are finished and results are saved. Shutting down in 2 minutes. Run shutdown /a to cancel.'"

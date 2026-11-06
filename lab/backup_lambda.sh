#!/bin/bash
# Pull the small, hard-to-redo files from the Lambda instance to data/lambda_backup/ every 30 min:
# labels, the train/validation split, every training run (checkpoints + logs), job logs.
# Audio clips are left out; they can be fetched again. Stop with Ctrl+C.
#   bash lab/backup_lambda.sh
cd "$(dirname "$0")/.."
DEST=data/lambda_backup
mkdir -p "$DEST"
while true; do
  if ssh -n -o ConnectTimeout=15 -o BatchMode=yes lambda \
      'cd /lambda/nfs/Audio/lpf && tar -czf - --ignore-failed-read data/realspeech/yodas/*.jsonl data/realspeech/yodas/*.txt data/realspeech/yodas/*.json lab/detector/runs *.log jobs*.txt 2>/dev/null' \
      > "$DEST/latest.tgz.part" && [ -s "$DEST/latest.tgz.part" ]; then
    mv "$DEST/latest.tgz.part" "$DEST/latest.tgz"
    rm -rf "$DEST/files" && mkdir -p "$DEST/files" && tar -xzf "$DEST/latest.tgz" -C "$DEST/files"
    echo "$(date +%H:%M) backup ok, $(du -sh "$DEST/latest.tgz" | cut -f1)"
  else
    echo "$(date +%H:%M) backup FAILED (Lambda unreachable?) - keeping the previous one"
  fi
  sleep 1800
done

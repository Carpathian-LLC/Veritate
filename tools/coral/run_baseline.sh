#!/usr/bin/env bash
# Launches the 50M baseline on the pre-mixed distill_v1_mix_tinystories corpus.
# This is the apples-to-apples comparison for the blended 30M output of merge.py.
# Same throughput caveat — don't run concurrently with other training.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p /tmp/coral
LOG=/tmp/coral/coral_baseline.log
nohup "$REPO/.venv/bin/python" "$REPO/tools/coral/run_coral.py" \
  --name coral_baseline_50m \
  --corpus distill_v1_mix_tinystories \
  --size 50m \
  --total_steps 6000 \
  --batch 16 \
  --seq 256 \
  --base_lr 3e-4 \
  --min_lr 3e-5 \
  --warmup 200 \
  --log_every 25 \
  --eval_every 250 \
  --eval_iters 8 \
  --ckpt_every 1000 \
  --description "Coral baseline 50M on the mixed corpus" \
  >"$LOG" 2>&1 &
echo "Coral-Baseline PID: $!  log: $LOG"
echo "Watch in Coral Lab → slot CMP → coral_baseline_50m"

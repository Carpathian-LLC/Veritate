#!/usr/bin/env bash
# Launches Coral-B (constituent on distill_v1) in the background.
# Pre-req: nothing — independent of Coral-A. But running concurrently with
# Coral-A on the same M1 GPU will halve throughput, so let A finish first.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p /tmp/coral
LOG=/tmp/coral/coral_b.log
nohup "$REPO/.venv/bin/python" "$REPO/tools/coral/run_coral.py" \
  --name coral_b_distill_v1_30m \
  --corpus distill_v1 \
  --size 30m \
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
  --description "Coral-B constituent on distill_v1" \
  >"$LOG" 2>&1 &
echo "Coral-B PID: $!  log: $LOG"
echo "Watch in Coral Lab → slot B → coral_b_distill_v1_30m"

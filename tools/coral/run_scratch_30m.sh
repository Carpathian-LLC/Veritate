#!/usr/bin/env bash
# 30M trained from scratch on the mixed corpus — the true apples-to-apples
# baseline for the Coral blend (same params, same data, no merge).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p /tmp/coral
LOG=/tmp/coral/coral_scratch_30m.log
nohup "$REPO/.venv/bin/python" "$REPO/tools/coral/run_coral.py" \
  --name coral_scratch_30m_mix \
  --corpus distill_v1_mix_tinystories \
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
  --description "30M scratch on mixed corpus — apples-to-apples baseline for Coral blend" \
  >"$LOG" 2>&1 &
echo "Coral-Scratch-30M PID: $!  log: $LOG"

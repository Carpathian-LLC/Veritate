#!/usr/bin/env bash
# Runs the Coral Merge algorithm — produces coral_blend_30m from
# coral_a_tinystories_30m and coral_b_distill_v1_30m. Requires both
# constituents trained first.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p /tmp/coral
LOG=/tmp/coral/coral_merge.log
nohup "$REPO/.venv/bin/python" "$REPO/tools/coral/merge.py" \
  --name_a   coral_a_tinystories_30m \
  --name_b   coral_b_distill_v1_30m \
  --out_name coral_blend_30m \
  --corpus   distill_v1_mix_tinystories \
  --align_samples 2048 \
  --refine_steps 1500 \
  --scalar_phase_frac 0.05 \
  --batch 16 \
  --seq 256 \
  --lr_scalar 1e-2 \
  --lr_weight 3e-5 \
  --temperature 2.0 \
  --lambda_ce 0.5 \
  --lambda_kl 0.25 \
  --log_every 25 \
  --eval_every 250 \
  --eval_iters 8 \
  --description "Coral Merge — polyphase distill-merge of A+B" \
  >"$LOG" 2>&1 &
echo "Coral-Merge PID: $!  log: $LOG"
echo "Watch in Coral Lab → slot CMP → coral_blend_30m"

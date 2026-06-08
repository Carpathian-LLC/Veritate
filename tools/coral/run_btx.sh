#!/usr/bin/env bash
# BTX-style merge: keep both FFNs as MoE experts, average attn/embed/norm,
# learn a per-block router. Phase 1: router only. Phase 2: joint refine.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p /tmp/coral
LOG=/tmp/coral/coral_btx.log
nohup "$REPO/.venv/bin/python" "$REPO/tools/coral/btx_merge.py" \
  --name_a coral_a_tinystories_30m \
  --name_b coral_b_distill_v1_30m \
  --out_name coral_btx_30m \
  --corpus distill_v1_mix_tinystories \
  --router_steps 500 \
  --joint_steps 1500 \
  --batch 16 --seq 256 \
  --lr_router 3e-3 --lr_joint 3e-5 \
  --warmup_frac 0.1 \
  --log_every 25 --eval_every 250 --eval_iters 8 \
  --description "BTX merge — averaged shared + MoE FFN with 2 experts (A, B)" \
  >"$LOG" 2>&1 &
echo "BTX-Merge PID: $!  log: $LOG"

#!/bin/bash
set -euo pipefail

MODEL="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
LOCAL_MODEL="/root/.cache/modelscope/hub/models/Wan-AI/Wan2___2-T2V-A14B-Diffusers"
if [[ -d "${LOCAL_MODEL}" ]]; then
  MODEL="${LOCAL_MODEL}"
fi

rm -rf /tmp/wan22_super_p95_beam_backfill
mkdir -p /tmp/wan22_super_p95_beam_backfill/trace

exec env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/wan22_super_p95_dispatcher.py \
  --host 127.0.0.1 \
  --port 8080 \
  --num-servers 8 \
  --device-ids '0;1;2;3;4;5;6;7' \
  --model "${MODEL}" \
  --backend-start-port 8091 \
  --backend-hardware-profiles 910B3 \
  --wan22-estimator-profile 8xusp1_inferred \
  --backend-scheduler super_p95_step \
  --wan22-scheduling-mode release_calendar_tail_pack_backfill \
  --backend-log-dir /tmp/wan22_super_p95_beam_backfill \
  --trace-log-dir /tmp/wan22_super_p95_beam_backfill/trace \
  --request-timeout-s 1000000 \
  --backend-health-timeout-s 1800 \
  --backend-health-poll-interval-s 10 \
  --backend-args=--omni \
  --backend-args=--usp \
  --backend-args=1 \
  --backend-args=--enable-layerwise-offload \
  --backend-args=--boundary-ratio \
  --backend-args=0.875 \
  --backend-args=--flow-shift \
  --backend-args=5.0 \
  --backend-args=--vae-use-slicing \
  --backend-args=--vae-use-tiling

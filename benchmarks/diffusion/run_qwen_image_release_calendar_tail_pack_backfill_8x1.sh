#!/bin/bash
set -euo pipefail

MODEL="Qwen/Qwen-Image"
LOCAL_MODEL="/root/.cache/modelscope/hub/models/Qwen/Qwen-Image"
if [[ -d "${LOCAL_MODEL}" ]]; then
  MODEL="${LOCAL_MODEL}"
fi

rm -rf /tmp/qwen_image_release_calendar_tail_pack_backfill
mkdir -p /tmp/qwen_image_release_calendar_tail_pack_backfill/trace

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_MODELSCOPE=true
export HCCL_CONNECT_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

exec env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/qwen_image_super_p95_dispatcher.py \
  --host 127.0.0.1 \
  --port 8080 \
  --num-servers 8 \
  --device-ids 0,1,2,3,4,5,6,7 \
  --model "${MODEL}" \
  --backend-start-port 8091 \
  --backend-hardware-profiles 910B3 \
  --qwen-image-scheduling-mode release_calendar_tail_pack_backfill \
  --backend-log-dir /tmp/qwen_image_release_calendar_tail_pack_backfill \
  --trace-log-dir /tmp/qwen_image_release_calendar_tail_pack_backfill/trace \
  --request-timeout-s 1000000 \
  --backend-health-timeout-s 1800 \
  --backend-health-poll-interval-s 10 \
  --backend-env VLLM_PLUGINS=ascend \
  --backend-env HF_HUB_OFFLINE=1 \
  --backend-args=--omni \
  --backend-args=--vae-use-slicing \
  --backend-args=--vae-use-tiling

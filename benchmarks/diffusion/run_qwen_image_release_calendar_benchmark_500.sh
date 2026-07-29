#!/bin/bash
set -euo pipefail

RESULT_DIR="/tmp/qwen_image_release_calendar_tail_pack_backfill_req500"
TRACE_DIR="/tmp/qwen_image_release_calendar_tail_pack_backfill/trace"

mkdir -p "${RESULT_DIR}" "${TRACE_DIR}"
rm -f "${TRACE_DIR}/client.jsonl"

env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8080 \
  --backend vllm-omni \
  --model Qwen/Qwen-Image \
  --dataset random \
  --task t2i \
  --num-prompts 500 \
  --max-concurrency 1000 \
  --request-rate 0.5 \
  --warmup-requests 1 \
  --warmup-num-inference-steps 1 \
  --client-timeout-s 1000000 \
  --seed 0 \
  --random-request-seed 8 \
  --arrival-seed 8 \
  --random-request-config '[
    {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
    {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
    {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
    {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
  ]' \
  --trace-log-file "${TRACE_DIR}/client.jsonl" \
  --trace-label client \
  --output-file "${RESULT_DIR}/result.json"

python3 benchmarks/diffusion/wan22_request_trace.py \
  --trace-dir "${TRACE_DIR}" \
  --output-prefix "${RESULT_DIR}/request_trace"

#!/bin/bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
RESULT_DIR="${RESULT_DIR:-/tmp/wan22_benchmark_50_trace}"
TRACE_LOG_DIR="${TRACE_LOG_DIR:-${RESULT_DIR}/trace}"

mkdir -p "${RESULT_DIR}" "${TRACE_LOG_DIR}"
rm -f "${TRACE_LOG_DIR}/client.jsonl"

env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url "${BASE_URL}" \
  --backend v1/videos \
  --model "${MODEL}" \
  --dataset random \
  --task t2v \
  --num-prompts 50 \
  --max-concurrency 50 \
  --request-rate 0.05 \
  --warmup-requests 1 \
  --client-timeout-s 1000000 \
  --enable-negative-prompt \
  --disable-tqdm \
  --seed 42 \
  --random-request-seed 42 \
  --arrival-seed 42 \
  --random-request-config '[
    {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":0.15},
    {"width":854,"height":480,"num_inference_steps":4,"num_frames":120,"fps":24,"weight":0.25},
    {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":0.60}
  ]' \
  --trace-log-file "${TRACE_LOG_DIR}/client.jsonl" \
  --trace-label client \
  --output-file "${RESULT_DIR}/result.json"

python3 benchmarks/diffusion/wan22_request_trace.py \
  --trace-dir "${TRACE_LOG_DIR}" \
  --output-prefix "${RESULT_DIR}/request_trace"

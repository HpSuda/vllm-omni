#!/bin/bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
DEVICE_IDS="${DEVICE_IDS:-0;1;2;3;4;5;6;7}"
BACKEND_START_PORT="${BACKEND_START_PORT:-8091}"
BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-/tmp/wan22_t2v_super_p95_8x1_cost_damped_risk}"
TRACE_LOG_DIR="${TRACE_LOG_DIR:-${BACKEND_LOG_DIR}/trace}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-910B3}"
WAN22_ESTIMATOR_PROFILE="${WAN22_ESTIMATOR_PROFILE:-8xusp1_inferred}"
CENTRAL_PULL_RISK_BETA="${CENTRAL_PULL_RISK_BETA:-0.5}"
NORMAL_ROUTING_POLICY="${NORMAL_ROUTING_POLICY:-central_pull_cost_damped_risk}"
CENTRAL_PULL_BAND_RISK_BETA="${CENTRAL_PULL_BAND_RISK_BETA:-0.625}"
CENTRAL_PULL_BAND_MIN_PENDING="${CENTRAL_PULL_BAND_MIN_PENDING:-10}"
CENTRAL_PULL_BAND_MAX_PENDING="${CENTRAL_PULL_BAND_MAX_PENDING:-27}"
CENTRAL_PULL_MIX_RISK_BETA="${CENTRAL_PULL_MIX_RISK_BETA:-0.4}"
CENTRAL_PULL_MIX_MIN_PENDING="${CENTRAL_PULL_MIX_MIN_PENDING:-16}"
CENTRAL_PULL_MIX_MAX_PENDING="${CENTRAL_PULL_MIX_MAX_PENDING:-26}"
CENTRAL_PULL_MIX_MAX_LONG_FRACTION="${CENTRAL_PULL_MIX_MAX_LONG_FRACTION:-0.32}"
CENTRAL_PULL_BEAM_HORIZON="${CENTRAL_PULL_BEAM_HORIZON:-4}"
CENTRAL_PULL_BEAM_WIDTH="${CENTRAL_PULL_BEAM_WIDTH:-16}"
CENTRAL_PULL_BEAM_BRANCH_WIDTH="${CENTRAL_PULL_BEAM_BRANCH_WIDTH:-6}"
CENTRAL_PULL_BEAM_RISK_SLACK_S="${CENTRAL_PULL_BEAM_RISK_SLACK_S:-100.0}"
CENTRAL_PULL_BEAM_MIN_PENDING="${CENTRAL_PULL_BEAM_MIN_PENDING:-10}"
CENTRAL_PULL_BEAM_MAX_PENDING="${CENTRAL_PULL_BEAM_MAX_PENDING:-27}"
CENTRAL_PULL_BEAM_HISTORY_SIZE="${CENTRAL_PULL_BEAM_HISTORY_SIZE:-128}"
CENTRAL_PULL_BEAM_CANDIDATE_CAP="${CENTRAL_PULL_BEAM_CANDIDATE_CAP:-4096}"
CENTRAL_PULL_LEVELING_TRIGGER_PENDING="${CENTRAL_PULL_LEVELING_TRIGGER_PENDING:-16}"
CENTRAL_PULL_LEVELING_COMMIT_SIZE="${CENTRAL_PULL_LEVELING_COMMIT_SIZE:-0}"
CENTRAL_PULL_LEVELING_MAX_DESCENT_ROUNDS="${CENTRAL_PULL_LEVELING_MAX_DESCENT_ROUNDS:-6}"
CENTRAL_PULL_LEVELING_CANDIDATE_CAP="${CENTRAL_PULL_LEVELING_CANDIDATE_CAP:-20000}"
TAIL_ROUTING_MODE="${TAIL_ROUTING_MODE:-spread}"
TAIL_DISPATCH_MODE="${TAIL_DISPATCH_MODE:-immediate}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1000000}"
BACKEND_HEALTH_TIMEOUT_S="${BACKEND_HEALTH_TIMEOUT_S:-1800}"
BACKEND_HEALTH_POLL_INTERVAL_S="${BACKEND_HEALTH_POLL_INTERVAL_S:-10}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
QUOTA_EVERY="${QUOTA_EVERY:-20}"
QUOTA_AMOUNT="${QUOTA_AMOUNT:-1}"
THRESHOLD_RATIO="${THRESHOLD_RATIO:-0.8}"
SACRIFICIAL_LOAD_FACTOR="${SACRIFICIAL_LOAD_FACTOR:-0.1}"
CLEAN_BACKEND_LOG_DIR="${CLEAN_BACKEND_LOG_DIR:-1}"

if [[ "${CLEAN_BACKEND_LOG_DIR}" == "1" ]]; then
  rm -rf "${BACKEND_LOG_DIR}"
fi
mkdir -p "${BACKEND_LOG_DIR}" "${TRACE_LOG_DIR}"

exec env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/wan22_super_p95_dispatcher.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-servers 8 \
  --device-ids "${DEVICE_IDS}" \
  --model "${MODEL}" \
  --backend-start-port "${BACKEND_START_PORT}" \
  --backend-hardware-profiles "${HARDWARE_PROFILE}" \
  --wan22-estimator-profile "${WAN22_ESTIMATOR_PROFILE}" \
  --backend-scheduler super_p95_step \
  --quota-every "${QUOTA_EVERY}" \
  --quota-amount "${QUOTA_AMOUNT}" \
  --threshold-ratio "${THRESHOLD_RATIO}" \
  --sacrificial-load-factor "${SACRIFICIAL_LOAD_FACTOR}" \
  --normal-routing-policy "${NORMAL_ROUTING_POLICY}" \
  --central-pull-risk-beta "${CENTRAL_PULL_RISK_BETA}" \
  --central-pull-band-risk-beta "${CENTRAL_PULL_BAND_RISK_BETA}" \
  --central-pull-band-min-pending "${CENTRAL_PULL_BAND_MIN_PENDING}" \
  --central-pull-band-max-pending "${CENTRAL_PULL_BAND_MAX_PENDING}" \
  --central-pull-mix-risk-beta "${CENTRAL_PULL_MIX_RISK_BETA}" \
  --central-pull-mix-min-pending "${CENTRAL_PULL_MIX_MIN_PENDING}" \
  --central-pull-mix-max-pending "${CENTRAL_PULL_MIX_MAX_PENDING}" \
  --central-pull-mix-max-long-fraction "${CENTRAL_PULL_MIX_MAX_LONG_FRACTION}" \
  --central-pull-beam-horizon "${CENTRAL_PULL_BEAM_HORIZON}" \
  --central-pull-beam-width "${CENTRAL_PULL_BEAM_WIDTH}" \
  --central-pull-beam-branch-width "${CENTRAL_PULL_BEAM_BRANCH_WIDTH}" \
  --central-pull-beam-risk-slack-s "${CENTRAL_PULL_BEAM_RISK_SLACK_S}" \
  --central-pull-beam-min-pending "${CENTRAL_PULL_BEAM_MIN_PENDING}" \
  --central-pull-beam-max-pending "${CENTRAL_PULL_BEAM_MAX_PENDING}" \
  --central-pull-beam-history-size "${CENTRAL_PULL_BEAM_HISTORY_SIZE}" \
  --central-pull-beam-candidate-cap "${CENTRAL_PULL_BEAM_CANDIDATE_CAP}" \
  --central-pull-leveling-trigger-pending "${CENTRAL_PULL_LEVELING_TRIGGER_PENDING}" \
  --central-pull-leveling-commit-size "${CENTRAL_PULL_LEVELING_COMMIT_SIZE}" \
  --central-pull-leveling-max-descent-rounds "${CENTRAL_PULL_LEVELING_MAX_DESCENT_ROUNDS}" \
  --central-pull-leveling-candidate-cap "${CENTRAL_PULL_LEVELING_CANDIDATE_CAP}" \
  --tail-routing-mode "${TAIL_ROUTING_MODE}" \
  --tail-dispatch-mode "${TAIL_DISPATCH_MODE}" \
  --backend-log-dir "${BACKEND_LOG_DIR}" \
  --trace-log-dir "${TRACE_LOG_DIR}" \
  --request-timeout-s "${REQUEST_TIMEOUT_S}" \
  --backend-health-timeout-s "${BACKEND_HEALTH_TIMEOUT_S}" \
  --backend-health-poll-interval-s "${BACKEND_HEALTH_POLL_INTERVAL_S}" \
  --backend-args=--omni \
  --backend-args=--usp \
  --backend-args=1 \
  --backend-args=--enable-layerwise-offload \
  --backend-args=--boundary-ratio \
  --backend-args="${BOUNDARY_RATIO}" \
  --backend-args=--flow-shift \
  --backend-args="${FLOW_SHIFT}" \
  --backend-args=--vae-use-slicing \
  --backend-args=--vae-use-tiling

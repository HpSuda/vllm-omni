#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${1:-${REPO_ROOT}/results/simulator/wan22_8xusp1_five_policy_behavior_100}"
TRACE_POOL="${REPO_ROOT}/benchmarks/diffusion/simulator/configs/wan22_8xusp1_calibrated_trace_pool.yaml"
TRACE_POOL_PECT="${REPO_ROOT}/benchmarks/diffusion/simulator/configs/wan22_8xusp1_calibrated_trace_pool_pect.yaml"
BEAM_CONFIG="${REPO_ROOT}/benchmarks/diffusion/simulator/configs/wan22_8xusp1_tail_aware_release_calendar_beam_50.yaml"
PYTHON=(python3)

if ! python3 -c 'import yaml' >/dev/null 2>&1; then
  PYTHON=(uv run --no-project --with pyyaml python)
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

run_policy() {
  local slug=$1
  local config=$2
  shift 2
  local output_dir="${OUTPUT_ROOT}/${slug}"
  local args=()
  local override

  mkdir -p "${output_dir}"
  for override in "$@"; do
    args+=(--set "${override}")
  done

  "${PYTHON[@]}" -m benchmarks.diffusion.simulator \
    --config "${config}" \
    --seed 42 \
    --runs 1 \
    "${args[@]}" \
    --output "${output_dir}/summary.json" \
    --requests-output "${output_dir}/requests.csv" \
    --trace-output "${output_dir}/events.jsonl"

  "${PYTHON[@]}" benchmarks/diffusion/simulator/trace_report.py \
    --events "${output_dir}/events.jsonl" \
    --requests "${output_dir}/requests.csv" \
    --output "${output_dir}/trace.html" \
    --seed 42 \
    --bubble-threshold-s 1 \
    --max-requests 20
}

run_policy \
  original \
  "${TRACE_POOL}" \
  "workload.request_types.0.estimated_service_s=38.07" \
  "workload.request_types.1.estimated_service_s=71.34" \
  "workload.request_types.2.estimated_service_s=119.71" \
  "policy.scheduler.global_protected_pull=false"

run_policy \
  beta05_spread \
  "${TRACE_POOL}"

run_policy \
  beta095_pack \
  "${TRACE_POOL_PECT}" \
  "policy.scheduler.protected_pull_risk_beta=0.95" \
  "policy.router.tail_mode=pack" \
  "policy.scheduler.protected_pull_tail_head_start=true"

run_policy \
  beta085_pack_gate \
  "${TRACE_POOL_PECT}" \
  "policy.scheduler.protected_pull_risk_beta=0.85" \
  "policy.router.tail_mode=pack" \
  "policy.scheduler.protected_pull_tail_head_start=false"

run_policy \
  beam_pack_gate \
  "${BEAM_CONFIG}" \
  "workload.num_requests=100" \
  "workload.request_rate=0.03" \
  "workload.request_types.0.nominal_service_s=105.61243595590904" \
  "workload.request_types.1.nominal_service_s=221.69386408977994" \
  "workload.request_types.2.nominal_service_s=627.7826891910798" \
  "service.actual_jitter_sigma=0.011239140586559346"

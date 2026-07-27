#!/bin/bash
set -euo pipefail

# Wan2.2 8×USP1 pure-P95 preset: level the complete online-visible Normal
# backlog, then commit only the next bounded dispatch wave. The wave stores a
# request order, not request-to-backend bindings; whichever backend becomes
# free next consumes the next request.
export NORMAL_ROUTING_POLICY="central_pull_backlog_leveling_wave_commit"
export CENTRAL_PULL_RISK_BETA="0.85"
export CENTRAL_PULL_BAND_RISK_BETA="0.625"
export CENTRAL_PULL_BAND_MIN_PENDING="10"
export CENTRAL_PULL_BAND_MAX_PENDING="27"
export CENTRAL_PULL_LEVELING_TRIGGER_PENDING="16"
# Zero resolves to one full backend wave (8 for this preset).
export CENTRAL_PULL_LEVELING_COMMIT_SIZE="0"
export CENTRAL_PULL_LEVELING_MAX_DESCENT_ROUNDS="6"
# The req50 seed-42 epochs need at most 2,269 evaluations. Keep a bounded
# synchronous planning budget for the experimental NPU run.
export CENTRAL_PULL_LEVELING_CANDIDATE_CAP="4096"
export CENTRAL_PULL_BEAM_HISTORY_SIZE="128"
export TAIL_ROUTING_MODE="pack"
export TAIL_DISPATCH_MODE="protected_drain"
export BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-/tmp/wan22_t2v_super_p95_8x1_backlog_leveling_wave_commit}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_wan22_super_p95_dispatcher_8x1_cost_damped_risk.sh"

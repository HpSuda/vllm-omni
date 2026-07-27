#!/bin/bash
set -euo pipefail

# Reproducible Wan2.2 8×USP1 Beam preset. Keep every coupled policy choice
# explicit so invoking this entry point cannot silently fall back to the
# cost-damped/spread defaults of the reusable launcher.
export NORMAL_ROUTING_POLICY="central_pull_tail_aware_release_calendar_beam"
export CENTRAL_PULL_RISK_BETA="0.85"
export CENTRAL_PULL_BAND_RISK_BETA="0.625"
export CENTRAL_PULL_BAND_MIN_PENDING="10"
export CENTRAL_PULL_BAND_MAX_PENDING="27"
export CENTRAL_PULL_BEAM_HORIZON="4"
export CENTRAL_PULL_BEAM_WIDTH="16"
export CENTRAL_PULL_BEAM_BRANCH_WIDTH="6"
export CENTRAL_PULL_BEAM_RISK_SLACK_S="100.0"
export CENTRAL_PULL_BEAM_MIN_PENDING="10"
export CENTRAL_PULL_BEAM_MAX_PENDING="27"
export CENTRAL_PULL_BEAM_HISTORY_SIZE="128"
export CENTRAL_PULL_BEAM_CANDIDATE_CAP="4096"
export TAIL_ROUTING_MODE="pack"
export TAIL_DISPATCH_MODE="protected_drain"
export BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-/tmp/wan22_t2v_super_p95_8x1_tail_aware_release_calendar_beam}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_wan22_super_p95_dispatcher_8x1_cost_damped_risk.sh"

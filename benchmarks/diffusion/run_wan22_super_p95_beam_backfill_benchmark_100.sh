#!/bin/bash
set -euo pipefail

export RESULT_DIR="/tmp/wan22_super_p95_beam_backfill_req100"
export TRACE_LOG_DIR="/tmp/wan22_super_p95_beam_backfill/trace"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_wan22_benchmark_100_trace.sh"

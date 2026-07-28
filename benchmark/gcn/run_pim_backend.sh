#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRIPT_DIR}"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

ID_X="${1:-0}"
ID_Y="${2:-1}"
WORKLOAD_FILE="${3:-$BENCHMARK_ROOT/reports/pim_pe_workload.txt}"
RUNTIME_PLAN_FILE="${4:-$BENCHMARK_ROOT/reports/runtime_plan_pim.txt}"
INTRA_DELAY_PROFILE="${5:-$BENCHMARK_ROOT/reports/pim_intra_delay_profile.txt}"
PIM_NATIVE_BIN="$BENCHMARK_ROOT/bin/pim"

if [[ ! -x "$PIM_NATIVE_BIN" ]]; then
  echo "[run_pim_backend] ERROR: native PIM binary not found: $PIM_NATIVE_BIN"
  exit 1
fi

# Keep the workload argument for call-site compatibility even though the native
# PIM path currently consumes the runtime plan and intra-delay profile only.
if [[ -n "$WORKLOAD_FILE" && ! -f "$WORKLOAD_FILE" ]]; then
  echo "[run_pim_backend] WARN: workload summary not found, continuing: $WORKLOAD_FILE"
fi

exec "$PIM_NATIVE_BIN" "$ID_X" "$ID_Y" "$RUNTIME_PLAN_FILE" "$INTRA_DELAY_PROFILE"

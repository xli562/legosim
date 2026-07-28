#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

POPNET_BIN="$SIMULATOR_ROOT/popnet_chiplet/build/popnet"
BENCH_FILE="$SCRIPT_DIR/bench_pim_intra.txt"
DELAY_FILE="$SCRIPT_DIR/delayInfo_pim_intra.txt"
TOPO_FILE="$SCRIPT_DIR/topology/pim_vertical_4_4.gv"
NORMALIZED_BENCH="$SCRIPT_DIR/bench_pim_intra_local.txt"
POPNET_LOG_FILE="${POPNET_LOG_FILE:-$SCRIPT_DIR/run_popnet_intra_pim.log}"
POPNET_LOG_MODE="${POPNET_LOG_MODE:-summary}"

if [[ ! -x "$POPNET_BIN" ]]; then
  echo "[run_popnet_intra_pim] ERROR: popnet binary not found: $POPNET_BIN"
  echo "[run_popnet_intra_pim] Build it first: (cd $SIMULATOR_ROOT/popnet_chiplet && make)"
  exit 1
fi

if [[ ! -f "$BENCH_FILE" ]]; then
  echo "[run_popnet_intra_pim] ERROR: bench file not found: $BENCH_FILE"
  echo "[run_popnet_intra_pim] Generate it first: (cd $SCRIPT_DIR && make ir-plan)"
  exit 1
fi
if [[ ! -f "$TOPO_FILE" ]]; then
  echo "[run_popnet_intra_pim] ERROR: topology file not found: $TOPO_FILE"
  exit 1
fi

echo "[run_popnet_intra_pim] Running PIM intra-chiplet PopNet"
echo "  popnet: $POPNET_BIN"
echo "  bench : $BENCH_FILE"
echo "  topo  : $TOPO_FILE"
echo "  log   : $POPNET_LOG_FILE"
echo "  mode  : $POPNET_LOG_MODE"

# bench_pim_intra may use global router IDs (e.g., 4..19).
# For standalone 17-node intra run:
# - keep global controller router id 1 as local node 16
# - normalize global PIM NPU router IDs (typically 4..19) to local 0..15
awk '
  NF >= 6 {
    src = $3
    dst = $4
    if (src == 1) {
      $3 = 16
    } else if (src >= 4 && src <= 19) {
      $3 = src - 4
    }
    if (dst == 1) {
      $4 = 16
    } else if (dst >= 4 && dst <= 19) {
      $4 = dst - 4
    }
    print $1" "$2" "$3" "$4" "$5" "$6
  }
' "$BENCH_FILE" > "$NORMALIZED_BENCH"

run_popnet() {
  "$POPNET_BIN" \
    -A 17 \
    -c 1 \
    -V 3 \
    -B 12 \
    -O 12 \
    -F 4 \
    -L 1000 \
    -T 1000000000 \
    -r 1 \
    -I "$NORMALIZED_BENCH" \
    -G "$TOPO_FILE" \
    -R 4 \
    -D "$DELAY_FILE" \
    -P
}

if [[ "$POPNET_LOG_MODE" == "silent" ]]; then
  if ! run_popnet >"$POPNET_LOG_FILE" 2>&1; then
    echo "[run_popnet_intra_pim] ERROR: PopNet failed. See log: $POPNET_LOG_FILE"
    tail -n 50 "$POPNET_LOG_FILE" || true
    exit 1
  fi
elif [[ "$POPNET_LOG_MODE" == "full" ]]; then
  if ! run_popnet >"$POPNET_LOG_FILE" 2>&1; then
    echo "[run_popnet_intra_pim] ERROR: PopNet failed. See log: $POPNET_LOG_FILE"
    tail -n 50 "$POPNET_LOG_FILE" || true
    exit 1
  fi
else
  if ! run_popnet 2>&1 | awk '
    /^\*+configuration\*+$/ { print; next }
    /^ary size:/ { print; next }
    /^cube dimension:/ { print; next }
    /^virtual channel number:/ { print; next }
    /^buffer size:/ { print; next }
    /^outbuffer size:/ { print; next }
    /^flit size:/ { print; next }
    /^link length:/ { print; next }
    /^simulation length:/ { print; next }
    /^trace file:/ { print; next }
    /^Routing algorithm:/ { print; next }
    /^Transaction count:/ { print; next }
    /^Packet count:/ { print; next }
    /^Direct Forward to cycle:/ { print; next }
    /^total finished:/ { print; next }
    /^average Delay:/ { print; next }
    /^total mem power:/ { print; next }
    /^total crossbar power:/ { print; next }
    /^total arbiter power:/ { print; next }
    /^total link power:/ { print; next }
    /^total power:/ { print; next }
    /^simulation empty check is correct\./ { print; next }
  ' >"$POPNET_LOG_FILE"; then
    echo "[run_popnet_intra_pim] ERROR: PopNet failed. See log: $POPNET_LOG_FILE"
    tail -n 50 "$POPNET_LOG_FILE" || true
    exit 1
  fi
fi

echo "[run_popnet_intra_pim] Done. Delay file: $DELAY_FILE"


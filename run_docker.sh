#!/bin/bash
# 用法: ./run_chiplet.sh [name]
NAME="${1:-chiplet}"

docker run -it --rm \
    --name "$NAME" \
    --hostname coord \
    --network host \
    -e http_proxy=http://127.0.0.1:10081 \
    -e https_proxy=http://127.0.0.1:10081 \
    -v ~/single_stage_simulator:/workspace/sim \
    -w /workspace/sim \
    chiplet-sim:u1804 bash

#!/usr/bin/env bash

# This is the template for run.sh

# Build and run the GEMM 2-D mesh simulation inside the `chiplet` container.
# Run from the host; prints the fresh run directory it creates.
set -eo pipefail

docker exec -u $(id -u):$(id -g) chiplet bash -c '
set -eo pipefail
cd /workspace/sim
source scripts/setup_env.sh

make -C "<SIMULATION_ROOT>"

mkdir -p run_outputs
timestamp=$(date +%y-%m-%d_%H-%M-%S)
run_dir=$(mktemp -d "/workspace/sim/run_outputs/${timestamp}_gemm_2dmesh.XXX")
chmod 755 "$run_dir"

cp <SIM_CONFIG_YAML_PATH> "$run_dir/input.sim_config.yml"
cp <GRAPHVIZ_FILE_PATH> "$run_dir/2dmesh.gv"
cp "<SIMULATION_ROOT>/src/comms_protocol.h" "$run_dir/comms_protocol.h"
cp "<SIMULATION_ROOT>/bin/asic" "<SIMULATION_ROOT>/bin/cpu" "$run_dir/"

printf "Output directory: %s\n" "$run_dir"
cd "$run_dir"
/workspace/sim/interchiplet/bin/interchiplet \
    <SIM_CONFIG_YAML_PATH> \
    -w <MESH_WIDTH> \
    -H <MESH_HEIGHT> \
    -f <FLITS_PER_PACKAGE> \
    2>&1 | tee coordinator.log
printf "Done. See %s/coordinator.log\n" "$run_dir"
'

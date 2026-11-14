#!/usr/bin/env bash
# Build and run the GEMM 2-D mesh simulation inside the `chiplet` container.
# Run from the host; prints the fresh run directory it creates.
set -e

docker exec chiplet bash -c '
set -e
cd /workspace/sim
source setup_env.sh

bench=/workspace/sim/benchmark/gemm_2dmesh
make -C "$bench"

mkdir -p run_outputs
run_dir=$(mktemp -d /workspace/sim/run_outputs/gemm_2dmesh.XXXXXX)
chmod 755 "$run_dir"

# Record the inputs. The coordinator still reads the YAML from its original
# location because $BENCHMARK_ROOT is derived from the YAML path.
cp "$bench/gemm_2dmesh.yml"       "$run_dir/input.gemm_2dmesh.yml"
cp "$bench/topology/2dmesh.gv"    "$run_dir/2dmesh.gv"
cp "$bench/comms_protocol.h"      "$run_dir/comms_protocol.h"
cp "$bench/bin/asic" "$bench/bin/cpu" "$run_dir/"

printf "Output directory: %s\n" "$run_dir"
cd "$run_dir"
/workspace/sim/interchiplet/bin/interchiplet "$bench/gemm_2dmesh.yml" -w 3 -f 2 \
    > coordinator.log 2>&1
printf "Done. See %s/coordinator.log\n" "$run_dir"
'

#!/bin/bash

if [[ -n "${SIMULATOR_ROOT}" ]]; then
    echo "SIMULATOR_ROOT is: ${SIMULATOR_ROOT}"
else
    echo "The environment variable SIMULATOR_ROOT is not defined."
    exit
fi

# Pathc for Sniper
cd ${SIMULATOR_ROOT}/dependencies/snipersim
git apply ${SIMULATOR_ROOT}/interchiplet/patch/snipersim.diff

# Patch for GPGPUSim
cd ${SIMULATOR_ROOT}/dependencies/gpgpu-sim
git apply ${SIMULATOR_ROOT}/interchiplet/patch/gpgpu-sim.diff

# Patch for GEM5
cd ${SIMULATOR_ROOT}/dependencies/gem5
git apply ${SIMULATOR_ROOT}/interchiplet/patch/gem5.diff

# Patch for Popnet
cd ${SIMULATOR_ROOT}/dependencies/popnet_chiplet
git apply ${SIMULATOR_ROOT}/interchiplet/patch/popnet.diff

# Patch for scale-sim
cd ${SIMULATOR_ROOT}/dependencies/scale-sim-v2
git apply ${SIMULATOR_ROOT}/interchiplet/patch/scale-sim.diff
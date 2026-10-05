# Use GCC 10 for LEGOSim / GPGPU-Sim
export PATH="$HOME/.local/gcc10/bin:$PATH"
export CC=gcc-10
export CXX=g++-10
hash -r

# Python2 support
export PATH="$HOME/.local/opt/lego-python2-shim:$PATH"

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64
export CUDA_INSTALL_PATH="$HOME/opt/cuda-11.3"
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
export SIMULATOR_ROOT="$HOME/LEGOSIM_MICRO"

# Fix SniperSim
# DynamoRIO libraries required by Sniper's DR frontend
export DR_BUILD="$SIMULATOR_ROOT/snipersim/dynamorio/build"
export SNIPER_SIM_LD_LIBRARY_PATH="$DR_BUILD/lib64/debug:$DR_BUILD/ext/lib64/debug"

source gpgpu-sim/setup_environment

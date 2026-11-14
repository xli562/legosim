# Keep a valid caller-provided CUDA_INSTALL_PATH. Otherwise, derive it from
# nvcc so the setup keeps working when CUDA is installed outside /usr/local.
if [ -z "${CUDA_INSTALL_PATH:-}" ] || [ ! -d "$CUDA_INSTALL_PATH" ]; then
    LEGOSIM_NVCC="$(command -v nvcc 2>/dev/null)"
    if [ -z "$LEGOSIM_NVCC" ]; then
        echo "ERROR ** CUDA Toolkit not found; set CUDA_INSTALL_PATH or add nvcc to PATH."
        return 1
    fi
    export CUDA_INSTALL_PATH="$(cd "$(dirname "$LEGOSIM_NVCC")/.." && pwd)"
fi

export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
export SIMULATOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$SIMULATOR_ROOT/dependencies/gpgpu-sim/setup_environment"

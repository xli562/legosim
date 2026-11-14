// Functional 10x10 GEMM tile with an explicit, replaceable accelerator latency.
// The CPU/GPU benchmark sends the same two 100x100 matrices to each worker.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "pipe_comm.h"
#include "sync_protocol.h"

namespace {
const int kMatrixWidth = 100;
const int kTileWidth = 10;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: matmul_npu NPU_X NPU_Y CPU_X CPU_Y COMPUTE_CYCLES\n";
        return 2;
    }
    const int npu_x = std::atoi(argv[1]);
    const int npu_y = std::atoi(argv[2]);
    const int cpu_x = std::atoi(argv[3]);
    const int cpu_y = std::atoi(argv[4]);
    const long long compute_cycles = std::atoll(argv[5]);
    if (npu_x < 0 || npu_y < 0 || cpu_x < 0 || cpu_y < 0 || compute_cycles <= 0) {
        std::cerr << "Invalid accelerator arguments\n";
        return 2;
    }

    std::vector<int64_t> a(kMatrixWidth * kMatrixWidth);
    std::vector<int64_t> b(kMatrixWidth * kMatrixWidth);
    std::vector<int64_t> result(kTileWidth * kTileWidth);
    InterChiplet::PipeComm pipes;
    const int input_bytes = static_cast<int>(a.size() * sizeof(int64_t));
    const int output_bytes = static_cast<int>(result.size() * sizeof(int64_t));

    InterChiplet::TimeType cycle = 1;
    for (int input = 0; input < 2; ++input) {
        std::string pipe_name = InterChiplet::receiveSync(cpu_x, cpu_y, npu_x, npu_y);
        int64_t *destination = input == 0 ? a.data() : b.data();
        if (pipes.read_data(pipe_name.c_str(), destination, input_bytes) != input_bytes) {
            std::cerr << "NPU input transfer was incomplete\n";
            return 1;
        }
        cycle = InterChiplet::readSync(cycle, cpu_x, cpu_y, npu_x, npu_y,
                                       input_bytes, 0);
        if (cycle == static_cast<InterChiplet::TimeType>(-1)) {
            std::cerr << "NPU input timing synchronization failed\n";
            return 1;
        }
    }

    int64_t checksum = 0;
    for (int row = 0; row < kTileWidth; ++row) {
        for (int col = 0; col < kTileWidth; ++col) {
            int64_t sum = 0;
            for (int k = 0; k < kMatrixWidth; ++k)
                sum += a[row * kMatrixWidth + k] * b[k * kMatrixWidth + col];
            result[row * kTileWidth + col] = sum;
            checksum += sum;
        }
    }

    const std::string pipe_name = InterChiplet::sendSync(npu_x, npu_y, cpu_x, cpu_y);
    if (pipes.write_data(pipe_name.c_str(), result.data(), output_bytes) != output_bytes) {
        std::cerr << "NPU output transfer was incomplete\n";
        return 1;
    }
    const InterChiplet::TimeType done = InterChiplet::writeSync(
        cycle + compute_cycles, npu_x, npu_y, cpu_x, cpu_y, output_bytes, 0);
    if (done == static_cast<InterChiplet::TimeType>(-1)) {
        std::cerr << "NPU output timing synchronization failed\n";
        return 1;
    }
    std::cerr << "NPU_CHECK PASS checksum=" << checksum
              << " compute_cycles=" << compute_cycles << '\n';
    return 0;
}

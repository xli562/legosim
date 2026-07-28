#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "apis_c.h"
#include "gcn_runtime.h"

namespace {
constexpr int kMemoryChannels = 2;
constexpr int kCpuX = 0;
constexpr int kCpuY = 0;

enum class IoExecMode {
    kAnalytic,
    kReal,
};

struct IoMemoryLevel {
    const char* name = "L";
    std::uint64_t capacity_bytes = 0;
    double bandwidth_bytes_per_cycle = 1.0;
    std::uint64_t access_latency_cycles = 0;
    bool unlimited = false;
};

struct IoTimingConfig {
    std::uint64_t memory_channels = kMemoryChannels;
    std::uint64_t memory_unit_bytes = 131072;
    std::uint64_t miss_penalty_cycles = 120;
    std::uint64_t pipeline_cycles = 6;
    std::string compose_mode = "sum";
    double partial_overlap_ratio = 0.5;
    std::vector<IoMemoryLevel> levels;
};

struct IoEstimate {
    std::uint64_t compute_cycles = 0;
    std::uint64_t memory_cycles = 0;
    std::uint64_t miss_count = 0;
    std::uint64_t miss_cycles = 0;
    std::uint64_t core_cycles = 0;
    std::uint64_t total_cycles = 1;
};

struct IoTouchSummary {
    std::uint64_t checksum = 0;
    double l1_norm = 0.0;
    std::uint64_t touched_bytes = 0;
};

std::string to_lower_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
    );
    return value;
}

IoExecMode load_exec_mode() {
    const std::string mode =
        to_lower_copy(gcn_runtime::get_env_string("GCN_IO_EXEC_MODE", "real"));
    if (mode == "real" || mode == "store") {
        return IoExecMode::kReal;
    }
    return IoExecMode::kAnalytic;
}

std::uint64_t mix_hash(std::uint64_t state, std::uint64_t value) {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
    return state;
}

std::uint64_t ceil_div_by_bandwidth(std::uint64_t bytes, double bw_bytes_per_cycle) {
    if (bytes == 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(
        std::ceil(static_cast<double>(bytes) / std::max(1e-12, bw_bytes_per_cycle)));
}

std::uint64_t ceil_div_u64(std::uint64_t a, std::uint64_t b) {
    const std::uint64_t d = std::max<std::uint64_t>(1, b);
    return (a + d - 1) / d;
}

std::uint64_t compose_compute_memory(
    std::uint64_t compute_cycles,
    std::uint64_t memory_cycles,
    const std::string& mode,
    double partial_ratio
) {
    if (mode == "max") {
        return std::max(compute_cycles, memory_cycles);
    }
    if (mode == "partial") {
        const std::uint64_t a = std::max(compute_cycles, memory_cycles);
        const std::uint64_t b = std::min(compute_cycles, memory_cycles);
        return a + static_cast<std::uint64_t>(
            std::llround(static_cast<double>(b) * partial_ratio));
    }
    return compute_cycles + memory_cycles;
}

IoTimingConfig load_io_config() {
    IoTimingConfig cfg;
    cfg.memory_channels = gcn_runtime::get_env_u64("GCN_IO_MEMORY_CHANNELS", kMemoryChannels);
    const double channel_bw = gcn_runtime::get_env_double("GCN_IO_CHANNEL_BW_BPC", 64.0);
    const double total_mem_bw = gcn_runtime::get_env_double(
        "GCN_IO_MEM_BW_BPC", channel_bw * static_cast<double>(cfg.memory_channels));

    cfg.memory_unit_bytes = gcn_runtime::get_env_u64("GCN_IO_MEMORY_UNIT_BYTES", 131072);
    cfg.miss_penalty_cycles = gcn_runtime::get_env_u64("GCN_IO_CACHE_MISS_PENALTY", 120);
    cfg.pipeline_cycles = gcn_runtime::get_env_u64("GCN_IO_PIPELINE_CYCLES", 6);
    cfg.compose_mode = gcn_runtime::get_env_string("GCN_IO_COMPUTE_MEMORY_MODE", "sum");
    cfg.partial_overlap_ratio = gcn_runtime::get_env_double("GCN_IO_COMPUTE_MEMORY_PARTIAL", 0.5);
    cfg.partial_overlap_ratio = std::max(0.0, std::min(1.0, cfg.partial_overlap_ratio));

    cfg.levels.push_back(
        {
            "L1",
            gcn_runtime::get_env_u64("GCN_IO_L1_BYTES", 32ULL * 1024ULL),
            gcn_runtime::get_env_double("GCN_IO_L1_BW_BPC", 64.0),
            gcn_runtime::get_env_u64("GCN_IO_L1_LAT", 1),
            false,
        }
    );
    cfg.levels.push_back(
        {
            "L2",
            gcn_runtime::get_env_u64("GCN_IO_L2_BYTES", 64ULL * 1024ULL),
            gcn_runtime::get_env_double("GCN_IO_L2_BW_BPC", 32.0),
            gcn_runtime::get_env_u64("GCN_IO_L2_LAT", 3),
            false,
        }
    );
    cfg.levels.push_back(
        {
            "L3",
            gcn_runtime::get_env_u64("GCN_IO_L3_BYTES", 128ULL * 1024ULL),
            gcn_runtime::get_env_double("GCN_IO_L3_BW_BPC", 16.0),
            gcn_runtime::get_env_u64("GCN_IO_L3_LAT", 8),
            false,
        }
    );
    cfg.levels.push_back(
        {
            "LLC",
            gcn_runtime::get_env_u64("GCN_IO_LLC_BYTES", 2ULL * 1024ULL * 1024ULL),
            gcn_runtime::get_env_double("GCN_IO_LLC_BW_BPC", 16.0),
            gcn_runtime::get_env_u64("GCN_IO_LLC_LAT", 12),
            false,
        }
    );
    cfg.levels.push_back(
        {
            "MEM",
            0,
            total_mem_bw,
            gcn_runtime::get_env_u64("GCN_IO_MEM_LAT", 0),
            true,
        }
    );
    return cfg;
}

IoEstimate estimate_store_cycles(std::uint64_t bytes, const IoTimingConfig& cfg) {
    IoEstimate out;
    std::uint64_t remaining = bytes;

    for (const IoMemoryLevel& lv : cfg.levels) {
        if (remaining == 0) {
            break;
        }
        const std::uint64_t hit_bytes =
            lv.unlimited ? remaining : std::min(remaining, lv.capacity_bytes);
        const std::uint64_t transfer_cycles =
            ceil_div_by_bandwidth(hit_bytes, lv.bandwidth_bytes_per_cycle);
        const std::uint64_t access_count =
            ceil_div_u64(hit_bytes, cfg.memory_unit_bytes);

        out.memory_cycles += transfer_cycles + access_count * lv.access_latency_cycles;
        if (std::string(lv.name) == "MEM") {
            out.miss_count = access_count;
        }
        remaining -= hit_bytes;
    }

    out.miss_cycles = out.miss_count * cfg.miss_penalty_cycles;
    out.core_cycles = compose_compute_memory(
        out.compute_cycles,
        out.memory_cycles,
        cfg.compose_mode,
        cfg.partial_overlap_ratio
    );
    out.total_cycles = std::max<std::uint64_t>(
        1, out.core_cycles + out.miss_cycles + cfg.pipeline_cycles);
    return out;
}

IoTouchSummary touch_output_for_store(
    std::vector<float>& output,
    int num_nodes,
    int output_dim
) {
    IoTouchSummary summary;
    if (output.empty() || num_nodes <= 0 || output_dim <= 0) {
        return summary;
    }

    std::vector<float> staging(output.size(), 0.0f);
    const int tile_rows = static_cast<int>(std::max<std::uint64_t>(
        1, gcn_runtime::get_env_u64("GCN_IO_STAGE_TILE_ROWS", 64)));
    std::uint64_t checksum = 0x0f0e0d0c0b0a0908ULL;
    double l1_norm = 0.0;

    for (int row_base = 0; row_base < num_nodes; row_base += tile_rows) {
        const int row_end = std::min(row_base + tile_rows, num_nodes);
        for (int row = row_base; row < row_end; ++row) {
            double row_norm = 0.0;
            for (int col = 0; col < output_dim; ++col) {
                const std::size_t idx =
                    static_cast<std::size_t>(row) * static_cast<std::size_t>(output_dim) +
                    static_cast<std::size_t>(col);
                const float value = output[idx];
                staging[idx] = value;
                row_norm += std::abs(static_cast<double>(value));
                checksum = mix_hash(
                    checksum,
                    static_cast<std::uint64_t>(idx + 1U) * 11400714819323198485ULL +
                        static_cast<std::uint64_t>(std::llround((value + 17.0f) * 65536.0f))
                );
            }
            l1_norm += row_norm;
            checksum = mix_hash(
                checksum,
                static_cast<std::uint64_t>(std::llround(row_norm * 4096.0))
            );
        }
    }

    output.swap(staging);
    summary.checksum = checksum;
    summary.l1_norm = l1_norm;
    summary.touched_bytes = static_cast<std::uint64_t>(output.size() * sizeof(float) * 2ULL);
    return summary;
}

std::vector<std::uint64_t> load_plan_store_cycles(const std::string& runtime_plan_path) {
    std::vector<gcn_runtime::Event> fallback_events;
    fallback_events.push_back(
        {gcn_runtime::EventType::kCompute, 0, 0, 0, 1, 0, 0, 0, "IO_STORE"});
    const auto plan_events =
        gcn_runtime::load_plan_file(runtime_plan_path, fallback_events, "GCN IO");

    std::vector<std::uint64_t> cycles;
    for (const auto& ev : plan_events) {
        if (ev.type == gcn_runtime::EventType::kCompute) {
            cycles.push_back(std::max<std::uint64_t>(1, ev.cycles));
        }
    }
    if (cycles.empty()) {
        cycles.push_back(1);
    }
    return cycles;
}
}  // namespace

void save_output(const float* output, int num_nodes, int output_dim, int epoch) {
    const std::string filename = "output_epoch_" + std::to_string(epoch) + ".txt";
    std::ofstream file(filename);
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            file << output[i * output_dim + j];
            if (j + 1 < output_dim) {
                file << " ";
            }
        }
        file << "\n";
    }
    file.close();
    std::cout << "[GCN IO] output saved: " << filename << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "[GCN IO] Usage: io <idX> <idY> [runtime_plan]\n";
        return EXIT_FAILURE;
    }

    const int id_x = std::atoi(argv[1]);
    const int id_y = std::atoi(argv[2]);
    const std::string runtime_plan_path =
        (argc >= 4) ? std::string(argv[3]) : std::string("../reports/runtime_plan_io.txt");
    const IoExecMode exec_mode = load_exec_mode();
    const bool verbose = gcn_runtime::get_env_u64("GCN_IO_REAL_VERBOSE", 1) != 0;
    const std::uint64_t burn_scale = gcn_runtime::get_env_u64("GCN_IO_BURN_SCALE", 1);

    std::vector<std::uint64_t> plan_store_cycles;
    if (exec_mode == IoExecMode::kAnalytic) {
        plan_store_cycles = load_plan_store_cycles(runtime_plan_path);
    }

    const IoTimingConfig io_cfg = load_io_config();
    std::cout << "[GCN IO] mode=" << ((exec_mode == IoExecMode::kReal) ? "real" : "analytic")
              << ", chiplet=(" << id_x << "," << id_y << "), mem_channels=" << kMemoryChannels
              << ", configured_mem_channels=" << io_cfg.memory_channels
              << ", burn_scale=" << burn_scale
              << ", runtime_plan=" << runtime_plan_path << "\n";

    int epoch = 0;

    while (true) {
        std::int64_t size_info[2] = {0, 0};
        const auto size_rc = InterChiplet::receiveMessage(
            id_x,
            id_y,
            kCpuX,
            kCpuY,
            static_cast<void*>(size_info),
            static_cast<std::int64_t>(sizeof(size_info))
        );
        if (size_rc < 0) {
            std::cerr << "[GCN IO] receiveMessage failed for size header\n";
            return EXIT_FAILURE;
        }

        const int num_nodes = static_cast<int>(size_info[0]);
        const int output_dim = static_cast<int>(size_info[1]);
        if (num_nodes == -1 && output_dim == -1) {
            std::cout << "[GCN IO] received DONE, exit\n";
            break;
        }
        if (num_nodes <= 0 || output_dim <= 0) {
            std::cerr << "[GCN IO] invalid shape from CPU: " << num_nodes << "x" << output_dim << "\n";
            return EXIT_FAILURE;
        }

        const std::size_t elem_cnt = static_cast<std::size_t>(num_nodes) * static_cast<std::size_t>(output_dim);
        std::vector<float> output(elem_cnt, 0.0f);
        const auto data_rc = InterChiplet::receiveMessage(
            id_x,
            id_y,
            kCpuX,
            kCpuY,
            static_cast<void*>(output.data()),
            static_cast<std::int64_t>(elem_cnt * sizeof(float))
        );
        if (data_rc < 0) {
            std::cerr << "[GCN IO] receiveMessage failed for payload\n";
            return EXIT_FAILURE;
        }

        if (exec_mode == IoExecMode::kReal) {
            const IoTouchSummary touch = touch_output_for_store(output, num_nodes, output_dim);
            if (verbose) {
                std::cout << "[GCN IO] epoch=" << epoch
                          << " real-store checksum=" << touch.checksum
                          << " l1_norm=" << touch.l1_norm
                          << " touched_bytes=" << touch.touched_bytes << "\n";
            }
        }
        if (exec_mode == IoExecMode::kAnalytic) {
            const IoEstimate est = estimate_store_cycles(
                static_cast<std::uint64_t>(elem_cnt * sizeof(float)), io_cfg);
            const std::uint64_t plan_cycles = plan_store_cycles[std::min<std::size_t>(
                static_cast<std::size_t>(epoch), plan_store_cycles.size() - 1)];
            const std::uint64_t store_cycles = std::max(est.total_cycles, plan_cycles);
            if (verbose) {
                std::cout << "[GCN IO] epoch=" << epoch
                          << " analytic-store est=" << est.total_cycles
                          << " plan=" << plan_cycles
                          << " burn=" << store_cycles << "\n";
            }
            gcn_runtime::burn_cycles(store_cycles, burn_scale);
        }

        save_output(output.data(), num_nodes, output_dim, epoch);
        ++epoch;
    }

    std::cout << "[GCN IO] shutdown, epochs=" << epoch << "\n";
    return EXIT_SUCCESS;
}

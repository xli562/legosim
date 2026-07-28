#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "apis_c.h"
#include "gcn_runtime.h"
#include "../../interchiplet/includes/pipe_comm.h"

namespace {
constexpr int kNumNpus = 16;
constexpr int kPimSramBytesPerNpu = 128 * 1024;
constexpr int kCpuX = 0;
constexpr int kCpuY = 0;
constexpr int kHiddenDim = 16;
constexpr int kOutputDim = 7;
constexpr int kEpochsDefault = 100;

struct GCNData {
    std::vector<int> feature_indices;
    std::vector<int> feature_indptr;
    std::vector<float> feature_values;
    std::vector<int> graph_indices;
    std::vector<int> graph_indptr;
    std::vector<int> labels;
};

struct IntraDelayProfile {
    bool loaded = false;
    std::uint64_t total_delay_cycles = 0;
    std::uint64_t packet_count = 0;
    std::vector<std::uint64_t> per_pe_delay_cycles;
};

std::string basename_of(const std::string& path) {
    const std::size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

bool file_exists(const std::string& path) {
    std::ifstream fin(path);
    return fin.good();
}

std::string resolve_required_input_path(const char* env_key, const char* default_rel) {
    std::vector<std::string> candidates;
    const std::string env_path = gcn_runtime::get_env_string(env_key, "");
    if (!env_path.empty()) {
        candidates.push_back(env_path);
    }

    const std::string def(default_rel);
    candidates.push_back(def);
    if (def.rfind("../", 0) == 0) {
        candidates.push_back(def.substr(3));
    }
    if (def.rfind("./", 0) == 0) {
        candidates.push_back(def.substr(2));
    }

    const std::string base = basename_of(def);
    candidates.push_back("data/" + base);
    candidates.push_back("./data/" + base);
    candidates.push_back("../data/" + base);
    candidates.push_back("benchmark/gcn/data/" + base);

    for (const std::string& candidate : candidates) {
        if (!candidate.empty() && file_exists(candidate)) {
            return candidate;
        }
    }

    std::ostringstream oss;
    oss << "cannot resolve dataset path for " << env_key << ", tried:";
    for (const std::string& candidate : candidates) {
        if (!candidate.empty()) {
            oss << " " << candidate;
        }
    }
    throw std::runtime_error(oss.str());
}

std::uint64_t ceil_div_u64(std::uint64_t a, std::uint64_t b) {
    const std::uint64_t d = std::max<std::uint64_t>(1, b);
    return (a + d - 1) / d;
}

void parse_graph(const std::string& filename, GCNData& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open graph file: " + filename);
    }
    std::string line;
    int node = 0;
    data.graph_indptr.clear();
    data.graph_indices.clear();
    data.graph_indptr.push_back(0);

    while (std::getline(file, line)) {
        data.graph_indices.push_back(node);
        data.graph_indptr.push_back(data.graph_indptr.back() + 1);
        ++node;

        std::istringstream ss(line);
        int neighbor = 0;
        while (ss >> neighbor) {
            data.graph_indices.push_back(neighbor);
            data.graph_indptr.back()++;
        }
    }
}

void parse_features(const std::string& filename, GCNData& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open feature file: " + filename);
    }
    std::string line;
    data.feature_indptr.clear();
    data.feature_indices.clear();
    data.feature_values.clear();
    data.labels.clear();
    data.feature_indptr.push_back(0);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        int label = 0;
        ss >> label;
        if (ss.fail()) {
            continue;
        }
        data.labels.push_back(label);
        data.feature_indptr.push_back(data.feature_indptr.back());

        std::string token;
        while (ss >> token) {
            std::istringstream kv(token);
            int k = 0;
            float v = 0.0f;
            char colon = ':';
            kv >> k >> colon >> v;
            if (kv.fail()) {
                continue;
            }
            data.feature_values.push_back(v);
            data.feature_indices.push_back(k);
            data.feature_indptr.back()++;
        }
    }
}

void graph_sum(
    const float* input,
    const std::vector<int>& indices,
    const std::vector<int>& indptr,
    float* output,
    int num_nodes,
    int dim
) {
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j) {
            const int neighbor = indices[j];
            for (int k = 0; k < dim; ++k) {
                output[i * dim + k] += input[neighbor * dim + k];
            }
        }
    }
}

void sparse_matmul(
    const float* features,
    const std::vector<int>& indices,
    const std::vector<int>& indptr,
    const float* weights,
    float* output,
    int num_nodes,
    int hidden_dim
) {
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j) {
            const int feature_idx = indices[j];
            const float feature_val = features[j];
            for (int k = 0; k < hidden_dim; ++k) {
                output[i * hidden_dim + k] += feature_val * weights[feature_idx * hidden_dim + k];
            }
        }
    }
}

void relu_activation(float* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
    }
}

void dense_matmul_npu(
    int npu_id,
    const float* input,
    const float* weights,
    float* output,
    int num_nodes,
    int hidden_dim,
    int output_dim
) {
    const int rows_per_npu = (num_nodes + kNumNpus - 1) / kNumNpus;
    const int start_row = npu_id * rows_per_npu;
    const int end_row = std::min(start_row + rows_per_npu, num_nodes);

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < hidden_dim; ++k) {
                sum += input[i * hidden_dim + k] * weights[k * output_dim + j];
            }
            output[i * output_dim + j] = sum;
        }
    }
}

std::uint64_t estimate_cycles(
    std::uint64_t ops,
    std::uint64_t bytes,
    double ops_per_cycle,
    double mem_bw_bpc,
    std::uint64_t cache_bytes,
    std::uint64_t memory_unit_bytes,
    std::uint64_t miss_penalty_cycles,
    std::uint64_t pipeline_cycles
) {
    const std::uint64_t compute_cycles = (ops == 0)
        ? 0
        : static_cast<std::uint64_t>(std::ceil(static_cast<double>(ops) / std::max(1e-12, ops_per_cycle)));
    const std::uint64_t memory_cycles = (bytes == 0)
        ? 0
        : static_cast<std::uint64_t>(std::ceil(static_cast<double>(bytes) / std::max(1e-12, mem_bw_bpc)));
    const std::uint64_t overflow = (bytes > cache_bytes) ? (bytes - cache_bytes) : 0;
    const std::uint64_t miss_count = ceil_div_u64(overflow, std::max<std::uint64_t>(1, memory_unit_bytes));
    const std::uint64_t miss_cycles = miss_count * miss_penalty_cycles;

    const std::string compose_mode = gcn_runtime::get_env_string("GCN_PIM_COMPUTE_MEMORY_MODE", "sum");
    std::uint64_t core = compute_cycles + memory_cycles;
    if (compose_mode == "max") {
        core = std::max(compute_cycles, memory_cycles);
    } else if (compose_mode == "partial") {
        const double p = std::max(0.0, std::min(1.0, gcn_runtime::get_env_double("GCN_PIM_COMPUTE_MEMORY_PARTIAL", 0.5)));
        const std::uint64_t a = std::max(compute_cycles, memory_cycles);
        const std::uint64_t b = std::min(compute_cycles, memory_cycles);
        core = a + static_cast<std::uint64_t>(std::llround(static_cast<double>(b) * p));
    }
    return std::max<std::uint64_t>(1, core + miss_cycles + pipeline_cycles);
}

IntraDelayProfile load_intra_delay_profile(const std::string& path) {
    IntraDelayProfile p;
    p.per_pe_delay_cycles.assign(kNumNpus, 0);
    std::ifstream fin(path);
    if (!fin.is_open()) {
        return p;
    }
    std::string line;
    while (std::getline(fin, line)) {
        const std::string t = gcn_runtime::trim(line);
        if (t.empty() || t[0] == '#') {
            continue;
        }
        std::istringstream iss(t);
        std::string key;
        iss >> key;
        if (key == "TOTAL") {
            std::uint64_t v = 0;
            if (iss >> v) {
                p.total_delay_cycles = v;
            }
            continue;
        }
        if (key == "PACKETS") {
            std::uint64_t v = 0;
            if (iss >> v) {
                p.packet_count = v;
            }
            continue;
        }
        if (key == "PE") {
            int pe = -1;
            std::uint64_t dst_delay = 0;
            if (iss >> pe >> dst_delay) {
                if (pe >= 0 && pe < kNumNpus) {
                    p.per_pe_delay_cycles[static_cast<std::size_t>(pe)] = dst_delay;
                }
            }
            continue;
        }
    }
    p.loaded = true;
    return p;
}

std::uint64_t mix_hash(std::uint64_t state, std::uint64_t value) {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
    return state;
}

std::uint64_t checksum_dataset_payload(const GCNData& data) {
    std::uint64_t checksum = 0x1234fedcba987654ULL;
    for (std::size_t i = 0; i < data.feature_indices.size(); ++i) {
        checksum = mix_hash(
            checksum,
            static_cast<std::uint64_t>(data.feature_indices[i] + 1) * 1315423911ULL
        );
    }
    for (std::size_t i = 0; i < data.feature_values.size(); ++i) {
        checksum = mix_hash(
            checksum,
            static_cast<std::uint64_t>(std::llround(data.feature_values[i] * 4096.0f)) +
                static_cast<std::uint64_t>(i + 1U) * 2654435761ULL
        );
    }
    for (std::size_t i = 0; i < data.feature_indptr.size(); ++i) {
        checksum = mix_hash(
            checksum,
            static_cast<std::uint64_t>(data.feature_indptr[i] + 1) * 40503ULL
        );
    }
    for (std::size_t i = 0; i < data.graph_indices.size(); ++i) {
        checksum = mix_hash(
            checksum,
            static_cast<std::uint64_t>(data.graph_indices[i] + 1) * 2166136261ULL
        );
    }
    for (std::size_t i = 0; i < data.graph_indptr.size(); ++i) {
        checksum = mix_hash(
            checksum,
            static_cast<std::uint64_t>(data.graph_indptr[i] + 1) * 709607ULL
        );
    }
    for (std::size_t i = 0; i < data.labels.size(); ++i) {
        checksum = mix_hash(
            checksum,
            static_cast<std::uint64_t>(data.labels[i] + 1) * 11400714819323198485ULL
        );
    }
    return checksum;
}

template <typename T>
bool receive_payload(
    InterChiplet::PipeComm& pipe_comm,
    std::uint64_t& time_now,
    int src_x,
    int src_y,
    int dst_x,
    int dst_y,
    T* payload,
    std::size_t count,
    const char* label
) {
    if (payload == nullptr || count == 0) {
        return true;
    }
    const int bytes = static_cast<int>(count * sizeof(T));
    const std::string file_name = InterChiplet::receiveSync(src_x, src_y, dst_x, dst_y);
    pipe_comm.read_data(file_name.c_str(), payload, bytes);
    time_now = InterChiplet::readSync(time_now, src_x, src_y, dst_x, dst_y, bytes, 0);
    std::cout << "[GCN PIM] received " << label << ", bytes=" << bytes << "\n";
    return true;
}

template <typename T>
bool send_payload(
    InterChiplet::PipeComm& pipe_comm,
    std::uint64_t& time_now,
    int src_x,
    int src_y,
    int dst_x,
    int dst_y,
    const T* payload,
    std::size_t count
) {
    if (payload == nullptr || count == 0) {
        return true;
    }
    const int bytes = static_cast<int>(count * sizeof(T));
    const std::string file_name = InterChiplet::sendSync(src_x, src_y, dst_x, dst_y);
    pipe_comm.write_data(file_name.c_str(), const_cast<T*>(payload), bytes);
    time_now = InterChiplet::writeSync(time_now, src_x, src_y, dst_x, dst_y, bytes, 0);
    return true;
}

bool receive_dataset_from_cpu(
    InterChiplet::PipeComm& pipe_comm,
    std::uint64_t& time_now,
    int id_x,
    int id_y,
    GCNData& data,
    gcn_dataset_transfer::DatasetHeader& header
) {
    static_assert(sizeof(int) == 4, "dataset transfer expects 32-bit int");

    if (!receive_payload(pipe_comm, time_now, kCpuX, kCpuY, id_x, id_y, &header, 1, "dataset header")) {
        return false;
    }
    if (header.magic != gcn_dataset_transfer::kDatasetHeaderMagic) {
        std::cerr << "[GCN PIM] invalid dataset header magic: " << header.magic << "\n";
        return false;
    }
    if (header.num_nodes <= 0 || header.feature_nnz < 0 || header.edge_items < 0) {
        std::cerr << "[GCN PIM] invalid dataset header counts\n";
        return false;
    }

    const std::size_t num_nodes = static_cast<std::size_t>(header.num_nodes);
    data.graph_indptr.resize(num_nodes + 1U);
    data.graph_indices.resize(static_cast<std::size_t>(header.edge_items));
    data.feature_indptr.resize(num_nodes + 1U);
    data.feature_indices.resize(static_cast<std::size_t>(header.feature_nnz));
    data.feature_values.resize(static_cast<std::size_t>(header.feature_nnz));
    data.labels.resize(num_nodes);

    return receive_payload(
               pipe_comm, time_now, kCpuX, kCpuY, id_x, id_y,
               data.graph_indptr.data(), data.graph_indptr.size(), "graph_indptr"
           ) &&
           receive_payload(
               pipe_comm, time_now, kCpuX, kCpuY, id_x, id_y,
               data.graph_indices.data(), data.graph_indices.size(), "graph_indices"
           ) &&
           receive_payload(
               pipe_comm, time_now, kCpuX, kCpuY, id_x, id_y,
               data.feature_indptr.data(), data.feature_indptr.size(), "feature_indptr"
           ) &&
           receive_payload(
               pipe_comm, time_now, kCpuX, kCpuY, id_x, id_y,
               data.feature_indices.data(), data.feature_indices.size(), "feature_indices"
           ) &&
           receive_payload(
               pipe_comm, time_now, kCpuX, kCpuY, id_x, id_y,
               data.feature_values.data(), data.feature_values.size(), "feature_values"
           ) &&
           receive_payload(
               pipe_comm, time_now, kCpuX, kCpuY, id_x, id_y,
               data.labels.data(), data.labels.size(), "labels"
           );
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "[GCN PIM] Usage: pim <idX> <idY> [runtime_plan] [intra_delay_profile]\n";
        return EXIT_FAILURE;
    }

    const int id_x = std::atoi(argv[1]);
    const int id_y = std::atoi(argv[2]);
    const std::string runtime_plan_path =
        (argc >= 4) ? std::string(argv[3]) : std::string("../reports/runtime_plan_pim.txt");
    const std::string intra_delay_profile_path =
        (argc >= 5) ? std::string(argv[4]) : std::string("../reports/pim_intra_delay_profile.txt");

    const auto runtime_plan = gcn_runtime::load_plan_file(runtime_plan_path, {}, "GCN PIM");
    std::vector<std::uint64_t> plan_compute_cycles;
    std::uint64_t total_plan_compute_cycles = 0;
    for (const auto& ev : runtime_plan) {
        if (ev.type == gcn_runtime::EventType::kCompute) {
            const std::uint64_t cycles = std::max<std::uint64_t>(1, ev.cycles);
            plan_compute_cycles.push_back(cycles);
            total_plan_compute_cycles += cycles;
        }
    }

    const bool intra_delay_enable = gcn_runtime::get_env_u64("GCN_PIM_INTRA_DELAY_ENABLE", 1) != 0;
    const double intra_delay_scale = std::max(0.0, gcn_runtime::get_env_double("GCN_PIM_INTRA_DELAY_SCALE", 1.0));
    IntraDelayProfile delay_profile;
    if (intra_delay_enable) {
        delay_profile = load_intra_delay_profile(intra_delay_profile_path);
        if (delay_profile.loaded) {
            delay_profile.total_delay_cycles = static_cast<std::uint64_t>(
                std::llround(static_cast<double>(delay_profile.total_delay_cycles) * intra_delay_scale));
        }
    }

    std::cout << "[GCN PIM] chiplet=(" << id_x << "," << id_y << "), runtime_plan=" << runtime_plan_path
              << ", intra_delay_enable=" << (intra_delay_enable ? 1 : 0)
              << ", intra_delay_total=" << delay_profile.total_delay_cycles
              << ", plan_compute_events=" << plan_compute_cycles.size()
              << ", plan_compute_total=" << total_plan_compute_cycles << "\n";

    InterChiplet::PipeComm pipe_comm;
    std::uint64_t time_now = 1;
    const std::string input_mode =
        gcn_runtime::get_env_string("GCN_PIM_INPUT_MODE", "cpu_stream");
    GCNData data;
    gcn_dataset_transfer::DatasetHeader dataset_header;
    if (input_mode == "cpu_stream") {
        std::cout << "[GCN PIM] waiting dataset stream from CPU...\n";
        if (!receive_dataset_from_cpu(pipe_comm, time_now, id_x, id_y, data, dataset_header)) {
            return EXIT_FAILURE;
        }
        const std::uint64_t observed_checksum = checksum_dataset_payload(data);
        std::cout << "[GCN PIM] dataset received from CPU, nodes=" << dataset_header.num_nodes
                  << ", nnz=" << dataset_header.feature_nnz
                  << ", edges=" << dataset_header.edge_items
                  << ", checksum=" << observed_checksum
                  << ", header_checksum=" << dataset_header.checksum << "\n";
        if (observed_checksum != dataset_header.checksum) {
            std::cerr << "[GCN PIM] dataset checksum mismatch after transfer\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "[GCN PIM] waiting START from CPU...\n";
        std::string file_name = InterChiplet::receiveSync(kCpuX, kCpuY, id_x, id_y);
        char start_signal[256] = {};
        pipe_comm.read_data(file_name.c_str(), start_signal, 256);
        time_now = InterChiplet::readSync(time_now, kCpuX, kCpuY, id_x, id_y, 256, 0);
        std::cout << "[GCN PIM] START received: " << start_signal << "\n";
        try {
            parse_graph(resolve_required_input_path("GCN_PIM_GRAPH_PATH", "../data/cora.graph"), data);
            parse_features(
                resolve_required_input_path("GCN_PIM_FEATURE_PATH", "../data/cora.svmlight"),
                data
            );
        } catch (const std::exception& ex) {
            std::cerr << "[GCN PIM] data loading error: " << ex.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    const int num_nodes = static_cast<int>(data.labels.size());
    if (num_nodes <= 0) {
        std::cerr << "[GCN PIM] invalid num_nodes: " << num_nodes << "\n";
        return EXIT_FAILURE;
    }
    int input_dim = 0;
    for (int idx : data.feature_indices) {
        input_dim = std::max(input_dim, idx + 1);
    }
    if (input_dim <= 0) {
        std::cerr << "[GCN PIM] invalid input_dim parsed from feature file\n";
        return EXIT_FAILURE;
    }
    const int hidden_dim = kHiddenDim;
    const int output_dim = kOutputDim;
    const int epochs = std::max(1, static_cast<int>(gcn_runtime::get_env_u64("GCN_EPOCHS", kEpochsDefault)));

    std::vector<float> weight1(static_cast<std::size_t>(input_dim) * hidden_dim);
    std::vector<float> weight2(static_cast<std::size_t>(hidden_dim) * output_dim);
    for (std::size_t i = 0; i < weight1.size(); ++i) {
        weight1[i] = (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 0.1f;
    }
    for (std::size_t i = 0; i < weight2.size(); ++i) {
        weight2[i] = (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 0.1f;
    }

    std::vector<float> layer1_input(static_cast<std::size_t>(num_nodes) * hidden_dim, 0.0f);
    std::vector<float> layer1_output(static_cast<std::size_t>(num_nodes) * hidden_dim, 0.0f);
    std::vector<float> layer2_input(static_cast<std::size_t>(num_nodes) * output_dim, 0.0f);
    std::vector<float> output(static_cast<std::size_t>(num_nodes) * output_dim, 0.0f);

    const std::uint64_t edges = static_cast<std::uint64_t>(data.graph_indices.size());
    const std::uint64_t nnz = static_cast<std::uint64_t>(data.feature_values.size());
    const std::uint64_t graph_bytes = edges * sizeof(int);
    const std::uint64_t feature_bytes = nnz * sizeof(float);
    const std::uint64_t layer1_bytes = static_cast<std::uint64_t>(layer1_output.size() * sizeof(float));
    const std::uint64_t layer2_bytes = static_cast<std::uint64_t>(output.size() * sizeof(float));
    const std::uint64_t weight1_bytes = static_cast<std::uint64_t>(weight1.size() * sizeof(float));
    const std::uint64_t weight2_bytes = static_cast<std::uint64_t>(weight2.size() * sizeof(float));

    const double pim_ops_per_cycle = gcn_runtime::get_env_double("GCN_PIM_OPS_PER_CYCLE", 6400.0);
    const double pim_mem_bw = gcn_runtime::get_env_double("GCN_PIM_MEM_BW_BPC", 64.0);
    const std::uint64_t pim_cache_bytes = gcn_runtime::get_env_u64(
        "GCN_PIM_CACHE_BYTES", static_cast<std::uint64_t>(kPimSramBytesPerNpu) * kNumNpus);
    const std::uint64_t memory_unit_bytes = gcn_runtime::get_env_u64("GCN_PIM_MEMORY_UNIT_BYTES", 131072);
    const std::uint64_t miss_penalty = gcn_runtime::get_env_u64("GCN_PIM_CACHE_MISS_PENALTY", 120);
    const std::uint64_t pipeline_cycles = gcn_runtime::get_env_u64("GCN_PIM_PIPELINE_CYCLES", 16);

    const std::uint64_t sparse_ops = 2ULL * nnz * hidden_dim;
    const std::uint64_t graph1_ops = 2ULL * edges * hidden_dim;
    const std::uint64_t relu_ops = static_cast<std::uint64_t>(num_nodes) * hidden_dim;
    const std::uint64_t dense_ops = 2ULL * static_cast<std::uint64_t>(num_nodes) * hidden_dim * output_dim;
    const std::uint64_t graph2_ops = 2ULL * edges * output_dim;
    const std::uint64_t sparse_cycles = estimate_cycles(
        sparse_ops, feature_bytes + weight1_bytes + layer1_bytes, pim_ops_per_cycle, pim_mem_bw,
        pim_cache_bytes, memory_unit_bytes, miss_penalty, pipeline_cycles);
    const std::uint64_t graph1_cycles = estimate_cycles(
        graph1_ops, layer1_bytes + graph_bytes, pim_ops_per_cycle, pim_mem_bw,
        pim_cache_bytes, memory_unit_bytes, miss_penalty, pipeline_cycles);
    const std::uint64_t relu_cycles = estimate_cycles(
        relu_ops, layer1_bytes, pim_ops_per_cycle, pim_mem_bw,
        pim_cache_bytes, memory_unit_bytes, miss_penalty, pipeline_cycles);
    const std::uint64_t dense_cycles = estimate_cycles(
        dense_ops, layer1_bytes + weight2_bytes + layer2_bytes, pim_ops_per_cycle, pim_mem_bw,
        pim_cache_bytes, memory_unit_bytes, miss_penalty, pipeline_cycles);
    const std::uint64_t graph2_cycles = estimate_cycles(
        graph2_ops, layer2_bytes + graph_bytes, pim_ops_per_cycle, pim_mem_bw,
        pim_cache_bytes, memory_unit_bytes, miss_penalty, pipeline_cycles);
    const std::uint64_t epoch_est_cycles =
        sparse_cycles + graph1_cycles + relu_cycles + dense_cycles + graph2_cycles;

    const std::uint64_t base_extra = (epochs > 0) ? (delay_profile.total_delay_cycles / epochs) : 0;
    const std::uint64_t rem_extra = (epochs > 0) ? (delay_profile.total_delay_cycles % epochs) : 0;

    std::cout << "[GCN PIM] nodes=" << num_nodes
              << ", nnz=" << nnz
              << ", edges=" << edges
              << ", epochs=" << epochs
              << ", est_epoch_cycles=" << epoch_est_cycles << "\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::fill(layer1_input.begin(), layer1_input.end(), 0.0f);
        sparse_matmul(
            data.feature_values.data(),
            data.feature_indices,
            data.feature_indptr,
            weight1.data(),
            layer1_input.data(),
            num_nodes,
            hidden_dim
        );

        std::fill(layer1_output.begin(), layer1_output.end(), 0.0f);
        graph_sum(
            layer1_input.data(),
            data.graph_indices,
            data.graph_indptr,
            layer1_output.data(),
            num_nodes,
            hidden_dim
        );

        relu_activation(layer1_output.data(), layer1_output.size());

        std::fill(layer2_input.begin(), layer2_input.end(), 0.0f);
        std::vector<std::thread> threads;
        threads.reserve(kNumNpus);
        for (int pe = 0; pe < kNumNpus; ++pe) {
            threads.emplace_back(
                dense_matmul_npu,
                pe,
                layer1_output.data(),
                weight2.data(),
                layer2_input.data(),
                num_nodes,
                hidden_dim,
                output_dim
            );
        }
        for (auto& t : threads) {
            t.join();
        }

        std::fill(output.begin(), output.end(), 0.0f);
        graph_sum(
            layer2_input.data(),
            data.graph_indices,
            data.graph_indptr,
            output.data(),
            num_nodes,
            output_dim
        );

        std::uint64_t runtime_plan_cycles = epoch_est_cycles;
        if (!plan_compute_cycles.empty()) {
            const bool plan_matches_epochs =
                plan_compute_cycles.size() == static_cast<std::size_t>(epochs);
            if (plan_matches_epochs) {
                runtime_plan_cycles = plan_compute_cycles[static_cast<std::size_t>(epoch)];
            } else {
                runtime_plan_cycles = std::max<std::uint64_t>(1, total_plan_compute_cycles);
            }
        }
        const std::uint64_t extra = base_extra + ((static_cast<std::uint64_t>(epoch) < rem_extra) ? 1 : 0);
        const std::uint64_t epoch_cycles = std::max(epoch_est_cycles, runtime_plan_cycles) + extra;
        time_now += epoch_cycles;

        std::int64_t size_info[2] = {num_nodes, output_dim};
        if (!send_payload(pipe_comm, time_now, id_x, id_y, kCpuX, kCpuY, size_info, 2) ||
            !send_payload(pipe_comm, time_now, id_x, id_y, kCpuX, kCpuY, output.data(), output.size())) {
            std::cerr << "[GCN PIM] failed to send epoch logits to CPU\n";
            return EXIT_FAILURE;
        }

        if (epoch == 0 || (epoch + 1) % 10 == 0 || epoch + 1 == epochs) {
            std::cout << "[GCN PIM] epoch " << (epoch + 1) << "/" << epochs
                      << " epoch_cycles=" << epoch_cycles
                      << " send_logits_bytes_to_cpu=" << (sizeof(size_info) + output.size() * sizeof(float)) << "\n";
        }
    }

    std::int64_t done_signal[2] = {-1, -1};
    if (!send_payload(pipe_comm, time_now, id_x, id_y, kCpuX, kCpuY, done_signal, 2)) {
        std::cerr << "[GCN PIM] failed to send done signal to CPU\n";
        return EXIT_FAILURE;
    }

    std::cout << "[GCN PIM] shutdown, final_cycle=" << time_now
              << ", injected_intra_delay=" << delay_profile.total_delay_cycles << "\n";
    return EXIT_SUCCESS;
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "apis_c.h"
#include "gcn_runtime.h"

namespace {
constexpr int kDefaultPimX = 0;
constexpr int kDefaultPimY = 1;
constexpr int kDefaultIoX = 1;
constexpr int kDefaultIoY = 0;

struct GCNData {
    std::vector<int> feature_indices;
    std::vector<int> feature_indptr;
    std::vector<float> feature_values;
    std::vector<int> graph_indices;
    std::vector<int> graph_indptr;
    std::vector<int> labels;
};

struct CpuDatasetSummary {
    int num_nodes = 0;
    int input_dim = 0;
    std::uint64_t nnz = 0;
    std::uint64_t edges_with_self = 0;
    std::uint64_t working_set_bytes = 0;
    std::uint64_t checksum = 0;
};

struct CpuDatasetPackage {
    GCNData data;
    CpuDatasetSummary summary;
};

void log_cpu_progress(const std::string& message) {
    std::cout << "[GCN CPU] " << message << std::endl;
}

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

void parse_graph(const std::string& path, GCNData& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open graph file: " + path);
    }

    data.graph_indices.clear();
    data.graph_indptr.clear();
    data.graph_indptr.push_back(0);

    std::string line;
    int node = 0;
    while (std::getline(file, line)) {
        data.graph_indices.push_back(node);
        data.graph_indptr.push_back(data.graph_indptr.back() + 1);
        ++node;

        std::istringstream iss(line);
        int neighbor = 0;
        while (iss >> neighbor) {
            data.graph_indices.push_back(neighbor);
            data.graph_indptr.back()++;
        }
    }
}

void parse_features(const std::string& path, GCNData& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open feature file: " + path);
    }

    data.feature_indices.clear();
    data.feature_indptr.clear();
    data.feature_values.clear();
    data.labels.clear();
    data.feature_indptr.push_back(0);

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int label = 0;
        iss >> label;
        if (iss.fail()) {
            continue;
        }

        data.labels.push_back(label);
        data.feature_indptr.push_back(data.feature_indptr.back());

        std::string token;
        while (iss >> token) {
            const std::size_t colon_pos = token.find(':');
            if (colon_pos == std::string::npos) {
                continue;
            }

            const std::string key = token.substr(0, colon_pos);
            const std::string value = token.substr(colon_pos + 1);
            try {
                data.feature_indices.push_back(std::stoi(key));
                data.feature_values.push_back(std::stof(value));
                data.feature_indptr.back()++;
            } catch (...) {
                continue;
            }
        }
    }
}

std::uint64_t mix_hash(std::uint64_t state, std::uint64_t value) {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
    return state;
}

void trim_dataset_to_active_nodes(GCNData& data, int num_nodes) {
    if (num_nodes <= 0) {
        data.feature_indices.clear();
        data.feature_indptr.assign(1, 0);
        data.feature_values.clear();
        data.graph_indices.clear();
        data.graph_indptr.assign(1, 0);
        data.labels.clear();
        return;
    }

    const std::vector<int> old_feature_indices = data.feature_indices;
    const std::vector<int> old_feature_indptr = data.feature_indptr;
    const std::vector<float> old_feature_values = data.feature_values;
    data.feature_indices.clear();
    data.feature_values.clear();
    data.feature_indptr.clear();
    data.feature_indptr.push_back(0);
    for (int node = 0; node < num_nodes; ++node) {
        const int begin = old_feature_indptr[static_cast<std::size_t>(node)];
        const int end = old_feature_indptr[static_cast<std::size_t>(node) + 1];
        for (int idx = begin; idx < end; ++idx) {
            data.feature_indices.push_back(old_feature_indices[static_cast<std::size_t>(idx)]);
            data.feature_values.push_back(old_feature_values[static_cast<std::size_t>(idx)]);
        }
        data.feature_indptr.push_back(static_cast<int>(data.feature_indices.size()));
    }

    const std::vector<int> old_graph_indices = data.graph_indices;
    const std::vector<int> old_graph_indptr = data.graph_indptr;
    data.graph_indices.clear();
    data.graph_indptr.clear();
    data.graph_indptr.push_back(0);
    for (int node = 0; node < num_nodes; ++node) {
        const int begin = old_graph_indptr[static_cast<std::size_t>(node)];
        const int end = old_graph_indptr[static_cast<std::size_t>(node) + 1];
        for (int idx = begin; idx < end; ++idx) {
            const int neighbor = old_graph_indices[static_cast<std::size_t>(idx)];
            if (neighbor >= 0 && neighbor < num_nodes) {
                data.graph_indices.push_back(neighbor);
            }
        }
        data.graph_indptr.push_back(static_cast<int>(data.graph_indices.size()));
    }

    if (static_cast<int>(data.labels.size()) > num_nodes) {
        data.labels.resize(static_cast<std::size_t>(num_nodes));
    }
}

int infer_input_dim(const GCNData& data) {
    int input_dim = 0;
    for (int feat_idx : data.feature_indices) {
        input_dim = std::max(input_dim, feat_idx + 1);
    }
    return input_dim;
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

CpuDatasetPackage run_real_cpu_workload() {
    const std::string graph_path =
        resolve_required_input_path("GCN_CPU_GRAPH_PATH", "../data/cora.graph");
    const std::string feature_path =
        resolve_required_input_path("GCN_CPU_FEATURE_PATH", "../data/cora.svmlight");
    const int max_nodes = static_cast<int>(
        gcn_runtime::get_env_u64("GCN_CPU_REAL_MAX_NODES", 0));

    log_cpu_progress("real workload: graph_path=" + graph_path);
    log_cpu_progress("real workload: feature_path=" + feature_path);

    GCNData data;
    log_cpu_progress("real workload: parsing graph...");
    parse_graph(graph_path, data);
    log_cpu_progress(
        "real workload: graph parsed, edges_with_self=" +
        std::to_string(data.graph_indices.size()));
    log_cpu_progress("real workload: parsing features...");
    parse_features(feature_path, data);
    log_cpu_progress(
        "real workload: features parsed, labels=" +
        std::to_string(data.labels.size()) +
        ", nnz=" + std::to_string(data.feature_indices.size()));

    const int graph_nodes = std::max<int>(0, static_cast<int>(data.graph_indptr.size()) - 1);
    const int feature_nodes = static_cast<int>(data.labels.size());
    int num_nodes = std::min(graph_nodes, feature_nodes);
    if (max_nodes > 0) {
        num_nodes = std::min(num_nodes, max_nodes);
        log_cpu_progress("real workload: limit active, max_nodes=" + std::to_string(max_nodes));
    }
    if (num_nodes <= 0) {
        throw std::runtime_error("dataset parsing produced zero nodes");
    }

    trim_dataset_to_active_nodes(data, num_nodes);
    const int input_dim = infer_input_dim(data);
    log_cpu_progress("real workload: packaging dataset buffers for PIM...");

    CpuDatasetSummary summary;
    summary.num_nodes = num_nodes;
    summary.input_dim = input_dim;
    summary.nnz = static_cast<std::uint64_t>(data.feature_values.size());
    summary.edges_with_self = static_cast<std::uint64_t>(data.graph_indices.size());
    summary.working_set_bytes =
        data.feature_indices.size() * sizeof(int) +
        data.feature_indptr.size() * sizeof(int) +
        data.feature_values.size() * sizeof(float) +
        data.graph_indices.size() * sizeof(int) +
        data.graph_indptr.size() * sizeof(int) +
        data.labels.size() * sizeof(int);
    summary.checksum = checksum_dataset_payload(data);
    log_cpu_progress(
        "real workload: dataset ready, active_nodes=" +
        std::to_string(summary.num_nodes) +
        ", checksum=" + std::to_string(summary.checksum));
    CpuDatasetPackage package;
    package.data = std::move(data);
    package.summary = summary;
    return package;
}

bool send_payload_to_pim(
    int cpu_x,
    int cpu_y,
    const void* payload,
    std::int64_t bytes,
    const char* label
) {
    if (payload == nullptr || bytes <= 0) {
        return true;
    }
    const auto rc = InterChiplet::sendMessage(
        kDefaultPimX,
        kDefaultPimY,
        cpu_x,
        cpu_y,
        const_cast<void*>(payload),
        bytes
    );
    if (rc < 0) {
        std::cerr << "[GCN CPU] sendMessage failed while sending " << label << "\n";
        return false;
    }
    return true;
}

std::uint64_t checksum_tensor_payload(const std::vector<float>& tensor) {
    std::uint64_t checksum = 0x6a09e667f3bcc909ULL;
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        checksum = mix_hash(
            checksum,
            static_cast<std::uint64_t>(std::llround((tensor[i] + 17.0f) * 65536.0f)) +
                static_cast<std::uint64_t>(i + 1U) * 16777619ULL
        );
    }
    return checksum;
}

std::string to_lower_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );
    return value;
}

enum class CpuPostMode {
    kNone,
    kSoftmax,
    kLogSoftmax,
};

CpuPostMode parse_post_mode(const std::string& text) {
    const std::string mode = to_lower_copy(text);
    if (mode == "none" || mode == "identity" || mode == "passthrough") {
        return CpuPostMode::kNone;
    }
    if (mode == "logsoftmax" || mode == "log_softmax") {
        return CpuPostMode::kLogSoftmax;
    }
    return CpuPostMode::kSoftmax;
}

const char* post_mode_name(CpuPostMode mode) {
    switch (mode) {
        case CpuPostMode::kNone:
            return "none";
        case CpuPostMode::kLogSoftmax:
            return "logsoftmax";
        case CpuPostMode::kSoftmax:
        default:
            return "softmax";
    }
}

void softmax_builtin(
    const std::vector<float>& logits,
    int rows,
    int cols,
    bool log_softmax,
    std::vector<float>* output
) {
    output->assign(logits.size(), 0.0f);
    const int safe_rows = std::max(1, rows);
    const int safe_cols = std::max(1, cols);
    for (int row = 0; row < safe_rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(safe_cols);
        float row_max = logits[base];
        for (int col = 1; col < safe_cols; ++col) {
            row_max = std::max(row_max, logits[base + static_cast<std::size_t>(col)]);
        }
        double sum_exp = 0.0;
        for (int col = 0; col < safe_cols; ++col) {
            sum_exp += std::exp(static_cast<double>(logits[base + static_cast<std::size_t>(col)] - row_max));
        }
        const double log_sum = std::log(std::max(1e-30, sum_exp));
        for (int col = 0; col < safe_cols; ++col) {
            const double shifted =
                static_cast<double>(logits[base + static_cast<std::size_t>(col)] - row_max);
            if (log_softmax) {
                (*output)[base + static_cast<std::size_t>(col)] = static_cast<float>(shifted - log_sum);
            } else {
                (*output)[base + static_cast<std::size_t>(col)] =
                    static_cast<float>(std::exp(shifted) / std::max(1e-30, sum_exp));
            }
        }
    }
}

void apply_cpu_postprocess(
    const std::vector<float>& logits,
    int rows,
    int cols,
    CpuPostMode mode,
    std::vector<float>* output,
    int* out_rows,
    int* out_cols
) {
    if (output == nullptr || out_rows == nullptr || out_cols == nullptr) {
        return;
    }
    const int safe_rows = std::max(1, rows);
    const int safe_cols = std::max(1, cols);
    *out_rows = safe_rows;
    *out_cols = safe_cols;
    if (mode == CpuPostMode::kNone) {
        *output = logits;
        return;
    }
    softmax_builtin(logits, safe_rows, safe_cols, mode == CpuPostMode::kLogSoftmax, output);
}

int run_real_mode(int id_x, int id_y, const std::string& post_mode_text) {
    static_assert(sizeof(int) == 4, "dataset transfer expects 32-bit int");
    const bool verbose = gcn_runtime::get_env_u64("GCN_CPU_REAL_VERBOSE", 1) != 0;
    const std::string env_post_mode = gcn_runtime::get_env_string("GCN_CPU_POST_MODE", "softmax");
    const CpuPostMode post_mode = parse_post_mode(post_mode_text.empty() ? env_post_mode : post_mode_text);
    log_cpu_progress(
        "enter real mode, chiplet=(" + std::to_string(id_x) + "," + std::to_string(id_y) + ")");
    const CpuDatasetPackage dataset = run_real_cpu_workload();
    const CpuDatasetSummary& summary = dataset.summary;
    if (verbose) {
        std::ostringstream oss;
        oss << "CPU post process mode=" << post_mode_name(post_mode);
        log_cpu_progress(oss.str());
    }

    if (verbose) {
        log_cpu_progress(
            "mode=real, chiplet=(" + std::to_string(id_x) + "," + std::to_string(id_y) + ")" +
            ", nodes=" + std::to_string(summary.num_nodes) +
            ", input_dim=" + std::to_string(summary.input_dim) +
            ", nnz=" + std::to_string(summary.nnz) +
            ", edges_with_self=" + std::to_string(summary.edges_with_self) +
            ", working_set_bytes=" + std::to_string(summary.working_set_bytes) +
            ", checksum=" + std::to_string(summary.checksum));
    }

    gcn_dataset_transfer::DatasetHeader dataset_header;
    dataset_header.num_nodes = summary.num_nodes;
    dataset_header.input_dim = summary.input_dim;
    dataset_header.feature_nnz = static_cast<std::int64_t>(summary.nnz);
    dataset_header.edge_items = static_cast<std::int64_t>(summary.edges_with_self);
    dataset_header.working_set_bytes = summary.working_set_bytes;
    dataset_header.checksum = summary.checksum;

    if (!send_payload_to_pim(id_x, id_y, &dataset_header, sizeof(dataset_header), "dataset header") ||
        !send_payload_to_pim(
            id_x,
            id_y,
            dataset.data.graph_indptr.data(),
            static_cast<std::int64_t>(dataset.data.graph_indptr.size() * sizeof(int)),
            "graph_indptr"
        ) ||
        !send_payload_to_pim(
            id_x,
            id_y,
            dataset.data.graph_indices.data(),
            static_cast<std::int64_t>(dataset.data.graph_indices.size() * sizeof(int)),
            "graph_indices"
        ) ||
        !send_payload_to_pim(
            id_x,
            id_y,
            dataset.data.feature_indptr.data(),
            static_cast<std::int64_t>(dataset.data.feature_indptr.size() * sizeof(int)),
            "feature_indptr"
        ) ||
        !send_payload_to_pim(
            id_x,
            id_y,
            dataset.data.feature_indices.data(),
            static_cast<std::int64_t>(dataset.data.feature_indices.size() * sizeof(int)),
            "feature_indices"
        ) ||
        !send_payload_to_pim(
            id_x,
            id_y,
            dataset.data.feature_values.data(),
            static_cast<std::int64_t>(dataset.data.feature_values.size() * sizeof(float)),
            "feature_values"
        ) ||
        !send_payload_to_pim(
            id_x,
            id_y,
            dataset.data.labels.data(),
            static_cast<std::int64_t>(dataset.data.labels.size() * sizeof(int)),
            "labels"
        )) {
        return EXIT_FAILURE;
    }
    log_cpu_progress("dataset sent to PIM, waiting final output stream...");

    std::uint64_t forwarded_checksum = 0;
    int epoch = 0;
    while (true) {
        std::int64_t size_info[2] = {0, 0};
        const auto size_rc = InterChiplet::receiveMessage(
            id_x,
            id_y,
            kDefaultPimX,
            kDefaultPimY,
            static_cast<void*>(size_info),
            static_cast<std::int64_t>(sizeof(size_info))
        );
        if (size_rc < 0) {
            std::cerr << "[GCN CPU] receiveMessage failed while waiting PIM shape header\n";
            return EXIT_FAILURE;
        }

        const int rows = static_cast<int>(size_info[0]);
        const int cols = static_cast<int>(size_info[1]);
        if (rows == -1 && cols == -1) {
            break;
        }
        if (rows <= 0 || cols <= 0) {
            std::cerr << "[GCN CPU] invalid tensor shape from PIM: " << rows << "x" << cols << "\n";
            return EXIT_FAILURE;
        }

        const std::size_t elem_cnt =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        std::vector<float> logits(elem_cnt, 0.0f);
        const auto data_rc = InterChiplet::receiveMessage(
            id_x,
            id_y,
            kDefaultPimX,
            kDefaultPimY,
            static_cast<void*>(logits.data()),
            static_cast<std::int64_t>(elem_cnt * sizeof(float))
        );
        if (data_rc < 0) {
            std::cerr << "[GCN CPU] receiveMessage failed while reading PIM payload\n";
            return EXIT_FAILURE;
        }

        std::vector<float> cpu_output = logits;
        int cpu_rows = rows;
        int cpu_cols = cols;
        std::uint64_t cpu_output_checksum = checksum_tensor_payload(cpu_output);
        apply_cpu_postprocess(logits, rows, cols, post_mode, &cpu_output, &cpu_rows, &cpu_cols);
        cpu_output_checksum = checksum_tensor_payload(cpu_output);

        forwarded_checksum = mix_hash(forwarded_checksum ^ summary.checksum, cpu_output_checksum);

        std::int64_t out_shape[2] = {cpu_rows, cpu_cols};
        const auto send_shape_rc = InterChiplet::sendMessage(
            kDefaultIoX,
            kDefaultIoY,
            id_x,
            id_y,
            static_cast<void*>(out_shape),
            static_cast<std::int64_t>(sizeof(out_shape))
        );
        if (send_shape_rc < 0) {
            std::cerr << "[GCN CPU] sendMessage failed while forwarding output shape to IO\n";
            return EXIT_FAILURE;
        }
        const auto send_data_rc = InterChiplet::sendMessage(
            kDefaultIoX,
            kDefaultIoY,
            id_x,
            id_y,
            static_cast<void*>(cpu_output.data()),
            static_cast<std::int64_t>(cpu_output.size() * sizeof(float))
        );
        if (send_data_rc < 0) {
            std::cerr << "[GCN CPU] sendMessage failed while forwarding output payload to IO\n";
            return EXIT_FAILURE;
        }

        if (verbose) {
            std::ostringstream oss;
            oss << "epoch=" << epoch
                << " pim_shape=" << rows << "x" << cols
                << " cpu_shape=" << cpu_rows << "x" << cpu_cols
                << " payload_bytes=" << (cpu_output.size() * sizeof(float))
                << " post_mode=" << post_mode_name(post_mode)
                << " output_checksum=" << cpu_output_checksum
                << " forwarded_checksum=" << forwarded_checksum;
            log_cpu_progress(oss.str());
        }
        ++epoch;
    }

    std::int64_t done_header[2] = {-1, -1};
    const auto io_done_rc = InterChiplet::sendMessage(
        kDefaultIoX,
        kDefaultIoY,
        id_x,
        id_y,
        static_cast<void*>(done_header),
        static_cast<std::int64_t>(sizeof(done_header))
    );
    if (io_done_rc < 0) {
        std::cerr << "[GCN CPU] sendMessage failed while sending IO DONE\n";
        return EXIT_FAILURE;
    }

    log_cpu_progress(
        "PIM stream completed, epochs=" + std::to_string(epoch) +
        ", checksum=" + std::to_string(summary.checksum) +
        ", forwarded_checksum=" + std::to_string(forwarded_checksum));
    return EXIT_SUCCESS;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "[GCN CPU] Usage: cpu <idX> <idY> [post_mode]\n";
        return EXIT_FAILURE;
    }

    const int id_x = std::atoi(argv[1]);
    const int id_y = std::atoi(argv[2]);
    const std::string post_mode_text = (argc >= 4) ? std::string(argv[3]) : std::string();
    try {
        return run_real_mode(id_x, id_y, post_mode_text);
    } catch (const std::exception& ex) {
        std::cerr << "[GCN CPU] fatal: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}

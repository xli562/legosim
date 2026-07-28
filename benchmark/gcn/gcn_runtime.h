#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gcn_dataset_transfer {

constexpr std::uint64_t kDatasetHeaderMagic = 0x47434E5F44533131ULL;  // "GCN_DS11"

struct DatasetHeader {
    std::uint64_t magic = kDatasetHeaderMagic;
    std::int64_t num_nodes = 0;
    std::int64_t input_dim = 0;
    std::int64_t feature_nnz = 0;
    std::int64_t edge_items = 0;
    std::uint64_t working_set_bytes = 0;
    std::uint64_t checksum = 0;
};

}  // namespace gcn_dataset_transfer

namespace gcn_runtime {

enum class EventType { kCompute, kSend, kRecv };

struct Event {
    EventType type = EventType::kCompute;
    int peer_x = 0;
    int peer_y = 0;
    std::int64_t bytes = 0;
    std::uint64_t cycles = 0;
    std::uint64_t ops = 0;
    std::uint64_t tensor_bytes = 0;
    std::uint64_t working_set_bytes = 0;
    std::string label;
};

inline std::uint64_t get_env_u64(const char* key, std::uint64_t default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    try {
        return static_cast<std::uint64_t>(std::stoull(std::string(raw)));
    } catch (...) {
        return default_value;
    }
}

inline double get_env_double(const char* key, double default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    try {
        return std::stod(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

inline std::string get_env_string(const char* key, const std::string& default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    return std::string(raw);
}

inline std::uint64_t parse_u64_token(const std::string& s, std::uint64_t default_value = 0) {
    try {
        return static_cast<std::uint64_t>(std::stoull(s));
    } catch (...) {
        return default_value;
    }
}

inline std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
        ++b;
    }
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
        --e;
    }
    return s.substr(b, e - b);
}

inline std::vector<Event> load_plan_file(
    const std::string& path,
    const std::vector<Event>& fallback_events,
    const std::string& tag
) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[" << tag << "] runtime plan not found, fallback: " << path << "\n";
        return fallback_events;
    }

    std::vector<Event> events;
    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        ++line_no;
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') {
            continue;
        }

        std::istringstream iss(t);
        std::string op;
        iss >> op;
        if (op == "COMPUTE") {
            std::uint64_t cycles = 0;
            if (!(iss >> cycles)) {
                std::cerr << "[" << tag << "] ignore invalid COMPUTE line " << line_no << "\n";
                continue;
            }
            std::uint64_t ops = 0;
            std::uint64_t tensor_bytes = 0;
            std::uint64_t working_set_bytes = 0;
            std::string label;
            std::string token;
            while (iss >> token) {
                const auto eq_pos = token.find('=');
                if (eq_pos != std::string::npos) {
                    const std::string key = token.substr(0, eq_pos);
                    const std::string val = token.substr(eq_pos + 1);
                    if (key == "ops") {
                        ops = parse_u64_token(val, ops);
                    } else if (key == "bytes") {
                        tensor_bytes = parse_u64_token(val, tensor_bytes);
                    } else if (key == "workset" || key == "workset_bytes") {
                        working_set_bytes = parse_u64_token(val, working_set_bytes);
                    }
                    continue;
                }
                if (label.empty()) {
                    label = token;
                }
            }
            Event ev;
            ev.type = EventType::kCompute;
            ev.cycles = std::max<std::uint64_t>(1, cycles);
            ev.ops = ops;
            ev.tensor_bytes = tensor_bytes;
            ev.working_set_bytes = working_set_bytes;
            ev.label = label;
            events.push_back(ev);
            continue;
        }

        if (op == "SEND" || op == "RECV") {
            int px = 0;
            int py = 0;
            std::int64_t bytes = 0;
            if (!(iss >> px >> py >> bytes)) {
                std::cerr << "[" << tag << "] ignore invalid " << op << " line " << line_no << "\n";
                continue;
            }
            std::string label;
            iss >> label;
            Event ev;
            ev.type = (op == "SEND") ? EventType::kSend : EventType::kRecv;
            ev.peer_x = px;
            ev.peer_y = py;
            ev.bytes = std::max<std::int64_t>(1, bytes);
            ev.label = label;
            events.push_back(ev);
            continue;
        }

        std::cerr << "[" << tag << "] ignore unknown op on line " << line_no << ": " << op << "\n";
    }

    if (events.empty()) {
        std::cerr << "[" << tag << "] runtime plan empty, fallback path: " << path << "\n";
        return fallback_events;
    }
    return events;
}

inline void burn_cycles(std::uint64_t cycles, std::uint64_t scale) {
    const std::uint64_t iters =
        std::max<std::uint64_t>(1, cycles) * std::max<std::uint64_t>(1, scale);
    volatile std::uint64_t acc = 0x9e3779b97f4a7c15ULL;
    for (std::uint64_t i = 0; i < iters; ++i) {
        acc ^= (acc << 7) + (acc >> 3) + (i | 1ULL);
    }
    if (acc == 0x12345678ULL) {
        std::cout << "[runtime] unlikely marker\n";
    }
}

}  // namespace gcn_runtime

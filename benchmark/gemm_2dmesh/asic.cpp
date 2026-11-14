#include <cstdlib>      // std::atoi
#include <iostream>     // std::cerr
#include <string>
#include <vector>
#include "comms_protocol.h"
#include "pipe_comm.h"
#include "sync_protocol.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG   // must precede the include
#define SPDLOG_LEVEL_NAMES {"TRACE", "DEBUG", "INFO ", "WARN ", "ERROR", "CRIT ", "OFF  "}
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
// @brief Logs to stderr, and to log_path if non-empty.
// @note  Call once at the top of main(). 
static void initLogging(const std::string& log_path) {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
    if (!log_path.empty())
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path, true));

    auto logger = std::make_shared<spdlog::logger>("asic", sinks.begin(), sinks.end());
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);
}

const gemm2DMesh::Mesh2DCoord srcNode = gemm2DMesh::CPU_SLOTS[0];

// @brief Receives an input sent from `srcNode` to `localNode` (i.e., the ASIC)
// @param opcode      (Opcode)       Either A_BLOCK or B_BLOCK
// @param asicId      (int)          Id of the ASIC instance
// @param localNode   (Mesh2DCoord)  Coordinate of the ASIC instance
// @param tileRow     (tileRowCol_t) Row block index of A_BLOCK that the ASIC receives
// @param tileCol     (tileRowCol_t) Column block index of B_BLOCK that the ASIC receives
// @param startCycle  (TimeType)     The starting cycle of the receive operation
// @param pipes       (PipeComm&)    Linux file handle used for data passing
// @param blockBuffer (vector<element_t>&, output) Received data
// @return `cycle`    (TimeType)     Cycles elapsed
InterChiplet::TimeType receiveInput(
    gemm2DMesh::Opcode opcode,
    int asicId,
    gemm2DMesh::Mesh2DCoord localNode,
    gemm2DMesh::tileRowCol_t tileRow,
    gemm2DMesh::tileRowCol_t tileCol,
    InterChiplet::TimeType startCycle,
    InterChiplet::PipeComm& pipes,
    std::vector<gemm2DMesh::element_t>& blockBuffer
) {
    using namespace gemm2DMesh;
    using InterChiplet::TimeType;

    if (!((opcode == Opcode::A_BLOCK) || (opcode == Opcode::B_BLOCK))) {
        SPDLOG_ERROR("ASIC {}: Input must be received from A or B, "
                     "got opcode {}", asicId, opcodeName(opcode));
        return static_cast<TimeType>(-1);
    }

    TimeType cycle = startCycle;
    MsgHeader header;
    blockBuffer.resize(getExpectedPayloadByteCnt(opcode) / 
                       sizeof(element_t));
    std::string pipeName = InterChiplet::receiveSync(srcNode.x, srcNode.y,
                                                     localNode.x, localNode.y);
    
    // Header validation
    int receivedHeaderSize = pipes.read_data(pipeName.c_str(), &header,
                                             sizeof(header));
    if (receivedHeaderSize != static_cast<int>(sizeof(header))) {
        SPDLOG_ERROR("ASIC {}: Received truncated header. "
                     "Expected header size {}, got {}",
                     asicId, sizeof(header), receivedHeaderSize);
        return static_cast<TimeType>(-1);
    }
    if (header.opcode != opcode) {
        SPDLOG_ERROR("ASIC {}: Header opcode mismatch. "
                     "Expected {}, got {}", asicId,
                     opcodeName(opcode), opcodeName(header.opcode));
        return static_cast<TimeType>(-1);
    }
    if (header.tileRow != tileRow || header.tileCol != tileCol) {
        SPDLOG_ERROR("ASIC {}: Header tile mismatch. "
                     "Expected ({}, {}), got ({}, {})", asicId,
                    tileRow, tileCol, header.tileRow, header.tileCol);
        return static_cast<TimeType>(-1);
    }
    if (header.payloadByteCnt != getExpectedPayloadByteCnt(opcode)) {
        SPDLOG_ERROR("ASIC {}: Mismatch in "
                     "header's payload byte count information. "
                     "Expected {}, got {}", asicId,
                     getExpectedPayloadByteCnt(opcode),
                     header.payloadByteCnt);
        return static_cast<TimeType>(-1);
    }

    // Payload validation
    const int intPayloadByteCnt = static_cast<int>(header.payloadByteCnt);
    int receivedPayloadByteCnt = pipes.read_data(pipeName.c_str(),
                                                 blockBuffer.data(),
                                                 intPayloadByteCnt);
    if (receivedPayloadByteCnt != intPayloadByteCnt) {
        SPDLOG_ERROR("ASIC {}: Received truncated payload. "
                     "Expected payload size {}, got {}", asicId,
                     intPayloadByteCnt, receivedPayloadByteCnt);
        return static_cast<TimeType>(-1);
    }

    cycle = InterChiplet::readSync(cycle, srcNode.x, srcNode.y,
                                   localNode.x, localNode.y,
                                   static_cast<int>(sizeof(header)) + 
                                                       intPayloadByteCnt,
                                   0);
    if (cycle == static_cast<TimeType>(-1)) {
        SPDLOG_ERROR("ASIC {}: readSync failed for block {}",
                     asicId, opcodeName(opcode));
        return static_cast<TimeType>(-1);
    }

    return cycle;
}

// @brief Sends the C tile from the ASIC to `srcNode`
// @param asicId      (int)          Id of the ASIC instance
// @param localNode   (Mesh2DCoord)  Coordinate of the ASIC instance
// @param tileRow     (tileRowCol_t) Row block index of A_BLOCK that the ASIC receives
// @param tileCol     (tileRowCol_t) Column block index of B_BLOCK that the ASIC receives
// @param startCycle  (TimeType)     The starting cycle of the send operation
// @param pipes       (PipeComm&)    Linux file handle used for data passing
// @param tileBuffer  (vector<element_t>&, output) Results being sent
// @return `cycle`    (TimeType)     Cycles elapsed
InterChiplet::TimeType sendResult(
    int asicId,
    gemm2DMesh::Mesh2DCoord localNode,
    gemm2DMesh::tileRowCol_t tileRow,
    gemm2DMesh::tileRowCol_t tileCol,
    InterChiplet::TimeType startCycle,
    InterChiplet::PipeComm& pipes,
    std::vector<gemm2DMesh::element_t>& tileBuffer
) {
    using namespace gemm2DMesh;
    using InterChiplet::TimeType;

    TimeType cycle = startCycle;
    MsgHeader header = MsgHeader{Opcode::C_TILE, tileRow, tileCol, 0,
                                 getExpectedPayloadByteCnt(Opcode::C_TILE), 0};
    std::string pipeName = InterChiplet::sendSync(localNode.x, localNode.y,
                                                  srcNode.x, srcNode.y);
    
    // Send header
    int sentHeaderSize = pipes.write_data(pipeName.c_str(), &header,
                                          sizeof(header));
    if (sentHeaderSize != static_cast<int>(sizeof(header))) {
        SPDLOG_ERROR("ASIC {}: Dispatched truncated header. "
                     "Expected header size {}, got {}",
                     asicId, sizeof(header), sentHeaderSize);
        return static_cast<TimeType>(-1);
    }

    // Send payload
    const int intPayloadByteCnt = static_cast<int>(header.payloadByteCnt);
    int sentPayloadByteCnt = pipes.write_data(pipeName.c_str(),
                                              tileBuffer.data(),
                                              intPayloadByteCnt);
    if (sentPayloadByteCnt != intPayloadByteCnt) {
        SPDLOG_ERROR("ASIC {}: Dispatched truncated payload. "
                     "Expected payload size {}, got {}", asicId,
                     intPayloadByteCnt, sentPayloadByteCnt);
        return static_cast<TimeType>(-1);
    }

    cycle = InterChiplet::writeSync(cycle, localNode.x, localNode.y,
                                    srcNode.x, srcNode.y,
                                    static_cast<int>(sizeof(header)) + 
                                                        intPayloadByteCnt,
                                    0);
    if (cycle == static_cast<TimeType>(-1)) {
        SPDLOG_ERROR("ASIC {}: writeSync failed for C_TILE", asicId);
        return static_cast<TimeType>(-1);
    }

    return cycle;
}

// @brief Computes the product of two matrices
// @param m           (int)                Number of rows of `matA`
// @param k           (int)                Number of columns of `matA` and number of rows of `matB`
// @param n           (int)                Number of columns of `matB`
// @param matA        (vector<element_t>&) The first matrix operand (shape: m*k)
// @param matB        (vector<element_t>&) The second matrix operand (shape: k*n)
// @param matC        (vector<element_t>&, output) The result
// @return `checksum` (long long)          Sum of all elements in the result matrix
long long matrixMultiply(
    int m, int k, int n,
    const std::vector<gemm2DMesh::element_t>& matA,
    const std::vector<gemm2DMesh::element_t>& matB,
    std::vector<gemm2DMesh::element_t>& matC
) {
    using namespace gemm2DMesh;
    long long checksum = 0;
    // c_{ij} = \sum_{p=1}^K a_{ip} b{pj}
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                element_t partialSum = matA[i*k+p] * matB[p*n+j];
                matC[i*n+j] += partialSum;
                checksum += partialSum;
            }
        }
    }
    return checksum;
}

// @param 1 ASIC id
// @param 2 ASIC latency
int main(int argc, char **argv) {
    initLogging("");
    using namespace gemm2DMesh;
    using InterChiplet::TimeType;

    //========================================================================
    // Argument checks
    //========================================================================
    if (argc != 3) {
        std::cerr << "[ERROR] require arguments <asic id> and <asic latency>\n";
        return 2;
    }
    const int asicId      = std::atoi(argv[1]);
    const int asicLatency = std::atoi(argv[2]);
    // ASIC id must be in [0, ASIC_CNT)
    if (asicId < 0) {
        SPDLOG_ERROR("ASIC id must be non-negative, got {}", asicId);
        return 2;
    }
    if (asicId >= ASIC_CNT) {
        SPDLOG_ERROR("ASIC id (got {}) must be smaller than "
                     "the total number of ASICs (specified as {})",
                     asicId, ASIC_CNT);
        return 2;
    }
    // ASIC latency must be positive
    if (asicLatency <= 0) {
        SPDLOG_ERROR("ASIC latency must be positive, got {}", asicLatency);
        return 2;
    }

    const Mesh2DCoord localNode = ASIC_SLOTS[asicId];
    const tileRowCol_t tileRow = getTileRow(asicId);
    const tileRowCol_t tileCol = getTileCol(asicId);
    
    //========================================================================
    // Receive inputs
    //========================================================================
    TimeType cycle = static_cast<TimeType>(1);
    InterChiplet::PipeComm pipes;
    std::vector<element_t> aBuffer, bBuffer;
    TimeType aCycles = receiveInput(Opcode::A_BLOCK, asicId, localNode,
                                    tileRow, tileCol, cycle, pipes, aBuffer);
    if (aCycles == static_cast<TimeType>(-1)) {
        return 1;
    }
    TimeType bCycles = receiveInput(Opcode::B_BLOCK, asicId, localNode,
                                    tileRow, tileCol, aCycles, pipes, bBuffer);
    if (bCycles == static_cast<TimeType>(-1)) {
        return 1;
    }

    //========================================================================
    // Compute the matrix product
    //========================================================================
    // C_TILE = A_BLOCK B_BLOCK
    // Sizes. A_BLOCK: (M / ROW_BLK_CNT) * K,
    //        B_BLOCK: K * (N / COL_BLK_CNT),
    //        C_TILE : (M / ROW_BLK_CNT) * (N / COL_BLK_CNT)
    std::vector<element_t> cBuffer((M / ROW_BLK_CNT) * (N / COL_BLK_CNT));
    long long checksum = matrixMultiply(M / ROW_BLK_CNT, K, N / COL_BLK_CNT,
                                        aBuffer, bBuffer, cBuffer);

    //========================================================================
    // Send the result
    //========================================================================
    TimeType cCycles = sendResult(asicId, localNode, tileRow, tileCol,
                                  bCycles + asicLatency, pipes, cBuffer);
    if (cCycles == static_cast<TimeType>(-1)) {
        return 1;
    }
    SPDLOG_INFO("ASIC {}: Sent C tile ({}, {}), t_last_input {}, t_done {} "
                "compute latency {}, checksum {}", asicId, tileRow, tileCol,
                bCycles, cCycles, asicLatency, checksum);

    spdlog::shutdown();
    return 0;
}

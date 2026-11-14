#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "apis_c.h"
#include "comms_protocol.h"

const gemm2DMesh::Mesh2DCoord srcNode = gemm2DMesh::CPU_SLOTS[0];

// @brief Cheap pseudorandom number genreator to fill a matrix
// @note Element (i,j) is assigned (i*p1 + j*p2 + seed) % 51
// @param pos    (int)       Position of the element in the matrix
// @param colCnt (int)       The number of columns of the matrix
// @param p1     (int)       A parameter of the RNG
// @param p2     (int)       A parameter of the RNG
// @param seed   (int)       The seed for random number generation
// @return elem  (element_t) The randomized element
gemm2DMesh::element_t matPrng(int pos, int colCnt, int p1, int p2, int seed) {
    int i = pos / colCnt;
    int j = pos % colCnt;
    return (i*p1 + j*p2 + seed) % 51;
}

// @brief Packs the header information and payload into a single message
// @param opcode         (Opcode)           Whether the payload is A_BLOCK, B_BLOCK, or C_TILE
// @param tileRow        (tileRowCol_t)     Row block index of A's row block
// @param tileCol        (tileRowCol_t)     Column block index of B's column block
// @param payloadByteCnt (payloadByteCnt_t) Size of the payload (unit: byte)
// @param payload        (element_t*)       The payload data
std::vector<uint8_t> packMessage(
    gemm2DMesh::Opcode opcode,
    gemm2DMesh::tileRowCol_t tileRow,
    gemm2DMesh::tileRowCol_t tileCol,
    gemm2DMesh::payloadByteCnt_t payloadByteCnt,
    const gemm2DMesh::element_t* payload
) {
    using namespace gemm2DMesh;

    if (payloadByteCnt > 0 && payload == nullptr) {
        throw std::invalid_argument("Empty payload");
    }

    MsgHeader header = {opcode, tileRow, tileCol, 0, payloadByteCnt, 0};
    std::vector<uint8_t> message(sizeof(MsgHeader) + payloadByteCnt);

    std::memcpy(message.data(), &header, sizeof(MsgHeader));
    if (payloadByteCnt > 0) {
        std::memcpy(message.data() + sizeof(MsgHeader),
                    payload, payloadByteCnt);
    }

    return message;
}

// @brief Unpacks a message, outputs header and payload
// @param message (vector<uint8_t>&) The message to be unpacked
// @param header  (MsgHeader, output) Header of the message
// @param payload (vector<element_t>, output) Payload of the message
int unpackMessage(
    std::vector<uint8_t>& message,
    gemm2DMesh::MsgHeader& header,
    std::vector<gemm2DMesh::element_t>& payload
) {
    using namespace gemm2DMesh;

    std::memcpy(&header, message.data(), sizeof(MsgHeader));
    if (message.size() != sizeof(MsgHeader) + header.payloadByteCnt) {
        std::cerr << "Received payload does not have the "
                  << "size specified by its header. "
                  << "Expected payload size: "
                  << header.payloadByteCnt
                  << ", got "
                  << message.size() - sizeof(MsgHeader)
                  << std::endl;
        return 1;
    }
    payload.resize(header.payloadByteCnt / sizeof(element_t));
    std::memcpy(payload.data(),
                message.data() + sizeof(MsgHeader),
                header.payloadByteCnt);
    return 0;
}

// @brief Sends A_BLOCK or B_BLOCK to their processing sites
// @param asicId (int)               Id of the ASIC instance
// @param opcode (Opcode)            Either A_BLOCK or B_BLOCK
// @param matA   (vector<element_t>) The entire matrix A
// @param matB   (vector<element_t>) The entire matrix B
void sendBlock(
    int asicId,
    gemm2DMesh::Opcode opcode,
    std::vector<gemm2DMesh::element_t>& matA,
    std::vector<gemm2DMesh::element_t>& matB
) { 
    using namespace gemm2DMesh;
    using InterChiplet::sendMessage;
    
    // Construct the row block (for matA) or column block (for matB) being sent
    std::vector<element_t> mat;
    if (opcode == Opcode::A_BLOCK) {
        mat.assign(matA.begin() + getRowBlockStart(asicId)*K,
                matA.begin() + getRowBlockEnd(asicId)*K);
    } else if (opcode == Opcode::B_BLOCK) {
        mat.resize(K * (N / COL_BLK_CNT));
        for (int a = 0; a < K * (N / COL_BLK_CNT); ++a) {
            // mat_{i,j} = matB_{i,colStart+j}
            int i = a / (N/COL_BLK_CNT);
            int j = a % (N/COL_BLK_CNT);
            mat[a] = matB[i*N + getColBlockStart(asicId) + j];
        }
    }

    std::vector<uint8_t> messageBuffer = packMessage(
        opcode,
        getTileRow(asicId),
        getTileCol(asicId),
        getExpectedPayloadByteCnt(opcode),
        mat.data());
    
    sendMessage(ASIC_SLOTS[asicId].x, ASIC_SLOTS[asicId].y,
                srcNode.x, srcNode.y, messageBuffer.data(),
                getExpectedPayloadByteCnt(opcode) + sizeof(MsgHeader));
}

// @brief Receives the tiles and re-assembles the full matrix C
// @param asicId (int) Id of the ASIC instance
// @param matC   (vector<element_t>&, output) The full matrix C
int receiveTile(
    int asicId,
    std::vector<gemm2DMesh::element_t>& matC
) {
    using namespace gemm2DMesh;
    using InterChiplet::receiveMessage;
    using std::cerr;
    using std::endl;

    std::vector<uint8_t> messageBuffer;
    messageBuffer.resize(sizeof(MsgHeader)
                         + getExpectedPayloadByteCnt(Opcode::C_TILE));
    receiveMessage(
        srcNode.x, srcNode.y,
        ASIC_SLOTS[asicId].x, ASIC_SLOTS[asicId].y,
        messageBuffer.data(),
        getExpectedPayloadByteCnt(Opcode::C_TILE) + sizeof(MsgHeader)
    );
    
    MsgHeader header;
    std::vector<element_t> cTile((M / ROW_BLK_CNT) * (N / COL_BLK_CNT));
    int retcode = unpackMessage(messageBuffer, header, cTile);
    if (retcode != 0) {
        cerr << "Cannot unpack message from ASIC " << asicId << endl;
        return 1;
    }

    // Data integrity check
    if (header.opcode != Opcode::C_TILE) {
        cerr << "Message from ASIC " << asicId
             << " has opcode " << opcodeName(header.opcode)
             << ", " << opcodeName(Opcode::C_TILE) << " expected" << endl;
        return 1;
    }
    if ((header.tileRow != getTileRow(asicId))
        || (header.tileCol != getTileCol(asicId))) {
        cerr << "Message from ASIC " << asicId
             << " claims tile (row, column) (" 
             << header.tileRow << ", " << header.tileCol << "), expected ("
             << getTileRow(asicId) << ", " << getTileCol(asicId) << ")"
             << endl;
        return 1;
    }
    if (header.payloadByteCnt != getExpectedPayloadByteCnt(Opcode::C_TILE)) {
        cerr << "Message from ASIC " << asicId
             << " claims to have payload size " << header.payloadByteCnt 
             << ", " << getExpectedPayloadByteCnt(Opcode::C_TILE)
             << " expected" << endl;
        return 1;
    }

    // Place the tile in matrix C
    int tileIterator = 0;
    for (int i = getRowBlockStart(asicId);
            i < getRowBlockEnd(asicId); ++i) {
        for (int j = getColBlockStart(asicId);
                j < getColBlockEnd(asicId); ++j) {
            matC[i*N+j] = cTile[tileIterator];
            ++ tileIterator;
        }
    }
    return 0;
}

int main() {
    using namespace gemm2DMesh;
    using std::vector;
    using std::cerr;
    using std::endl;
    Opcode A_BLOCK = Opcode::A_BLOCK;
    Opcode B_BLOCK = Opcode::B_BLOCK;

    vector<element_t> matA(M * K);
    vector<element_t> matB(K * N);
    vector<element_t> matC(M * N);

    for (int i = 0; i < M*K; ++i) matA[i] = matPrng(i, K, RNG_P1, RNG_P2, SEED_A);
    for (int i = 0; i < K*N; ++i) matB[i] = matPrng(i, N, RNG_P1, RNG_P2, SEED_B);

    //========================================================================
    // Send A and B blocks
    //========================================================================
    for (int asicId = 0; asicId < ASIC_CNT; ++asicId) {
        for (Opcode opcode : {A_BLOCK, B_BLOCK}) {
            sendBlock(asicId, opcode, matA, matB);
        }
    }

    //========================================================================
    // Receive C tiles
    //========================================================================
    for (int asicId = 0; asicId < ASIC_CNT; ++asicId) {
        int retcode = receiveTile(asicId, matC);
        if (retcode != 0) return 1;
    }

    std::ofstream out("matC.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(matC.data()),
              matC.size() * sizeof(element_t));
    cerr << "GEMM_DUMP matC.bin bytes=" 
         << matC.size() * sizeof(element_t) << endl;

    return 0;
}
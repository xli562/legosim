#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

namespace gemm2DMesh {
    // Matrices: AB = C
    // Sizes. A: M*K, B: K*N, C: M*N
    // Everything is row-major
    static constexpr int M = 150;   // Must be divisible by ROW_BLK_CNT
    static constexpr int K = 60;
    static constexpr int N = 200;   // Must be divisible by COL_BLK_CNT
    using element_t = int64_t;
    // Parameters for randomly initializing a matrix
    static constexpr int RNG_P1 = 7;
    static constexpr int RNG_P2 = 13;
    static constexpr int SEED_A = 321;
    static constexpr int SEED_B = 37;

    // Divide A into several row blocks and B into several column blocks. Each
    // ASIC computes a tile, which is the product of 1 row block and 1 column
    // block
    // For example,
    // +--------+--------+--------+
    // | Tile 0 | Tile 1 | Tile 2 |
    // +--------+--------+--------+
    // | Tile 3 | Tile 4 | Tile 5 |
    // +--------+--------+--------+
    // Tile 4 has tileRow=1 and tileCol = 1.
    static constexpr int ROW_BLK_CNT = 2;
    static constexpr int COL_BLK_CNT = 4;
    using tileRowCol_t = uint32_t;  // Changing this affects header structure
    constexpr tileRowCol_t getTileRow(int asicId) {return asicId / COL_BLK_CNT;}
    constexpr tileRowCol_t getTileCol(int asicId) {return asicId % COL_BLK_CNT;}
    constexpr int getRowBlockStart(int asicId) {
        return getTileRow(asicId) * (M/ROW_BLK_CNT);}
    constexpr int getRowBlockEnd  (int asicId) {
        return (getTileRow(asicId) + 1) * (M/ROW_BLK_CNT);}
    constexpr int getColBlockStart(int asicId) {
        return getTileCol(asicId) * (N/COL_BLK_CNT);};
    constexpr int getColBlockEnd  (int asicId) {
        return (getTileCol(asicId) + 1) * (N/COL_BLK_CNT);}

    // 2D mesh topology
    static constexpr int MESH_X_DIM = 3;
    static constexpr int MESH_Y_DIM = 3;
    struct Mesh2DCoord {int x; int y;};
    static constexpr Mesh2DCoord ASIC_SLOTS[] = {
        {0,0}, {0,1}, {0,2},
        {1,0},        {1,2},
        {2,0}, {2,1}, {2,2}
    };
    static constexpr Mesh2DCoord CPU_SLOTS[] = {
        {1,1}
    };

    static constexpr int ASIC_CNT = sizeof(ASIC_SLOTS) / sizeof(ASIC_SLOTS[0]);
    static_assert(ROW_BLK_CNT * COL_BLK_CNT == ASIC_CNT,
                  "Number of tiles does not equal number of ASICs");
    static_assert(M % ROW_BLK_CNT == 0,
                  "Cannot split matrix A's rows into ROW_BLK_CNT blocks");
    static_assert(N % COL_BLK_CNT == 0,
                  "Cannot split matrix B's columns into COL_BLK_CNT blocks");

    // Message format
    static constexpr int FLIT_SIZE = 16;    // Unit: byte
    enum class Opcode: uint32_t {A_BLOCK, B_BLOCK, C_TILE};
    inline std::string opcodeName(Opcode opcode) {return
        (opcode == Opcode::A_BLOCK) ? "A_BLOCK"
      : (opcode == Opcode::B_BLOCK) ? "B_BLOCK"
      : "C_TILE";
    }
    using payloadByteCnt_t = uint64_t;  // Changing this affects header structure
    struct MsgHeader {
        Opcode           opcode;
        tileRowCol_t     tileRow;
        tileRowCol_t     tileCol;
        uint32_t         align8BytesGap;    // Keep payloadByteCnt 8-byte aligned
        payloadByteCnt_t payloadByteCnt;
        uint64_t         padding;           // Pads the header to snugly fit flits
    };
    static_assert(sizeof(MsgHeader) % FLIT_SIZE == 0,
                  "Size of message header is not divisible by flit size");
    constexpr payloadByteCnt_t getExpectedPayloadByteCnt(Opcode opcode) {return
        (opcode == Opcode::A_BLOCK) ? (M / ROW_BLK_CNT) * K * sizeof(element_t)
      : (opcode == Opcode::B_BLOCK) ? K * (N / COL_BLK_CNT) * sizeof(element_t)
      : (M / ROW_BLK_CNT) * (N / COL_BLK_CNT) * sizeof(element_t);
    }
}

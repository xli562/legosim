#pragma once
#include <cstdint>
#include <cstddef>

namespace gemmPnoc {
    // Matrices: AB = C
    // Sizes. A: M*K, B: K*N, C: M*N
    static constexpr int M = 150;   // Must be divisible by ROW_BLK_CNT
    static constexpr int K = 60;
    static constexpr int N = 200;   // Must be divisible by COL_BLK_CNT
    using element_t = int64_t;

    // Divide A into several row blocks and B into several column blocks. Each
    // ASIC computes a tile, which is the product of 1 row block and 1 column
    // block
    static constexpr int ROW_BLK_CNT = 2;
    static constexpr int COL_BLK_CNT = 4;
    constexpr int getTileRow(int asicId) {return asicId / COL_BLK_CNT;}
    constexpr int getTileCol(int asicId) {return asicId % COL_BLK_CNT;}

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
    struct MsgHeader {
        Opcode opcode;
        uint32_t tileRow;
        uint32_t tileCol;
        uint32_t align8BytesGap;    // Keep payloadByteCnt 8-byte aligned
        uint64_t payloadByteCnt;
        uint64_t padding;           // Pads the header to snugly fit flits
    };
    static_assert(sizeof(MsgHeader) % FLIT_SIZE == 0,
                  "Size of message header is not divisible by flit size");
    constexpr int getPayloadByteCnt(Opcode opcode) {return
        (opcode == Opcode::A_BLOCK) ? (M / ROW_BLK_CNT) * K * sizeof(element_t)
      : (opcode == Opcode::B_BLOCK) ? K * (N / COL_BLK_CNT) * sizeof(element_t)
      : (M / ROW_BLK_CNT) * (N / COL_BLK_CNT) * sizeof(element_t);
    }
}

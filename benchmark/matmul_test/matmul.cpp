#include <cstdlib>
#include <fstream>
#include <iostream>

#include "apis_c.h"

#define Row 100
#define Col 100

int idX, idY;

int main(int argc, char **argv) {
    if (argc != 3 && argc != 9) {
        std::cerr << "usage: matmul_checked_c CPU_X CPU_Y [GPU1_X GPU1_Y GPU2_X GPU2_Y GPU3_X GPU3_Y]\n";
        return 2;
    }
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);
    int gpuX[3] = {0, 1, 1};
    int gpuY[3] = {1, 0, 1};
    if (argc == 9) {
        for (int gpu = 0; gpu < 3; ++gpu) {
            gpuX[gpu] = atoi(argv[3 + gpu * 2]);
            gpuY[gpu] = atoi(argv[4 + gpu * 2]);
        }
    }
    for (int gpu = 0; gpu < 3; ++gpu) {
        if (gpuX[gpu] < 0 || gpuY[gpu] < 0 ||
            (gpuX[gpu] == idX && gpuY[gpu] == idY)) {
            std::cerr << "Invalid GPU position\n";
            return 2;
        }
        for (int earlier = 0; earlier < gpu; ++earlier) {
            if (gpuX[gpu] == gpuX[earlier] && gpuY[gpu] == gpuY[earlier]) {
                std::cerr << "Duplicate GPU position\n";
                return 2;
            }
        }
    }

    int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *B = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *C1 = (int64_t *)malloc(sizeof(int64_t) * Col);
    int64_t *C2 = (int64_t *)malloc(sizeof(int64_t) * Col);
    int64_t *C3 = (int64_t *)malloc(sizeof(int64_t) * Col);

    for (int i = 0; i < Row * Col; i++) {
        A[i] = rand() % 51;
        B[i] = rand() % 51;
    }

    for (int gpu = 0; gpu < 3; ++gpu)
        InterChiplet::sendMessage(gpuX[gpu], gpuY[gpu], idX, idY, A, 10000 * sizeof(int64_t));

    for (int gpu = 0; gpu < 3; ++gpu)
        InterChiplet::sendMessage(gpuX[gpu], gpuY[gpu], idX, idY, B, 10000 * sizeof(int64_t));

    int64_t *results[3] = {C1, C2, C3};
    for (int gpu = 0; gpu < 3; ++gpu)
        InterChiplet::receiveMessage(idX, idY, gpuX[gpu], gpuY[gpu], results[gpu],
                                     100 * sizeof(int64_t));

    for (int i = 0; i < 100; i++) {
        C1[i] += C2[i];
        C1[i] += C3[i];
    }

#ifdef VALIDATE_MATMUL
    // The three GPUs receive identical matrices. Each returns the top-left
    // 10x10 tile of A*B, so the reduced result should be three times that tile.
    int64_t checksum = 0;
    for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < 10; ++col) {
            int64_t expected = 0;
            for (int k = 0; k < Col; ++k) {
                expected += A[row * Col + k] * B[k * Col + col];
            }
            expected *= 3;
            int index = row * 10 + col;
            if (C1[index] != expected) {
                std::cerr << "MATMUL_CHECK FAIL index=" << index
                          << " expected=" << expected << " actual=" << C1[index] << '\n';
                return 1;
            }
            checksum += C1[index];
        }
    }
    std::cerr << "MATMUL_CHECK PASS checksum=" << checksum << '\n';
#endif
    return 0;
}

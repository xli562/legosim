#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <string>

#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
 * Communication example: three GPU chiplets receive identical 100x100 matrices.
 * Each GPU computes the same top-left 10x10 tile of A*B. The CPU sums the
 * three returned tiles; this is redundant work, not a partitioned product.
 */

#define Row 100
#define Col 100

/**
 * Core function for matrix multiplication. Each thread runs this function once,
 * computing different positions in the result matrix based on thread ID.
 */

__global__ void matrix_mul_gpu(int64_t* M, int64_t* N, int64_t* P, int width) {
    int sumNum = threadIdx.x + threadIdx.y * 10;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int64_t sum = 0;
    for (int k = 0; k < width; k++) {
        int64_t a = M[j * width + k];
        int64_t b = N[k * width + i];
        sum += a * b;
    }
    P[sumNum] = sum;
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        std::cerr << "usage: matmul_cu GPU_X GPU_Y [CPU_X CPU_Y]\n";
        return 1;
    }
    // Read the chiplet number represented by this process
    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    int cpuX = argc == 5 ? atoi(argv[3]) : 0;
    int cpuY = argc == 5 ? atoi(argv[4]) : 0;
    int64_t *d_dataA, *d_dataB, *d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(int64_t) * Row * Col);
    cudaMalloc((void**)&d_dataB, sizeof(int64_t) * Row * Col);
    cudaMalloc((void**)&d_dataC, sizeof(int64_t) * Col);

    receiveMessage(idX, idY, cpuX, cpuY, d_dataA, sizeof(int64_t) * Row * Col);
    receiveMessage(idX, idY, cpuX, cpuY, d_dataB, sizeof(int64_t) * Row * Col);

    // calculate
    dim3 threadPerBlock(10, 10);
    dim3 blockNumber(1);
    matrix_mul_gpu<<<blockNumber, threadPerBlock>>>(d_dataA, d_dataB, d_dataC, Col);

    sendMessage(cpuX, cpuY, idX, idY, d_dataC, 100 * sizeof(int64_t));
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>

const int N = 256;

const int TILE_WIDTH = 32;

using calculateType = float;

__global__ void matrixMulKernel(calculateType* A, calculateType* B, calculateType* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        calculateType value = 0;
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

//Matrix multiplication on GPU
__global__ void matrixMulTileKernel(calculateType *A, calculateType *B, calculateType *C) {
    __shared__ calculateType Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ calculateType Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    calculateType Cval = 0.0;

    // loop over tiles
    for (int64_t m = 0; m < N / TILE_WIDTH; m++) {
        Ads[ty][tx] = A[Row * N + (m * TILE_WIDTH + tx)];
        Bds[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        __syncthreads();
        for (int64_t k = 0; k < TILE_WIDTH; k++) {
            // loop within tile
            Cval += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    C[Row * N + Col] = Cval; /* write back to global memory */
}

//Matrix multiplication on CPU
void matrixMulCPU(calculateType *A, calculateType *B, calculateType *C) {
    calculateType val = 0;
    for (int64_t row = 0; row < N; ++row)
        for (int64_t col = 0; col < N; ++col) {
            val = 0;
            for (int64_t k = 0; k < N; ++k)
                val += A[row * N + k] * B[k * N + col];
            C[row * N + col] = val;
        }
}

int checkResults(calculateType *A, calculateType *B, calculateType *RC) {
    calculateType *r_cpu = (calculateType *) malloc(N * N * sizeof(calculateType));
    matrixMulCPU(A, B, r_cpu);

    //struct timeval start, end;
    //gettimeofday(&start, NULL);

    for (int64_t row = 0; row < N; ++row)
        for (int64_t col = 0; col < N; ++col) {
            if (RC[row * N + col] != r_cpu[row * N + col]) {
                printf("Wrong calculation at [%d][%d], expected GPU: %d, CPU: %d\n", row, col, RC[row * N + col], r_cpu[row * N + col]);
                //free(r_cpu);
                //return 0;
            }
        }
    //gettimeofday(&end, NULL);
    //float elapsed_ms = (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec) / 1000.0;
    //printf("On CPU it took %f milliseconds\n", elapsed_ms);

    free(r_cpu);
    return 1;
}

int main() {
    printf("Started ...\n");
    calculateType *a_cpu, *b_cpu, *a_gpu, *b_gpu, *c_cpu, *c_gpu;
    int64_t size = N * N * sizeof(calculateType);
    a_cpu = (calculateType *) malloc(size);
    b_cpu = (calculateType *) malloc(size);
    c_cpu = (calculateType *) malloc(size);
    auto ret = cudaMalloc(&a_gpu, size);
    if(ret == cudaSuccess){
        std::cout << "works";
    }
    ret = cudaMalloc(&b_gpu, size);
    if(ret == cudaSuccess){
        std::cout << "works";
    }
    ret = cudaMalloc(&c_gpu, size);
    if(ret == cudaSuccess){
        std::cout << "works";
    }
    // Fill the matrices with some random data
    time_t t;
    srand((unsigned) time(&t));
    for (int64_t i = 0; i < N * N; ++i) {
        a_cpu[i] = rand() % 512;
        //a_cpu[i] = i;
        //b_cpu[i] = i;
        b_cpu[i] = rand() % 512;
        c_cpu[i] = 0;
    }
    cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    dim3 bSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gSize(N / TILE_WIDTH, N / TILE_WIDTH);
    matrixMulKernel <<< gSize, bSize >>>(a_gpu, b_gpu, c_gpu);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsedTimeMS = 0; //time in milliseconds
    cudaEventElapsedTime(&elapsedTimeMS, start, end);

    std::cout << "**GPU-" << N << "^2 :           " << elapsedTimeMS * 1000.f << " (milliseconds)\n";

    cudaMemcpy(c_cpu, c_gpu, size, cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Run time error: %s\n", cudaGetErrorString(error));
    }
    if (checkResults(a_cpu, b_cpu, c_cpu))
        printf("Results are correct!\n");
    free(a_cpu);
    free(b_cpu);
    free(c_cpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    printf("Done!\n");
    return 0;
}
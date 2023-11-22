#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>

const int N = 256;

const int TILE_WIDTH = 32;

__global__ void matrixMulKernel(int* A, int* B, int* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

//Matrix multiplication on GPU
__global__ void matrixMulTileKernel(int *A, int *B, int *C) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cval = 0.0;

    // loop over tiles
    for (int m = 0; m < N / TILE_WIDTH; m++) {
        Ads[ty][tx] = A[Row * N + (m * TILE_WIDTH + tx)];
        Bds[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            // loop within tile
            Cval += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    C[Row * N + Col] = Cval; /* write back to global memory */
}

//Matrix multiplication on CPU
void matrixMulCPU(int *A, int *B, int *C) {
    int val = 0;
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            val = 0;
            for (int k = 0; k < N; ++k)
                val += A[row * N + k] * B[k * N + col];
            C[row * N + col] = val;
        }
}

int checkResults(int *A, int *B, int *RC) {
    int *r_cpu = (int *) malloc(N * N * sizeof(int));
    matrixMulCPU(A, B, r_cpu);

    //struct timeval start, end;
    //gettimeofday(&start, NULL);

    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            if (RC[row * N + col] != r_cpu[row * N + col]) {
                printf("Wrong calculation at [%d][%d]\n", row, col);
                free(r_cpu);
                return 0;
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
    int *a_cpu, *b_cpu, *a_gpu, *b_gpu, *c_cpu, *c_gpu;
    int size = N * N * sizeof(int);
    a_cpu = (int *) malloc(size);
    b_cpu = (int *) malloc(size);
    c_cpu = (int *) malloc(size);
    cudaMalloc(&a_gpu, size);
    cudaMalloc(&b_gpu, size);
    cudaMalloc(&c_gpu, size);
    // Fill the matrices with some random data
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < N * N; ++i) {
        a_cpu[i] = rand() % N;
        b_cpu[i] = rand() % N;
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
    matrixMulTileKernel <<< gSize, bSize >>>(a_gpu, b_gpu, c_gpu);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsedTimeMS = 0; //time in milliseconds
    cudaEventElapsedTime(&elapsedTimeMS, start, end);

    std::cout << "**GPU-" << N << "^2 :           " << elapsedTimeMS * 1000.f << " (microseconds)\n";

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
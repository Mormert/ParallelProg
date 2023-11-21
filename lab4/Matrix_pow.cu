// Matrix power2, Farhang Nemati 2021

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int N = 1600;

__global__ void gpuPow(long *d, long *r) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    r[idx] = d[idx] * d[idx];
}

/* void cpuPow(long *d, long *r) {
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
            r[row * N + col] = d[row * N + col] * d[row * N + col];
} */

int main() {
    const int size = N * N * sizeof(long);
    long *data, *result;
    data = (long *) malloc(size);
    result = (long *) malloc(size);

    //Fill data matrix with random (<100) numbers
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < N * N; ++i) {
        result[i] = 0;
        data[i] = rand() % 100;
    }

    long *gData, *gResult;

    cudaMalloc(&gData, size);
    cudaMalloc(&gResult, size);

    cudaMemcpy(gData, data, size, cudaMemcpyHostToDevice);

#define BLOCK_SIZE 1024
    int numBlocks = (N * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpuPow<<<numBlocks, BLOCK_SIZE>>>(gData, gResult);

    {
        // I got errors when using wrong block sizes, this helped find why:
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess)
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    }

    cudaMemcpy(result, gResult, size, cudaMemcpyDeviceToHost);

    // cpuPow(data, result); Don't run cpuPow, only gpuPow!

    //Check if the results are correct
    int error = 0;
    for (int row = 0; row < N && !error; ++row)
        for (int col = 0; col < N && !error; ++col)
            if (result[row * N + col] != data[row * N + col] * data[row * N + col]) {
                printf("Calculation at result[%d][%d] is wrong \n", row, col);
                error = 1;
                break;
            }
    if (!error)
        printf("Correct!\n");
    free(data);
    free(result);

    return 0;
}

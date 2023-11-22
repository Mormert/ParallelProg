// Matrix power2, Farhang Nemati 2021

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <cassert>

#include "scope_profile.h"

__global__ void gpuPow(long *d, long *r, const int N) {
    // Method assuming square matrix:
    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // r[idx] = d[idx] * d[idx];

    // row-col
    /*{
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        int element = row * N + col;

        r[element] = d[element] * d[element];
    }*/

    // col-row
    {
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        int element = col * N + row;

        r[element] = d[element] * d[element];
    }
}

void cpuPow(long *d, long *r, const int N) {
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
            r[row * N + col] = d[row * N + col] * d[row * N + col];
}

void testMatMul(const int N, bool testGpu, int blockSize = 1024) {
    const int size = N * N * sizeof(long);
    long *data, *result;
    data = (long *) malloc(size);
    result = (long *) malloc(size);

    // Fill data matrix with random (<100) numbers
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < N * N; ++i) {
        result[i] = 0;
        data[i] = rand() % 100;
    }

    if (testGpu) {
        std::cout << "BlockSize " << blockSize << ":\n";
        long *gData, *gResult;

        cudaMalloc(&gData, size);
        cudaMalloc(&gResult, size);

        cudaMemcpy(gData, data, size, cudaMemcpyHostToDevice);

        {
            SCOPED_PROFILE_LOG(std::string{"-*CPU/GPU-TOTAL-" + std::to_string(N) + "^2"});
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventRecord(start);

            int numBlocks = (N * N + blockSize - 1) / blockSize;
            gpuPow<<<numBlocks, blockSize>>>(gData, gResult, N);

            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float elapsedTimeMS = 0; //time in milliseconds
            cudaEventElapsedTime(&elapsedTimeMS, start, end);

            std::cout << "**GPU-" << N << "^2 :           " << elapsedTimeMS * 1000.f << " (microseconds)\n";
        }


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

        cudaFree(gData);
        cudaFree(gResult);
    } else {
        SCOPED_PROFILE_LOG(std::string{"--Only CPU-" + std::to_string(N) + "^2      "});
        cpuPow(data, result, N);
    }

    //Check if the results are correct
    int error = 0;
    for (int row = 0; row < N && !error; ++row)
        for (int col = 0; col < N && !error; ++col)
            if (result[row * N + col] != data[row * N + col] * data[row * N + col]) {
                //printf("Calculation at result[%d][%d] is wrong \n", row, col);
                error = 1;
                break;
            }
    if (error) {
        printf("Not Correct!\n");
        assert(false);
    }
    free(data);
    free(result);
}

int main() {

    for (int i = 1; i < 11; i++) {
        for (int j = 4; j < 11; j += 2) {
            int blockSize = pow(2, j);
            testMatMul(pow(2, i), true, blockSize);
        }
        std::cout << '\n';
        testMatMul(pow(2, i), false);
        std::cout << std::endl;
    }

    return 0;
}

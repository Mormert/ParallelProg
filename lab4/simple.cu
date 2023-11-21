// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
//Modified by Farhang Nemati 2021
// Assigns every element in an array with its index.

#include <stdio.h>

const int N = 16;
const int blocksize = 16;

__global__
void simple(int *c) {
    c[threadIdx.x] = threadIdx.x;
}

int main() {
    int *c = new int[N];
    int *cd;
    const int size = N * sizeof(int);

    cudaMalloc((void **) &cd, size);
    dim3 dimBlock(blocksize, 1);
    dim3 dimGrid(1, 1);
    simple<<<dimGrid, dimBlock>>>(cd);
    cudaThreadSynchronize();
    cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);
    cudaFree(cd);

    for (int i = 0; i < N; i++)
        printf("%d ", c[i]);
    printf("\n");
    delete[] c;
    printf("done\n");
    return EXIT_SUCCESS;
}

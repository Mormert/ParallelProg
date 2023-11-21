#include <stdio.h>
#include <stdlib.h>
#include <ctime>

const int  N = 32;

const int TILE_WIDTH = 32;
//Matrix multiplication on GPU
__global__ void matrixMulTileKernel(int* A, int* B, int* C) 
{

}

//Matrix multiplication on CPU
void matrixMulCPU( int* A, int* B, int* C )
{
	int val = 0;
	for( int row = 0; row < N; ++row )
		for( int col = 0; col < N; ++col )
		{
			val = 0;
			for ( int k = 0; k < N; ++k )
			val += A[row * N + k] * B[k * N + col];
			C[row * N + col] = val;
		}
}

int checkResults(int* A, int* B, int* RC)
{
	int* r_cpu = (int*)malloc(N * N * sizeof (int));
	matrixMulCPU( A, B, r_cpu);

	//struct timeval start, end;	
	//gettimeofday(&start, NULL);	

	for( int row = 0; row < N; ++row )
		for( int col = 0; col < N; ++col )
		{
			if(RC[row*N + col] != r_cpu[row*N + col]){
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

int main()
{
    printf("Started ...\n");
	int *a_cpu, *b_cpu, *a_gpu, *b_gpu, *c_cpu, *c_gpu; 
	int size = N * N * sizeof (int); 
	a_cpu = (int*)malloc(size);
	b_cpu = (int*)malloc(size);
	c_cpu = (int*)malloc(size);
	cudaMalloc(&a_gpu, size);
	cudaMalloc(&b_gpu, size);
	cudaMalloc(&c_gpu, size);
	// Fill the matrices with some random data
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < N * N; ++i) {
		a_cpu[i] = rand()%N;
		b_cpu[i] = rand()%N;
		c_cpu[i] = 0;
	}
	cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice);	
	dim3 bSize(32, 32, 1);
	dim3 gSize (N / 32, N / 32);
	matrixMulTileKernel <<< gSize, bSize >>> ( a_gpu, b_gpu, c_gpu );
	cudaMemcpy(c_cpu, c_gpu, size, cudaMemcpyDeviceToHost);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		printf("Run time error: %s\n", cudaGetErrorString(error));
	}
	if(checkResults(a_cpu, b_cpu, c_cpu))
		printf("Results are correct!\n");
	free(a_cpu);
	free(b_cpu);
	free(c_cpu);
	cudaFree(a_gpu); 
	cudaFree(b_gpu);
	cudaFree(c_gpu ); 

	printf("Done!\n");
	return 0;
}
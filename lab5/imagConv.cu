//By Farhang Nemati 2022

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "scope_profile.h"

#include "helper.h"

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define TWidth 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//Sequential filtering
void sequential(unsigned char *inputImageData, int *maskData, unsigned char *outputImageData,
                const int imageWidth, const int imageHeight, const int channels, const int maskWidth, int divideBy) {
    int maskRadius = maskWidth / 2;

    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            for (int k = 0; k < channels; k++) {
                float accum = 0;
                for (int y = -maskRadius; y < maskRadius; y++) {
                    for (int x = -maskRadius; x < maskRadius; x++) {
                        unsigned int xOffset = j + x;
                        unsigned int yOffset = i + y;
                        if (xOffset >= 0 && xOffset < imageWidth && yOffset >= 0 && yOffset < imageHeight) {
                            unsigned char imagePixel = inputImageData[(yOffset * imageWidth + xOffset) * channels + k];
                            int maskValue = maskData[(y + maskRadius) * maskWidth + x + maskRadius];
                            accum += imagePixel * maskValue;
                        }
                    }
                }
                // pixels are in the range of 0 to 1
                outputImageData[(i * imageWidth + j) * channels + k] = accum / divideBy;
            }
        }
    }
}

__device__ void
assignPixelValue(int imageIdx, int imageHeight, int imageWidth, int channels, unsigned char *inputImageData,
                 unsigned char MShared[][TWidth + 2 * Mask_radius], int x, int y) {
    if (imageIdx > 0 && imageIdx < imageHeight * imageWidth * channels)
        MShared[y][x] = inputImageData[imageIdx];
    else
        MShared[y][x] = 0;
}

__device__ void
edgePixelLoader(int row, int col, int imageWidth, int imageHeight, int channels, int channelIdx, unsigned char *inputImageData,
                unsigned char MShared[][TWidth + 2 * Mask_radius], int xSh, int ySh,
                int shiftX, int shiftY) {
   // auto imageIdx = ((row + shiftY) * imageWidth + col + shiftX) * channels;
    auto imageIdx = ((row + shiftY) * imageWidth + col + shiftX) * channels + channelIdx;

    assignPixelValue(imageIdx, imageHeight, imageWidth, channels, inputImageData, MShared, xSh + shiftX, ySh + shiftY);
}

__global__ void convolution(unsigned char *inputImageData, int *maskData, unsigned char *outputImageData,
                            const int imageWidth, const int imageHeight, const int channels, const int maskWidth,
                            int divideBy) {

    __shared__ unsigned char MShared[TWidth + 2 * Mask_radius][TWidth + 2 * Mask_radius];

    auto ySh = threadIdx.y + Mask_radius;
    auto xSh = threadIdx.x + Mask_radius;

    auto col = blockIdx.x * TWidth + threadIdx.x;
    auto row = blockIdx.y * TWidth + threadIdx.y;

    for (int channelIdx = 0; channelIdx < channels; channelIdx++) {
        // the threads in the green and red regions
        if (threadIdx.x == 0 || threadIdx.x == TWidth - 1 || threadIdx.y == 0 || threadIdx.y == TWidth - 1) {

            if (threadIdx.x == 0 && threadIdx.y == 0) {
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx,inputImageData, MShared, xSh, ySh, -1, -1);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx,inputImageData, MShared, xSh, ySh, -1, -2);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx,inputImageData, MShared, xSh, ySh, -2, -1);
            }
            if (threadIdx.x == 0) {
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx,inputImageData, MShared, xSh, ySh, 0, 0);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, -1, 0);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, -2, 0);
            }
            if (threadIdx.y == 0) {
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 0, 0);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx,inputImageData, MShared, xSh, ySh, 0, -1);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 0, -2);
            }

            edgePixelLoader(row, col, imageWidth, imageHeight, channels, channelIdx, inputImageData, MShared, xSh, ySh, -2, -2);
            edgePixelLoader(row, col, imageWidth, imageHeight, channels, channelIdx,inputImageData, MShared, xSh, ySh, 2, -2);
            edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, -2, 2);
            edgePixelLoader(row, col, imageWidth, imageHeight, channels, channelIdx,inputImageData, MShared, xSh, ySh, 2, 2);

            if (threadIdx.x == TWidth - 1 && threadIdx.y == TWidth - 1) {
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 1, 1);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 1, 2);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx,inputImageData, MShared, xSh, ySh, 2, 1);
            }
            if (threadIdx.x == TWidth - 1) {
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 0, 0);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 1, 0);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 2, 0);
            }
            if (threadIdx.y == TWidth - 1) {
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 0, 0);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx,inputImageData, MShared, xSh, ySh, 0, 1);
                edgePixelLoader(row, col, imageWidth, imageHeight, channels,channelIdx, inputImageData, MShared, xSh, ySh, 0, 2);
            }

        } else { //the thread in the blue region
            edgePixelLoader(row, col, imageWidth, imageHeight, channels, channelIdx, inputImageData, MShared, xSh, ySh, 0, 0);
        }

        // Wait until all the elements are read by the threads of the block into the shared memory
        __syncthreads();

        // The calculation done by each thread for case MWidth = 5 -> MRadius = 2
        // Replace 2s by MRadius for general case
        auto accum = 0;
        for (int x = -2; x <= 2; ++x) {
            for (int y = -2; y <= 2; ++y) {
                accum += MShared[ySh + x][xSh + y] * maskData[(x + 2) * maskWidth + y + 2];
            }
        }

        auto imageIdx = (row * imageWidth + col) * channels + channelIdx;
        outputImageData[imageIdx] = accum / divideBy;

        // Wait until all the the threads are done with the calculations
        __syncthreads();
    }
}


int maskRows;
int maskColumns;
int imageChannels;
int imageWidth;
int imageHeight;
unsigned char *hostInputImageData;
unsigned char *hostOutputImageData;
int *hostMaskData;
unsigned char *deviceInputImageData;
unsigned char *deviceOutputImageData;
int *deviceMaskData;

//To be divided by 256
int mask1[5][5] = {
        {1, 4,  6,  4,  1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4,  6,  4,  1}
};

//To be divided by 25
int mask2[5][5] = {
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1}
};

int mask3[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
};

int mask4[3][3] = {
        {0,  -1, 0},
        {-1, 5,  -1},
        {0,  -1, 0}
};

int main(int argc, char *argv[]) {
    maskRows = 5;
    maskColumns = 5;
    hostMaskData = (int *) malloc(maskRows * maskColumns * sizeof(int));


    if (argc == 2) {
        hostInputImageData = readppm(argv[1], (int *) &imageWidth, (int *) &imageHeight);
    } else if (argc > 2) {
        hostInputImageData = readppm(argv[1], (int *) &imageWidth, (int *) &imageHeight);

        FILE *fp = fopen(argv[2], "rb");
        if (fp == NULL) {
            printf("Mask file could not be found\n");
            return -1;
        }
        fread(hostMaskData, sizeof(int), maskRows * maskColumns, fp);
    } else {
        hostInputImageData = readppm((char *) "joanne512.ppm", (int *) &imageWidth, (int *) &imageHeight);

        for (int i = 0; i < maskRows; i++)
            for (int j = 0; j < maskColumns; j++)
                hostMaskData[i * maskColumns + j] = mask1[i][j];
    }

    imageChannels = 3;
    hostOutputImageData = (unsigned char *) malloc(imageWidth * imageHeight * sizeof(unsigned char) * imageChannels);

    {
        SCOPED_PROFILE_LOG("SEQUENTIAL")

        //Mask1: Gaussian filter
        sequential(hostInputImageData, hostMaskData, hostOutputImageData, imageWidth, imageHeight, imageChannels, 5,
                   256);

        //Mask2
        // sequential(hostInputImageData, hostMaskData, hostOutputImageData, imageWidth, imageHeight, imageChannels, 5, 25);

        //Mask3
        // sequential(hostInputImageData, hostMaskData, hostOutputImageData, imageWidth, imageHeight, imageChannels, 3, 1);

    }

    writeppm("outputImageCpu.ppm", imageWidth, imageHeight, hostOutputImageData);


    {
        SCOPED_PROFILE_LOG("ON GPU")

        const int imageSize = imageWidth * imageHeight * sizeof(unsigned char) * imageChannels;
        const int maskSize = maskRows * maskColumns * sizeof(int);

        printf("Doing GPU memory allocation\n");

        cudaError_t err;

        err = cudaMalloc(&deviceInputImageData, imageSize);
        if (err != cudaSuccess) {
            printf("Failed to allocate device memory for input image data (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMalloc(&deviceMaskData, maskSize);
        if (err != cudaSuccess) {
            printf("Failed to allocate device memory for mask data (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMalloc(&deviceOutputImageData, imageSize);
        if (err != cudaSuccess) {
            printf("Failed to allocate device memory for output image data (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printf("Copying data to the GPU\n");

        err = cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Failed to copy input image data from host to device (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(deviceMaskData, hostMaskData, maskSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Failed to copy mask data from host to device (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        printf("Doing the computation on the GPU\n");

        dim3 dBlock(TWidth, TWidth);
        dim3 dGrid(ceil((float) imageWidth / TWidth), ceil((float) imageHeight / TWidth));

        convolution<<<dGrid, dBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
                                       imageWidth, imageHeight, imageChannels, 5, 256);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

        printf("Copying data from the GPU\n");

        err = cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Failed to copy output image data from device to host (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    }


    writeppm("outputImageGpu.ppm", imageWidth, imageHeight, hostOutputImageData);
    return 0;
}
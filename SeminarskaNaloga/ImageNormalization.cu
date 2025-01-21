#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

/*
https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
*/

// Compile with: srun --partition=gpu nvcc -o prog ImageNormalization.cu
// Run with: srun --partition=gpu --ntasks=1 --gpus=1 ./prog kolesar_input.jpg kolesar_output.jpg


#define BSX 16
#define BSY 16
#define COLOR_CHANNELS 1
#define GRAYLEVELS 256

int* get_histogram(int width, int height, unsigned char *d_imageIn);
int* get_cfd(int* hist);
int get_min(int *array, int size);
unsigned char* get_equalized_image(unsigned char* h_imageIn, int* cfd, int cfd_min, int height, int width, int imageSize);

__global__ void histo_kernel(unsigned char* image, int width, int height, int *histogram) {
    __shared__ int localHistogram[GRAYLEVELS];
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    localHistogram[threadId] = 0;

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixelValue = image[y * width + x];
        atomicAdd(&localHistogram[pixelValue], 1);
    }
    __syncthreads();

    if (threadId < GRAYLEVELS) {
        atomicAdd(&histogram[threadId], localHistogram[threadId]);
    }
}

__device__ inline unsigned char Scale(unsigned int cdf, unsigned int cdfmin, unsigned int imageSize){
    float scale;
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    scale = roundf(scale * (float)(GRAYLEVELS-1));
    return (unsigned char)scale;
}
__global__ void cfd_kernel(int *cfd, int *histogram) {
    extern __shared__ int temp[2 * GRAYLEVELS];
    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2 * thid] = histogram[2 * thid];
    temp[2 * thid + 1] = histogram[2 * thid + 1];

    // Build sum in place up the tree
    for (int d = GRAYLEVELS >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element for exclusive scan
    if (thid == 0) temp[GRAYLEVELS - 1] = 0;

    // Traverse down tree & build scan
    for (int d = 1; d < GRAYLEVELS; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    // Write results to device memory
    __syncthreads();
    cfd[2 * thid] = temp[2 * thid];
    cfd[2 * thid + 1] = temp[2 * thid + 1];
}

__global__ void equalize_kernel(unsigned char* d_imageIn, unsigned char* d_imageOut, int width, int height, int* cfd, int cfd_min) {
    extern __shared__ int local_cfd[];

    int threadId = threadIdx.x + blockIdx.x * blockDim.x + (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x;

    if (threadIdx.x < GRAYLEVELS) {
        local_cfd[threadIdx.x] = cfd[threadIdx.x];
    }
    __syncthreads();

    unsigned long imageSize = width * height;

    if (threadId < imageSize) {
        unsigned char pixelValue = d_imageIn[threadId];
        d_imageOut[threadId] = Scale(local_cfd[pixelValue], cfd_min, imageSize);
    }
}

int main(int argc, char* argv[]){
    if (argc < 3)
    {
        printf("USAGE: prog input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    char szImage_out_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);
    
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(argv[1], &width, &height, &cpp, COLOR_CHANNELS);
    if (h_imageIn == NULL) {
        printf("Error loading image\n");
        return -1;
    }

    size_t imageSize = width * height * COLOR_CHANNELS * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char*)malloc(imageSize);

    unsigned char *d_imageIn, *d_imageOut;
    cudaMalloc((void**)&d_imageIn, imageSize);
    cudaMalloc((void**)&d_imageOut, imageSize);

    cudaMemcpy(d_imageIn, h_imageIn, imageSize, cudaMemcpyHostToDevice);

    int* hist_out = get_histogram(width, height, d_imageIn);
    int* cfd = get_cfd(hist_out);
    int min_cfd = get_min(cfd, 256);
    h_imageOut = get_equalized_image(h_imageIn, cfd, min_cfd, height, width, imageSize);

    for (size_t i = 0; i < GRAYLEVELS; i++)
    {
        printf("%3d: cdf: %6d, \t hist: %5d\n", i, cfd[i], hist_out[i]);
    }
    

    stbi_write_png(argv[2], width, height, COLOR_CHANNELS, h_imageOut, width * COLOR_CHANNELS);
}

int* get_histogram(int width, int height, unsigned char *d_imageIn){
    dim3 blockSize(BSX, BSY);
    dim3 gridSize((width + BSX - 1) / BSX, (height + BSY - 1) / BSY);

    int* hist = (int*)malloc(GRAYLEVELS * sizeof(int));
    memset(hist, 0, GRAYLEVELS);

    int *d_hist;
    cudaMalloc((void**)&d_hist, GRAYLEVELS * sizeof(int));
    cudaMemcpy(d_hist, hist, GRAYLEVELS * sizeof(int), cudaMemcpyHostToDevice);
    
    histo_kernel<<<gridSize, blockSize>>>(d_imageIn, width, height, d_hist);

    cudaMemcpy(hist, d_hist, GRAYLEVELS * sizeof(int), cudaMemcpyDeviceToHost);

    return hist;
}

int* get_cfd(int* hist) {
    int* cfd = (int*) calloc(GRAYLEVELS, sizeof(int));

    int* d_cfd;
    cudaMalloc((void**) &d_cfd, GRAYLEVELS * sizeof(int));
    cudaMemcpy(d_cfd, cfd, GRAYLEVELS * sizeof(int), cudaMemcpyHostToDevice);

    int* d_hist;
    cudaMalloc((void**) &d_hist, GRAYLEVELS * sizeof(int));
    cudaMemcpy(d_hist, hist, GRAYLEVELS * sizeof(int), cudaMemcpyHostToDevice);
    
    cfd_kernel<<<1, GRAYLEVELS>>>(d_cfd, d_hist);

    cudaMemcpy(cfd, d_cfd, GRAYLEVELS * sizeof(int), cudaMemcpyDeviceToHost);

    return cfd;
}

int get_min(int *array, int size) {
    int min_val = INT_MAX;

    for (int i = 0; i < size; i++) {
        if (array[i] < min_val && array[i] > 0) {
            min_val = array[i];
        }
    }

    return min_val;
}

unsigned char* get_equalized_image(unsigned char* h_imageIn, int* cfd, int cfd_min, int height, int width, int imageSize){
    dim3 blockSize(BSX, BSY);
    dim3 gridSize((width + BSX - 1) / BSX, (height + BSY - 1) / BSY);

    // Allocate memory for device and host outputs
    unsigned char* h_imageOut = (unsigned char*)malloc(imageSize);
    unsigned char* d_imageIn;
    unsigned char* d_imageOut;

    cudaMalloc((void**)&d_imageIn, imageSize);
    cudaMalloc((void**)&d_imageOut, imageSize);

    // Copy input image to device
    cudaMemcpy(d_imageIn, h_imageIn, imageSize, cudaMemcpyHostToDevice);

    // Launch kernel
    equalize_kernel<<<gridSize, blockSize, GRAYLEVELS * sizeof(int)>>>(d_imageIn, d_imageOut, width, height, cfd, cfd_min);

    // Copy output back to host
    cudaMemcpy(h_imageOut, d_imageOut, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_imageIn);
    cudaFree(d_imageOut);

    return h_imageOut;
}
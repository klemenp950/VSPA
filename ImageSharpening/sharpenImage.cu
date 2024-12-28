#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

/*
https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 4
#define BSX 16
#define BSY 16

__device__ inline unsigned char get_pixel(unsigned char* image, int width, int height, int x, int y, int c, int cpp){
    if (x < 0 || x >= width)
        return 0;
    if (y < 0 || y >= height)
        return 0;
    return image[(y * width + x) * cpp + c];
}

__global__ void enhance_image(unsigned char* h_imageIn, unsigned char* h_imageOut, int width, int height, int cpp){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height){
        for(int i = 0; i < cpp; i++){
            unsigned char center = get_pixel(h_imageIn, width, height, x, y, i, cpp);
            unsigned char gor = get_pixel(h_imageIn, width, height, x, y + 1, i, cpp);
            unsigned char dol = get_pixel(h_imageIn, width, height, x, y - 1, i, cpp);
            unsigned char levo = get_pixel(h_imageIn, width, height, x - 1, y, i, cpp);
            unsigned char desno = get_pixel(h_imageIn, width, height, x + 1, y, i, cpp);

            int novPixel = 5 * center - gor - dol - levo - desno;
            novPixel = max(min(novPixel, 255), 0);

            h_imageOut[(y * width + x) * cpp + i] = (unsigned char)novPixel;
        }
    }
}

int main(int argc, char *argv[]){

    if (argc < 3)
    {
        printf("USAGE: prog input_image\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    char szImage_out_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
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

    dim3 blockSize(BSX, BSY);
    dim3 gridSize((width + BSX - 1) / BSX, (height + BSY - 1) / BSY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    enhance_image<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, COLOR_CHANNELS);
    cudaEventRecord(stop);

    cudaMemcpy(h_imageOut, d_imageOut, imageSize, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

    stbi_write_png(argv[2], width, height, COLOR_CHANNELS, h_imageOut, width * COLOR_CHANNELS);

    stbi_image_free(h_imageIn);
    free(h_imageOut);
    cudaFree(d_imageIn);
    cudaFree(d_imageOut);

    return 0;
}

/*
REPORT

----------------CPU---------------
| IMG_7783_10.png (151 × 202)    |
| 3.945 miliseconds              |
|                                |
| IMG_7783_25.png (378 × 504)    |
| 25.056 miliseconds             |
|                                |
| IMG_7783_50.png (756 × 1008)   |
| 99.563 miliseconds             |
|                                |
| IMG_7783_100.png (1512 × 2016) |
| 399.369 miliseconds            |
|                                |
| IMG_7783_100.png (3024 × 4032) |
| 1474.249 miliseconds           |
----------------------------------

-------------------GPU------------------
| IMG_7783_10.png (151 × 202)          |
| 0.797 milliseconds                   |
| Speedup: 3.945 / 0.797 = 4.949       |
|                                      |
| IMG_7783_25.png (378 × 504)          |
| 0.964 milliseconds                   |
| Speedup: 25.056 / 0.964 = 25.991     |
|                                      |    
| IMG_7783_50.png (756 × 1008)         |
| 0.884 milliseconds                   |
| Speedup: 99.563 / 0.884 = 112.627    |
|                                      |
| IMG_7783_100.png (1512 × 2016)       |
| 0.838 milliseconds                   |
| Speedup: 399.369 / 0.838 = 476.573   |
|                                      |
| IMG_7783_100.png (3024 × 4032)       |
| 1.120 milliseconds                   |
| Speedup: 1474.249 / 1.120 = 1316.293 |
----------------------------------------
*/
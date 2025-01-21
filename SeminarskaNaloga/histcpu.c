#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


#define COLOR_CHANNELS 1

#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1
#define BLOCK_SIZE 16


/* timespec is a struct defined in ctime as
   * struct timespec {
   *   time_t   tv_sec;        // seconds 
   *   long     tv_nsec;       // nanoseconds
   * };
   */
struct timespec timeStart, timeEnd;



unsigned int findMin(unsigned int* cdf){
    
    unsigned int min = 0;
    // grem skozi CDF dokler ne najdem prvi nenicelni element ali pridem do konca
    // namig: na GPU uporabi redukcijo v bloku niti velikosti 256
    for (int i = 0; min == 0 && i < GRAYLEVELS; i++) {
		min = cdf[i];
    }
    
    return min;
}


unsigned char Scale(unsigned int cdf, unsigned int cdfmin, unsigned int imageSize){
    
    float scale;
    
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    
    scale = roundf(scale * (float)(GRAYLEVELS-1));
    
    return (unsigned char)scale;
}


void Equalize(unsigned char * image_in, unsigned char * image_out, int width, int height, unsigned int* cdf){
     
    unsigned int imageSize = width * height;

    //Equalize: namig: blok niti naj si CDF naloži v skupni pomnilnik
    // findmin implementiraj z redukcijo
    
    unsigned int cdfmin = findMin(cdf);
    
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            image_out[(i*width + j)] = Scale(cdf[image_in[i*width + j]], cdfmin, imageSize);
        }
    }
}




/*
NAMIG: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda 
*/
void CalculateCDF(unsigned int* histogram, unsigned int* cdf){
    
    // clear cdf:
    for (int x=0; x<GRAYLEVELS; x++) {
        cdf[x] = 0;
    }
    
    // calculate cdf from histogram
    cdf[0] = histogram[0];
    for (int x=1; x<GRAYLEVELS; x++) {
        cdf[x] = cdf[x-1] + histogram[x];
    }
}



void CalculateHistogram(unsigned char* image, int width, int height, unsigned int* histogram){
    
    //Clear histogram:
    for (int x=0; x<GRAYLEVELS; x++) {
        histogram[x] = 0;
    }
    
    //Calculate histogram. namig: Cuda by Example, poglavje 9, str. 179
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            histogram[image[y*width + x]]++;
        }
    }
}




int main(){

    // Read image from file
    int width, height, cpp;
    // read only DESIRED_NCHANNELS channels from the input image:
    unsigned char *imageIn = stbi_load("kolesar_input.jpg", &width, &height, &cpp, DESIRED_NCHANNELS);
    if(imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }
    printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);

    // Allocate memory for raw output image data, histogram, and CDF 
	unsigned char *imageOut = (unsigned char *)malloc(height * width * sizeof(unsigned int));
    unsigned int *histogram_CPU = (unsigned int *)malloc(GRAYLEVELS * sizeof(unsigned int));
    unsigned int *new_histogram = (unsigned int *)malloc(GRAYLEVELS * sizeof(unsigned int));
    unsigned int *CDF_CPU = (unsigned int *)malloc(GRAYLEVELS * sizeof(unsigned int));

    /*********************************************************************************
    Izenačenje histograma na CPU:
    **********************************************************************************/
    clock_gettime(CLOCK_REALTIME, &timeStart);

    // 1. Izračun histograma:
    CalculateHistogram(imageIn, width, height, histogram_CPU);

    // 2. izračun CDF: 
    CalculateCDF(histogram_CPU, CDF_CPU);
    
    // 3. Izenači histogram:
    Equalize(imageIn, imageOut, width, height, CDF_CPU);

    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double cpu_elapsedTime = ((timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9) * 1000;    // in miliseconds 


    

    /***************************************************************
    Izpis rezultatov in slik
    ****************************************************************/


    // Izračun histograma nove slike:
    CalculateHistogram(imageOut, width, height, new_histogram);

    printf("Beam       Hist_CPU      CDF_CPU      newHist\n");
    for (size_t i = 0; i < GRAYLEVELS; i++)
    {
        printf("%3d      %6d        %6d         %6d \n", i, histogram_CPU[i], CDF_CPU[i], new_histogram[i]);
    }

    printf("\n\nSlika velikosti %d x %d\n", width, height);
    printf("CPU time: %.4f ms\n", cpu_elapsedTime);


    stbi_write_jpg("slikaCPU.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);

    

    free(imageIn);
    free(imageOut);
    free(histogram_CPU);
    free(CDF_CPU);

    return 0;
}
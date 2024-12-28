#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define COLOR_CHANNELS 4
#define BSX 16
#define BSY 16
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


unsigned char get_pixel(unsigned char* image, int width, int height, int x, int y, int c, int cpp){
    if (x < 0 || x >= width)
        return 0;
    if (y < 0 || y >= height)
        return 0;
    return image[(y * width + x) * cpp + c];
}

void enhance_image(unsigned char* h_imageIn, unsigned char* h_imageOut, int width, int height, int cpp){
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            for(int i = 0; i < cpp; i++){
                unsigned char center = get_pixel(h_imageIn, width, height, x, y, i, cpp);
                unsigned char gor = get_pixel(h_imageIn, width, height, x, y + 1, i, cpp);
                unsigned char dol = get_pixel(h_imageIn, width, height, x, y - 1, i, cpp);
                unsigned char levo = get_pixel(h_imageIn, width, height, x - 1, y, i, cpp);
                unsigned char desno = get_pixel(h_imageIn, width, height, x + 1, y, i, cpp);

                int novPixel = 5 * center - gor - dol - levo - desno;
                novPixel = MAX(MIN(novPixel, 255), 0);

                h_imageOut[(y * width + x) * cpp + i] = (unsigned char)novPixel;
            }
        }
    }
}

int main(int argc, char *argv[]){

    if (argc < 3)
    {
        printf("USAGE: prog input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    clock_t start, stop;
    
    char szImage_in_name[255];
    char szImage_out_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    if (h_imageIn == NULL) {
        printf("Error loading image\n");
        return -1;
    }

    size_t imageSize = width * height * COLOR_CHANNELS * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char*)malloc(imageSize);

    start = clock();
    enhance_image(h_imageIn, h_imageOut, width, height, COLOR_CHANNELS);
    stop = clock();

    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds\n", elapsed);


    stbi_write_png(szImage_out_name, width, height, COLOR_CHANNELS, h_imageOut, width * COLOR_CHANNELS);

    stbi_image_free(h_imageIn);
    free(h_imageOut);

    return 0;
}

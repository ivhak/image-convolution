#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bitmap.h"


#define BMP_HEADER_SIZE 54

void free_bmp_data(bmp_image_t *image) {
    if (image->data != NULL) {
        free(image->data);
        image->data = NULL;
    }
    if (image->rawdata != NULL) {
        free(image->rawdata);
        image->rawdata = NULL;
    }
}


void free_bmp_image(bmp_image_t *image) {
    free_bmp_data(image);
    if (image) {
        free(image);
    }
}

int reallocate_bmp_buffer(bmp_image_t *image, unsigned int const width, unsigned int const height) {
    free_bmp_data(image);
    if (height * width > 0) {
        image->rawdata = calloc(image->height * image->width, sizeof(pixel));
        if (image->rawdata == NULL) {
            return 1;
        }
        image->data = malloc(image->height * sizeof(pixel *));
        if (image->data == NULL) {
            free_bmp_data(image);
            return 1;
        }
        for (unsigned int i = 0; i < height; i++) {
            image->data[i] = &(image->rawdata[i * width]);
        }
    }
    return 0;
}


bmp_image_t * new_bmp_image(unsigned int const width, unsigned int const height) {
    bmp_image_t *new = malloc(sizeof(bmp_image_t));
    if (new == NULL)
        return NULL;
    new->width = width;
    new->height = height;
    new->data = NULL;
    new->rawdata = NULL;
    reallocate_bmp_buffer(new, width, height);
    return new;
}

bmp_image_soa_t * new_bmp_image_soa(unsigned int const width, unsigned int const height) {
    bmp_image_soa_t *new = malloc(sizeof(bmp_image_soa_t));
    if (new == NULL)
        return NULL;
    new->width = width;
    new->height = height;
    new->r = malloc(width*height*sizeof(unsigned char));;
    new->g = malloc(width*height*sizeof(unsigned char));;
    new->b = malloc(width*height*sizeof(unsigned char));;
    return new;
}

void free_image_soa(bmp_image_soa_t *image)
{
    free(image->r);
    free(image->g);
    free(image->b);
}



int load_bmp_image(bmp_image_t *image, char const *filename) {
    int ret = 1;
    FILE* fImage = fopen(filename, "rb");   //read the file
    if (!fImage) {
        goto failed_file;
    }

    unsigned char header[BMP_HEADER_SIZE];
    if (fread(header, sizeof(unsigned char), BMP_HEADER_SIZE, fImage) < BMP_HEADER_SIZE) {
        goto failed_read;
    }
    image->width = *(int *) &header[18];
    image->height = *(int *) &header[22];

    reallocate_bmp_buffer(image, image->width, image->height);
    if (image->rawdata == NULL) {
        goto failed_read;
    }

    int padding=0;
    while ((image->width * 3 + padding) % 4 != 0)
        padding++;

    size_t lineSize = (image->width * 3);
    size_t paddedLineSize = lineSize + padding;
    unsigned char* data = malloc(paddedLineSize * sizeof(unsigned char));

    for (unsigned int y=0; y < image->height; y++ ) {
        if (fread( data, sizeof(unsigned char), paddedLineSize, fImage) < paddedLineSize) {
            goto failed_read;
        }
        memcpy(image->data[y], data, lineSize);
    }
    ret = 0;
failed_read:
    fclose(fImage); //close the file
failed_file:
    return ret;
}

int save_bmp_image(bmp_image_t *image, char const *filename) {
    int ret = 0;
    FILE *fImage=fopen(filename,"wb");
    if(!fImage) {
        return 1;
    }

    char padBuffer[4] = {};
    const size_t dataSize = image->width * image->height * sizeof(pixel);
    size_t lineWidth = image->width * sizeof(pixel);
    size_t padding = 0;
    if (lineWidth % 4 != 0) {
        padding = 4 - (lineWidth % 4);
    }

    const size_t size= dataSize + BMP_HEADER_SIZE;

    unsigned char header[BMP_HEADER_SIZE]= {
        'B', 'M', size & 255, (size >> 8) & 255, (size >> 16) & 255, size >> 24, 0,
        0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, image->width & 255, image->width >> 8, 0,
        0, image->height & 255, image->height >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    if (fwrite(header, sizeof(unsigned char), BMP_HEADER_SIZE,fImage) < BMP_HEADER_SIZE) {
        ret = 1;
    } else {
        for (unsigned int i = 0; i < image->height; i++) {
            if (fwrite(image->data[i], sizeof(pixel), image->width ,fImage) < image->width)  {
                ret = 1;
                break;
            }
            if (padding > 0) {
                if (fwrite(padBuffer, sizeof(char), padding ,fImage) < padding)  {
                    ret = 1;
                    break;
                }
            }
        }
    }
    fclose(fImage);
    return ret;
}


int image_to_soa_image(bmp_image_t *image, bmp_image_soa_t *soa_image)
{
    if (image->width  != soa_image->width)  return 0;
    if (image->height != soa_image->height) return 0;

    for (int i = 0; i < image->height*image->width; i++) {
        soa_image->r[i] = image->rawdata[i].r;
        soa_image->g[i] = image->rawdata[i].g;
        soa_image->b[i] = image->rawdata[i].b;
    }
    return 1;
}

int soa_image_to_image(bmp_image_soa_t *soa_image, bmp_image_t *image)
{
    if (image->width  != soa_image->width)  return 0;
    if (image->height != soa_image->height) return 0;
    for (int i = 0; i < image->height*image->width; i++) {
        image->rawdata[i].r = soa_image->r[i];
        image->rawdata[i].g = soa_image->g[i];
        image->rawdata[i].b = soa_image->b[i];
    }
    return 1;
}

void swap_image_channels(unsigned char **in, unsigned char **out)
{
    unsigned char *tmp = *in;
    *in = *out;
    *out = tmp;
}

// Helper function to swap bmpImageChannel pointers
void swap_image_rawdata(pixel **one, pixel **two) {
    pixel *helper = *two;
    *two = *one;
    *one = helper;
}

void swap_bmp_image(bmp_image_t **one, bmp_image_t **two) {
    bmp_image_t *helper = *two;
    *two = *one;
    *one = helper;
}

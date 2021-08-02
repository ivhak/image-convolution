#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "bitmap.h"


#define BMP_HEADER_SIZE 54

void free_image_data(image_t *image) {
    if (image->data != NULL) {
        free(image->data);
        image->data = NULL;
    }
    if (image->rawdata != NULL) {
        free(image->rawdata);
        image->rawdata = NULL;
    }
}


void free_image(image_t *image) {
    free_image_data(image);
    if (image) {
        free(image);
    }
}

int reallocate_buffer(image_t *image, unsigned int const width, unsigned int const height) {
    free_image_data(image);
    if (height * width > 0) {
        image->rawdata = calloc(image->height * image->width, sizeof(pixel));
        if (image->rawdata == NULL) {
            return 1;
        }
        image->data = malloc(image->height * sizeof(pixel *));
        if (image->data == NULL) {
            free_image_data(image);
            return 1;
        }
        for (unsigned int i = 0; i < height; i++) {
            image->data[i] = &(image->rawdata[i * width]);
        }
    }
    return 0;
}


image_t * new_image(unsigned int const width, unsigned int const height) {
    image_t *new = malloc(sizeof(image_t));
    if (new == NULL)
        return NULL;
    new->width = width;
    new->height = height;
    new->data = NULL;
    new->rawdata = NULL;

    reallocate_buffer(new, width, height);
    return new;
}

int load_image(image_t *image, char const *filename) {
    int ret = 1;
    FILE *fimage;


    if (!(fimage = fopen(filename, "rb")))
        goto failed_file;

    // Read in the bitmap header, retrieve the width and height of the image
    unsigned char header[BMP_HEADER_SIZE];
    if (fread(header, sizeof(unsigned char), BMP_HEADER_SIZE, fimage) < BMP_HEADER_SIZE)
        goto failed_read;
    image->width  = *(int *) &header[18];
    image->height = *(int *) &header[22];

    // Resize the image so that it fits the bitmap image
    reallocate_buffer(image, image->width, image->height);
    if (image->rawdata == NULL) {
        goto failed_read;
    }

    int padding=0;
    while ((image->width * 3 + padding) % 4 != 0)
        padding++;

    size_t line_size = (image->width * 3);
    size_t padded_line_size = line_size + padding;

    unsigned char* data = malloc(padded_line_size * sizeof(unsigned char));

    for (unsigned int y = 0; y < image->height; y++ ) {
        if (fread(data, sizeof(unsigned char), padded_line_size, fimage) < padded_line_size)
            goto failed_read;
        memcpy(image->data[y], data, line_size);
    }
    ret = 0;
failed_read:
    fclose(fimage);
    free(data);
failed_file:
    return ret;
}

int save_image(image_t *image, char const *filename) {
    int ret = 0;
    FILE *fimage;

    if(!(fimage=fopen(filename,"wb")))
        return 1;

    char pad_buffer[4] = {};
    const size_t data_size = image->width * image->height * sizeof(pixel);
    size_t line_width = image->width * sizeof(pixel);
    size_t padding = 0;
    if (line_width % 4 != 0) {
        padding = 4 - (line_width % 4);
    }

    const size_t size = data_size + BMP_HEADER_SIZE;

    unsigned char header[BMP_HEADER_SIZE]= {
        'B', 'M', size & 255, (size >> 8) & 255, (size >> 16) & 255, size >> 24, 0,
        0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, image->width & 255, image->width >> 8, 0,
        0, image->height & 255, image->height >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    if (fwrite(header, sizeof(unsigned char), BMP_HEADER_SIZE,fimage) < BMP_HEADER_SIZE) {
        ret = 1;
    } else {
        for (unsigned int i = 0; i < image->height; i++) {
            if (fwrite(image->data[i], sizeof(pixel), image->width ,fimage) < image->width)  {
                ret = 1;
                break;
            }
            if (padding > 0) {
                if (fwrite(pad_buffer, sizeof(char), padding ,fimage) < padding)  {
                    ret = 1;
                    break;
                }
            }
        }
    }
    fclose(fimage);
    return ret;
}

imageSOA_t * new_imageSOA(unsigned int const width, unsigned int const height) {
    imageSOA_t *new = malloc(sizeof(imageSOA_t));
    if (new == NULL)
        return NULL;
    new->width = width;
    new->height = height;
    new->r = malloc(width*height*sizeof(unsigned char));;
    new->g = malloc(width*height*sizeof(unsigned char));;
    new->b = malloc(width*height*sizeof(unsigned char));;
    return new;
}

void free_imageSOA(imageSOA_t *image)
{
    free(image->r);
    free(image->g);
    free(image->b);
    free(image);
}


/* Copy the image data from AOS to SOA */
int image_to_imageSOA(image_t *image, imageSOA_t *soa_image)
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

/* Copy the image data from SOA to AOS */
int imageSOA_to_image(imageSOA_t *soa_image, image_t *image)
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

void swap_image_rawdata(pixel **one, pixel **two) {
    pixel *helper = *two;
    *two = *one;
    *one = helper;
}

void swap_image(image_t **one, image_t **two) {
    image_t *helper = *two;
    *two = *one;
    *one = helper;
}

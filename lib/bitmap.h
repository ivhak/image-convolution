#ifndef BITMAP_H
#define BITMAP_H

// NOTE: The struct members are in the "wrong" order as  bitmap images are
// stored "backwards" of what one might assume, i.e., the first pixel after the
// header in a bitmap file is in bottom right corner, rather than the top left.
typedef struct {
    unsigned char b;
    unsigned char g;
    unsigned char r;
} pixel;

typedef struct {
    unsigned int width;
    unsigned int height;
    pixel *rawdata;
    pixel **data;
} image_t;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned char *r;
    unsigned char *g;
    unsigned char *b;
} imageSOA_t;

image_t *new_image(unsigned int const width, unsigned int const height);
void free_image(image_t *image);
int  load_image(image_t *image, char const *filename);
int  save_image(image_t *image, char const *filename);

imageSOA_t *new_imageSOA(unsigned int const width, unsigned int const height);
void free_imageSOA(imageSOA_t *image);

int image_to_imageSOA(image_t    *image,     imageSOA_t *soa_image);
int imageSOA_to_image(imageSOA_t *image_soa, image_t    *image);

void swap_image_rawdata(pixel **one, pixel **two);
void swap_image(image_t **one, image_t **two);
void swap_image_channels(unsigned char **in, unsigned char **out);

#endif

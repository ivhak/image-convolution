#ifndef BITMAP_H
#define BITMAP_H

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
} bmp_image_t;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned char *r;
    unsigned char *g;
    unsigned char *b;
} bmp_image_soa_t;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned char *rawdata;
    unsigned char **data;
} bmp_image_channel_t;


bmp_image_t *new_bmp_image(unsigned int const width, unsigned int const height);
void free_bmp_image(bmp_image_t *image);
int load_bmp_image(bmp_image_t *image, char const *filename);
int save_bmp_image(bmp_image_t *image, char const *filename);

bmp_image_soa_t *new_bmp_image_soa(unsigned int const width, unsigned int const height);
void free_image_soa(bmp_image_soa_t *image);

int image_to_soa_image(bmp_image_t     *image,     bmp_image_soa_t *soa_image);
int soa_image_to_image(bmp_image_soa_t *soa_image, bmp_image_t     *image);

void swap_image_rawdata(pixel **one, pixel **two);
void swap_image(bmp_image_t **one, bmp_image_t **two);
void swap_image_channels(unsigned char **in, unsigned char **out);

#endif

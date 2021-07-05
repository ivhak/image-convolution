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
    unsigned char *rawdata;
    unsigned char **data;
} bmp_image_channel_t;

bmp_image_t *new_bmp_image(unsigned int const width, unsigned int const height);
void free_bmp_image(bmp_image_t *image);
int load_bmp_image(bmp_image_t *image, char const *filename);
int save_bmp_image(bmp_image_t *image, char const *filename);

bmp_image_channel_t *new_bmp_image_channel(unsigned int const width, unsigned int const height);
void free_bmp_image_channel(bmp_image_channel_t *imageChannel);

int  extract_image_channel(bmp_image_channel_t *to, bmp_image_t *from, unsigned char extract_method(pixel from));
void map_image_channels_to_image(bmp_image_t *image, bmp_image_channel_t *red, bmp_image_channel_t *green, bmp_image_channel_t *blue);

unsigned char extract_red(pixel from);
unsigned char extract_green(pixel from);
unsigned char extract_blue(pixel from);

void swap_image_rawdata(pixel **one, pixel **two);
void swap_image(bmp_image_t **one, bmp_image_t **two);
void swap_image_channels(unsigned char **in, unsigned char **out);

#endif

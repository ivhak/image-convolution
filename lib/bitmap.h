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
} bmpImage;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned char *rawdata;
    unsigned char **data;
} bmpImageChannel;

bmpImage *newBmpImage(unsigned int const width, unsigned int const height);
void freeBmpImage(bmpImage *image);
int loadBmpImage(bmpImage *image, char const *filename);
int saveBmpImage(bmpImage *image, char const *filename);

bmpImageChannel * newBmpImageChannel(unsigned int const width, unsigned int const height);
void freeBmpImageChannel(bmpImageChannel *imageChannel);
int extractImageChannel(bmpImageChannel *to, bmpImage *from, unsigned char extractMethod(pixel from));
void map_image_channels_to_image(bmpImage *image, bmpImageChannel *red, bmpImageChannel *green, bmpImageChannel *blue);

unsigned char extractRed(pixel from);
unsigned char extractGreen(pixel from);
unsigned char extractBlue(pixel from);
//
// Helper function to swap bmpImageChannel pointers
void swapImageRawdata(pixel **one, pixel **two);
void swapImage(bmpImage **one, bmpImage **two);

#endif

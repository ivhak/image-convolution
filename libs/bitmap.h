#ifndef BITMAP_H
#define BITMAP_H

typedef struct {
    float b;
    float g;
    float r;
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
int mapImageChannel(bmpImage *to, bmpImageChannel *from, pixel extractMethod(unsigned char from));
pixel mapRedChannel(unsigned char from);
unsigned char extractRedChannel(pixel from);

pixel mapRed(float from);
pixel mapGreen(float from);
pixel mapBlue(float from);
float extractRed(pixel from);
float extractGreen(pixel from);
float extractBlue(pixel from);
float extractAverage(pixel from);
pixel mapEqual(float from);

#endif

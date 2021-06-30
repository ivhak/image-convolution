#include <stdlib.h>
#include <stdio.h>

#include "../../lib/bitmap.h"

bmpImage *load_image(const char *filename)
{
    bmpImage *image = newBmpImage(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
    }

    if (loadBmpImage(image, filename) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", filename);
        freeBmpImage(image);
    }
    return image;
}

int pixel_equal(pixel pixel1, pixel pixel2) {
    return pixel1.r == pixel2.r && pixel1.g == pixel2.g && pixel1.b == pixel2.b;
}

int main(int argc, char *argv[])
{
    if (argc != 3) exit(1);
    const char *filename1 = argv[1];
    const char *filename2 = argv[2];

    bmpImage *image1 = load_image(filename1);
    bmpImage *image2 = load_image(filename2);

    bmpImage *diff = newBmpImage(image1->width, image1->height);

    if (image1->width != image2->width) {
        fprintf(stderr, "Images differ in width. \"%s\" is %dpx, \"%s\" is %dpx\n",
                filename1, image1->width, filename2, image2->width);
        exit(1);
    }

    if (image1->height != image2->height) {
        fprintf(stderr, "Images differ in width. \"%s\" is %dpx, \"%s\" is %dpx\n",
                filename1, image1->width, filename2, image2->width);
        exit(1);
    }

    pixel diff_pixel = { .r = 255, .g = 0, .b = 0 };
    int pixels_differ_count = 0;
    for (int i = 0; i < image1->height; i++)
        for (int j = 0; j < image1->width; j++) {
            pixel *pixel1 = &(image1->data[i][j]);
            pixel *pixel2 = &(image2->data[i][j]);
            if (!pixel_equal(*pixel1, *pixel2)) {
                // printf("Images differ at [%d][%d]: image1 = %x%x%x, image2 = %x%x%x\n",
                //         i, j, pixel1.r, pixel1.g, pixel1.b, pixel2.r, pixel2.g, pixel2.b);
                diff->data[i][j].r = diff_pixel.r;
                diff->data[i][j].g = diff_pixel.g;
                diff->data[i][j].b = diff_pixel.b;
                pixels_differ_count++;
            } else {
                diff->data[i][j].r = pixel1->r;
                diff->data[i][j].g = pixel1->g;
                diff->data[i][j].b = pixel1->b;
            }
        }

    printf("Total number of differing pixels: %d\n", pixels_differ_count);
    if (pixels_differ_count > 0) {
        char diff_filename[1024];
        snprintf(diff_filename, 1024, "%s-%s.diff.bmp", filename1, filename2);
        printf("More than 0 pixels differ, saving diff to %s", diff_filename);
        saveBmpImage(diff, diff_filename);
    }
}

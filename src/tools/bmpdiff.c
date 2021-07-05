#include <stdlib.h>
#include <stdio.h>

#include "../../lib/bitmap.h"
#include "tools.h"

const char * diff_filename = "diff.bmp";

int pixel_equal(pixel pixel1, pixel pixel2) {
    return pixel1.r == pixel2.r && pixel1.g == pixel2.g && pixel1.b == pixel2.b;
}

int main(int argc, char *argv[])
{
    if (argc != 3) exit(1);
    const char *filename1 = argv[1];
    const char *filename2 = argv[2];

    image_t *image1, *image2;

    if ((image1 = image_from_filename(filename1)) == NULL)  exit(1);
    if ((image2 = image_from_filename(filename2)) == NULL)  exit(1);

    image_t *diff = new_image(image1->width, image1->height);

    if (image1->width != image2->width) {
        fprintf(stderr, "Images differ in width. \"%s\" is %dpx, \"%s\" is %dpx\n",
                filename1, image1->width, filename2, image2->width);
        exit(1);
    }

    if (image1->height != image2->height) {
        fprintf(stderr, "Images differ in hiehgt. \"%s\" is %dpx, \"%s\" is %dpx\n",
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
        printf("More than 0 pixels differ, saving diff to %s\n", diff_filename);
        save_image(diff, diff_filename);
    }
}

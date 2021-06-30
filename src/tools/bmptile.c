#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
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

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: bmptile file.bmp x y\n");
        exit(1);
    }
    int tiles_x_direction = atoi(argv[1]);
    int tiles_y_direction = atoi(argv[2]);
    char *filename_in  = argv[3];
    char *filename_out = argv[4];

    bmpImage *original = load_image(filename_in);
    bmpImage *new = newBmpImage(tiles_x_direction*original->width, tiles_y_direction*original->height);

    for (int i = 0; i < new->height; i++)
        for (int j = 0; j < new->width; j++) {
            new->data[i][j] = original->data[i%(original->height)][j%(original->width)];
        }
    saveBmpImage(new, filename_out);

}
